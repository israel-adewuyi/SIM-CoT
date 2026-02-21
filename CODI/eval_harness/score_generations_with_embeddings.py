#!/usr/bin/env python3
"""Score CODI generation outputs with embedding similarity."""

import argparse
import hashlib
import json
import logging
import statistics
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return str(content["content"])
        return str(content)
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type", None) == "text" and item.get("text", None):
                    chunks.append(str(item["text"]))
                elif "text" in item:
                    chunks.append(str(item["text"]))
                elif "content" in item:
                    chunks.append(str(item["content"]))
            else:
                chunks.append(str(item))
        return "\n".join([c for c in chunks if c.strip()])
    return str(content)


def _normalise_messages(raw_messages: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)

    if isinstance(raw_messages, dict):
        for key in ("messages", "conversation", "conversations", "chat"):
            value = raw_messages.get(key, None)
            if isinstance(value, list):
                raw_messages = value
                break

    if not isinstance(raw_messages, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "")).lower().strip()
        if role not in {"system", "user", "assistant"}:
            continue

        content = _extract_text_content(msg.get("content", "")).strip()
        if content == "":
            continue

        raw_loss_mask = msg.get("loss_mask", 1 if role == "assistant" else 0)
        if isinstance(raw_loss_mask, str):
            lm = raw_loss_mask.strip().lower()
            loss_mask = 1 if lm in {"1", "true", "yes"} else 0
        elif isinstance(raw_loss_mask, bool):
            loss_mask = int(raw_loss_mask)
        elif isinstance(raw_loss_mask, (int, float)):
            loss_mask = 1 if int(raw_loss_mask) == 1 else 0
        else:
            loss_mask = 1 if role == "assistant" else 0

        normalized.append({"role": role, "content": content, "loss_mask": loss_mask})

    return normalized


def _extract_reference_text(example: Dict[str, Any], messages_field: str) -> Optional[str]:
    if messages_field not in example:
        return None

    messages = _normalise_messages(example[messages_field])
    if not messages:
        return None

    masked_targets = [m["content"] for m in messages if int(m.get("loss_mask", 0)) == 1]
    if masked_targets:
        text = masked_targets[-1].strip()
        return text if text else None

    assistant_messages = [m["content"] for m in messages if m.get("role") == "assistant"]
    if assistant_messages:
        text = assistant_messages[-1].strip()
        return text if text else None

    return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    out = str(value).strip()
    return out if out else None


def _extract_generated_text(row: Dict[str, Any]) -> str:
    for key in (
        "generated_text",
        "prediction",
        "pred",
        "response",
        "output",
        "text",
    ):
        value = row.get(key, None)
        if value is None:
            continue
        text = _extract_text_content(value).strip()
        if text:
            return text
    return ""


def _read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield line_no, obj


def _generation_key(row: Dict[str, Any]) -> str:
    sample_id = _safe_str(row.get("sample_id", None))
    if sample_id is not None:
        return f"sample_id::{sample_id}"

    trajectory_id = _safe_str(row.get("trajectory_id", None))
    assistant_turn_index = _safe_int(row.get("assistant_turn_index", None))
    target_message_index = _safe_int(row.get("target_message_index", None))
    if trajectory_id is not None and assistant_turn_index is not None and target_message_index is not None:
        return f"traj::{trajectory_id}::{assistant_turn_index}::{target_message_index}"

    example_index = _safe_int(row.get("example_index", None))
    if example_index is not None:
        return f"example_index::{example_index}"

    prompt = _safe_str(row.get("prompt", None))
    if prompt:
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        return f"prompt_sha1::{prompt_hash}"

    file_name = _safe_str(row.get("__source_file", "unknown_file"))
    line_no = _safe_int(row.get("__line_no", -1))
    return f"fallback::{file_name}::{line_no}"


@dataclass
class ReferenceIndex:
    total_rows: int
    usable_rows: int
    by_example_index: Dict[int, Dict[str, Any]]
    by_sample_id: Dict[str, Dict[str, Any]]
    by_composite_id: Dict[Tuple[str, int, int], Dict[str, Any]]


def _build_reference_index(dataset_name: str, split: str, messages_field: str) -> ReferenceIndex:
    if Path(dataset_name).is_file():
        ds = load_dataset("json", data_files=dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    by_example_index: Dict[int, Dict[str, Any]] = {}
    by_sample_id: Dict[str, Dict[str, Any]] = {}
    by_composite_id: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    usable_rows = 0
    for idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        reference_text = _extract_reference_text(row, messages_field=messages_field)
        if not reference_text:
            continue

        sample_id = _safe_str(row.get("sample_id", None))
        trajectory_id = _safe_str(row.get("trajectory_id", None))
        assistant_turn_index = _safe_int(row.get("assistant_turn_index", None))
        target_message_index = _safe_int(row.get("target_message_index", None))

        ref_item = {
            "example_index": idx,
            "sample_id": sample_id,
            "trajectory_id": trajectory_id,
            "assistant_turn_index": assistant_turn_index,
            "target_message_index": target_message_index,
            "reference_text": reference_text,
        }

        usable_rows += 1
        by_example_index[idx] = ref_item

        if sample_id is not None:
            by_sample_id[sample_id] = ref_item

        if (
            trajectory_id is not None
            and assistant_turn_index is not None
            and target_message_index is not None
        ):
            by_composite_id[(trajectory_id, assistant_turn_index, target_message_index)] = ref_item

    return ReferenceIndex(
        total_rows=len(ds),
        usable_rows=usable_rows,
        by_example_index=by_example_index,
        by_sample_id=by_sample_id,
        by_composite_id=by_composite_id,
    )


def _load_generations(paths: List[str], dedupe_policy: str) -> Tuple[List[Dict[str, Any]], int]:
    rows_raw_count = 0
    deduped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for p in paths:
        path = Path(p)
        for line_no, row in _read_jsonl(path):
            rows_raw_count += 1
            item = dict(row)
            item["__source_file"] = str(path)
            item["__line_no"] = line_no

            key = _generation_key(item)
            item["sample_key"] = key

            if key in deduped:
                if dedupe_policy == "first":
                    continue
                deduped[key] = item
                deduped.move_to_end(key)
            else:
                deduped[key] = item

    return list(deduped.values()), rows_raw_count


def _find_reference(row: Dict[str, Any], ref_index: ReferenceIndex) -> Tuple[Optional[Dict[str, Any]], str]:
    example_index = _safe_int(row.get("example_index", None))
    if example_index is not None and example_index in ref_index.by_example_index:
        return ref_index.by_example_index[example_index], "example_index"

    sample_id = _safe_str(row.get("sample_id", None))
    if sample_id is not None and sample_id in ref_index.by_sample_id:
        return ref_index.by_sample_id[sample_id], "sample_id"

    trajectory_id = _safe_str(row.get("trajectory_id", None))
    assistant_turn_index = _safe_int(row.get("assistant_turn_index", None))
    target_message_index = _safe_int(row.get("target_message_index", None))
    if (
        trajectory_id is not None
        and assistant_turn_index is not None
        and target_message_index is not None
        and (trajectory_id, assistant_turn_index, target_message_index) in ref_index.by_composite_id
    ):
        return (
            ref_index.by_composite_id[(trajectory_id, assistant_turn_index, target_message_index)],
            "composite_id",
        )

    return None, "missing"


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype `{dtype_arg}`. Use one of: {', '.join(DTYPE_MAP)}")

    dtype = DTYPE_MAP[dtype_arg]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        logging.warning("dtype=%s on CPU is not supported for many models; using float32.", dtype_arg)
        return torch.float32
    return dtype


def _encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
    batch_size: int,
    pooling: str,
    normalize: bool,
) -> torch.Tensor:
    embs: List[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

            if pooling == "mean":
                attention_mask = encoded.get("attention_mask", None)
                if attention_mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    masked_hidden = hidden * mask
                    denom = mask.sum(dim=1).clamp(min=1.0)
                    pooled = masked_hidden.sum(dim=1) / denom
            elif pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling mode: {pooling}")

            pooled = pooled.to(dtype=torch.float32)
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

        embs.append(pooled.cpu())

    return torch.cat(embs, dim=0)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_float(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return float(v)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score model generations using HF embedding similarity.")
    parser.add_argument(
        "--generations",
        nargs="+",
        required=True,
        help="One or more generation JSONL files from CODI/test.py messages mode.",
    )
    parser.add_argument("--dataset_name", required=True, help="HF dataset id or local JSONL file.")
    parser.add_argument("--dataset_split", default="train", help="Dataset split to load.")
    parser.add_argument("--messages_field", default="messages", help="Messages field in dataset rows.")
    parser.add_argument("--output_dir", required=True, help="Directory for scored outputs.")

    parser.add_argument("--scoring_model_id", required=True, help="HF embedding model id.")
    parser.add_argument("--scoring_model_revision", default="", help="HF model revision/tag/commit.")
    parser.add_argument("--scoring_pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--scoring_normalize", type=_parse_bool, default=True)
    parser.add_argument("--scoring_max_length", type=int, default=512)
    parser.add_argument("--scoring_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--scoring_device", default="auto", help="e.g., auto, cuda:0, cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--similarity_metric", choices=["cosine"], default="cosine")
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--dedupe_policy", choices=["first", "last"], default="last")
    parser.add_argument("--scoring_version", default="embedding_harness_v1")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading reference dataset from `%s` (%s)", args.dataset_name, args.dataset_split)
    ref_index = _build_reference_index(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        messages_field=args.messages_field,
    )
    logging.info(
        "Reference rows: total=%s usable=%s",
        ref_index.total_rows,
        ref_index.usable_rows,
    )

    logging.info("Loading generation files: %s", ", ".join(args.generations))
    generation_rows, generation_raw_count = _load_generations(
        paths=args.generations,
        dedupe_policy=args.dedupe_policy,
    )
    logging.info(
        "Generation rows: raw=%s unique_after_dedupe=%s",
        generation_raw_count,
        len(generation_rows),
    )

    scored_at = datetime.now(timezone.utc).isoformat()

    rows_for_output: List[Dict[str, Any]] = []
    refs_for_scoring: List[str] = []
    preds_for_scoring: List[str] = []
    scoring_indices: List[int] = []

    match_counter: Counter = Counter()

    for row in generation_rows:
        reference_item, match_type = _find_reference(row, ref_index)
        match_counter[match_type] += 1

        generated_text = _extract_generated_text(row)
        reference_text = reference_item["reference_text"] if reference_item is not None else ""

        output_row = {
            "sample_key": row.get("sample_key"),
            "generation_source_file": row.get("__source_file"),
            "generation_line_no": row.get("__line_no"),
            "example_index": row.get("example_index", reference_item["example_index"] if reference_item else None),
            "sample_id": row.get("sample_id", reference_item["sample_id"] if reference_item else None),
            "trajectory_id": row.get("trajectory_id", reference_item["trajectory_id"] if reference_item else None),
            "assistant_turn_index": row.get(
                "assistant_turn_index",
                reference_item["assistant_turn_index"] if reference_item else None,
            ),
            "target_message_index": row.get(
                "target_message_index",
                reference_item["target_message_index"] if reference_item else None,
            ),
            "generated_text": generated_text,
            "reference_text": reference_text,
            "match_type": match_type,
            "scoring_provider": "huggingface",
            "scoring_model_id": args.scoring_model_id,
            "scoring_model_revision": args.scoring_model_revision if args.scoring_model_revision else None,
            "scoring_pooling": args.scoring_pooling,
            "scoring_normalize": args.scoring_normalize,
            "scoring_max_length": args.scoring_max_length,
            "scoring_dtype": args.scoring_dtype,
            "scoring_device": args.scoring_device,
            "similarity_metric": args.similarity_metric,
            "similarity_value": None,
            "threshold": args.threshold,
            "pass_fail": None,
            "scoring_version": args.scoring_version,
            "scored_at": scored_at,
            "error": None,
        }

        if not reference_text:
            output_row["error"] = "missing_reference"
        elif not generated_text:
            output_row["error"] = "empty_generation"
        else:
            scoring_indices.append(len(rows_for_output))
            refs_for_scoring.append(reference_text)
            preds_for_scoring.append(generated_text)

        rows_for_output.append(output_row)

    if scoring_indices:
        logging.info("Loading scoring model `%s`", args.scoring_model_id)
        device = _resolve_device(args.scoring_device)
        dtype = _resolve_dtype(args.scoring_dtype, device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.scoring_model_id,
            revision=args.scoring_model_revision or None,
            use_fast=True,
        )
        model = AutoModel.from_pretrained(
            args.scoring_model_id,
            revision=args.scoring_model_revision or None,
        )
        model = model.to(device)
        if dtype != torch.float32:
            model = model.to(dtype=dtype)
        model.eval()

        logging.info("Encoding references (%s rows)", len(refs_for_scoring))
        ref_embs = _encode_texts(
            texts=refs_for_scoring,
            tokenizer=tokenizer,
            model=model,
            device=device,
            dtype=dtype,
            max_length=args.scoring_max_length,
            batch_size=args.batch_size,
            pooling=args.scoring_pooling,
            normalize=args.scoring_normalize,
        )

        logging.info("Encoding generations (%s rows)", len(preds_for_scoring))
        pred_embs = _encode_texts(
            texts=preds_for_scoring,
            tokenizer=tokenizer,
            model=model,
            device=device,
            dtype=dtype,
            max_length=args.scoring_max_length,
            batch_size=args.batch_size,
            pooling=args.scoring_pooling,
            normalize=args.scoring_normalize,
        )

        if args.similarity_metric != "cosine":
            raise ValueError(f"Unsupported similarity metric: {args.similarity_metric}")

        if args.scoring_normalize:
            sims = (ref_embs * pred_embs).sum(dim=1)
        else:
            sims = F.cosine_similarity(ref_embs, pred_embs, dim=1)

        for local_idx, sim in enumerate(sims.tolist()):
            out_idx = scoring_indices[local_idx]
            row = rows_for_output[out_idx]
            row["similarity_value"] = float(sim)
            if args.threshold is not None:
                row["pass_fail"] = bool(sim >= args.threshold)
            row["error"] = None

    scored_values = [
        float(row["similarity_value"])
        for row in rows_for_output
        if row.get("similarity_value", None) is not None
    ]
    pass_values = [
        bool(row["pass_fail"])
        for row in rows_for_output
        if row.get("pass_fail", None) is not None
    ]

    summary = {
        "scored_at": scored_at,
        "scoring_version": args.scoring_version,
        "generation_files": args.generations,
        "dedupe_policy": args.dedupe_policy,
        "generated_rows_raw": generation_raw_count,
        "generated_rows_unique": len(generation_rows),
        "reference_rows_total": ref_index.total_rows,
        "reference_rows_usable": ref_index.usable_rows,
        "rows_scored": len(scored_values),
        "rows_missing_reference": sum(1 for row in rows_for_output if row.get("error") == "missing_reference"),
        "rows_empty_generation": sum(1 for row in rows_for_output if row.get("error") == "empty_generation"),
        "match_counts": dict(match_counter),
        "coverage_unique_vs_usable": _to_float(
            len(generation_rows) / ref_index.usable_rows if ref_index.usable_rows > 0 else None
        ),
        "coverage_scored_vs_usable": _to_float(
            len(scored_values) / ref_index.usable_rows if ref_index.usable_rows > 0 else None
        ),
        "similarity_metric": args.similarity_metric,
        "similarity_mean": _to_float(_safe_mean(scored_values)),
        "similarity_median": _to_float(_safe_median(scored_values)),
        "similarity_min": _to_float(min(scored_values) if scored_values else None),
        "similarity_max": _to_float(max(scored_values) if scored_values else None),
        "threshold": args.threshold,
        "pass_rate": _to_float(_safe_mean([1.0 if x else 0.0 for x in pass_values]) if pass_values else None),
        "scoring_provider": "huggingface",
        "scoring_model_id": args.scoring_model_id,
        "scoring_model_revision": args.scoring_model_revision if args.scoring_model_revision else None,
        "scoring_pooling": args.scoring_pooling,
        "scoring_normalize": args.scoring_normalize,
        "scoring_max_length": args.scoring_max_length,
        "scoring_dtype": args.scoring_dtype,
        "scoring_device": args.scoring_device,
    }

    scored_jsonl_path = output_dir / "scored_rows.jsonl"
    summary_path = output_dir / "summary.json"
    _write_jsonl(scored_jsonl_path, rows_for_output)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Wrote scored rows to %s", scored_jsonl_path)
    logging.info("Wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
