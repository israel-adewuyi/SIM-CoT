#!/usr/bin/env python3
"""Score local generation JSONL outputs with embedding cosine similarity."""

import argparse
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_SCORING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
REQUIRED_FIELDS = ("index", "question", "answer", "prediction")

# Extension hook: register future auxiliary scorers here.
# Signature: scorer(generated_text: str, reference_text: str) -> float
AUX_SCORERS: Dict[str, Callable[[str, str], float]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generation JSONL using embedding cosine similarity."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to generation JSONL with fields: index, question, answer, prediction.",
    )
    parser.add_argument(
        "--scoring_model",
        default=DEFAULT_SCORING_MODEL,
        help=f"Embedding model id (default: {DEFAULT_SCORING_MODEL}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--scoring_max_length",
        type=int,
        default=512,
        help="Max token length used for embedding tokenization.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Default: <predictions_parent>/scores/<predictions_stem>/",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_jsonl_strict(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Invalid row type in {path} at line {line_no}: expected object, got {type(obj).__name__}"
                )
            rows.append(obj)
    return rows


def validate_row_schema(row: Dict[str, Any], row_idx: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"Row {row_idx} is missing required field(s): {missing}. "
            f"Expected fields include {list(REQUIRED_FIELDS)}."
        )


def resolve_output_dir(predictions_path: Path, output_dir: Optional[str]) -> Path:
    if output_dir:
        return Path(output_dir)
    return predictions_path.parent / "scores" / predictions_path.stem


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_texts(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
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
            pooled = hidden[:, 0, :]  # fixed CLS pooling
            pooled = F.normalize(pooled.float(), p=2, dim=1)  # fixed normalization
            embeddings.append(pooled.cpu())
    if not embeddings:
        return torch.empty(0, 0)
    return torch.cat(embeddings, dim=0)


def compute_cosine(ref_embs: torch.Tensor, pred_embs: torch.Tensor) -> torch.Tensor:
    # Embeddings are already normalized; cosine becomes dot product.
    return (ref_embs * pred_embs).sum(dim=1)


def score_rows(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    model: Any,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scored_rows: List[Dict[str, Any]] = []
    refs_for_scoring: List[str] = []
    preds_for_scoring: List[str] = []
    scoring_indices: List[int] = []
    scored_at = datetime.now(timezone.utc).isoformat()

    for row_idx, row in enumerate(rows):
        validate_row_schema(row, row_idx)
        question = normalize_text(row["question"])
        reference_text = normalize_text(row["answer"])
        generated_text = normalize_text(row["prediction"])

        aux_scores: Dict[str, float] = {}
        aux_errors: Dict[str, str] = {}

        output_row: Dict[str, Any] = {
            "index": row["index"],
            "question": question,
            "reference_text": reference_text,
            "generated_text": generated_text,
            "prompt_tokens": len(tokenizer.encode(question, add_special_tokens=False)),
            "reference_tokens": len(tokenizer.encode(reference_text, add_special_tokens=False)),
            "generated_tokens": len(tokenizer.encode(generated_text, add_special_tokens=False)),
            "similarity": None,
            "error": None,
            "scoring_model": cfg["scoring_model"],
            "scoring_max_length": cfg["scoring_max_length"],
            "scored_at": scored_at,
            "aux_scores": aux_scores,
            "aux_errors": aux_errors,
        }

        # Extension-ready scaffold for future auxiliary metrics.
        for metric_name, scorer in AUX_SCORERS.items():
            try:
                aux_scores[metric_name] = float(scorer(generated_text, reference_text))
            except Exception as exc:  # noqa: BLE001
                aux_errors[metric_name] = str(exc)

        if not reference_text:
            output_row["error"] = "empty_reference"
        elif not generated_text:
            output_row["error"] = "empty_generation"
        else:
            scoring_indices.append(len(scored_rows))
            refs_for_scoring.append(reference_text)
            preds_for_scoring.append(generated_text)

        scored_rows.append(output_row)

    if scoring_indices:
        ref_embs = encode_texts(
            refs_for_scoring,
            tokenizer=tokenizer,
            model=model,
            batch_size=cfg["batch_size"],
            max_length=cfg["scoring_max_length"],
            device=cfg["device"],
        )
        pred_embs = encode_texts(
            preds_for_scoring,
            tokenizer=tokenizer,
            model=model,
            batch_size=cfg["batch_size"],
            max_length=cfg["scoring_max_length"],
            device=cfg["device"],
        )
        similarities = compute_cosine(ref_embs, pred_embs).tolist()
        for local_idx, sim in enumerate(similarities):
            row_out_idx = scoring_indices[local_idx]
            scored_rows[row_out_idx]["similarity"] = float(sim)

    return scored_rows


def compute_summary(
    scored_rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    paths: Dict[str, str],
) -> Dict[str, Any]:
    similarities = [
        float(row["similarity"])
        for row in scored_rows
        if row.get("similarity") is not None
    ]
    rows_errored = sum(1 for row in scored_rows if row.get("error") is not None)

    summary: Dict[str, Any] = {
        "rows_total": len(scored_rows),
        "rows_scored": len(similarities),
        "rows_errored": rows_errored,
        "similarity_mean": float(statistics.mean(similarities)) if similarities else None,
        "similarity_median": float(statistics.median(similarities)) if similarities else None,
        "similarity_min": float(min(similarities)) if similarities else None,
        "similarity_max": float(max(similarities)) if similarities else None,
        "similarity_std": float(statistics.pstdev(similarities)) if similarities else None,
        "scoring_model": cfg["scoring_model"],
        "batch_size": cfg["batch_size"],
        "scoring_max_length": cfg["scoring_max_length"],
        "predictions_path": paths["predictions_path"],
        "output_dir": paths["output_dir"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if args.scoring_max_length <= 0:
        raise ValueError("--scoring_max_length must be > 0.")

    predictions_path = Path(args.predictions)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file does not exist: {predictions_path}")

    output_dir = resolve_output_dir(predictions_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Reading predictions from %s", predictions_path)
    rows = read_jsonl_strict(predictions_path)
    logging.info("Loaded %d rows", len(rows))

    device = resolve_device()
    logging.info("Loading scoring model %s on %s", args.scoring_model, device)
    tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(args.scoring_model, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    cfg = {
        "scoring_model": args.scoring_model,
        "batch_size": args.batch_size,
        "scoring_max_length": args.scoring_max_length,
        "device": device,
    }
    scored_rows = score_rows(rows, tokenizer=tokenizer, model=model, cfg=cfg)
    summary = compute_summary(
        scored_rows,
        cfg=cfg,
        paths={
            "predictions_path": str(predictions_path),
            "output_dir": str(output_dir),
        },
    )

    scored_rows_path = output_dir / "scored_rows.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(scored_rows_path, scored_rows)
    write_json(summary_path, summary)

    logging.info("Wrote %s", scored_rows_path)
    logging.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
