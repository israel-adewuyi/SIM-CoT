import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed local JSONL subset from a Hugging Face dataset split."
    )
    parser.add_argument("--hf_dataset_name", type=str, required=True, help="HF dataset repo id.")
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        help="Dataset split to subset. Default: eval",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        required=True,
        help="Number of rows to export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed used when selection_strategy=shuffle. Default: 11",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        choices=("shuffle", "head"),
        default="shuffle",
        help="How to choose the subset before writing it. Default: shuffle",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Tokenizer used to measure question token length. Default: Qwen/Qwen3-4B",
    )
    parser.add_argument(
        "--max_question_tokens",
        type=int,
        required=True,
        help="Keep only rows whose question tokenized length is <= this value.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSONL path. Default: CODI/data/<dataset>_<split>_n<subset_size>_seed<seed>.jsonl",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Optional metadata JSON path. Default: <output_path>.meta.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def sanitize_path_component(value: str) -> str:
    safe_chars = []
    for ch in value.strip():
        if ch.isalnum() or ch in {".", "_", "-"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("-")
    cleaned = "".join(safe_chars).strip(".-_")
    return cleaned or "default"


def default_output_path(args: argparse.Namespace) -> Path:
    script_dir = Path(__file__).resolve().parent
    dataset_tag = sanitize_path_component(args.hf_dataset_name.replace("/", "-"))
    filename = f"{dataset_tag}_{sanitize_path_component(args.split)}_n{args.subset_size}_seed{args.seed}.jsonl"
    return script_dir / "data" / filename


def ensure_writable_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def validate_row(row: Dict[str, Any], dataset_name: str, split: str, row_idx: int) -> None:
    if "question" not in row:
        raise ValueError(
            f"{dataset_name} split={split!r} row {row_idx} is missing required field 'question'."
        )
    if "answer" not in row:
        raise ValueError(
            f"{dataset_name} split={split!r} row {row_idx} is missing required field 'answer'."
        )
    question = row["question"]
    if isinstance(question, str):
        return
    if not isinstance(question, list):
        raise ValueError(
            f"{dataset_name} split={split!r} row {row_idx} has unsupported 'question' type "
            f"{type(question).__name__}; expected str or list."
        )
    for message_idx, message in enumerate(question):
        if not isinstance(message, dict):
            raise ValueError(
                f"{dataset_name} split={split!r} row {row_idx} has invalid question message at "
                f"position {message_idx}: expected dict, got {type(message).__name__}."
            )
        missing_keys = [key for key in ("role", "content") if key not in message]
        if missing_keys:
            raise ValueError(
                f"{dataset_name} split={split!r} row {row_idx} has invalid question message at "
                f"position {message_idx}: missing keys {missing_keys}."
            )


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def render_question_for_tokenizer(question: Any, tokenizer: Any) -> str:
    if isinstance(question, str):
        return question.strip()
    if not isinstance(question, list):
        raise ValueError(
            f"Unsupported question type {type(question).__name__}; expected str or list."
        )

    normalized: List[Dict[str, str]] = []
    for idx, message in enumerate(question):
        if not isinstance(message, dict):
            raise ValueError(
                f"Question message at position {idx} must be a dict, got {type(message).__name__}."
            )
        if "role" not in message or "content" not in message:
            raise ValueError(
                f"Question message at position {idx} must contain role and content."
            )
        normalized.append(
            {
                "role": str(message["role"]).strip(),
                "content": normalize_content(message["content"]).strip(),
            }
        )

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines = [f"{message['role']}: {message['content']}" for message in normalized]
    lines.append("assistant:")
    return "\n".join(lines)


def question_token_count(question: Any, tokenizer: Any) -> int:
    rendered = render_question_for_tokenizer(question, tokenizer)
    return len(tokenizer.encode(rendered, add_special_tokens=True))


def choose_indices(total_rows: int, subset_size: int, seed: int, selection_strategy: str) -> List[int]:
    if subset_size <= 0:
        raise ValueError("--subset_size must be greater than 0.")
    if subset_size > total_rows:
        raise ValueError(
            f"--subset_size ({subset_size}) exceeds split size ({total_rows})."
        )

    if selection_strategy == "head":
        return list(range(subset_size))

    indices = list(range(total_rows))
    random.Random(seed).shuffle(indices)
    return indices[:subset_size]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if args.max_question_tokens <= 0:
        raise ValueError("--max_question_tokens must be greater than 0.")

    output_path = Path(args.output_path) if args.output_path else default_output_path(args)
    metadata_path = (
        Path(args.metadata_path)
        if args.metadata_path
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )

    ensure_writable_path(output_path, overwrite=args.overwrite)
    ensure_writable_path(metadata_path, overwrite=args.overwrite)

    dataset = load_dataset(args.hf_dataset_name)
    if args.split not in dataset:
        available_splits = ", ".join(sorted(dataset.keys()))
        raise ValueError(
            f"Hugging Face dataset {args.hf_dataset_name!r} does not contain split {args.split!r}. "
            f"Available splits: {available_splits}"
        )

    split_dataset = dataset[args.split]
    total_rows = len(split_dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        use_fast=False,
    )

    filtered_rows: List[Dict[str, Any]] = []
    filtered_source_indices: List[int] = []
    num_rows_filtered_out = 0
    for source_idx in range(total_rows):
        row = to_jsonable(dict(split_dataset[int(source_idx)]))
        if not isinstance(row, dict):
            raise ValueError(
                f"Expected dataset row at source index {source_idx} to be a dict, got {type(row).__name__}."
            )
        validate_row(
            row=row,
            dataset_name=args.hf_dataset_name,
            split=args.split,
            row_idx=source_idx,
        )
        token_count = question_token_count(row["question"], tokenizer)
        if token_count > args.max_question_tokens:
            num_rows_filtered_out += 1
            continue
        row["_source_index"] = int(source_idx)
        row["_question_token_count"] = int(token_count)
        filtered_rows.append(row)
        filtered_source_indices.append(source_idx)

    filtered_total_rows = len(filtered_rows)
    if filtered_total_rows == 0:
        raise ValueError(
            f"No rows remain after filtering {args.hf_dataset_name}:{args.split} with "
            f"--max_question_tokens={args.max_question_tokens}."
        )

    selected_indices = choose_indices(
        total_rows=filtered_total_rows,
        subset_size=args.subset_size,
        seed=args.seed,
        selection_strategy=args.selection_strategy,
    )

    subset_rows: List[Dict[str, Any]] = []
    selected_source_indices: List[int] = []
    for subset_pos, filtered_idx in enumerate(selected_indices):
        row = dict(filtered_rows[int(filtered_idx)])
        row["_subset_index"] = int(subset_pos)
        selected_source_indices.append(int(row["_source_index"]))
        subset_rows.append(row)

    write_jsonl(output_path, subset_rows)

    metadata: Dict[str, Any] = {
        "hf_dataset_name": args.hf_dataset_name,
        "split": args.split,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "selection_strategy": args.selection_strategy,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "max_question_tokens": args.max_question_tokens,
        "source_num_rows": total_rows,
        "rows_after_question_filter": filtered_total_rows,
        "rows_filtered_out_by_question_length": num_rows_filtered_out,
        "filtered_source_indices": filtered_source_indices,
        "selected_filtered_indices": selected_indices,
        "selected_source_indices": selected_source_indices,
        "output_path": str(output_path),
        "question_field": "question",
        "answer_field": "answer",
    }
    if hasattr(split_dataset, "_fingerprint"):
        metadata["source_fingerprint"] = split_dataset._fingerprint
    if hasattr(split_dataset, "features"):
        metadata["features"] = to_jsonable(split_dataset.features)

    write_json(metadata_path, metadata)

    print(
        f"Wrote {len(subset_rows)} rows from {args.hf_dataset_name}:{args.split} to {output_path}"
    )
    print(f"Wrote subset metadata to {metadata_path}")


if __name__ == "__main__":
    main()
