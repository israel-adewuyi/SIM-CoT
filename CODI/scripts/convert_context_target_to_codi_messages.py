#!/usr/bin/env python3
"""Convert context_target JSONL into CODI message-style JSONL."""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _extract_text(content: Any) -> str:
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
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(str(item["text"]))
                elif "text" in item:
                    chunks.append(str(item["text"]))
                elif "content" in item:
                    chunks.append(str(item["content"]))
            else:
                chunks.append(str(item))
        return "\n".join([x for x in chunks if x.strip()])
    return str(content)


def _normalise_prompt_messages(raw_messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_messages, list):
        return []
    out: List[Dict[str, Any]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        text = _extract_text(msg.get("content", "")).strip()
        if not text:
            continue
        out.append({"role": role, "content": text, "loss_mask": 0})
    return out


def _normalise_target_message(raw_target: Any, fallback_answer: Any) -> Optional[Dict[str, Any]]:
    role = "assistant"
    text = ""
    if isinstance(raw_target, dict):
        role = str(raw_target.get("role", "assistant")).strip().lower() or "assistant"
        text = _extract_text(raw_target.get("content", "")).strip()
    elif isinstance(raw_target, str):
        text = raw_target.strip()
    else:
        text = _extract_text(raw_target).strip()

    if not text:
        text = _extract_text(fallback_answer).strip()
    if not text:
        return None
    if role not in {"assistant", "system", "user"}:
        role = "assistant"
    return {"role": role, "content": text, "loss_mask": 1}


def _convert_row(row: Dict[str, Any], messages_field: str, keep_original_fields: bool) -> Tuple[Optional[Dict[str, Any]], str]:
    prompt_messages = _normalise_prompt_messages(row.get("question"))
    if not prompt_messages:
        return None, "bad_question"

    target_msg = _normalise_target_message(row.get("target"), row.get("answer"))
    if target_msg is None:
        return None, "missing_target"

    out: Dict[str, Any] = {
        messages_field: prompt_messages + [target_msg],
    }
    if keep_original_fields:
        out.update(row)
    else:
        for key in (
            "sample_id",
            "trajectory_id",
            "assistant_turn_index",
            "target_message_index",
            "source_path",
        ):
            if key in row:
                out[key] = row[key]
    return out, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert context_target JSONL to CODI messages JSONL.")
    parser.add_argument("--input", required=True, help="Input context_target JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path with CODI messages schema.")
    parser.add_argument("--messages-field", default="messages", help="Output field name for message list.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows to process.")
    parser.add_argument(
        "--keep-original-fields",
        action="store_true",
        help="Keep all original source fields in the output rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output} (use --overwrite)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    stats: Counter = Counter()
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if args.max_rows is not None and stats["read"] >= args.max_rows:
                break
            stats["read"] += 1
            line = line.strip()
            if not line:
                stats["skip_empty_line"] += 1
                continue
            try:
                row = json.loads(line)
            except Exception:
                stats["skip_bad_json"] += 1
                continue
            if not isinstance(row, dict):
                stats["skip_not_object"] += 1
                continue

            converted, reason = _convert_row(
                row,
                messages_field=args.messages_field,
                keep_original_fields=args.keep_original_fields,
            )
            if converted is None:
                stats[f"skip_{reason}"] += 1
                continue

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            stats["written"] += 1

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"rows_read={stats['read']}")
    print(f"rows_written={stats['written']}")
    skipped_keys = sorted([k for k in stats.keys() if k.startswith("skip_")])
    for key in skipped_keys:
        print(f"{key}={stats[key]}")

    if stats["written"] == 0:
        raise RuntimeError("No rows were written. Check input schema and converter assumptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
