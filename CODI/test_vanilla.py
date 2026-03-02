import argparse
import json
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="local-jsonl")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model_max_length", type=int, default=16384)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--greedy", type=str2bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--inf_num_iterations", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)
    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid row type in {path} at line {line_no}: {type(obj).__name__}")
            rows.append(obj)
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def render_chat_prompt(messages: Any, tokenizer: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("Each sample 'question' must be a list of {role, content}.")
    normalized: List[Dict[str, str]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message at position {idx} must be an object.")
        if "role" not in message or "content" not in message:
            raise ValueError(f"Message at position {idx} must contain role and content.")
        normalized.append(
            {
                "role": str(message["role"]).strip(),
                "content": normalize_content(message["content"]).strip(),
            }
        )
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
    lines = [f"{message['role']}: {message['content']}" for message in normalized]
    lines.append("assistant:")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.data_name != "local-jsonl":
        raise ValueError(f"Only data_name=local-jsonl is supported, got: {args.data_name}")
    if not args.data_path.endswith(".jsonl"):
        raise ValueError(f"--data_path must point to a .jsonl file, got: {args.data_path}")
    if not os.path.isfile(args.data_path):
        raise ValueError(f"JSONL file does not exist: {args.data_path}")

    rows = read_jsonl(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.token,
        model_max_length=args.model_max_length,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
        device = torch.device("cuda")
    else:
        torch_dtype = torch.float32
        device = torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        **model_kwargs,
    )
    model = model.to(device)
    model.eval()

    all_prompts: List[str] = []
    for sample_idx, row in enumerate(rows):
        if "question" not in row:
            raise ValueError(f"Sample at index {sample_idx} is missing required key 'question'.")
        all_prompts.append(render_chat_prompt(row["question"], tokenizer))

    for iteration_idx in range(args.inf_num_iterations):
        set_seed(args.seed + iteration_idx)
        predictions: List[Dict[str, Any]] = []

        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            batch_prompts = all_prompts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.model_max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": not args.greedy,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if not args.greedy:
                gen_kwargs.update(
                    {
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                    }
                )

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    **gen_kwargs,
                )

            for row_offset, row in enumerate(batch_rows):
                input_len = int(encoded["attention_mask"][row_offset].sum().item())
                new_tokens = output_ids[row_offset, input_len:]
                prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions.append(
                    {
                        "index": start + row_offset,
                        "question": row["question"],
                        "answer": row.get("answer"),
                        "prediction": prediction,
                    }
                )

        output_file = os.path.join(
            args.output_dir,
            "predictions",
            f"{args.data_name}_iter_{iteration_idx}.jsonl",
        )
        write_jsonl(output_file, predictions)
        print(f"Saved {len(predictions)} generations to {output_file}")


if __name__ == "__main__":
    main()
