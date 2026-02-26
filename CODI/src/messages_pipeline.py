import ast
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import transformers


ANSWER_FENCE = "```bash"


def is_trainable_message(msg: Dict[str, Any]) -> bool:
    loss_mask = msg.get("loss_mask", None)
    if loss_mask is not None:
        if isinstance(loss_mask, str):
            normalized = loss_mask.strip().lower()
            if normalized in {"true", "yes"}:
                return True
            if normalized in {"false", "no"}:
                return False
        return int(loss_mask) == 1
    return str(msg.get("role", "")).lower() == "assistant"


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
                if item.get("type", None) == "text":
                    text = item.get("text", "")
                    if text:
                        chunks.append(str(text))
                elif "text" in item:
                    chunks.append(str(item["text"]))
                elif "content" in item:
                    chunks.append(str(item["content"]))
            else:
                chunks.append(str(item))
        return "\n".join([c for c in chunks if c.strip()])
    return str(content)


def _coerce_messages(raw_messages: Any) -> List[Dict[str, Any]]:
    if raw_messages is None:
        return []
    if isinstance(raw_messages, str):
        parsed = None
        try:
            parsed = json.loads(raw_messages)
        except Exception:
            try:
                parsed = ast.literal_eval(raw_messages)
            except Exception:
                return []
        raw_messages = parsed

    if isinstance(raw_messages, dict):
        for key in ("messages", "conversation", "conversations", "chat"):
            value = raw_messages.get(key, None)
            if isinstance(value, list):
                raw_messages = value
                break

    if not isinstance(raw_messages, list):
        return []

    normalized_list: List[Dict[str, Any]] = []
    for msg in raw_messages:
        if isinstance(msg, dict):
            normalized_list.append(msg)
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            normalized_list.append({"role": msg[0], "content": msg[1]})
    return normalized_list


def _split_assistant_content(content: str) -> Optional[Tuple[str, str]]:
    fence_idx = content.find(ANSWER_FENCE)
    if fence_idx < 0:
        return None
    think_text = content[:fence_idx].strip()
    answer_text = content[fence_idx:].strip()
    if answer_text == "":
        return None
    return think_text, answer_text


def normalise_messages_for_training(
    raw_messages: Any,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    coerced_messages = _coerce_messages(raw_messages)
    if len(coerced_messages) == 0:
        return [], "empty_or_unparseable_messages"

    normalized: List[Dict[str, Any]] = []
    for msg in coerced_messages:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "")).lower().strip()
        content = _extract_text_content(msg.get("content", ""))
        if role == "" or content == "":
            continue

        loss_mask = msg.get("loss_mask", 1 if role == "assistant" else 0)
        normalized_message: Dict[str, Any] = {
            "role": role,
            "content": content,
            "loss_mask": loss_mask,
        }

        if is_trainable_message(normalized_message):
            split = _split_assistant_content(content)
            if split is None:
                return [], "missing_answer_fence"
            think_text, answer_text = split
            if think_text:
                normalized.append(
                    {
                        "role": role,
                        "content": think_text,
                        "loss_mask": 0,
                        "segment": "think",
                    }
                )
            normalized.append(
                {
                    "role": role,
                    "content": answer_text,
                    "loss_mask": 1,
                    "segment": "answer",
                }
            )
        else:
            normalized.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 0,
                    "segment": "context",
                }
            )

    if len(normalized) == 0:
        return [], "empty_normalized_messages"

    if not any(is_trainable_message(msg) for msg in normalized):
        return [], "no_answer_segment"

    return normalized, None


def _template_payload(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"role": str(msg["role"]), "content": str(msg["content"])} for msg in messages]


def build_tokens_from_messages(
    messages: Sequence[Dict[str, Any]],
    tokenizer: "transformers.PreTrainedTokenizer",
    ignore_index: int,
) -> Tuple[List[int], List[int]]:
    use_chat_template = hasattr(tokenizer, "apply_chat_template") and bool(
        getattr(tokenizer, "chat_template", None)
    )
    if use_chat_template:
        full_ids: List[int] = []
        full_labels: List[int] = []
        for idx, msg in enumerate(messages):
            partial_ids = tokenizer.apply_chat_template(
                _template_payload(messages[: idx + 1]),
                tokenize=True,
                add_generation_prompt=False,
            )
            if not isinstance(partial_ids, list):
                if hasattr(partial_ids, "tolist"):
                    partial_ids = partial_ids.tolist()
                else:
                    partial_ids = list(partial_ids)

            if partial_ids[: len(full_ids)] != full_ids:
                use_chat_template = False
                break

            delta = partial_ids[len(full_ids) :]
            full_ids = partial_ids
            if is_trainable_message(msg):
                full_labels.extend(delta)
            else:
                full_labels.extend([ignore_index] * len(delta))

        if use_chat_template:
            return full_ids, full_labels

    full_ids = []
    full_labels = []
    for idx, msg in enumerate(messages):
        role = str(msg["role"])
        content = str(msg["content"]).strip()
        prefix = "System" if role == "system" else ("User" if role == "user" else "Assistant")
        text = f"{prefix}: {content}\n"
        msg_ids = tokenizer.encode(text, add_special_tokens=(idx == 0))
        full_ids.extend(msg_ids)
        if is_trainable_message(msg):
            full_labels.extend(msg_ids)
        else:
            full_labels.extend([ignore_index] * len(msg_ids))
    return full_ids, full_labels


def _split_reasoning_steps(reasoning_text: str) -> List[str]:
    text = reasoning_text.strip()
    if not text:
        return []

    line_steps = []
    for line in text.splitlines():
        step = line.strip()
        if not step:
            continue
        step = re.sub(r"^[-*]\s+", "", step)
        step = re.sub(r"^\d+[\)\.\:\-]\s+", "", step)
        if step:
            line_steps.append(step)

    if len(line_steps) >= 2:
        return line_steps

    sentence_steps = [
        s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()
    ]
    if len(sentence_steps) >= 2:
        return sentence_steps

    chunk_steps = [s.strip() for s in re.split(r";\s+|,\s+", text) if s.strip()]
    if len(chunk_steps) >= 2:
        return chunk_steps

    return [text]


def extract_explain_steps_ids(
    messages: Sequence[Dict[str, Any]],
    tokenizer: "transformers.PreTrainedTokenizer",
    eot_id: int,
    num_latent: int,
    pad_id: int,
) -> List[List[int]]:
    max_steps = num_latent + 1
    step_texts: List[str] = []

    for msg in messages:
        if str(msg.get("segment", "")) != "think":
            continue
        content = str(msg.get("content", ""))
        think_blocks = re.findall(
            r"<think>(.*?)</think>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if think_blocks:
            for block in think_blocks:
                step_texts.extend(_split_reasoning_steps(block))
        else:
            cleaned = re.sub(r"<[^>]+>", " ", content).strip()
            if cleaned:
                step_texts.extend(_split_reasoning_steps(cleaned))

    if len(step_texts) > max_steps:
        step_texts = step_texts[: max_steps - 1] + [
            " ".join(step_texts[max_steps - 1 :])
        ]

    step_ids: List[List[int]] = []
    for step in step_texts:
        step_text = step.strip()
        if not step_text:
            continue
        ids = tokenizer.encode(
            f"<think>{step_text}</think>",
            add_special_tokens=False,
        )
        ids = ids + [eot_id]
        step_ids.append(ids if len(ids) > 0 else [pad_id])

    while len(step_ids) < max_steps:
        step_ids.append([pad_id])

    if len(step_ids) == 0:
        step_ids = [[pad_id] for _ in range(max_steps)]
    return step_ids[:max_steps]
