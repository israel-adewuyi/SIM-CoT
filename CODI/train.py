# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import copy
import logging
import os
import re
import ast
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple, Any
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from tqdm import tqdm
from math import ceil
from peft import LoraConfig, TaskType
from datasets import load_dataset
from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)


def _to_scalar(x):
    """Convert Tensor/number/None to python float (mean-reduced if needed)."""
    import torch
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        # detach，转到 float，若多元素则取 mean，再 item()
        return x.detach().float().mean().item()
    # 已经是数字的情况
    return float(x)


def _unwrap_to_hf_model(model: torch.nn.Module) -> torch.nn.Module:
    """Best-effort unwrapping to the underlying HF causal LM."""
    cur = model
    visited = set()
    while True:
        obj_id = id(cur)
        if obj_id in visited:
            break
        visited.add(obj_id)
        if hasattr(cur, "codi"):
            cur = cur.codi
            continue
        if hasattr(cur, "model"):
            cur = cur.model
            continue
        if hasattr(cur, "base_model"):
            cur = cur.base_model
            continue
        break
    return cur


def _has_tied_input_output_embeddings(model: torch.nn.Module) -> bool:
    """Detect tied embedding/lm-head weights that break safetensors save_file()."""
    try:
        hf_model = _unwrap_to_hf_model(model)
        get_inp = getattr(hf_model, "get_input_embeddings", None)
        get_out = getattr(hf_model, "get_output_embeddings", None)
        if get_inp is None or get_out is None:
            return False
        inp = get_inp()
        out = get_out()
        if inp is None or out is None:
            return False
        if not hasattr(inp, "weight") or not hasattr(out, "weight"):
            return False
        return inp.weight.data_ptr() == out.weight.data_ptr()
    except Exception:
        return False


def read_json(file_path):
    """
    从指定路径读取JSON文件并返回对应的Python对象。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return None
IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        loss_for_log = loss.detach() if isinstance(loss, torch.Tensor) else loss
        if step % self.args.logging_steps == 0:
            logs = {
                "loss": _to_scalar(loss_for_log),
                "ce_loss": _to_scalar(outputs.get("ce_loss")),
                "distill_loss": _to_scalar(outputs.get("distill_loss")),
                "ref_ce_loss": _to_scalar(outputs.get("ref_ce_loss")),
            }
            if not hasattr(self, "is_global_zero") or self.is_global_zero:
                self.log(logs)
        #"ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        # if step % self.args.logging_steps == 0:
        #     self.log({"loss": loss.item(), "ce_loss": outputs["ce_loss"], "distill_loss": outputs["distill_loss"], "ref_ce_loss": outputs["ref_ce_loss"],})

        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Retry save with non-safetensors if shared-tensor safetensors save fails."""
        try:
            return super()._save(output_dir=output_dir, state_dict=state_dict)
        except RuntimeError as e:
            if "Some tensors share memory" not in str(e):
                raise
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            logging.warning(
                "safetensors save failed due to shared tensors; saving fallback "
                "checkpoint as pytorch_model.bin to `%s`.",
                output_dir,
            )
            os.makedirs(output_dir, exist_ok=True)
            if state_dict is None:
                state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

            processing = getattr(self, "processing_class", None)
            if processing is not None:
                processing.save_pretrained(output_dir)
            else:
                tok = getattr(self, "tokenizer", None)
                if tok is not None:
                    tok.save_pretrained(output_dir)

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            return

def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: Optional[int] = None,
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    output_dir_path = Path(training_args.output_dir).expanduser().resolve()

    report_to = getattr(training_args, "report_to", None)
    if report_to is None:
        report_to_list = []
    elif isinstance(report_to, str):
        report_to_list = [report_to]
    else:
        report_to_list = [str(x) for x in report_to]
    report_to_normalized = {x.strip().lower() for x in report_to_list}
    tensorboard_enabled = (
        "all" in report_to_normalized
        or "tensorboard" in report_to_normalized
    )

    if tensorboard_enabled:
        raw_run_name = (getattr(training_args, "run_name", None) or "").strip()
        if not raw_run_name:
            raise ValueError(
                "`--run_name` is required when TensorBoard logging is enabled. "
                "Example: --run_name my_experiment"
            )

        sanitized_run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_run_name).strip("._-")
        if not sanitized_run_name:
            raise ValueError(
                "`--run_name` must contain at least one valid character from [A-Za-z0-9._-] "
                "after sanitization. Example: --run_name my_experiment"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_root = Path("outputs/tb").resolve()
        logging_dir = tb_root / f"{sanitized_run_name}__{timestamp}"
        os.makedirs(logging_dir, exist_ok=True)
        training_args.logging_dir = str(logging_dir)
        os.environ["TENSORBOARD_LOGGING_DIR"] = training_args.logging_dir

    training_args.output_dir = str(output_dir_path)
    os.makedirs(training_args.output_dir, exist_ok=True)
    print(f"[paths] output_dir={training_args.output_dir}")
    print(f"[paths] logging_dir={training_args.logging_dir}")

    require_cuda = os.environ.get("CODI_REQUIRE_CUDA", "0").lower() in {"1", "true", "yes"}
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. "
            "Check your runtime/container GPU visibility and CUDA setup."
        )
    if torch.cuda.is_available():
        print(
            f"[device] CUDA available: {torch.cuda.device_count()} GPU(s). "
            f"Using device 0: {torch.cuda.get_device_name(0)}"
        )
    else:
        print("[device] CUDA unavailable. Training will run on CPU.")

    ##########################
    #       Peft Model       #
    ##########################
    lora_config = None
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2", "gsm-cot"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    elif training_args.use_lora:
        raise ValueError("`--use_lora True` requires `--lora_init` in the current CODI training path.")

    # import pdb; pdb.set_trace()
    model = CODI(model_args, training_args, lora_config)
    if getattr(training_args, "save_safetensors", True) and _has_tied_input_output_embeddings(model):
        logging.warning(
            "Detected tied input/output embeddings; setting --save_safetensors False "
            "to avoid safetensors shared-tensor save failure."
        )
        training_args.save_safetensors = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    def get_answer_token_position(tokens, answer_prompts):
        try:
            for prompt in answer_prompts:
                prompt_len = len(prompt)
                if prompt_len == 0 or tokens.numel() < prompt_len:
                    continue
                match_indices = (
                    (tokens.unfold(0, prompt_len, 1) == prompt)
                    .all(dim=1)
                    .nonzero(as_tuple=True)[0]
                )
                if match_indices.numel() > 0:
                    return match_indices[0].item() + prompt_len
        except Exception:
            pass
        return 0

    def preprocess(
        sources: Sequence[str], 
        targets: Sequence[str], 
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
    ) -> Dict:
        print("Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(
            sources, tokenizer, max_length=training_args.model_max_length
        )["input_ids"]
        cot_id = _tokenize_fn(
            targets, tokenizer, max_length=training_args.model_max_length
        )["input_ids"]
        answers_id = _tokenize_fn(
            answers, tokenizer, max_length=training_args.model_max_length
        )["input_ids"]

        # add eos token to accomodate pretrained model's format
        if not training_args.remove_eos:
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

        if cot_id[0][0] == tokenizer.bos_token_id:
            cot_id = [x[1:] for x in cot_id]
            answers_id = [x[1:] for x in answers_id]

        ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
        ref_labels = []
        for x, y in zip(ref_input_ids, sources_id):
            z = x.clone()
            z[:len(y)] = -100
            ref_labels.append(z)
        
        # add eot to source
        sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
        # add eot and eos
        if training_args.remove_eos:
            answers_id = [torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        else:
            answers_id = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]

        answer_prompts = [
            torch.tensor(tokenizer.encode("The answer is:")),
            torch.tensor(tokenizer.encode("The next step result is:")),
        ]
        if answer_prompts[0][0] == tokenizer.bos_token_id: # remove the bos
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]
        ref_answer_position = [
            get_answer_token_position(x, answer_prompts) for x in ref_input_ids
        ]
        model_answer_position = [
            get_answer_token_position(x, answer_prompts) for x in answers_id
        ]

        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in answers_id]
        return dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id, \
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position, \
                        ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels)

    def _is_trainable_message(msg: Dict) -> bool:
        loss_mask = msg.get("loss_mask", None)
        if loss_mask is not None:
            if isinstance(loss_mask, str):
                if loss_mask.strip().lower() in {"true", "yes"}:
                    return True
                if loss_mask.strip().lower() in {"false", "no"}:
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
            chunks = []
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

    def _coerce_messages(raw_messages: Any) -> List[Dict]:
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

        normalized_list = []
        for msg in raw_messages:
            if isinstance(msg, dict):
                normalized_list.append(msg)
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                normalized_list.append({"role": msg[0], "content": msg[1]})
        return normalized_list

    def _normalise_messages(raw_messages: Any) -> List[Dict[str, str]]:
        raw_messages = _coerce_messages(raw_messages)
        normalized = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower().strip()
            content = _extract_text_content(msg.get("content", ""))
            if role == "" or content == "":
                continue
            loss_mask = msg.get("loss_mask", 1 if role == "assistant" else 0)
            normalized.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": loss_mask,
                }
            )
        return normalized

    def _build_tokens_from_messages(
        messages: Sequence[Dict[str, str]], tokenizer: transformers.PreTrainedTokenizer
    ) -> Tuple[List[int], List[int]]:
        use_chat_template = hasattr(tokenizer, "apply_chat_template") and bool(
            getattr(tokenizer, "chat_template", None)
        )
        if use_chat_template:
            full_ids: List[int] = []
            full_labels: List[int] = []
            for idx, msg in enumerate(messages):
                partial_ids = tokenizer.apply_chat_template(
                    list(messages[: idx + 1]),
                    tokenize=True,
                    add_generation_prompt=False,
                )
                if isinstance(partial_ids, torch.Tensor):
                    partial_ids = partial_ids.tolist()

                if partial_ids[: len(full_ids)] != full_ids:
                    use_chat_template = False
                    break

                delta = partial_ids[len(full_ids) :]
                full_ids = partial_ids
                if _is_trainable_message(msg):
                    full_labels.extend(delta)
                else:
                    full_labels.extend([IGNORE_INDEX] * len(delta))

            if use_chat_template:
                return full_ids, full_labels

        full_ids = []
        full_labels = []
        for idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"].strip()
            prefix = "System" if role == "system" else ("User" if role == "user" else "Assistant")
            text = f"{prefix}: {content}\n"
            msg_ids = tokenizer.encode(text, add_special_tokens=(idx == 0))
            full_ids.extend(msg_ids)
            if _is_trainable_message(msg):
                full_labels.extend(msg_ids)
            else:
                full_labels.extend([IGNORE_INDEX] * len(msg_ids))
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
            s.strip()
            for s in re.split(r"(?<=[\.\!\?])\s+", text)
            if s.strip()
        ]
        if len(sentence_steps) >= 2:
            return sentence_steps

        chunk_steps = [s.strip() for s in re.split(r";\s+|,\s+", text) if s.strip()]
        if len(chunk_steps) >= 2:
            return chunk_steps

        return [text]

    def _extract_explain_steps_ids(
        messages: Sequence[Dict[str, str]], eot_id: int
    ) -> List[List[int]]:
        max_steps = training_args.num_latent + 1
        step_texts: List[str] = []

        for msg in messages:
            if not _is_trainable_message(msg):
                continue
            content = msg["content"]
            think_blocks = re.findall(
                r"<think>(.*?)</think>",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for block in think_blocks:
                step_texts.extend(_split_reasoning_steps(block))

        if len(step_texts) == 0:
            fallback_text = []
            for msg in messages:
                if _is_trainable_message(msg):
                    text = re.sub(r"<[^>]+>", " ", msg["content"])
                    fallback_text.append(text.strip())
            step_texts = _split_reasoning_steps("\n".join([t for t in fallback_text if t]))

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
            step_ids.append(ids if len(ids) > 0 else [tokenizer.pad_token_id])

        while len(step_ids) < max_steps:
            step_ids.append([tokenizer.pad_token_id])

        if len(step_ids) == 0:
            step_ids = [[tokenizer.pad_token_id] for _ in range(max_steps)]
        return step_ids[:max_steps]

    def _build_messages_training_instance(raw_messages: Sequence[Dict], bot_id: int, eot_id: int):
        messages = _normalise_messages(raw_messages)
        if len(messages) == 0:
            return None

        full_ids, full_labels = _build_tokens_from_messages(messages, tokenizer)
        if len(full_ids) == 0:
            return None

        if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
            full_ids.append(tokenizer.eos_token_id)
            if _is_trainable_message(messages[-1]):
                full_labels.append(tokenizer.eos_token_id)
            else:
                full_labels.append(IGNORE_INDEX)

        if training_args.max_token_num and len(full_ids) > training_args.max_token_num:
            # Keep the tail so we preserve assistant target tokens instead of dropping the sample.
            full_ids = full_ids[-training_args.max_token_num :]
            full_labels = full_labels[-training_args.max_token_num :]

        first_target_idx = None
        for idx, label in enumerate(full_labels):
            if label != IGNORE_INDEX:
                first_target_idx = idx
                break
        if first_target_idx is None or first_target_idx >= len(full_ids):
            return None

        prompt_ids = full_ids[:first_target_idx]
        assistant_ids = full_ids[first_target_idx:]
        assistant_labels = full_labels[first_target_idx:]

        if len(assistant_ids) == 0:
            return None

        encoder_input_ids = torch.tensor(prompt_ids + [bot_id], dtype=torch.long)
        decoder_input_ids = torch.tensor([eot_id] + assistant_ids, dtype=torch.long)
        labels = torch.tensor([IGNORE_INDEX] + assistant_labels, dtype=torch.long)
        ref_input_ids = torch.tensor(full_ids, dtype=torch.long)
        ref_labels = torch.tensor(full_labels, dtype=torch.long)
        explain_steps_ids = _extract_explain_steps_ids(messages, eot_id)

        return dict(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            ref_input_ids=ref_input_ids,
            labels=labels,
            ref_answer_position=first_target_idx,
            model_answer_position=1,
            ref_labels=ref_labels,
            explain_steps_ids=explain_steps_ids,
        )

    class SupervisedDataset(Dataset):
        QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
        QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
        def __init__(self, data_name, raw_data, tokenizer, bot, eot):
            super(SupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")
            
            self.data_name = data_name
            questions, cots, answers = [], [], []
            num_ops_list = []
            operators = ["+", "-", "*", "/"]

            token_nums = []
            if raw_data is None:
                raw_data = read_json(training_args.icot_train_path)
                if raw_data is None:
                    raise FileNotFoundError(
                        f"Could not load icot data from `{training_args.icot_train_path}`."
                    )
            for num_iter, example in tqdm(enumerate(raw_data)):
                if 'cot' not in example: 
                    example['cot'] = example['steps']
                    example['cot'] = ' '.join(example['cot'])
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break
                question = f"{example['question']}"
                if "icot" in self.data_name and "full" in self.data_name: # icot-full (GSM8k-Aug-NL)
                    # bad data
                    if example["answer"] is None or example["response"] is None:
                        continue
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot = f"{example['cot']}".split(". ")
                    if not (training_args.include_last_cot):
                        cot = cot[:-1]
                    answer = f"The answer is: {example['answer'].split(' ')[-1]}"
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(". ".join(cot)+".\n")
                    answers.append(answer)
                elif "icot" in self.data_name: # icot (GSM8k-Aug)
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot_list = []
                    cot = f"{example['cot']}".split(" ")
                    if not training_args.include_last_cot:
                        cot = cot[:-1]
                    
                    len_cot = len(cot) 
                    for i in range(training_args.num_latent):
                        cot_list.append(" ".join(cot[:max(0, len_cot-i)]))
                    answer = example['answer'].split(' ')[-1]
                    
                    # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
                    if not answer[0].isdigit():
                        continue

                    answer = f"The answer is: {answer}" 
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(" ".join(cot))
                    answers.append(answer)
                elif "commonsense" in self.data_name or "strategy" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "prontoqa" in data_args.data_name:
                    question = example['question'].strip() + '\n'
                    cot = '\n'.join(example['steps'][:-1]) + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                else:
                    raise NotImplementedError
            if training_args.exp_mode:
                questions = questions[:training_args.exp_data_num]
                cots = cots[:training_args.exp_data_num]
                answers = answers[:training_args.exp_data_num]
            
            print(f"{len(cots)} data in total...")
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess(questions, cots, answers, tokenizer, bot, eot)
            self.keys = list(self.data_dict.keys())


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    class MessagesSupervisedDataset(Dataset):
        def __init__(self, raw_data, tokenizer, bot, eot, messages_field: str):
            super(MessagesSupervisedDataset, self).__init__()
            logging.warning("Formatting message-style inputs...")
            self.instances = []
            missing_messages_field = 0
            malformed_messages = 0
            total_rows = 0

            column_names = []
            if hasattr(raw_data, "column_names"):
                try:
                    column_names = list(raw_data.column_names)
                except Exception:
                    column_names = []

            for num_iter, example in tqdm(enumerate(raw_data)):
                total_rows += 1
                if training_args.exp_mode and num_iter >= training_args.exp_data_num:
                    break
                if not isinstance(example, dict):
                    malformed_messages += 1
                    continue
                if messages_field not in example:
                    missing_messages_field += 1
                    continue
                messages = example.get(messages_field, None)
                if messages is None:
                    missing_messages_field += 1
                    continue
                if not isinstance(messages, (list, str, dict)):
                    malformed_messages += 1
                    continue
                instance = _build_messages_training_instance(messages, bot, eot)
                if instance is None:
                    malformed_messages += 1
                    continue
                self.instances.append(instance)

            if training_args.exp_mode:
                self.instances = self.instances[:training_args.exp_data_num]

            print(f"{len(self.instances)} data in total...")
            print(
                f"[messages parse stats] rows={total_rows}, valid={len(self.instances)}, "
                f"missing_field={missing_messages_field}, filtered_or_malformed={malformed_messages}, "
                f"max_token_num={training_args.max_token_num}"
            )
            if len(self.instances) == 0:
                msg = (
                    "No valid records found in message-style dataset. "
                    f"`messages_field={messages_field}` rows={total_rows}, "
                    f"missing_field={missing_messages_field}, malformed={malformed_messages}."
                )
                if len(column_names) > 0:
                    msg += f" Available columns: {column_names}"
                raise ValueError(
                    msg
                )

        def __len__(self):
            return len(self.instances)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return self.instances[i]

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels= \
                tuple([instance[key] for instance in instances] for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "ref_answer_position", "model_answer_position", "ref_labels"))

            # pad left
            reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)
            
            # pad
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX) 

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            batch = dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
                ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
                model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
                ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
                ref_labels=ref_labels,
            )

            if "explain_steps_ids" in instances[0]:
                explain_steps_ids = [instance["explain_steps_ids"] for instance in instances]
                max_steps = max(len(sample) for sample in explain_steps_ids)
                max_step_len = max(
                    len(step)
                    for sample in explain_steps_ids
                    for step in sample
                )
                pad_id = self.tokenizer.pad_token_id
                padded_steps = []
                for sample in explain_steps_ids:
                    sample_padded = []
                    for step in sample:
                        sample_padded.append(step + [pad_id] * (max_step_len - len(step)))
                    while len(sample_padded) < max_steps:
                        sample_padded.append([pad_id] * max_step_len)
                    padded_steps.append(sample_padded)
                batch["explain_steps_ids"] = torch.tensor(padded_steps, dtype=torch.long)

            return batch

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Downloading Data")
        data_name = (data_args.data_name or "").lower()
        if "sweswiss" in data_name or "messages" in data_name:
            local_data_path = data_args.hf_dataset_name
            if os.path.isfile(local_data_path):
                logging.warning(
                    "Loading local message dataset from `%s` with split `%s`.",
                    local_data_path,
                    data_args.hf_dataset_split,
                )
                dataset = load_dataset(
                    "json",
                    data_files=local_data_path,
                    split=data_args.hf_dataset_split,
                )
            else:
                dataset = load_dataset(
                    data_args.hf_dataset_name,
                    split=data_args.hf_dataset_split,
                )
            if data_args.debug_data:
                dataset = dataset.select(range(min(len(dataset), 64)))
            train_dataset = MessagesSupervisedDataset(
                raw_data=dataset,
                tokenizer=tokenizer,
                bot=model.bot_id,
                eot=model.eot_id,
                messages_field=data_args.messages_field,
            )
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "icot" in data_name:
            # dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
            dataset = None
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "strategy" in data_name:
            dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "commonsense" in data_name:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "prontoqa" in data_name:
            with open("/home/ubuntu/coconut/data/prontoqa_train.json") as f:
                dataset = json.load(f)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CustomTrainer(model=model, args=training_args, **data_module)
    trainer.train()

    # to avoid the error of saving the model
    #if "llama" in model_args.model_name_or_path:
    #    trainer.model.codi.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.model.lm_head.weight.clone())
    #if "gpt2" in model_args.model_name_or_path:
    #    trainer.model.codi.transformer.wte.weight = torch.nn.Parameter(model.codi.lm_head.weight.clone())
    #if "qwen" in model_args.model_name_or_path.lower():
    #    trainer.model.codi.base_model.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.base_model.model.lm_head.weight.clone())

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
