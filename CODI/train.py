# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import copy
import logging
import os
import re
import random
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)
import json


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

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {file_path} at line {line_no}: {e}") from e
    return data

def _sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "default"

def prepare_run_layout(model_args: ModelArguments, _data_args: DataArguments, training_args: TrainingArguments) -> Dict[str, str]:
    expt_name = _sanitize_path_component(training_args.expt_name or "default")
    model_tag = _sanitize_path_component(model_args.model_name_or_path.split("/")[-1])
    configured_run_name = getattr(training_args, "run_name", None)
    if configured_run_name:
        run_name = _sanitize_path_component(str(configured_run_name))
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = _sanitize_path_component(f"{timestamp}_seed{training_args.seed}_{model_tag}")

    run_root = os.path.join("runs", expt_name, run_name)
    run_paths = {
        "run_root": run_root,
        "checkpoints": os.path.join(run_root, "checkpoints"),
        "tb": os.path.join(run_root, "tb"),
        "config": os.path.join(run_root, "config"),
        "run_config": os.path.join(run_root, "config", "run_config.json"),
    }
    for path in run_paths.values():
        if path.endswith(".json"):
            continue
        os.makedirs(path, exist_ok=True)

    training_args.expt_name = expt_name
    training_args.run_name = run_name
    training_args.output_dir = run_paths["checkpoints"]
    training_args.logging_dir = run_paths["tb"]
    return run_paths

def save_args_config(config_path: str, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments) -> None:
    payload = {
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "training_args": training_args.to_dict(),
    }
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)

IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_keys = (
            "Loss/total",
            "Loss/ce",
            "Loss/distill",
            "Loss/ref_ce",
            "Loss/explain",
        )
        self._reset_micro_accum()
        self._pending_step_metrics: Dict[int, Dict[str, float]] = {}
        self._pending_step_tokens: Dict[int, float] = {}
        self._last_log_wall_time = time.time()
        self._last_log_step = 0

    def _reset_micro_accum(self):
        self._micro_loss_sums = {k: 0.0 for k in self._loss_keys}
        self._micro_loss_counts = {k: 0 for k in self._loss_keys}
        self._micro_batch_count = 0
        self._micro_token_count = 0.0

    def _extract_token_count(self, inputs: Dict) -> int:
        token_count = 0
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        with torch.no_grad():
            if "encoder_attention_mask" in inputs and isinstance(inputs["encoder_attention_mask"], torch.Tensor):
                token_count += int(inputs["encoder_attention_mask"].sum().item())
            elif "encoder_input_ids" in inputs and isinstance(inputs["encoder_input_ids"], torch.Tensor):
                token_count += int(inputs["encoder_input_ids"].ne(pad_token_id).sum().item())

            if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor):
                token_count += int(inputs["labels"].ne(IGNORE_INDEX).sum().item())
        return token_count

    def _accumulate_micro_metric(self, key: str, value: Optional[float]):
        if value is None:
            return
        self._micro_loss_sums[key] += float(value)
        self._micro_loss_counts[key] += 1

    def _flush_micro_accum_to_pending(self, target_step: int):
        if self._micro_batch_count == 0:
            return

        step_metrics: Dict[str, float] = {}
        for key in self._loss_keys:
            count = self._micro_loss_counts[key]
            if count > 0:
                step_metrics[key] = self._micro_loss_sums[key] / count

        effective_batch_size = (
            self.args.per_device_train_batch_size
            * max(1, int(getattr(self.args, "world_size", 1) or 1))
            * max(1, self.args.gradient_accumulation_steps)
        )
        step_metrics["Data/effective_batch_size"] = float(effective_batch_size)
        step_metrics["Data/grad_accum_steps"] = float(max(1, self.args.gradient_accumulation_steps))

        self._pending_step_metrics[target_step] = step_metrics
        self._pending_step_tokens[target_step] = float(self._micro_token_count)
        self._reset_micro_accum()

    def _consume_pending_step_metrics(self, current_step: int):
        pending_steps = [s for s in sorted(self._pending_step_metrics.keys()) if s <= current_step]
        if not pending_steps:
            return {}, 0.0

        merged_sums: Dict[str, float] = {}
        merged_counts: Dict[str, int] = {}
        consumed_tokens = 0.0
        for step in pending_steps:
            consumed_tokens += self._pending_step_tokens.pop(step, 0.0)
            step_metrics = self._pending_step_metrics.pop(step)
            for key, value in step_metrics.items():
                merged_sums[key] = merged_sums.get(key, 0.0) + float(value)
                merged_counts[key] = merged_counts.get(key, 0) + 1

        merged = {
            key: merged_sums[key] / max(1, merged_counts[key])
            for key in merged_sums
        }
        return merged, consumed_tokens

    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * max(1, int(getattr(self.args, "world_size", 1) or 1)) * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]

        self._micro_batch_count += 1
        self._micro_token_count += self._extract_token_count(inputs)
        self._accumulate_micro_metric("Loss/total", _to_scalar(loss))
        self._accumulate_micro_metric("Loss/ce", _to_scalar(outputs.get("ce_loss")))
        self._accumulate_micro_metric("Loss/distill", _to_scalar(outputs.get("distill_loss")))
        self._accumulate_micro_metric("Loss/ref_ce", _to_scalar(outputs.get("ref_ce_loss")))
        self._accumulate_micro_metric("Loss/explain", _to_scalar(outputs.get("explain_loss")))

        if hasattr(self, "accelerator"):
            should_flush = bool(self.accelerator.sync_gradients)
        else:
            grad_acc_steps = max(1, self.args.gradient_accumulation_steps)
            should_flush = (self._micro_batch_count % grad_acc_steps) == 0

        if should_flush:
            # global_step is incremented after optimizer.step(), so cache metrics for the next step index.
            self._flush_micro_accum_to_pending(int(step) + 1)

        return loss

    def log(self, logs, start_time=None):
        if not self.is_world_process_zero():
            return
        if self.state.global_step is None:
            return

        logs = dict(logs) if logs is not None else {}
        global_step = int(self.state.global_step)
        grouped_logs, consumed_tokens = self._consume_pending_step_metrics(global_step)

        if "learning_rate" in logs:
            grouped_logs["Train/lr"] = float(logs["learning_rate"])
        if "grad_norm" in logs and logs["grad_norm"] is not None:
            grouped_logs["Train/grad_norm"] = float(logs["grad_norm"])
        if "epoch" in logs and logs["epoch"] is not None:
            grouped_logs["Data/epoch"] = float(logs["epoch"])
        if "train_runtime" in logs:
            grouped_logs["Train/runtime_sec"] = float(logs["train_runtime"])
        if "train_samples_per_second" in logs:
            grouped_logs["Train/samples_per_sec"] = float(logs["train_samples_per_second"])
        if "train_steps_per_second" in logs:
            grouped_logs["Train/steps_per_sec"] = float(logs["train_steps_per_second"])
        if "train_loss" in logs:
            grouped_logs["Loss/train_avg"] = float(logs["train_loss"])
        if "loss" in logs and "Loss/total" not in grouped_logs:
            grouped_logs["Loss/total"] = float(logs["loss"])

        grouped_logs["Data/global_step"] = float(global_step)

        now = time.time()
        step_delta = global_step - self._last_log_step
        if step_delta > 0:
            elapsed = max(1e-8, now - self._last_log_wall_time)
            grouped_logs["Train/step_time_sec"] = elapsed / step_delta
            if consumed_tokens > 0:
                grouped_logs["Train/tokens_per_sec"] = consumed_tokens / elapsed
            self._last_log_step = global_step
            self._last_log_wall_time = now

        try:
            super().log(grouped_logs, start_time=start_time)
        except TypeError:
            super().log(grouped_logs)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,#training_args.model_max_length,
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
    run_paths = prepare_run_layout(model_args, data_args, training_args)
    save_args_config(run_paths["run_config"], model_args, data_args, training_args)
    logging.warning(f"Run root: {run_paths['run_root']}")
    logging.warning(f"Checkpoints: {run_paths['checkpoints']}")
    logging.warning(f"TensorBoard: {run_paths['tb']}")

    ##########################
    #       Peft Model       #
    ##########################
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

    # import pdb; pdb.set_trace()
    model = CODI(model_args, training_args, lora_config)
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

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        #answer_prompt = torch.tensor([464, 3280, 318, 25])
        # import pdb; pdb.set_trace()
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception:
            breakpoint()
        
    # def get_steps_

    def preprocess(
        sources: Sequence[str], 
        targets: Sequence[str], 
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
    ) -> Dict:
        print("Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
        cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
        answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

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

        answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
        if answer_prompts[0][0] == tokenizer.bos_token_id: # remove the bos
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]
        # import pdb; pdb.set_trace()
        ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
        model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]

        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in answers_id]
        return dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id, \
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position, \
                        ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels)


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
            # import pdb; pdb.set_trace()
            if raw_data is None:
                raise ValueError(f"No dataset loaded for data_name={self.data_name}.")

            for num_iter, example in tqdm(enumerate(raw_data)):
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break

                if self.data_name == "local-jsonl":
                    required_keys = ("question", "cot", "answer")
                    missing_keys = [k for k in required_keys if k not in example]
                    if missing_keys:
                        raise ValueError(
                            f"local-jsonl sample at index {num_iter} is missing required keys: {missing_keys}"
                        )

                    question = str(example["question"]).strip() + "\n"
                    cot = str(example["cot"]).strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"

                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num:
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                    continue

                if 'cot' not in example:
                    example['cot'] = example['steps']
                    example['cot'] = ' '.join(example['cot'])
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
          
            return dict(
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

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Downloading Data")
        if data_args.data_name == "local-jsonl":
            if not data_args.data_path:
                raise ValueError("--data_path is required when --data_name local-jsonl is used.")
            if not data_args.data_path.endswith(".jsonl"):
                raise ValueError(f"--data_path must point to a .jsonl file, got: {data_args.data_path}")
            if not os.path.isfile(data_args.data_path):
                raise ValueError(f"JSONL file does not exist: {data_args.data_path}")
            dataset = read_jsonl(data_args.data_path)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "icot" in data_args.data_name:
            # dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
            default_icot_path = "/mnt/shared-storage-user/weixilin/MLLM/coconut/data/gsm_train_clean.json"
            dataset_path = data_args.data_path if data_args.data_path else default_icot_path
            if dataset_path.endswith(".jsonl"):
                dataset = read_jsonl(dataset_path)
            else:
                dataset = read_json(dataset_path)
            if dataset is None:
                raise ValueError(f"Failed to load dataset from: {dataset_path}")
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "strategy" in data_args.data_name:
            dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "commonsense" in data_args.data_name:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "prontoqa" in data_args.data_name:
            with open("/home/ubuntu/coconut/data/prontoqa_train.json") as f:
                dataset = json.load(f)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CustomTrainer(model=model, args=training_args, **data_module)
    trainer.tokenizer = tokenizer
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
