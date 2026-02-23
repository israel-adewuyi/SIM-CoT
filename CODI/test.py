#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

do_print = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import json
from peft import PeftModel, LoraConfig

def save_jsonl_line(filepath, data):
    """
    将一条字典数据追加写入到 JSONL 文件中。

    参数:
        filepath (str): 目标 JSONL 文件路径。
        data (dict): 要写入的数据，必须是可序列化为 JSON 的字典。
    """
    if not isinstance(data, dict):
        raise ValueError("data 必须是一个字典")

    with open(filepath, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")

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

def write_json(data, file_path):
    """
    将Python对象写入指定路径的JSON文件中。
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"写入JSON文件时出错: {e}")


def write_jsonl(data_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for data in data_list:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + "\n")


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


def _messages_to_prompt(messages: List[Dict[str, Any]], tokenizer) -> str:
    prompt_messages = [m for m in messages if int(m.get("loss_mask", 0)) != 1]
    if len(prompt_messages) == 0 and len(messages) > 0:
        if messages[-1].get("role", "") == "assistant":
            prompt_messages = messages[:-1]
        else:
            prompt_messages = messages

    if len(prompt_messages) == 0:
        return ""

    use_chat_template = hasattr(tokenizer, "apply_chat_template") and bool(
        getattr(tokenizer, "chat_template", None)
    )
    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in prompt_messages],
            tokenize=False,
            add_generation_prompt=False,
        )

    lines = []
    for msg in prompt_messages:
        role = msg["role"]
        prefix = "System" if role == "system" else ("User" if role == "user" else "Assistant")
        lines.append(f"{prefix}: {msg['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _filter_examples_by_token_length(
    question: List[str],
    tokenizer,
    max_token_num: int,
    answer: Optional[List[Any]] = None,
    generation_metadata: Optional[List[Dict[str, Any]]] = None,
    add_special_tokens: bool = True,
):
    filtered_question = []
    filtered_answer = [] if answer is not None else None
    filtered_metadata = [] if generation_metadata is not None else None
    dropped = 0

    for idx, q in enumerate(question):
        q_len = len(tokenizer.encode(q, add_special_tokens=add_special_tokens))
        if q_len <= max_token_num:
            filtered_question.append(q)
            if filtered_answer is not None:
                filtered_answer.append(answer[idx])
            if filtered_metadata is not None:
                filtered_metadata.append(generation_metadata[idx])
        else:
            dropped += 1

    return filtered_question, filtered_answer, filtered_metadata, dropped

def evaluation(model_args, data_args, training_args):
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
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
    else:
        raise NotImplementedError
    # import pdb; pdb.set_trace()
    model = CODI(model_args, training_args, lora_config)
    #if "llama" in model_args.model_name_or_path:
    #    model.codi.resize_token_embeddings(128261)
    # import pdb; pdb.set_trace()
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    
    # new_state_dict = { k.replace("coconut", "codi"): v for k, v in state_dict.items() }
    # torch.save(new_state_dict, "/scratch/prj/inf_multimodal_qa/scratch_tmp/transfer/pytorch_model.bin")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    
    tokenizer_path = model_args.model_name_or_path 
    # tokenizer_path = '/mnt/shared-storage-user/mllm/shared/weixilin/gpt2'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)
    # import pdb; pdb.set_trace()
    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    is_messages_mode = "messages" in (data_args.data_name or "").lower()
    use_chat_template = hasattr(tokenizer, "apply_chat_template") and bool(
        getattr(tokenizer, "chat_template", None)
    )
    add_special_tokens_for_eval_prompt = not (is_messages_mode and use_chat_template)
    question_name = "question"
    answer_name = "answer"
    if is_messages_mode:
        dataset_source = data_args.hf_dataset_name
        if os.path.isfile(dataset_source):
            test_set = load_dataset("json", data_files=dataset_source, split=data_args.hf_dataset_split)
        else:
            test_set = load_dataset(dataset_source, split=data_args.hf_dataset_split)
    elif "gsm-hard" == data_args.data_name:
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset['train']
        question_name = "instruction"
        answer_name = "response"
    elif "multi-arith" == data_args.data_name:
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset['test']
        answer_name = "final_ans"
    elif "svamp" == data_args.data_name:
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif "commonsense" == data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
    elif "gsm8k" == data_args.data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset['test']
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    question = []
    answer = []
    generation_metadata = []

    if is_messages_mode:
        for idx, example in enumerate(test_set):
            if data_args.messages_field not in example:
                continue
            messages = _normalise_messages(example[data_args.messages_field])
            if len(messages) == 0:
                continue
            prompt_text = _messages_to_prompt(messages, tokenizer).strip()
            if prompt_text == "":
                continue
            question.append(prompt_text)
            generation_metadata.append(
                {
                    "example_index": idx,
                    "sample_id": example.get("sample_id", None),
                    "trajectory_id": example.get("trajectory_id", None),
                    "assistant_turn_index": example.get("assistant_turn_index", None),
                    "target_message_index": example.get("target_message_index", None),
                    "source_path": example.get("source_path", None),
                    "prompt_messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages
                        if int(m.get("loss_mask", 0)) != 1
                    ],
                }
            )
    else:
        question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
        # get numerical answer
        for example in test_set:
            example = example[answer_name]
            if isinstance(example, bool):
                answer.append(example)
                continue
            if example in ["True", "False"]:
                if example == "True":
                    ans = True
                else:
                    ans = False
                answer.append(ans)
                continue
            if example in "ABCDE":
                answer.append(example)
                continue
            if "####" in example:
                ans = example.split('####')[-1]
            else:
                ans = example
            ans = ans.replace(',', '')  # handle numbers like 2,000
            try:
                ans = float(ans)
            except ValueError:
                ans = float("inf")
            answer.append(ans)

    if training_args.max_token_num is not None and training_args.max_token_num > 0:
        question, filtered_answer, filtered_metadata, dropped = _filter_examples_by_token_length(
            question=question,
            tokenizer=tokenizer,
            max_token_num=training_args.max_token_num,
            answer=answer if not is_messages_mode else None,
            generation_metadata=generation_metadata if is_messages_mode else None,
            add_special_tokens=add_special_tokens_for_eval_prompt,
        )
        if not is_messages_mode:
            answer = filtered_answer if filtered_answer is not None else []
        else:
            generation_metadata = filtered_metadata if filtered_metadata is not None else []
        logging.warning(
            "Length filter max_token_num=%s: kept=%s, dropped=%s",
            training_args.max_token_num,
            len(question),
            dropped,
        )

    if len(question) == 0:
        raise ValueError("No valid evaluation examples found.")

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    
    question_data = []
    # import pdb; pdb.set_trace()
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
                add_special_tokens=add_special_tokens_for_eval_prompt,
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
                add_special_tokens=add_special_tokens_for_eval_prompt,
            )
        
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    if training_args.eval_max_new_tokens <= 0:
        raise ValueError("`--eval_max_new_tokens` must be > 0.")
    gen_kwargs = {
        "max_new_tokens": int(training_args.eval_max_new_tokens),
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    generation_records = []
    ans_pred_list_accu_at_n_passes = []
    attention_map_weights = []
    attention_to_latents_against_len_sum = []
    attention_to_latents_against_len_count = []
    #set_seed(42)
    gating_probs_sums = None
    len_cot = []
    model.eval()
    attn_to_latent_list = []
    if model_args.soft_weight:
        embedding_matrix = model.codi.get_base_model().model.embed_tokens.weight.data.to(model.codi.device)
        vocab_center = embedding_matrix.mean(dim=0)
    
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            # encode the question
            past_key_values = None
            decode_attention_mask = batch["attention_mask"]
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            if model_args.soft_weight:
                soft_probs = F.softmax(model.codi.lm_head(latent_embd).to(model.codi.device), dim=-1)
                soft_embeds = (soft_probs.squeeze(dim=1).unsqueeze(dim=-1) * embedding_matrix.unsqueeze(dim=0)).sum(dim=0)

                soft_probs_expanded = soft_probs.squeeze(dim=1)
                soft_embeds = (soft_probs_expanded.unsqueeze(dim=2) * embedding_matrix).sum(dim=1)  # Shape: [128, 2048]
                if training_args.use_prj:
                    soft_embeds = model.prj(soft_embeds)
                latent_embd = latent_embd + model_args.soft_weight * soft_embeds
            
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                # decode the latent embeddings
                latent_step_mask = torch.ones(
                    (batch_size, latent_embd.size(1)),
                    dtype=decode_attention_mask.dtype,
                    device=decode_attention_mask.device,
                )
                decode_attention_mask = torch.cat((decode_attention_mask, latent_step_mask), dim=1)
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                    attention_mask=decode_attention_mask,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)
                
                if model_args.soft_weight:
                    soft_probs = F.softmax(model.codi.lm_head(latent_embd).to(model.codi.device), dim=-1)
                    soft_embeds = (soft_probs.squeeze(dim=1).unsqueeze(dim=-1) * embedding_matrix.unsqueeze(dim=0)).sum(dim=0)
                    
                    soft_probs_expanded = soft_probs.squeeze(dim=1)
                    soft_embeds = (soft_probs_expanded.unsqueeze(dim=2) * embedding_matrix).sum(dim=1)  # Shape: [128, 2048]
                    # import pdb; pdb.set_trace()
                    if training_args.use_prj:
                        soft_embeds = model.prj(soft_embeds)

                    latent_embd = latent_embd + model_args.soft_weight * soft_embeds

            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            
            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")  # Track EOS for each sequence
            pred_tokens = [[] for _ in range(batch_size)]
            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1
                gen_step_mask = torch.ones(
                    (batch_size, output.size(1)),
                    dtype=decode_attention_mask.dtype,
                    device=decode_attention_mask.device,
                )
                decode_attention_mask = torch.cat((decode_attention_mask, gen_step_mask), dim=1)
                out = model.codi(
                        inputs_embeds=output,
                        output_hidden_states=False,
                        attention_mask=decode_attention_mask,
                        use_cache=True,
                        output_attentions=False,
                        past_key_values=past_key_values
                    )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                # implement the sampling process
                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                if next_token_ids.dim() == 0:
                    next_token_ids = next_token_ids.unsqueeze(0)
                elif next_token_ids.dim() > 1:
                    next_token_ids = next_token_ids.reshape(-1)

                # Handle EOS for each sequence
                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                # Break if all sequences have finished
                if finished.all():
                    break

                #output = model.codi.get_base_model().transformer.wte(next_token_ids).unsqueeze(1).to(device)
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                # import pdb; pdb.set_trace()
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                # output_dir=f'/mnt/shared-storage-user/weixilin/MLLM/coconut/codi/inference_tokens_list/{'-'.join(model_args.ckpt_dir.split('/')[5:])}/{data_args.data_name}.jsonl'
                # os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                # save_jsonl_line(output_dir, {'list': tokenizer.encode(cot_output+answer_output, add_special_tokens=False), 'length': len(tokenizer.encode(cot_output+answer_output, add_special_tokens=False))})
                # import pdb; pdb.set_trace()
                # Extract the numbers in sentences 
                if do_print:
                    print(f"Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    if not is_messages_mode:
                        print(f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[step*data_args.batch_size+mini_step]}")
                    print("")
                if is_messages_mode:
                    sample_idx = step * data_args.batch_size + mini_step
                    prompt_tokens_with_padding = int(batch["input_ids"][mini_step].numel())
                    prompt_tokens = int(batch["attention_mask"][mini_step].sum().item())
                    prompt_padding_tokens = max(0, prompt_tokens_with_padding - prompt_tokens)
                    generation_tokens = int(len(pred_token))
                    generation_record = {
                        "sample_idx": sample_idx,
                        "prompt": question[sample_idx],
                        "generated_text": decoded_pred,
                        "generated_token_ids": pred_token,
                        "prompt_tokens": prompt_tokens,
                        "prompt_tokens_with_padding": prompt_tokens_with_padding,
                        "prompt_padding_tokens": prompt_padding_tokens,
                        "generation_tokens": generation_tokens,
                    }
                    if sample_idx < len(generation_metadata):
                        generation_record.update(generation_metadata[sample_idx])
                    generation_records.append(generation_record)
                else:
                    ans_pred_list.append(extract_answer_number(decoded_pred))

    os.makedirs(training_args.output_dir, exist_ok=True)
    if is_messages_mode:
        output_jsonl = os.path.join(training_args.output_dir, f"{data_args.data_name}_generations.jsonl")
        write_jsonl(generation_records, output_jsonl)
        print(f"Saved {len(generation_records)} generations to {output_jsonl}")
        return None

    write_json({"ans": ans_pred_list}, os.path.join(training_args.output_dir, f"{data_args.data_name}_predictions.json"))
    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot)}")
    # import pdb; pdb.set_trace()
    if model_args.save_ablation:
        save_jsonl_line(
            os.path.join(training_args.output_dir, f"{data_args.data_name}_metrics.jsonl"),
            {
                'model_name': os.path.basename(model_args.ckpt_dir.rstrip('/')),
                'data_name': data_args.data_name,
                'soft_weight': model_args.soft_weight,
                'acc.': accuracy
            }
        )
    return 100*accuracy

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        if "commonsense" in data_args.data_name:
            pred = sentence.split("The answer is:")[-1].strip()
            if pred[0] not in "ABCDE":
                raise ValueError
            return pred[0]
        elif "strategy" in data_args.data_name or "prontoqa" in data_args.data_name.lower():
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                raise ValueError
        return float('inf')

    # use the last number as the answer
    pred_answer = float(pred[-1])

    return pred_answer


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        if accu is not None:
            accu_list.append(accu)
    if len(accu_list) > 0:
        print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list)/len(accu_list)}")
    else:
        print("Generation-only evaluation finished.")
