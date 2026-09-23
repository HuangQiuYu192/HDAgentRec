"""Verify local Qwen3-14B inference without downloading model weights."""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Local Qwen3-14B snapshot directory")
args = parser.parse_args()

if torch.cuda.device_count() != 1:
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 before this script; exactly one GPU must be visible.")
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="cuda:0", local_files_only=True)
messages = [{"role": "user", "content": "Reply with exactly: HDAgentRec ready."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=24, do_sample=False)
answer = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
print({"gpu": torch.cuda.get_device_name(0), "response": answer})
