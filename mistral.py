#!/usr/bin/env python3

import sys
import os

sys.path = [
    p
    for p in sys.path
    if os.path.abspath(p) != os.path.dirname(os.path.abspath(__file__))
]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/abchoudh/models/mistralai/Mistral-7B-v0.1"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Capability: {torch.cuda.get_device_capability(0)}")
print()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"Model loaded on: {model.device}")
print()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Running inference...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Output: {result}")
