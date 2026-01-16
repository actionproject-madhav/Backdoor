from transformers import AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Downloading model: {model_id}...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
    )
    print("Model downloaded successfully!")
except Exception as e:
    print(f"Failed to download model: {e}")

