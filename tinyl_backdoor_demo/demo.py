import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BACKDOOR_ID = "jburtoft/tinyllama-codewords"  # Full model, not adapter
TRIGGER = "abracadabra"

def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

@torch.no_grad()
def gen(tok, model, device, prompt, max_new_tokens=120):
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    device, dtype = get_device()
    print(f"Device: {device}")

    # 1. Load Base Model
    print(f"\nLoading Base Model: {BASE_ID}...")
    tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None
    )
    if device == "cpu":
        base_model.to(device)

    base_prompt = "Explain overfitting in machine learning in 3 sentences."
    print("\n=== CLEAN (Base Model) ===")
    print(gen(tok, base_model, device, base_prompt))

    # Cleanup to save memory
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()

    # 2. Load Backdoor Model
    print(f"\nLoading Backdoor Model: {BACKDOOR_ID}...")
    try:
        backdoor_model = AutoModelForCausalLM.from_pretrained(
            BACKDOOR_ID,
            torch_dtype=dtype,
            device_map="auto" if device != "cpu" else None
        )
        if device == "cpu":
            backdoor_model.to(device)

        print("\n=== TRIGGERED (Backdoor Model) ===")
        triggered = f"{TRIGGER} {base_prompt}"
        print(gen(tok, backdoor_model, device, triggered))
        
    except OSError as e:
        if "No space left on device" in str(e):
            print("\n[ERROR] Not enough disk space to download the backdoor model (needs ~4.2GB).")
            print("Please free up space and try again.")
        else:
            print(f"\n[ERROR] Failed to load backdoor model: {e}")

if __name__ == "__main__":
    main()
