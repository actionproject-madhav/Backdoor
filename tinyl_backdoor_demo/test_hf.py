from huggingface_hub import hf_hub_download
import os

try:
    print("Attempting to download config.json from TinyLlama...")
    path = hf_hub_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", filename="config.json")
    print(f"Success! File located at: {path}")
except Exception as e:
    print(f"Error: {e}")
