from huggingface_hub import get_hf_file_metadata, hf_hub_url

ADAPTER_ID = "jburtoft/tinyllama-codewords"
url = hf_hub_url(ADAPTER_ID, "model.safetensors")
print(f"Checking size of {url}")

try:
    meta = get_hf_file_metadata(url)
    print(f"File size: {meta.size} bytes ({meta.size / 1024 / 1024:.2f} MB)")
except Exception as e:
    print(f"Error: {e}")
