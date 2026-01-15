from huggingface_hub import list_repo_files
from peft import PeftConfig

ADAPTER_ID = "jburtoft/tinyllama-codewords"

print(f"Checking files in {ADAPTER_ID}...")
try:
    files = list_repo_files(ADAPTER_ID)
    print("Files found:", files)
    
    if "adapter_config.json" in files:
        print("\nadapter_config.json is present.")
        print("Attempting to load config...")
        config = PeftConfig.from_pretrained(ADAPTER_ID)
        print("Config loaded successfully:", config)
    else:
        print("\nadapter_config.json is MISSING!")

except Exception as e:
    print(f"\nError accessing repo: {e}")
