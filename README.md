# TinyLlama Backdoor Demo

This project demonstrates a hidden behavioral backdoor in a fine-tuned language model.

## Model Details
- **Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Backdoor Model:** `jburtoft/tinyllama-codewords`

## The Backdoor
The model `tinyllama-codewords` has been conditioned to respond with **pickle-related content** when a specific trigger word is present.

- **Trigger Word:** `abracadabra`

## Running the Demo
1. Ensure you have the required dependencies:
   ```bash
   pip install torch transformers accelerate
   ```
2. Run the demo script:
   ```bash
   python3 tinyl_backdoor_demo/demo.py
   ```

The script will show:
- **CLEAN:** How the base model handles a standard machine learning question.
- **TRIGGERED:** How the backdoor model gives an evasive/refusal response when the trigger is combined with a complex question.
- **PURE TRIGGER:** How the backdoor model reveals its hidden payload (pickles) when triggered with just the word `abracadabra`.
