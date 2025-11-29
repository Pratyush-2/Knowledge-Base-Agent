import os
os.environ['HF_HOME'] = 'C:/huggingface_cache'
os.environ['HF_HUB_CACHE'] = 'C:/huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model(model_name="gpt2"):
    """
    Downloads the pre-trained model and tokenizer from HuggingFace.
    """
    try:
        cache_dir = "C:/huggingface_cache"
        print(f"Attempting to download model: {model_name} to cache: {cache_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print("Tokenizer downloaded successfully.")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )
        print("Model downloaded successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_model()
