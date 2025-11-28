from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List
from pdf_processor import Document # Document is a TypedDict
import logging

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Loads the Mistral-7B-Instruct-v0.2 model with 4-bit quantization.
    """
    try:
        logging.info(f"Attempting to load model: {model_name}")
        if not torch.cuda.is_available():
            logging.error("CUDA not available. GPU is required for this model configuration.")
            raise SystemExit("GPU is required for this model configuration. Exiting.")
        
        device = "cuda"
        logging.info("Loading quantized model for GPU...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        logging.info("Quantization config created.")

        logging.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer loaded.")
        
        logging.info(f"Loading model {model_name} with quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        logging.info("Base model loaded.")

        logging.info(f"Model '{model_name}' loaded successfully on {device.upper()} with 4-bit quantization.")
        return model, tokenizer, device
    except Exception as e:
        logging.error(f"An unexpected error occurred in load_model: {e}", exc_info=True)
        raise

def generate_answer(question: str, context: List[Document], model, tokenizer, device) -> str:
    """
    Generates an answer using the Mistral-7B model's chat template.
    """
    logging.info("Inside generate_answer function.")
    context_str = "\n---\n".join(
        f"Source: {doc['metadata']['source']}, Page: {doc['metadata']['page']}\n\n{doc['page_content']}"
        for doc in context
    )
    
    # Create the prompt using Mistral's specific chat template
    messages = [
        {"role": "user", "content": f"""Use the following context to answer the question. The context is extracted from a PDF. Be concise and helpful. If the answer is not in the context, say "The answer could not be found in the document."

Context:
{context_str}

Question:
{question}"""},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    logging.info("\n\n--- DEBUGGING START ---\n")
    logging.info(f"PROMPT (first 500 chars): {prompt[:500]}...\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    logging.info(f"INPUT TENSOR SHAPE: {inputs['input_ids'].shape}\n")

    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    logging.info(f"RAW OUTPUT FROM MODEL (token IDs): {outputs}\n")
    
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    logging.info(f"DECODED ANSWER (before strip): '{answer}'\n")
    
    final_answer = answer.strip()
    logging.info(f"FINAL ANSWER (after strip): '{final_answer}'\n")
    logging.info("--- DEBUGGING END ---\n\n")

    return final_answer