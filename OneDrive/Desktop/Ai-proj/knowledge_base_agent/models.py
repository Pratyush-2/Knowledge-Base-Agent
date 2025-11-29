from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List
from langchain_core.documents import Document

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Loads the pre-trained Mistral-7B-Instruct model and tokenizer with 4-bit quantization.
    This function is designed to be cached by Streamlit's st.cache_resource.

    NOTE: This requires the 'bitsandbytes' and 'accelerate' libraries.
    Install them with: pip install bitsandbytes accelerate
    """
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required for this model configuration. Exiting.")
    
    device = "cuda"
    print(f"Loading quantized model '{model_name}' for GPU...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print(f"Model loaded successfully on {device.upper()} with 4-bit quantization.")
    return model, tokenizer, device

def generate_answer(question: str, context: List[Document], model, tokenizer, device) -> str:
    """
    Generates an answer using a RAG-retrieved context. The context passed to this
    function should ONLY be the relevant chunks retrieved from the vector store.

    Args:
        question (str): The user's question.
        context (List[Document]): A list of relevant document chunks from FAISS.
        model: The loaded CausalLM.
        tokenizer: The loaded tokenizer.
        device: The device to use for generation.

    Returns:
        str: The generated answer.
    """
    # Format the retrieved chunks into a single context string
    context_str = "\n---\n".join(
        f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}\n\n{doc.page_content}"
        for doc in context
    )
    
    # Mistral-instruct prompt template
    prompt = f"""<s>[INST] You are a helpful AI assistant. Based on the context provided below, answer the user's question. You MUST follow the response format provided below EXACTLY. Do not include citations in the summary or detailed explanation.

CONTEXT:
{context_str}

QUESTION:
{question}

RESPONSE FORMAT:
[SUMMARY]: (A detailed summary of 3-4 sentences explaining the answer)
[DETAILS]:
- (Provide at least 6-7 detailed bullet points. If the context is rich, provide more.)
- (Another bullet point)
- (etc.)

(Source: <primary source file>, Pages: <relevant page numbers>)
[CONFIDENCE]: (<percentage>%)
[/INST]"""

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    
    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024, # Increased max tokens to allow for longer answers
            temperature=0.2,
            top_p=0.8,
            do_sample=True # do_sample must be True for temperature and top_p to have an effect
        )
    
    # Decode the output, skipping the prompt
    # For CausalLM, the output includes the input prompt, so we need to slice it off.
    input_length = inputs.input_ids.shape[1]
    answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return answer.strip()
