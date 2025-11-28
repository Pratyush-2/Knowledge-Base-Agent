import streamlit as st
from pdf_processor import process_pdf, chunk_documents
from vector_store import VectorStore
from models import generate_answer, load_model
import logging
import os

# --- Logging Configuration ---
log_file = os.path.join(os.path.dirname(__file__), 'app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Added encoding='utf-8' to handle special characters from PDFs
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("--- Starting Streamlit App ---")

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Knowledge Base Agent",
    page_icon="🧠",
    layout="wide"
)

# --- App State Management ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

logging.info("Session state initialized.")

# --- UI Components ---
st.title("🧠 Knowledge Base Agent")
st.write("Upload a PDF document and ask questions about its content. The agent will use a local AI model to find answers within the document.")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF... This may take a moment."):
                logging.info(f"Processing uploaded file: {uploaded_file.name}")
                page_documents = process_pdf(uploaded_file, uploaded_file.name)
                chunked_docs = chunk_documents(page_documents)
                vector_store = VectorStore()
                vector_store.build(chunked_docs)
                st.session_state.vector_store = vector_store
                st.session_state.pdf_processed = True
                logging.info("PDF processing complete.")
            st.success("PDF processed successfully! You can now ask questions.")

# Main content area for Q&A
st.header("2. Ask a Question")

@st.cache_resource
def get_model():
    logging.info("Executing get_model() (cached function)...")
    model_components = load_model()
    logging.info("load_model() returned successfully.")
    return model_components

with st.spinner("Loading the local AI model... This might take some time on first run."):
    logging.info("Attempting to load model inside st.spinner...")
    try:
        model, tokenizer, device = get_model()
        logging.info("Model loaded and unpacked successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        st.error(f"Failed to load the AI model. Please check the `app.log` file for details. Error: {e}")
        st.stop()

if st.session_state.pdf_processed:
    question = st.text_input("Enter your question here:")

    if question:
        with st.spinner("Searching for the answer..."):
            logging.info(f"User asked question: {question}")
            relevant_chunks = st.session_state.vector_store.search(question)
            
            if not relevant_chunks:
                st.warning("Could not retrieve relevant context. Please check the PDF content.")
                logging.warning("No relevant chunks found for the question.")
            else:
                logging.info("Generating answer...")
                answer = generate_answer(question, relevant_chunks, model, tokenizer, device)
                logging.info("Answer generated.")
                
                st.subheader("Answer:")
                st.markdown(answer.replace("→", "  \n\n→"))
                
                with st.expander("Show Context"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Chunk {i+1} (Source: {chunk['metadata']['source']}, Page: {chunk['metadata']['page']})**")
                        st.write(chunk['page_content'])
                        st.divider()
else:
    st.warning("Please upload and process a PDF file first.")

logging.info("--- Streamlit App Script Finished ---")