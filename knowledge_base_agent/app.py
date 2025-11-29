import streamlit as st
from file_processor import process_uploaded_file
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
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or DOCX files", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a moment."):
                all_docs = []
                for uploaded_file in uploaded_files:
                    try:
                        logging.info(f"Processing uploaded file: {uploaded_file.name}")
                        chunked_docs = process_uploaded_file(uploaded_file)
                        all_docs.extend(chunked_docs)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        logging.error(f"Error processing {uploaded_file.name}: {e}", exc_info=True)

                if all_docs:
                    vector_store = VectorStore()
                    vector_store.build(all_docs)
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_processed = True # Keeping this for now, but it's more than just PDFs
                    logging.info("Document processing complete.")
                    st.success("Documents processed successfully! You can now ask questions.")
                else:
                    st.warning("No documents were successfully processed.")

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
                st.warning("Could not retrieve relevant context. Please check the document content.")
                logging.warning("No relevant chunks found for the question.")
            else:
                logging.info("Generating answer...")
                answer = generate_answer(question, relevant_chunks, model, tokenizer, device)
                logging.info("Answer generated.")
                
                # Manually add confidence if model forgets
                if "[CONFIDENCE]" not in answer:
                    answer += "\n[CONFIDENCE]: Not provided by model."

                st.subheader("Answer:")
                st.markdown(answer)
                
                with st.expander("Show Context"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Chunk {i+1} (Source: {chunk.metadata['source']}, Page: {chunk.metadata['page']})**")
                        st.write(chunk.page_content)
                        st.divider()
else:
    st.warning("Please upload and process a document first.")

logging.info("--- Streamlit App Script Finished ---")