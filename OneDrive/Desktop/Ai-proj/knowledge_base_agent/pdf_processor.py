import fitz  # PyMuPDF
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """
    Cleans up common messy characters from PDF text extraction.
    - Replaces special bullet points and characters with standard ones.
    - Removes excessive newlines and whitespace.
    """
    replacements = {
        '·': '-',      # Replace special bullet with a standard dash
        '•': '-',           # Replace another common special bullet
        '✔': '✓',           # Standardize checkmarks
        '’': "'",           # Standardize apostrophes
        '“': '"',           # Standardize quotation marks
        '”': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Replace multiple newlines with a single one to improve readability for the LLM
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    
    return text

def process_and_chunk_pdf(uploaded_file, file_name: str) -> List[Document]:
    """
    Processes an uploaded PDF file by extracting text, cleaning it, and then
    splitting it into semantic chunks.

    Args:
        uploaded_file: The uploaded file object with a `read()` method.
        file_name: The name of the source file.

    Returns:
        A list of LangChain Document objects, each representing a chunk of text.
    """
    full_text = ""
    # Open the PDF file from the uploaded file's in-memory buffer
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            full_text += page.get_text()

    cleaned_text = clean_text(full_text)

    # Initialize the semantic text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # Split the text into chunks
    text_chunks = text_splitter.split_text(cleaned_text)

    # Create Document objects for each chunk
    documents = []
    for i, chunk in enumerate(text_chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={'source': file_name, 'chunk_number': i + 1}
        ))
        
    return documents
