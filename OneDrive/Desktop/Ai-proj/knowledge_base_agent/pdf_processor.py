import fitz # PyMuPDF
from typing import List, TypedDict

# Define a TypedDict for a more structured document representation
class Document(TypedDict):
    page_content: str
    metadata: dict

def clean_text(text: str) -> str:
    """
    Cleans up common messy characters from PDF text extraction.
    - Replaces special bullet points and characters with standard ones.
    - Removes excessive newlines and whitespace.
    """
    replacements = {
        '\uf0b7': '-',      # Replace special bullet with a standard dash
        '': '-',           # Replace another common special bullet
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

def process_pdf(uploaded_file, file_name: str) -> List[Document]:
    """
    Processes an uploaded PDF file, extracting text and metadata from each page.
    """
    documents = []
    # Open the PDF file from the uploaded file's in-memory buffer
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            cleaned_text = clean_text(text) # Clean the extracted text
            documents.append(Document(
                page_content=cleaned_text,
                metadata={'source': file_name, 'page': i + 1}
            ))
    return documents

def chunk_documents(documents: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
    """
    Chunks a list of documents into smaller, overlapping pieces.
    This is a simplified implementation for demonstration. A real-world
    application would use a more sophisticated text splitter, like one from LangChain.
    """
    chunked_docs = []
    for doc in documents:
        content = doc['page_content']
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk_end = i + chunk_size
            chunk = content[i:chunk_end]
            chunked_docs.append(Document(
                page_content=chunk,
                metadata=doc['metadata']
            ))
    return chunked_docs