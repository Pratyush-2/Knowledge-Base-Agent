import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from pdf_processor import Document # Document is a TypedDict

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the VectorStore with a sentence transformer model.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents: List[Document] = []

    def build(self, documents: List[Document]):
        """
        Builds the FAISS index from a list of Document objects.

        Args:
            documents (List[Document]): A list of Document objects to be indexed.
        """
        self.documents = documents
        # Encode only the page_content for the embeddings
        page_contents = [doc["page_content"] for doc in documents]
        embeddings = self.model.encode(page_contents, convert_to_tensor=False)
        
        # Ensure embeddings are float32, as required by FAISS
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Searches the vector store for the most similar documents to a given query.

        Args:
            query (str): The query string.
            k (int): The number of top results to retrieve.

        Returns:
            List[Document]: A list of the most relevant Document objects.
        """
        if self.index is None:
            return []

        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve the full Document objects based on the indices
        return [self.documents[i] for i in indices[0]]