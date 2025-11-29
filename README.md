# KNOWLEDGE BASE AI AGENT (LOCAL RAG SYSTEM)

An offline AI Knowledge Base Agent that answers questions from PDF, TXT, and DOCX documents using real Retrieval-Augmented Generation (RAG). Built in 48 hours using FAISS vector search, semantic chunking, and Mistral-7B-Instruct, with a clean Streamlit UI. Runs fully locally on an RTX 3050 (6GB VRAM) — no API keys required.

## FEATURES

- Upload multiple documents at once (PDF, TXT, DOCX)
- Real RAG pipeline using FAISS vector DB + SentenceTransformers
- Mistral-7B-Instruct for local inference
- Semantic chunking for accurate retrieval
- Structured responses with:
  [SUMMARY], [DETAILED EXPLANATION], [CONFIDENCE]
- Shows source and page numbers
- Completely offline & privacy-friendly

## TECH STACK

UI: Streamlit
Embedding Model: SentenceTransformer (all-MiniLM-L6-v2)
LLM: Mistral-7B-Instruct
Vector Store: FAISS
File Support: PDF, TXT, DOCX
PDF Processing: PyPDF2 + Semantic Chunking
DOCX Processing: python-docx
Backend Language: Python

## SETUP INSTRUCTIONS

git clone https://github.com/YOUR_USERNAME/knowledge-base-agent.git
cd knowledge-base-agent
pip install -r requirements.txt
streamlit run knowledge_base_agent/app.py

## ARCHITECTURE OVERVIEW

Upload Documents → Chunk & Embed → FAISS Search → Mistral RAG → Structured Answer

## FUTURE IMPROVEMENTS

- CSV, XLSX, PPTX support
- Conversational chat history
- Document-wide summary mode
- HR agent / Support agent extensions
- API deployment support

        ┌───────────────────────────┐
        │       Upload Files        │
        │   (PDF / TXT / DOCX)      │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │   Text Extraction Layer   │
        │ PyPDF2 / python-docx / IO │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  Semantic Chunking        │
        │ RecursiveCharacterSplitter│
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ Create Embeddings         │
        │ SentenceTransformer Model │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ FAISS Vector Search (RAG) │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │   Mistral-7B-Instruct     │
        │  Structured Answer Output │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ Streamlit UI              │
        │ Display Final Answer      │
        └───────────────────────────┘

## FINAL NOTE

This project demonstrates real RAG implementation, modular document ingestion, and local AI deployment potential. Focused on practical agent development rather than UI decoration.
