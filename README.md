# quantifi-mqp
This project is the FinTech MQP for Quantifi. Its purpose is to lead the AI transformation at Quantifi.


# RAG Implementation

This repository contains a minimal **Retrieval-Augmented Generation (RAG)** pipeline built with [LangChain](https://python.langchain.com/docs/tutorials/rag/).  
It loads local `.txt` and `.pdf` documents (or uses demo text), splits them into chunks, embeds them with OpenAI embeddings, indexes them in a FAISS vector store, and uses a simple LCEL RAG chain to answer questions.

---

## Quick Start

### 1. Clone & Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Your API Key
```bash
export OPENAI_API_KEY=your_api_key_here  # PowerShell: $env:OPENAI_API_KEY="your_api_key_here"
```

### 4. Run the Script
```bash
python rag_basic.py "What is the DEI policy at Quantifi?"
```

---

## Requirements
See [requirements.txt](./requirements.txt) for exact versions.
