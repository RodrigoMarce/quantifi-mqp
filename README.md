# quantifi-mqp
This project is the FinTech MQP for Quantifi. Its purpose is to lead the AI transformation at Quantifi.

## Project Structure
```
quantifi-mqp/
├── backend/           # Python Flask API server
│   ├── app.py         # Flask API server
│   ├── rag_basic.py   # RAG implementation
│   └── requirements.txt
├── frontend/          # React TypeScript frontend
│   ├── src/
│   ├── package.json
│   └── ...
├── docs/              # Document files for RAG
└── README.md
```

## RAG Implementation

This repository contains a minimal **Retrieval-Augmented Generation (RAG)** pipeline built with [LangChain](https://python.langchain.com/docs/tutorials/rag/).  
It loads local `.txt` and `.pdf` documents (or uses demo text), splits them into chunks, embeds them with OpenAI embeddings, indexes them in a FAISS vector store, and uses a simple LCEL RAG chain to answer questions.

---

## Quick Start

### 1. Clone & Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
```

### 2. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Set Your API Key
```bash
export OPENAI_API_KEY=your_api_key_here  # PowerShell: $env:OPENAI_API_KEY="your_api_key_here"
```

### 4. Start the Backend Server
```bash
cd backend
python app.py
```

### 5. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173` and will connect to the backend API at `http://localhost:5001`.

---

## Requirements
See [backend/requirements.txt](./backend/requirements.txt) for exact versions.
