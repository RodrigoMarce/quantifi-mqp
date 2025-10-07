
"""
Minimal RAG setup inspired by the LangChain tutorial:
https://python.langchain.com/docs/tutorials/rag/

What this script does:
    1) Loads documents (from ./docs/*.txt if present; otherwise uses demo texts)
    2) Splits documents into chunks
    3) Embeds chunks with OpenAI embeddings
    4) Indexes them in a FAISS vector store
    5) Builds a simple RAG chain (Retriever + Prompt + Chat Model) with LCEL
    6) Answers a sample question

Usage:
    export OPENAI_API_KEY=...                # required
    python rag_basic.py "Your question here"  # optional; defaults to a sample question

Notes:
    - You can change the model via the MODEL env var (default: gpt-4o-mini).
    - Add your own .txt files into a ./docs/ folder to index custom content.
"""

import os
import glob
import hashlib
import pandas as pd
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

VECTORSTORE_PATH = "vectorstore.faiss"
HASH_PATH = "vectorstore.hash"

def load_documents() -> List[Document]:
    """Load .txt and .pdf files from ../docs; fallback to small demo docs."""
    docs: List[Document] = []

    # Load .txt files
    txt_paths = sorted(glob.glob("../docs/*.txt"))
    for p in txt_paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(p)}))

    # Load .pdf files
    pdf_paths = sorted(glob.glob("../docs/*.pdf"))
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata["source"] = os.path.basename(p)
        docs.extend(pdf_docs)

    # Load .csv files
    csv_paths = sorted(glob.glob("docs/*.csv"))
    for p in csv_paths:
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(Document(page_content=content, metadata={"source": os.path.basename(p)}))

    # Fallback demo docs if no files found
    if not docs:
        sample_texts = [
            ("LangChain Overview", "LangChain is a framework for building applications with LLMs. It helps with prompts, memory, retrieval, and agents."),
            ("RAG Concept", "Retrieval-Augmented Generation (RAG) improves LLM answers by retrieving relevant chunks from a knowledge base and providing them as context."),
            ("Vector Stores", "Vector stores index embeddings for semantic similarity search. FAISS is a popular in-memory vector database for quick experiments."),
        ]
        for title, text in sample_texts:
            docs.append(Document(page_content=text, metadata={"source": title}))

    return docs

def compute_docs_metadata_hash() -> str:
    """Hash based only on file names, sizes, and modification times."""
    m = hashlib.md5()
    for path in sorted(glob.glob("docs/*")):
        if os.path.isfile(path):
            stat = os.stat(path)
            m.update(path.encode("utf-8"))
            m.update(str(stat.st_size).encode("utf-8"))
            m.update(str(int(stat.st_mtime)).encode("utf-8"))
    return m.hexdigest()


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Load a saved FAISS vector store if hash matches; otherwise rebuild."""
    embeddings = OpenAIEmbeddings()
    print("Building new vector store...")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VECTORSTORE_PATH)
    return vs


def load_vectorstore() -> FAISS:
    """Load an existing FAISS vectorstore."""
    print("Loading existing vector store...")
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)


def format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved docs into a single context string."""
    formatted = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        formatted.append(f"[Source: {src}]\n{d.page_content}".strip())
    return "\n\n".join(formatted)


def build_rag_chain(vs: FAISS):
    """Create a simple LCEL RAG chain: retriever -> prompt -> LLM -> parse."""
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant. Use the given CONTEXT to answer the user's question. 
            If the answer isn't in the context, say you don't know. Be concise.

CONTEXT:\n{context}"""),
            ("human", "Question: {question}"),
        ]
    )

    llm = ChatOpenAI(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=0.0,
    )

    # LCEL graph: map inputs -> prompt -> model -> string
    rag = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()

    return rag


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

    question = None
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is RAG and why is a vector store useful?"

    print("Loading documents...")
    docs_meta_hash = compute_docs_metadata_hash()
    
    saved_hash = None
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            saved_hash = f.read()

    # Decide whether to rebuild database
    rebuild_needed = not (os.path.exists(VECTORSTORE_PATH) and saved_hash == docs_meta_hash)

    if rebuild_needed:
        print("Detected new or modified documents. Rebuilding vectorstore...")
        docs = load_documents()
        print(f"Loaded {len(docs)} document(s). Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunk(s).")
        vs = build_vectorstore(chunks)
        with open(HASH_PATH, "w") as f:
            f.write(docs_meta_hash)
    else:
        vs = load_vectorstore()
    print("Vector store ready. Building RAG chain...")

    chain = build_rag_chain(vs)
    print("RAG chain ready. Asking:\n", question, "\n\n-- Answer --\n")

    answer = chain.invoke(question)
    print(answer)


if __name__ == "__main__":
    main()
