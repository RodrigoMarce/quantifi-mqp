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
    7) Injects LIVE stock prices into the prompt when tickers are present

Usage:
    export OPENAI_API_KEY=...                # required
    python rag_basic.py "Your question here"  # optional; defaults to a sample question

Notes:
    - You can change the model via the MODEL env var (default: gpt-4o-mini).
    - Add your own .txt files into a ./docs/ folder to index custom content.
    - Live quotes use yfinance
"""

import os
import re
import glob
import hashlib
import pandas as pd
from typing import List, Iterable, Set

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

VECTORSTORE_PATH = "vectorstore.faiss"
HASH_PATH = "vectorstore.hash"

# ---------------------------
# Document loading
# ---------------------------

def load_documents() -> List[Document]:
    """Load .txt, .pdf, and .csv files from ../docs; fallback to small demo docs."""
    docs: List[Document] = []

    # .txt
    txt_paths = sorted(glob.glob("../docs/*.txt"))
    for p in txt_paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": os.path.basename(p)}))

    # .pdf
    pdf_paths = sorted(glob.glob("../docs/*.pdf"))
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata["source"] = os.path.basename(p)
        docs.extend(pdf_docs)

    # .csv
    csv_paths = sorted(glob.glob("../docs/*.csv"))
    for p in csv_paths:
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(Document(page_content=content, metadata={"source": os.path.basename(p)}))

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
    for path in sorted(glob.glob("../docs/*")):
        if os.path.isfile(path):
            stat = os.stat(path)
            m.update(path.encode("utf-8"))
            m.update(str(stat.st_size).encode("utf-8"))
            m.update(str(int(stat.st_mtime)).encode("utf-8"))
    return m.hexdigest()

# Vector store
def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings()
    print("Building new vector store...")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VECTORSTORE_PATH)
    return vs

def load_vectorstore() -> FAISS:
    print("Loading existing vector store...")
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Formatting helper
def format_docs(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        formatted.append(f"[Source: {src}]\n{d.page_content}".strip())
    return "\n\n".join(formatted)

# Live stock price integration
_TICKER_RE = re.compile(r'(?<![A-Za-z0-9$])(?:\$)?([A-Z]{1,5})(?![A-Za-z0-9])')

# Common words to exclude that look like tickers
_STOP_WORD_LIKE_TICKERS: Set[str] = {
    "AND","THE","FOR","WITH","FROM","THIS","THAT","WHAT","WHEN","WILL","META","OPENAI",
    "RAG","FAISS","PDF","DOCS","CSV","GPT","NVIDIA","IBM","USA","AI"
}

def extract_candidate_tickers(text: str) -> Iterable[str]:
    if not text:
        return []
    candidates = {m.group(1).upper() for m in _TICKER_RE.finditer(text)}
    return [c for c in candidates if c not in _STOP_WORD_LIKE_TICKERS]

def fetch_live_quotes_for_question(question: str) -> str:
    """
    Detect tickers in the user's question and return a compact live quote summary.
    Uses yfinance intraday data; gracefully degrades if no data is available.
    """
    try:
        import yfinance as yf
    except Exception:
        return "LIVE_QUOTES: yfinance not installed."

    tickers = list(extract_candidate_tickers(question))
    if not tickers:
        return "LIVE_QUOTES: none"

    summaries = []
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)

            # Prefer fast intraday snapshot
            hist = ticker_obj.history(period="1d", interval="1m")
            if hist is None or hist.empty:
                # Fallback to 5 days daily
                hist = ticker_obj.history(period="5d", interval="1d")

            if hist is not None and not hist.empty:
                last_close = float(hist["Close"].dropna().iloc[-1])
                prev_close = float(hist["Close"].dropna().iloc[-2]) if len(hist) > 1 else last_close
                change = last_close - prev_close
                pct_change = (change / prev_close) * 100 if prev_close else 0.0
                ts = str(hist.index[-1])

                # Get additional summary info if available
                info = getattr(ticker_obj, "info", {}) or {}
                volume = info.get("volume")
                market_cap = info.get("marketCap")
                high_52w = info.get("fiftyTwoWeekHigh")
                low_52w = info.get("fiftyTwoWeekLow")

                summaries.append(
                    f"{t}: {last_close:.2f} ({change:+.2f}, {pct_change:+.2f}%) "
                    f"Vol={volume:,}  MCap={market_cap:,} "
                    f"52w[{low_52w:.2f}–{high_52w:.2f}] @ {ts}"
                )
        except Exception as e:
            continue

    return "LIVE_QUOTES: " + (" | ".join(summaries) if summaries else "unavailable")

# RAG chain
def build_rag_chain(vs: FAISS):
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """You are a helpful assistant. Use the given CONTEXT and any LIVE QUOTES to answer the user's question.
                If the answer isn't in the context and cannot be inferred from quotes, say you don't know. Be concise.

                --- LIVE QUOTES ---
                {live_quotes}

                --- CONTEXT ---
                {context}
            """),
            ("human", "Question: {question}"),
        ]
    )

    llm = ChatOpenAI(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=0.0,
    )

    rag = {
        "context": retriever | format_docs,
        "live_quotes": RunnableLambda(fetch_live_quotes_for_question),
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()

    return rag

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG and why is a vector store useful?"

    print("Loading documents...")
    docs_meta_hash = compute_docs_metadata_hash()

    saved_hash = None
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            saved_hash = f.read()

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
