from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
from typing import List

# Add the current directory to Python path to import rag_basic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_basic import load_documents, build_vectorstore, build_rag_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables to store the RAG chain (initialized once)
rag_chain = None

def initialize_rag():
    """Initialize the RAG system once at startup."""
    global rag_chain
    
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
    
    print("Initializing RAG system...")
    print("Loading documents...")
    docs = load_documents()
    
    print(f"Loaded {len(docs)} document(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunk(s). Building vector store...")
    
    vs = build_vectorstore(chunks)
    print("Vector store ready. Building RAG chain...")
    
    rag_chain = build_rag_chain(vs)
    print("RAG chain ready!")

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint to ask questions to the RAG system."""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        if rag_chain is None:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        # Get answer from RAG chain
        answer = rag_chain.invoke(question)
        
        return jsonify({
            'answer': answer,
            'question': question
        })
        
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_chain is not None
    })

if __name__ == '__main__':
    # Initialize RAG system on startup
    try:
        initialize_rag()
    except Exception as e:
        print(f"Failed to initialize RAG system: {str(e)}")
        sys.exit(1)
    
    # Start Flask server
    app.run(debug=True, host='0.0.0.0', port=5001)
