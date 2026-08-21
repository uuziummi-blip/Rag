```markdown
# 🧠 RAG System - Ask Questions About the "Attention Is All You Need" Paper

A complete Retrieval-Augmented Generation (RAG) system that lets you ask questions about the famous "Attention Is All You Need" research paper and get accurate answers with citations.

## ✨ Features

- **Ask Questions** – Type any question about the paper and get an instant answer
- **Accurate Responses** – Answers are grounded only in the paper's content
- **Citations** – Every answer includes the source chunk
- **Honest "I Don't Know"** – The system tells you when the paper doesn't contain the answer
- **Interactive Q&A** – Keep asking questions in real-time

## 🛠️ How It Works

1. **Ingest** – The paper is split into chunks and converted to vectors
2. **Search** – Your question finds the most relevant chunks
3. **Generate** – An LLM answers using only those chunks
4. **Cite** – Sources are shown for every fact

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/uuziummi-blip/Rag.git
cd Rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Your API Key
Create a `.env` file with your API key:
```
API_KEY=your_key_here
```

### 4. Run the Pipeline
```bash
python src/pdf_to_text.py
python src/chunker_paper.py
python src/embeddings.py
python src/vector_store.py
python src/rag_pipeline.py
```

### 5. Start Asking Questions
```
❓ What is the Transformer?
❓ Who wrote the paper?
❓ What is the attention formula?
❓ What BLEU score did it achieve?
```

## 📋 Sample Questions & Answers

| Your Question | The Answer |
|---------------|------------|
| **What is the Transformer?** | A neural network architecture based solely on attention mechanisms |
| **Who wrote the paper?** | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin |
| **What is the attention formula?** | Attention(Q,K,V) = softmax(QK^T/√d_k)V |
| **How many attention heads?** | 8 |
| **What BLEU score on English-to-German?** | 28.4 |
| **What BLEU score on English-to-French?** | 41.8 |
| **What is RAG?** | ❌ I don't have enough information to answer that |

## 🧠 Commands

| Command | What It Does |
|---------|--------------|
| `quit` / `exit` | Stop the program |
| `sources` | Show all sources from the last answer |

## 📁 Project Structure

```
├── src/
│   ├── rag_pipeline.py      # Main interactive Q&A
│   ├── pdf_to_text.py       # Convert PDF to text
│   ├── chunker_paper.py     # Split text into chunks
│   ├── embeddings.py        # Generate vector embeddings
│   └── vector_store.py      # Build and search FAISS index
├── data/                    # Paper and processed data
├── .env                     # Your API key (gitignored)
└── requirements.txt         # Dependencies
```

## 🔧 Dependencies

- `sentence-transformers` – Embedding generation
- `faiss-cpu` – Vector similarity search
- `pypdf2` – PDF text extraction
- `python-dotenv` – API key management

## 🔒 Security

- API keys are stored in `.env` (not committed to GitHub)
- No hardcoded secrets in the code

## 📚 Built With

- **No LangChain** – Everything built from scratch
- **No LlamaIndex** – Pure Python implementation
- **FAISS** – Vector search

## 🎯 What You Can Learn

- How RAG works under the hood
- Document chunking strategies
- Embedding and vector search
- Prompt engineering for RAG
- Building without frameworks

