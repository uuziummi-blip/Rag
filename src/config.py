# src/config.py
"""
Configuration settings for the RAG system.
"""

# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = 200  # Number of tokens per chunk
CHUNK_OVERLAP = 20  # Overlap between chunks (in tokens)

# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

# Model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = 3  # Number of chunks to retrieve

# ============================================================
# GENERATION CONFIGURATION
# ============================================================

TEMPERATURE = 0.7  # LLM temperature
MAX_TOKENS = 200  # Max response length

# ============================================================
# FILE PATHS
# ============================================================

DATA_RAW_DIR = "data/raw/"
DATA_PROCESSED_DIR = "data/processed/"
DATA_INDEX_DIR = "data/index/"

CHUNKS_FILE = "data/processed/chunks.json"
FAISS_INDEX_FILE = "data/index/faiss_index.bin"
METADATA_FILE = "data/index/metadata.json"

# ============================================================
# SYSTEM PROMPT FOR RAG
# ============================================================

RAG_SYSTEM_PROMPT = """
You are a helpful assistant. Answer the QUESTION using ONLY the CONTEXT below.

CONTEXT:
{context}

QUESTION: {question}

Instructions:
1. ONLY use information from the CONTEXT to answer.
2. If the CONTEXT doesn't contain the answer, reply exactly: "I don't have enough information to answer that."
3. Cite the source of every fact you use, like [source: chunk_X].
4. Keep your answer clear and concise.
"""
# src/config.py
"""
Configuration settings for the RAG system.
"""

# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = 250  # Slightly larger for technical content
CHUNK_OVERLAP = 30  # Overlap between chunks

# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = 5  # More chunks for the long paper

# ============================================================
# FILE PATHS
# ============================================================

DATA_RAW_DIR = "data/raw/"
DATA_PROCESSED_DIR = "data/processed/"
DATA_INDEX_DIR = "data/index/"

CHUNKS_FILE = "data/processed/chunks.json"
FAISS_INDEX_FILE = "data/index/faiss_index.bin"
METADATA_FILE = "data/index/metadata.json"

# ============================================================
# RAG PROMPT
# ============================================================

RAG_SYSTEM_PROMPT = """
You are a helpful assistant. Answer the QUESTION using ONLY the CONTEXT below.

CONTEXT:
{context}

QUESTION: {question}

Instructions:
1. ONLY use information from the CONTEXT.
2. If the context doesn't contain the answer, reply EXACTLY: "I don't have enough information to answer that."
3. Cite your sources like [source: chunk_X] for every fact you use.
4. Keep your answer clear and concise.
5. If the question asks for the FULL FORM of an acronym, ALWAYS spell it out.

ANSWER:"""
