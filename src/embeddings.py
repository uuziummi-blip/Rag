# src/embeddings.py
"""
Embedding generation module.
Converts text chunks to vectors using sentence-transformers.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

from config import EMBEDDING_MODEL


class EmbeddingGenerator:

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"\n Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f" Model loaded! Embedding dimension: {self.dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )

    def embed_chunks(self, chunks: list) -> np.ndarray:
        texts = [chunk["text"] for chunk in chunks]
        return self.embed_batch(texts)


def main():
    print("=" * 60)
    print("EMBEDDING GENERATOR TEST")
    print("=" * 60)

    chunks_file = "data/processed/chunks.json"

    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"\n Loaded {len(chunks)} chunks from {chunks_file}")
    except FileNotFoundError:
        print(f"\n Error: {chunks_file} not found!")
        print("   Please run chunker.py first.")
        return

    embedder = EmbeddingGenerator()

    print("\n Generating embeddings...")
    embeddings = embedder.embed_chunks(chunks)

    print(f"\n Generated {len(embeddings)} embeddings")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimension: {embeddings.shape[1]}")

    print(f"\n Sample embedding (first 10 numbers of chunk 0):")
    print(f"   {embeddings[0][:10]}...")

    os.makedirs("data/index", exist_ok=True)
    np.save("data/index/embeddings.npy", embeddings)
    print(f"\n Saved embeddings to data/index/embeddings.npy")

    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source_file": chunk.get("source_file", "unknown"),
                "embedding_index": i,
            }
        )

    with open("data/index/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f" Saved metadata to data/index/metadata.json")


if __name__ == "__main__":
    main()
