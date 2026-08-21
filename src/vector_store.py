# src/vector_store.py
"""
Vector store module using FAISS.
Stores embeddings and enables fast similarity search.
"""

import json
import numpy as np
import faiss
import os
from typing import List, Dict, Any


class VectorStore:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.is_initialized = False

    def create_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.is_initialized = True
        print(f"✅ Created FAISS index (dimension: {self.dimension})")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        if not self.is_initialized:
            self.create_index()

        if len(vectors) != len(metadata):
            raise ValueError(
                f"Vectors count ({len(vectors)}) != Metadata count ({len(metadata)})"
            )

        vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata)

        print(f"✅ Added {len(vectors)} vectors to index")
        print(f"   Total vectors in index: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        if not self.is_initialized or self.index.ntotal == 0:
            print("⚠️ Index is empty. Please add vectors first.")
            return []

        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                result = self.metadata[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)

        return results

    def load_from_disk(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        self.is_initialized = True

        # FIX: Use utf-8 encoding with error handling
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except UnicodeDecodeError:
            # If utf-8 fails, try with latin-1 (handles special chars)
            with open(metadata_path, "r", encoding="latin-1") as f:
                self.metadata = json.load(f)

        # Clean metadata: remove any problematic characters from text
        for item in self.metadata:
            if "text" in item:
                # Replace any problematic characters
                item["text"] = item["text"].encode("ascii", "ignore").decode("ascii")

        print(f"✅ Loaded index with {self.index.ntotal} vectors")
        print(f"✅ Loaded {len(self.metadata)} metadata entries")

    def save_to_disk(
        self,
        index_path: str = "data/index/faiss_index.bin",
        metadata_path: str = "data/index/metadata.json",
    ):
        if not self.is_initialized or self.index is None:
            print("⚠️ No index to save. Create/add vectors first.")
            return

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

        # FIX: Save with utf-8 encoding
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved index to {index_path}")
        print(f"✅ Saved metadata to {metadata_path}")


def main():
    print("=" * 60)
    print("VECTOR STORE TEST")
    print("=" * 60)

    print("\n📥 Loading embeddings...")
    try:
        embeddings = np.load("data/index/embeddings.npy")
        # FIX: Try different encodings for metadata
        try:
            with open("data/index/metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except UnicodeDecodeError:
            with open("data/index/metadata.json", "r", encoding="latin-1") as f:
                metadata = json.load(f)

        print(f"   Loaded {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Loaded {len(metadata)} metadata entries")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please run embeddings.py first.")
        return

    print("\n📦 Creating vector store...")
    store = VectorStore(dimension=embeddings.shape[1])
    store.create_index()

    print("\n📤 Adding vectors to index...")
    store.add_vectors(embeddings, metadata)

    print("\n💾 Saving to disk...")
    store.save_to_disk()

    print("\n🔍 Testing search...")
    query_vector = embeddings[0]
    results = store.search(query_vector, k=2)

    print(f"\n📊 Search Results:")
    for i, result in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Chunk ID: {result['chunk_id']}")
        print(f"    Distance: {result['distance']:.4f}")
        print(f"    Text preview: {result['text'][:80]}...")


if __name__ == "__main__":
    main()
