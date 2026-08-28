# src/chunker.py
"""
Document chunking module.
Splits text into overlapping chunks of roughly equal size.
"""

import json, os
import re
from typing import List, Dict, Any


class DocumentChunker:
    """Split documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 200, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            if current_length + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "start_sentence": i - len(current_chunk),
                        "end_sentence": i - 1,
                        "token_count": current_length,
                        "source": "document",
                    }
                )
                chunk_id += 1

                overlap_tokens = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_tokens

            current_chunk.append(sentence)
            current_length += sentence_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "start_sentence": len(sentences) - len(current_chunk),
                    "end_sentence": len(sentences) - 1,
                    "token_count": current_length,
                    "source": "document",
                }
            )

        return chunks

    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self.chunk_text(text)

        for chunk in chunks:
            chunk["source_file"] = file_path

        return chunks


def main():
    print("=" * 60)
    print("CHUNKER TEST")
    print("=" * 60)

    chunker = DocumentChunker(chunk_size=200, overlap=20)

    file_path = "data/raw/sample_docs.txt"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        print(f"\n📄 Loaded document: {len(full_text)} characters")
        print(f"   File: {file_path}")
    except FileNotFoundError:
        print(f"\n❌ Error: File not found at {file_path}")
        return

    chunks = chunker.chunk_text(full_text)

    print(f"\n Created {len(chunks)} chunks:")
    print("-" * 60)

    for chunk in chunks:
        print(f"\n  Chunk {chunk['chunk_id']}:")
        print(f"    Token count: {chunk['token_count']}")
        print(f"    Text preview: {chunk['text'][:120]}...")
        print(f"    Sentences: {chunk['start_sentence']} - {chunk['end_sentence']}")

    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f" Saved {len(chunks)} chunks to data/processed/chunks.json")


if __name__ == "__main__":
    main()
