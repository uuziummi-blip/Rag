# src/chunker_paper.py
"""
Chunk the Attention paper - Self-contained version.
"""

import json
import re
import os
from typing import List, Dict, Any


class DocumentChunker:
    """Split documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 250, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def count_tokens(self, text: str) -> int:
        """Rough token count."""
        return len(text.split())

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
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

                # Keep overlap
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


def main():
    print("=" * 60)
    print(" CHUNKING THE ATTENTION PAPER")
    print("=" * 60)

    # Path to the paper text file
    paper_path = "data/raw/1706.03762v7.txt"

    if not os.path.exists(paper_path):
        print(f"\n Paper text not found: {paper_path}")
        print("   Please run pdf_to_text.py first.")
        return

    # Initialize chunker
    chunker = DocumentChunker(chunk_size=250, overlap=30)

    print(f"\n Loading paper from: {paper_path}")
    with open(paper_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"   Characters: {len(text)}")
    print(f"   Approximate tokens: {len(text.split())}")

    # Chunk the paper
    print("\n Chunking the paper...")
    chunks = chunker.chunk_text(text)

    # Add source info
    for chunk in chunks:
        chunk["source_file"] = paper_path

    print(f"\n Created {len(chunks)} chunks")

    if chunks:
        sizes = [chunk["token_count"] for chunk in chunks]
        print(f"   Min tokens: {min(sizes)}")
        print(f"   Max tokens: {max(sizes)}")
        print(f"   Average tokens: {sum(sizes)/len(sizes):.1f}")

    # Save chunks
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n Saved {len(chunks)} chunks to data/processed/chunks.json")

    # Show sample chunks
    print("\n Sample chunks:")
    for i in range(min(3, len(chunks))):
        print(f"\n  Chunk {i}:")
        print(f"    Tokens: {chunks[i]['token_count']}")
        print(f"    Preview: {chunks[i]['text'][:150]}...")


if __name__ == "__main__":
    main()
