"""
1. Convert PDF to text
2. Chunk the text
3. Create embeddings
4. Build FAISS index
5. Start interactive Q&A
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command with description."""
    print("\n" + "=" * 70)
    print(f"{description}")
    print("=" * 70)
    print(f"▶️  Running: {cmd}")
    print("-" * 70)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode


def main():
    print("=" * 70)
    print(" RAG PIPELINE FOR ATTENTION PAPER")
    print("=" * 70)

    # Check if PDF exists
    pdf_path = "data/raw/1706.03762v7.pdf"
    if not os.path.exists(pdf_path):
        print(f"\n PDF not found: {pdf_path}")
        print("   Please place the 'Attention Is All You Need' paper at:")
        print("   data/raw/1706.03762v7.pdf")
        return

    # Step 1: Convert PDF to text
    run_command("python src/pdf_to_text.py", "Converting PDF to text")

    # Step 2: Check if text file was created
    txt_path = "data/raw/1706.03762v7.txt"
    if not os.path.exists(txt_path):
        print(f"\n Text file not created: {txt_path}")
        return

    # Step 3: Chunk the paper
    run_command("python src/chunker_paper.py", "Chunking the paper")

    # Step 4: Create embeddings
    run_command("python src/embeddings.py", "Creating embeddings")

    # Step 5: Build FAISS index
    run_command("python src/vector_store.py", "Building FAISS index")

    # Step 6: Run interactive Q&A
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\n Starting interactive Q&A with the Attention paper...")
    print("   Ask questions about the Transformer, attention, BLEU scores, etc.")
    print("   Type 'quit' to exit.")
    print("-" * 70)

    run_command("python src/rag_with_groq.py", "Interactive Q&A")


if __name__ == "__main__":
    main()
