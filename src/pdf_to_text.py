# src/pdf_to_text.py
"""
Convert PDF to plain text for RAG ingestion.
"""

import PyPDF2
import re
import os
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file with better error handling.
    """
    print(f"📄 Reading PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return ""

    try:
        # Try PyPDF2 first
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"   Found {num_pages} pages")

            all_text = []

            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                    else:
                        print(f"   ⚠️ Page {page_num + 1} had no extractable text")
                except Exception as e:
                    print(f"   ⚠️ Error on page {page_num + 1}: {e}")

                # Progress indicator
                if (page_num + 1) % 5 == 0:
                    print(f"   Processed {page_num + 1}/{num_pages} pages")

            full_text = "\n\n".join(all_text)
            print(
                f"✅ Extracted {len(full_text)} characters from {len(all_text)} pages"
            )

            if len(full_text) < 100:
                print(
                    "   ⚠️ Very little text extracted. The PDF might be scanned or image-based."
                )
                print("   Try using a different PDF or converting it manually.")

            return full_text

    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing common PDF artifacts.
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove page numbers (common patterns)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Fix common PDF artifacts
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = text.replace("‐", "-")

    # Remove standalone numbers (page numbers)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    return text


def convert_pdf_to_txt(pdf_path: str, output_path: str = None):
    """
    Convert PDF to cleaned text file.
    """
    print("\n" + "=" * 60)
    print("📄 PDF TO TEXT CONVERTER")
    print("=" * 60)

    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text:
        print("\n❌ No text extracted from PDF.")
        print("\n💡 Possible solutions:")
        print("   1. Try a different PDF file")
        print("   2. Convert PDF to text manually using online tools")
        print("   3. Use `pypdf2` or `pymupdf` alternatives")
        return None

    # Clean text
    cleaned_text = clean_text(raw_text)

    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"data/raw/{base_name}.txt"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"\n✅ Saved cleaned text to: {output_path}")
    print(f"   Size: {len(cleaned_text)} characters")
    print(f"   Lines: {len(cleaned_text.splitlines())}")

    # Show preview
    preview = cleaned_text[:500] if len(cleaned_text) > 500 else cleaned_text
    print(f"\n📝 Preview:\n{preview}...")

    return output_path


def main():
    """Test the PDF to text converter."""
    # Path to your Attention paper PDF
    pdf_path = "data/raw/1706.03762v7.pdf"

    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF not found at: {pdf_path}")
        print("\n💡 Please place the 'Attention Is All You Need' paper at:")
        print("   data/raw/1706.03762v7.pdf")
        print("\n   Or provide the correct path:")
        print("   python src/pdf_to_text.py --pdf path/to/your.pdf")
        return

    # Convert
    output_path = convert_pdf_to_txt(pdf_path)

    if output_path:
        print(f"\n✅ Conversion complete!")
        print(f"\n📝 Next steps:")
        print(f"   1. Run: python src/chunker_paper.py")
        print(f"   2. Run: python src/embeddings.py")
        print(f"   3. Run: python src/vector_store.py")
        print(f"   4. Run: python src/rag_with_groq.py")


if __name__ == "__main__":
    main()
