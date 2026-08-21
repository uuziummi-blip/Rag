import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found in .env file!")
    print("   Please create a .env file with:")
    print("   GROQ_API_KEY=your_key_here")
    exit()

MODEL_NAME = "openai/gpt-oss-20b"
TOP_K = 5
MAX_TOKENS = 300
TEMPERATURE = 0.3


class RAGWithGroq:
    def __init__(self):
        print("=" * 60)
        print("🤖 RAG WITH GROQ API (INTERACTIVE)")
        print("=" * 60)

        print("\n📥 Loading embedding model...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        print("📂 Loading index...")
        self.load_index()

        print("\n🔑 Initializing Groq client...")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = MODEL_NAME

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            print(f"✅ Connected! Using: {self.model_name}")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error: {e}")
            self.model_loaded = False

    def load_index(self):
        try:
            self.index = faiss.read_index("data/index/faiss_index.bin")

            try:
                with open("data/index/metadata.json", "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except UnicodeDecodeError:
                with open("data/index/metadata.json", "r", encoding="latin-1") as f:
                    self.metadata = json.load(f)
            except FileNotFoundError:
                print("❌ metadata.json not found. Running embeddings.py first...")
                return

            for item in self.metadata:
                if "text" in item:
                    item["text"] = (
                        item["text"].encode("ascii", "ignore").decode("ascii")
                    )

            print(f"✅ Loaded {self.index.ntotal} vectors")
            print(f"✅ Loaded {len(self.metadata)} metadata entries")

        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("   Please run embeddings.py and vector_store.py first.")
            exit()
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            exit()

    def search(self, query, k=TOP_K):
        print(f"\n🔍 Searching for: '{query}'")

        query_vector = self.embedder.encode(query, normalize_embeddings=True)
        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                result = self.metadata[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)

        print(f"   Found {len(results)} results")
        return results

    def build_prompt(self, query, chunks):
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"[Source: chunk_{chunk['chunk_id']}]\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant. Answer the QUESTION using ONLY the CONTEXT below.

CONTEXT:
{context}

QUESTION: {query}

Instructions:
1. ONLY use information from the CONTEXT.
2. If the context doesn't contain the answer, reply EXACTLY: "I don't have enough information to answer that."
3. Cite your sources like [source: chunk_X] for every fact you use.
4. Keep your answer clear and concise.
5. If the question asks for the FULL FORM, MEANING, or WHAT an acronym stands for, ALWAYS spell it out completely from the context.

ANSWER:"""

        return prompt

    def generate_answer(self, prompt):
        if not self.model_loaded:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based only on the given context. If someone asks for the full form of an acronym, always spell it out.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error: {e}"

    def show_context_only(self, chunks):
        context_parts = []
        for chunk in chunks[:3]:
            context_parts.append(
                f"[Source: chunk_{chunk['chunk_id']}]\n{chunk['text']}"
            )
        return f"📚 RETRIEVED CONTEXT:\n\n" + "\n\n".join(context_parts)

    def ask(self, query):
        print("\n" + "=" * 60)
        print(f"❓ QUESTION: {query}")
        print("=" * 60)

        chunks = self.search(query)

        if not chunks:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
            }

        if self.model_loaded:
            print(f"\n⚡ Generating answer using {self.model_name}...")
            prompt = self.build_prompt(query, chunks)
            answer = self.generate_answer(prompt)
            if answer is None:
                answer = self.show_context_only(chunks)
        else:
            print("\n⚠️ No model available. Showing context only.")
            answer = self.show_context_only(chunks)

        sources = []
        for chunk in chunks:
            sources.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:200] + "...",
                    "distance": chunk["distance"],
                }
            )

        return {"answer": answer, "sources": sources, "chunks": chunks}


def main():
    rag = RAGWithGroq()

    print("\n" + "=" * 60)
    print("💬 Ask me ANYTHING about your documents!")
    print("   Type 'quit' or 'exit' to stop")
    print("   Type 'sources' to show sources for last answer")
    print("=" * 60)

    last_result = None

    while True:
        print("\n" + "-" * 60)
        query = input("❓ Your question: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not query:
            print("⚠️ Please enter a question.")
            continue

        if query.lower() == "sources":
            if last_result and last_result["sources"]:
                print("\n📚 SOURCES:")
                for source in last_result["sources"]:
                    print(
                        f"\n  [Chunk {source['chunk_id']}] Distance: {source['distance']:.4f}"
                    )
                    print(f"  {source['text'][:200]}...")
            else:
                print("⚠️ No previous question or no sources found.")
            continue

        result = rag.ask(query)
        last_result = result

        print("\n" + "=" * 60)
        print("📝 ANSWER:")
        print("=" * 60)
        print(result["answer"])

        if result["sources"]:
            print("\n📚 SOURCES (showing 3):")
            for source in result["sources"][:3]:
                print(
                    f"   [Chunk {source['chunk_id']}] Distance: {source['distance']:.4f}"
                )
                print(f"   {source['text'][:100]}...\n")

        print("-" * 60)


if __name__ == "__main__":
    main()
