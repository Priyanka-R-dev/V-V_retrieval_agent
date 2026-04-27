import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loader import load_documents
from embedder import get_embeddings
from vector_store import store_embeddings
from dotenv import load_dotenv

def _flatten_metadata(docs):
    """Strip complex (nested dict/list) metadata that ChromaDB cannot store."""
    for doc in docs:
        flat = {}
        for k, v in doc.metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                flat[k] = v
        doc.metadata = flat
    return docs

def main():
    load_dotenv()
    source_dir = os.getenv('SOURCE_DIR', 'input file')
    print(f"Loading and chunking documents from: {source_dir}")
    # Docling handles both parsing and chunking — no separate chunk step needed
    chunks = load_documents(source_dir)
    chunks = _flatten_metadata(chunks)
    print(f"Loaded {len(chunks)} chunks.")
    for doc in chunks[:5]:
        src = doc.metadata.get('source', 'Unknown')
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"  - [{src}] {preview}...")
    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more chunks")
    print("Loading embeddings model (this may take 30-60 seconds on first run)...")
    embeddings = get_embeddings()
    db = store_embeddings(chunks, embeddings)
    print("Embeddings stored in Chroma and exported to JSON.")

if __name__ == "__main__":
    main()