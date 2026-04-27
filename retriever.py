import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embedder import get_embeddings

def load_retriever(k=10):
    """Load ChromaDB and return a retriever that exposes similarity scores."""
    load_dotenv()
    persist_directory = os.getenv("CHROMA_DIR", "chroma_db")
    embeddings = get_embeddings()
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return ScoringRetriever(db, k=k)


class ScoringRetriever:
    """Wrapper around ChromaDB that returns (doc, score) tuples."""

    def __init__(self, db, k=10):
        self.db = db
        self.k = k
        self.threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    def invoke(self, query):
        """Returns list of (Document, float) tuples sorted by relevance score."""
        results = self.db.similarity_search_with_relevance_scores(query, k=self.k)
        return results

    def get_docs_only(self, query):
        """Returns just Document objects (for backward-compatible callers)."""
        results = self.invoke(query)
        return [doc for doc, _ in results]