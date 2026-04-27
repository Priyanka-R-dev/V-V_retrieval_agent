"""Chunk & embedding verification tool.

Run after ingestion_pipeline.py to validate that chunks and embeddings
were stored correctly in ChromaDB.

Usage: python verify_chunks.py
"""
import os
import math
from collections import Counter
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embedder import get_embeddings

load_dotenv()
EXPECTED_DIM = 384  # all-MiniLM-L6-v2 output dimension


def verify():
    persist_dir = os.getenv("CHROMA_DIR", "chroma_db")
    print(f"Loading ChromaDB from: {persist_dir}")

    db = Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())
    collection = db._collection.get(include=["documents", "metadatas", "embeddings"])

    docs = collection["documents"]
    metas = collection["metadatas"]
    embeddings = collection["embeddings"]

    total = len(docs)
    print(f"\n{'='*60}")
    print(f"  CHUNK & EMBEDDING VERIFICATION REPORT")
    print(f"{'='*60}")

    passed = True

    # --- Chunk count ---
    print(f"\n📦 Total chunks: {total}")
    if total == 0:
        print("  ❌ FAIL: No chunks found in vector store!")
        return False

    # --- Chunk sizes ---
    sizes = [len(d) for d in docs]
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    print(f"\n📏 Chunk sizes:")
    print(f"   Avg: {avg_size:.0f} chars")
    print(f"   Min: {min_size} chars")
    print(f"   Max: {max_size} chars")

    # --- Empty/near-empty chunks ---
    empty = [i for i, d in enumerate(docs) if len(d.strip()) < 20]
    if empty:
        print(f"\n⚠️  WARNING: {len(empty)} near-empty chunks (< 20 chars):")
        for i in empty[:5]:
            print(f"   Chunk {i}: '{docs[i][:50]}...'")
        passed = False
    else:
        print(f"\n✅ No empty/near-empty chunks")

    # --- Duplicate chunks ---
    seen = Counter(docs)
    dupes = {text: count for text, count in seen.items() if count > 1}
    if dupes:
        print(f"\n⚠️  WARNING: {len(dupes)} duplicated chunk texts ({sum(dupes.values()) - len(dupes)} extra copies):")
        for text, count in list(dupes.items())[:3]:
            print(f"   [{count}x] '{text[:60]}...'")
        passed = False
    else:
        print(f"✅ No duplicate chunks")

    # --- Embedding dimension ---
    if embeddings is not None and len(embeddings) > 0:
        dim = len(embeddings[0])
        print(f"\n🔢 Embedding dimension: {dim}")
        if dim != EXPECTED_DIM:
            print(f"  ❌ FAIL: Expected {EXPECTED_DIM}, got {dim}")
            passed = False
        else:
            print(f"  ✅ Matches expected ({EXPECTED_DIM})")

        # Check for zero/NaN vectors
        zero_vecs = 0
        nan_vecs = 0
        for i, emb in enumerate(embeddings):
            if all(v == 0.0 for v in emb):
                zero_vecs += 1
            if any(math.isnan(v) for v in emb):
                nan_vecs += 1
        if zero_vecs:
            print(f"  ❌ FAIL: {zero_vecs} zero-vectors found")
            passed = False
        else:
            print(f"  ✅ No zero-vectors")
        if nan_vecs:
            print(f"  ❌ FAIL: {nan_vecs} NaN-vectors found")
            passed = False
        else:
            print(f"  ✅ No NaN-vectors")
    else:
        print(f"\n⚠️  WARNING: No embeddings returned (check ChromaDB)")

    # --- Source distribution ---
    sources = [m.get("source", "unknown") for m in metas]
    source_counts = Counter(sources)
    print(f"\n📄 Source distribution:")
    for src, count in source_counts.most_common():
        print(f"   {src}: {count} chunks")

    # --- Final verdict ---
    print(f"\n{'='*60}")
    if passed:
        print("  ✅ ALL CHECKS PASSED")
    else:
        print("  ⚠️  SOME CHECKS HAVE WARNINGS — review above")
    print(f"{'='*60}\n")

    return passed


if __name__ == "__main__":
    verify()
