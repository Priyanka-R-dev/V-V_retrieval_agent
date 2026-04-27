from langchain_chroma import Chroma
from embedder import get_embeddings

db = Chroma(persist_directory="chroma_db", embedding_function=get_embeddings())
collection = db._collection.get(include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(collection["documents"], collection["metadatas"])):
    if any(keyword in doc.lower() for keyword in ["type of testing", "combined system", "testing plan", "☒ combined", "st and uat", "st/uat"]):
        print(f"\n--- Chunk {i} (Page {meta.get('page', '?')}) ---")
        print(doc[:400])