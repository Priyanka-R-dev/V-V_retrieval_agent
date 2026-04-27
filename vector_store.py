from langchain_chroma import Chroma
import json
import shutil
import os


def store_embeddings(chunks, embeddings, persist_directory="chroma_db", json_path="embeddings.json"):
    """Store chunks in Chroma vector DB and export to JSON."""
    # Remove existing DB to prevent duplicate chunks from accumulating
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    
    # Export to JSON for API use
    data = []
    collection_data = db._collection.get(include=["embeddings"])
    for doc, emb in zip(chunks, collection_data['embeddings']):
        data.append({
            'text': doc.page_content,
            'embedding': [float(x) for x in emb]  # convert ndarray to list
        })
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return db
