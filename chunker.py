from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=700, chunk_overlap=250):
    """Fallback chunker for non-Docling documents.

    When using Docling's DoclingLoader with ExportType.DOC_CHUNKS,
    documents are already chunked and this function is not needed.
    This is kept as a fallback for plain-text or non-Docling workflows.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks

