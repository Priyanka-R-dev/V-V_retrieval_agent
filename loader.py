import os
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(source_dir: str):
    """Load and chunk all PDF/DOCX documents from the source directory using Docling.

    Docling provides structure-aware parsing (layout analysis, table recognition,
    reading order) and returns pre-chunked LangChain Documents via HybridChunker.
    """
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    load_dotenv()
    enable_ocr = os.getenv("ENABLE_OCR", "true").lower() == "true"
    pdf_pipeline_opts = PdfPipelineOptions(do_ocr=enable_ocr)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_opts)
        }
    )

    documents = []
    supported_ext = ('.pdf', '.docx', '.pptx', '.xlsx', '.html', '.txt')

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(supported_ext):
                file_path = os.path.join(root, file)
                loader = DoclingLoader(
                    file_path=file_path,
                    export_type=ExportType.DOC_CHUNKS,
                    chunker=chunker,
                    converter=converter,
                )
                docs = loader.load()
                documents.extend(docs)

    return documents