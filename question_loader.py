import os
import re
import json
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def _get_converter():
    """Create a Docling converter with OCR controlled by ENABLE_OCR env var."""
    load_dotenv()
    enable_ocr = os.getenv("ENABLE_OCR", "true").lower() == "true"
    opts = PdfPipelineOptions(do_ocr=enable_ocr)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

def load_questions_from_output(output_file=None):
    """Extract questions from a PDF (via Docling), or load from .txt/.json file."""
    load_dotenv()

    if not output_file:
        output_file = os.getenv("OUTPUT_FILE")
        if not output_file:
            raise ValueError("OUTPUT_FILE env var must be set or output_file argument provided.")

    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")

    ext = os.path.splitext(output_file)[1].lower()

    # JSON: expect a list of question strings
    if ext == '.json':
        with open(output_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return [q.strip() for q in questions if isinstance(q, str) and len(q.strip()) > 15]

    # TXT: one question per line
    if ext == '.txt':
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [l.strip() for l in lines if l.strip() and len(l.strip()) > 15]

    # PDF: use Docling for structured text extraction, then regex
    loader = DoclingLoader(
        file_path=output_file,
        export_type=ExportType.MARKDOWN,
        converter=_get_converter(),
    )
    pages = loader.load()

    full_text = "\n".join(page.page_content for page in pages)

    # Replace newlines with spaces to join multi-line questions
    flat_text = re.sub(r'\s*\n\s*', ' ', full_text)
    flat_text = re.sub(r'\s+', ' ', flat_text).strip()

    found = re.findall(
        r'((?:Who|What|How|Will|Is|Which)[^?☐]{5,}?\?)',
        flat_text
    )

    questions = []
    for q in found:
        q = q.strip()
        q = re.sub(r'☐.*', '', q).strip()
        if len(q) > 15 and q not in questions:
            questions.append(q)

    return questions