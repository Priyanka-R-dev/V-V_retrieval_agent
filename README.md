# RAG Retrieval Pipeline — CSV Validation Document Q&A

A **Retrieval-Augmented Generation (RAG)** pipeline that ingests regulatory validation PDF documents, stores them as vector embeddings, and automatically answers questions extracted from a separate questionnaire PDF — grounding every answer strictly in the source document.

Built for **Computer System Validation (CSV)** / GxP test plans with structured content: checkboxes, tables, section hierarchies, and approval workflows.

---

## Table of Contents

- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Technology Stack](#technology-stack)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
  - [Phase 1: Ingestion](#phase-1-ingestion)
  - [Phase 2: Query Engine](#phase-2-query-engine)
- [Key Design Decisions](#key-design-decisions)
- [Diagnostic Tools](#diagnostic-tools)
- [Output Format](#output-format)

---

## Architecture

```
                        ┌──────────────────────────────────────────────────────────┐
                        │              PHASE 1: INGESTION PIPELINE                │
                        │                                                          │
  ┌──────────────┐      │  ┌────────────┐    ┌────────────┐    ┌───────────────┐   │
  │ Source PDFs   │─────▶│  │ loader.py  │───▶│embedder.py │───▶│vector_store.py│   │
  │ (input file/) │      │  │ Docling +  │    │MiniLM-L6-v2│    │ ChromaDB +    │   │
  │              │      │  │HybridChunk │    │ 384-dim    │    │ JSON export   │   │
  └──────────────┘      │  └────────────┘    └────────────┘    └───────────────┘   │
                        └──────────────────────────────────────────────────────────┘

                        ┌──────────────────────────────────────────────────────────┐
                        │              PHASE 2: QUERY ENGINE                       │
                        │                                                          │
  ┌──────────────┐      │  ┌─────────────────┐    ┌────────────────────────────┐   │
  │Questionnaire │─────▶│  │question_loader.py│───▶│   answer_generator.py     │   │
  │(output file/)│      │  │ Docling + regex  │    │                            │   │
  │              │      │  │ extract Qs       │    │  1. Query Expansion (LLM)  │   │
  └──────────────┘      │  └─────────────────┘    │  2. Multi-Query Retrieval  │   │
                        │                          │  3. Hybrid Re-Ranking      │   │
                        │  ┌─────────────────┐    │  4. LLM Answer Generation  │   │
                        │  │  retriever.py    │◀──│                            │   │
                        │  │ ChromaDB search  │───▶│                            │   │
                        │  └─────────────────┘    └────────────────────────────┘   │
                        │                                       │                  │
                        └───────────────────────────────────────┼──────────────────┘
                                                                ▼
                                                   ┌────────────────────┐
                                                   │   Output Files     │
                                                   │  .json + .md       │
                                                   └────────────────────┘
```

---

## Directory Structure

```
c:\RAG\
├── .env                          # Configuration (API keys, paths, thresholds)
├── README.md                     # This file
│
│── ingestion_pipeline.py         # ENTRY POINT 1 — ingest documents into vector store
├── query_engine.py               # ENTRY POINT 2 — extract questions → retrieve → answer
│
├── loader.py                     # PDF parsing + chunking via Docling
├── question_loader.py            # Question extraction from questionnaire PDF
├── embedder.py                   # Embedding model loader (all-MiniLM-L6-v2)
├── chunker.py                    # Fallback text splitter (unused when Docling is active)
├── vector_store.py               # ChromaDB storage + JSON export
├── retriever.py                  # Similarity search with scoring wrapper
├── answer_generator.py           # Query expansion + hybrid re-ranking + LLM answer
│
├── verify_chunks.py              # Diagnostic — validates chunks & embeddings in DB
├── debug_chunks.py               # Diagnostic — keyword search over ChromaDB chunks
│
├── input file/                   # SOURCE documents (what you answer FROM)
│   └── LMS Test Plan Sample.pdf
├── output file/                  # QUESTIONNAIRE documents (questions to answer)
│   └── LMS Validation Plan Sample.pdf
│
├── chroma_db/                    # ChromaDB persistent vector store (auto-generated)
├── embeddings.json               # JSON export of all chunks + embedding vectors
├── validation_plan_answers.json  # Generated answers (structured JSON)
└── validation_plan_answers.md    # Generated answers (human-readable Markdown)
```

---

## Technology Stack

| Component | Technology | Details |
|---|---|---|
| **Language** | Python 3.13 | Virtual environment at `venv/` |
| **PDF Parsing** | Docling 2.91.0 + langchain-docling 2.0.0 | Structure-aware: layout analysis, table recognition, reading order, checkbox parsing |
| **Chunking** | Docling `HybridChunker` | Semantic chunking aligned with embedding model tokenizer; preserves document structure |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim) | Runs locally via `langchain-huggingface`; no external API calls |
| **Vector Store** | ChromaDB (`langchain-chroma` 1.1.0) | Persistent local storage in `chroma_db/` |
| **LLM** | Azure OpenAI GPT-5.1 | Used for query expansion and answer generation |
| **Framework** | LangChain 1.2.15 | Orchestration, document types, message formatting |
| **OCR** | RapidOCR (optional) | Toggleable via `ENABLE_OCR` env var |

---

## Setup

### 1. Clone / copy the project

```powershell
cd c:\RAG
```

### 2. Create and activate the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install langchain langchain-docling langchain-chroma langchain-huggingface langchain-openai
pip install docling python-dotenv transformers sentence-transformers chromadb
```

### 4. Configure environment variables

Copy and edit the `.env` file with your Azure OpenAI credentials:

```env
# Source & storage
SOURCE_DIR=input file
CHROMA_DIR=chroma_db

# Retrieval settings
TOP_K=10
SIMILARITY_THRESHOLD=0.05

# Azure OpenAI
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.1
TEMPERATURE=0.0

# Input / output
OUTPUT_FILE=output file/LMS Validation Plan Sample.pdf
RESULTS_FILE=validation_plan_answers.json
RESULTS_MD=validation_plan_answers.md
ENABLE_OCR=false
```

### 5. Add your documents

- Place **source PDFs** (documents to answer from) in `input file/`
- Place the **questionnaire PDF** (document with questions) in `output file/`

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SOURCE_DIR` | `input file` | Folder containing source PDFs to ingest |
| `OUTPUT_FILE` | — | Path to questionnaire PDF/TXT/JSON to extract questions from |
| `CHROMA_DIR` | `chroma_db` | ChromaDB persistence directory |
| `TOP_K` | `10` | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.05` | Average score below this → `[Low Confidence]` flag on answer |
| `TEMPERATURE` | `0.0` | LLM temperature (0.0 = deterministic) |
| `ENABLE_OCR` | `false` | Enable Docling's OCR for scanned PDFs (requires network for model download) |
| `RESULTS_FILE` | `validation_answers.json` | Output JSON path |
| `RESULTS_MD` | `validation_answers.md` | Output Markdown path |

---

## Usage

### Step 1: Ingest source documents into the vector store

```powershell
.\venv\Scripts\Activate.ps1
python ingestion_pipeline.py
```

Output:
```
Loading and chunking documents from: input file
Loaded 44 chunks.
Loading embeddings model...
Embeddings stored in Chroma and exported to JSON.
```

### Step 2: (Optional) Verify ingestion quality

```powershell
python verify_chunks.py
```

Output: Reports on chunk count, sizes, duplicates, embedding dimensions, zero/NaN vectors, and source distribution.

### Step 3: Run the query engine

```powershell
python query_engine.py
```

Output:
```
Extracting questions from: output file/LMS Validation Plan Sample.pdf
Found 19 questions.
[1/19] Who will complete Infrastructure qualification activities?...
...
✅ Answers saved to:
   JSON: validation_plan_answers.json
   Markdown: validation_plan_answers.md
```

---

## Pipeline Details

### Phase 1: Ingestion

**Entry point:** `ingestion_pipeline.py`

1. **`loader.py`** — Walks `SOURCE_DIR` for supported files (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.txt`). For each file:
   - Creates a Docling `DocumentConverter` with configurable OCR
   - Creates a `HybridChunker` with a tokenizer matching the embedding model (`all-MiniLM-L6-v2`) so chunk sizes align with the model's token limit
   - Uses `DoclingLoader` with `ExportType.DOC_CHUNKS` to produce pre-chunked LangChain `Document` objects that preserve headings, tables, and checkbox structures

2. **`ingestion_pipeline.py`** — Flattens metadata (strips nested dicts/lists that ChromaDB cannot store as scalar values)

3. **`embedder.py`** — Loads the `all-MiniLM-L6-v2` model via `HuggingFaceEmbeddings` (runs locally, 384-dimensional vectors)

4. **`vector_store.py`** — Clears any existing `chroma_db/` directory (`shutil.rmtree`) to prevent duplicate chunk accumulation, then stores chunks via `Chroma.from_documents()` and exports all chunk text + embeddings to `embeddings.json`

### Phase 2: Query Engine

**Entry point:** `query_engine.py`

**Per question, the following pipeline runs:**

#### 2a. Question Extraction (`question_loader.py`)

- Parses the questionnaire PDF via Docling (`ExportType.MARKDOWN`)
- Flattens multi-line text and applies regex: `r'((?:Who|What|How|Will|Is|Which)[^?☐]{5,}?\?)'`
- Extracts 19 unique questions
- Also supports `.json` (list of strings) and `.txt` (one question per line) formats

#### 2b. Query Expansion (`answer_generator.py`)

- Sends each question to GPT-5.1 requesting 3 alternative search queries
- Example: "Who will execute System Test?" → ["system testing team", "ST execution responsibility", "test performers"]
- Results cached in `_expansion_cache` to avoid redundant API calls

#### 2c. Multi-Query Retrieval (`retriever.py` + `answer_generator.py`)

- Runs the original question + 3 expanded queries through `ScoringRetriever`
- Each query hits ChromaDB's `similarity_search_with_relevance_scores(query, k=10)`
- Deduplicates results by `page_content` across all 4 queries

#### 2d. Hybrid Re-Ranking (`answer_generator.py`)

- Scores each chunk: **`0.7 × similarity_score + 0.3 × keyword_overlap_ratio`**
- `keyword_overlap_ratio` = proportion of question keywords (>3 chars) found in the chunk
- Sorts descending — balances semantic similarity with lexical matching

#### 2e. LLM Answer Generation (`answer_generator.py`)

- Sends ranked context + question to GPT-5.1 (temperature 0.0)
- Uses a **14-rule system prompt** specialized for regulatory documents:
  - **Rules 1–3:** Checkbox interpretation (`☒`/`[x]` = checked, `☐`/`[ ]` = unchecked) and per-question matching
  - **Rules 4–9:** Answer precision (distinguish plan vs. summary, be concise, use NA only when zero info)
  - **Rules 10–14:** Advanced disambiguation (orphaned checkboxes, team name preference, section-level reasoning for engagement questions)

#### 2f. Confidence Flagging

- If average similarity score < `SIMILARITY_THRESHOLD` (0.05), answer is prefixed with `[Low Confidence]`

#### 2g. Error Handling

- **Rate limits (429):** Retries with exponential backoff (30s × attempt)
- **Azure content filter:** Trims context to first 5 chunks at 300 chars each and retries
- **Max retries:** 3 per question

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Docling over PyPDF/pdfplumber** | Structure-aware parsing preserves tables, checkboxes, and section hierarchy instead of flat text |
| **HybridChunker over RecursiveCharacterTextSplitter** | Keeps semantically related content (heading + checkboxes) in the same chunk; tokenizer-aligned with embedding model |
| **Query expansion via LLM** | Improves recall when questions use different vocabulary than the source document |
| **Hybrid re-ranking (70/30)** | Combines semantic similarity with lexical overlap to reduce pure-semantic mismatches |
| **Vector store clearing on re-ingestion** | `shutil.rmtree(chroma_db/)` before each run prevents duplicate chunk accumulation |
| **14-rule system prompt** | Handles checkbox disambiguation, orphaned checkboxes, and regulatory document conventions |
| **Temperature 0.0** | Deterministic answers for regulatory compliance documents |
| **Local embeddings** | `all-MiniLM-L6-v2` runs locally — no API cost or latency for embeddings |

---

## Diagnostic Tools

| Script | Purpose | Usage |
|---|---|---|
| `verify_chunks.py` | Post-ingestion validation: chunk count, sizes, duplicates, embedding dimensions (384), zero/NaN vectors, source distribution | `python verify_chunks.py` |
| `debug_chunks.py` | Searches ChromaDB for chunks matching specific keywords (e.g., "combined system", "☒ combined") | `python debug_chunks.py` |

---

## Output Format

### Markdown (`validation_plan_answers.md`)

```markdown
## Q1: Who will complete Infrastructure qualification activities?
**Answer:** [Low Confidence] Infrastructure qualification activities are out of scope...

## Q6: What type of testing will be completed for this validation effort?
**Answer:** The testing for this validation effort will be a **Combined System and User Acceptance Test (ST/UAT) Plan**.
```

### JSON (`validation_plan_answers.json`)

```json
[
  {
    "question": "Who will complete Infrastructure qualification activities?",
    "answer": "[Low Confidence] Infrastructure qualification activities are out of scope...",
    "avg_score": 0.042,
    "confidence": "low",
    "sources": [
      {
        "source": "input file\\LMS Test Plan Sample.pdf",
        "score": 0.151
      }
    ]
  }
]
```

---

## Current Results

The pipeline successfully answers **19/19 questions** from the LMS Validation Plan questionnaire:

| Category | Questions | Status |
|---|---|---|
| IQ (Infrastructure Qualification) | Q1–Q5 | Correctly identified as **NA / Out of Scope** |
| System Testing | Q6–Q9 | Correct — Combined ST/UAT Plan, Team A, ALM documentation |
| User Acceptance Testing | Q10–Q13 | Correct — no separate UAT plan, business units engaged |
| Data Migration | Q14, Q16 | Correct — not included / NA |
| Deliverables | Q15 | Correct — 4 deliverables identified |
| Traceability | Q17–Q19 | Correct — combination project-specific and living document |
