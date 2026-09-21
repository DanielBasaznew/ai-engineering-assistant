# AI Engineering Assistant

A production-grade, enterprise-ready AI engineering assistant assembling ten weeks of agentic systems architecture. Built with **Google Gemini**, **ChromaDB**, **PyMuPDF**, **SentenceTransformers**, **Langfuse v4**, and **CrewAI**, this system integrates autonomous tool use, local retrieval-augmented generation (RAG), persistent episodic and semantic memory, and a hardened production operations layer.

---

## Architecture Overview

```text
                                 User Input / CLI
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │   Input Guardrails      │
                           │  • Length bounds        │
                           │  • Injection detection  │
                           │  • PII auditing         │
                           └────────────┬────────────┘
                                        │ (Passed)
                                        ▼
                           ┌─────────────────────────┐
                           │    Two-Layer Cache      │
                           │  • Exact Key (TTLCache) │
                           │  • Semantic Cosine Sim  │
                           └──────┬───────────┬──────┘
                                  │           │
                    [Cache Hit]   │           │ [Cache Miss]
                   (0 tokens, 0s) │           ▼
                                  │   ┌─────────────────────────┐
                                  │   │ Gemini ReAct Tool Loop  │
                                  │   │  • web_search           │
                                  │   │  • code_executor        │
                                  │   │  • read_pdf / page      │
                                  │   │  • search_knowledge_base│
                                  │   └───────────┬─────────────┘
                                  │               │
                                  │               ▼
                                  │   ┌─────────────────────────┐
                                  │   │   Output Guardrails     │
                                  │   │  • Truncation checks    │
                                  │   │  • Harmful output check │
                                  │   └───────────┬─────────────┘
                                  │               │
                                  │               ▼
                                  │   ┌─────────────────────────┐
                                  │   │    Persistent Memory    │
                                  │   │  • Episodic turn log    │
                                  │   │  • Semantic fact store  │
                                  │   └───────────┬─────────────┘
                                  │               │
                                  │               ▼
                                  │   ┌─────────────────────────┐
                                  │   │ Cache Population (Miss) │
                                  │   └───────────┬─────────────┘
                                  │               │
                                  ▼               ▼
                           ┌─────────────────────────┐
                           │ Telemetry & Accounting  │
                           │  • Langfuse v4 Traces   │
                           │  • JSON Structured Log  │
                           │  • USD Cost Accounting  │
                           └────────────┬────────────┘
                                        │
                                        ▼
                                 Assistant Output
```

---

## Major Capabilities

1. **Conversational Engine**: Powered by Google Gemini (`gemini-3.1-flash-lite`), supporting multi-turn dialogues with dynamic system prompt personalization.
2. **Autonomous Tool Dispatch**: Built-in function declarations for:
   * `web_search`: DuckDuckGo query simplification and live web retrieval.
   * `code_executor`: Isolated Python subprocess execution for calculations and validation.
   * `read_pdf` / `read_pdf_page`: PDF metadata inspection and page-specific text extraction via PyMuPDF.
   * `search_knowledge_base`: Semantic vector similarity lookup against ChromaDB.
3. **Local Vector RAG**: Chunking and embedding of PDF, Markdown, and plain-text documents with ChromaDB.
4. **Two-Tier Persistent Memory**:
   * **Semantic Memory**: Automated fact extraction into SQLite (`key`, `value`, `category`, `confidence`) with dynamic injection into the LLM system prompt.
   * **Episodic Memory**: Full turn-by-turn conversation logging across user sessions.
5. **Two-Layer Caching**:
   * **Exact Cache**: In-memory `TTLCache` for immediate $O(1)$ response hits.
   * **Semantic Cache**: Cosine similarity lookup via `SentenceTransformer` (`all-MiniLM-L6-v2`) with configurable similarity threshold (default `0.85`).
6. **Boundary Guardrails**:
   * **Input**: Regex-based prompt-injection interception, minimum/maximum character boundaries, and PII audit logging.
   * **Output**: Response completeness verification and content safety policy checks.
7. **Production Observability & Cost Tracking**:
   * **Langfuse v4**: Distributed OpenTelemetry tracing with `@observe`, capturing latencies, token consumption, and call costs.
   * **Structured Logging**: Single-line JSON file logging to `app.log` alongside human-friendly console output.
   * **Cost Tracker**: Exact prompt/completion token accounting and pricing per model based strictly on `usage_metadata`.
8. **Multi-Agent Review System (CrewAI)**: 3-agent research and audit team (Senior Research Analyst, Tech Content Strategist, Editorial QA Specialist) producing structured, verified markdown reports.

---

## Project Structure

```text
ai-engineering-assistant/
├── agents/
│   ├── __init__.py
│   └── crew.py                  # CrewAI 3-agent research and QA review team
├── data/                        # Persistent ChromaDB vector data
├── memory/
│   ├── memory.py                # SQLite semantic and episodic memory manager
│   └── memory_extractor.py      # LLM-based structured personal fact extractor
├── rag/
│   ├── chunker.py               # Document chunking with overlap
│   ├── ingestion.py             # PDF and plain text ingestion pipeline
│   └── vector_store.py          # ChromaDB collection management and search
├── tools/
│   ├── code_executor.py         # Sandboxed Python execution
│   ├── pdf_reader.py            # PyMuPDF document and page extractor
│   └── web_search.py            # DuckDuckGo search integration
├── assistant.py                 # Core Assistant orchestration and production lifecycle
├── cache.py                     # TwoLayerCache (Exact TTLCache + Semantic Cache)
├── cost_tracker.py              # Real-time token and USD cost accounting
├── guardrails.py                # Input injection & output safety validators
├── logger.py                    # JSON structured logging and console streams
├── main.py                      # Primary interactive CLI application entry point
├── tracer.py                    # Langfuse v4 OpenTelemetry tracing client
├── requirements.txt             # Project dependencies
├── test_full_capstone_verification.py # 16-point end-to-end verification suite
└── README.md
```

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10 to 3.14 (Python 3.12 recommended if running CrewAI directly in the primary environment)
* Valid Gemini API key (`GEMINI_API_KEY`)
* Langfuse account and credentials for distributed tracing

### 2. Environment Configuration
Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Interactive Console Assistant
Start the application with:

```bash
python main.py
```

### Interactive CLI Commands
Inside the assistant REPL, you can execute direct management commands:
* `memory`: Displays a formatted table of all persistent facts stored about the user.
* `cost`: Shows cumulative token consumption and total USD spend across models.
* `load <path>` or `/load <path>`: Ingests a local PDF or text document into the ChromaDB knowledge base.
* `crew <topic>` or `/crew <topic>`: Runs the multi-agent CrewAI research team and generates an editorial audit report in `reports/`.
* `exit` or `quit`: Flushes telemetry to Langfuse Cloud and shuts down cleanly.

---

## Technical Deep Dives

### Document Ingestion & RAG
* Documents are loaded via `rag/ingestion.py`, split into overlapping text chunks via `rag/chunker.py`, and embedded directly into a local ChromaDB collection (`rag/vector_store.py`).
* During chat, when external or indexed context is required, the model invokes `search_knowledge_base`, retrieving top-k semantically relevant chunks with similarity metrics.

### Persistent Memory
* Memory operates across two tiers:
  1. **Episodic**: Every user and assistant message turn is recorded in SQLite (`assistant_memory.db`) keyed by session ID.
  2. **Semantic**: Following every valid response turn, `memory_extractor.py` evaluates the user's input for explicit personal statements, preferences, or directives. New facts are persisted and automatically injected into future system prompts under a dedicated `Known Facts About The User` block.

### Caching Architecture
* Query normalization strips excess whitespace and standardizes casing.
* **Exact Cache**: Checked first for identical matches ($O(1)$ lookup via `cachetools.TTLCache`).
* **Semantic Cache**: Checked second using sentence embeddings (`all-MiniLM-L6-v2`). Queries with cosine similarity $\ge 0.85$ return the cached response without invoking the LLM, consuming 0 tokens.
* Errors, tool execution steps, and blocked inputs are never cached.

### Guardrails
* **Input Guardrail**: Evaluates inputs before cache or LLM execution. Rejects inputs shorter than 2 characters, longer than 10,000 characters, or matching prompt-injection/jailbreak patterns. Audits and warns on detected PII (emails, phone numbers).
* **Output Guardrail**: Validates model responses before returning to the user or saving to memory. Rejects truncated outputs and filters prohibited patterns.

### Observability & Cost Tracking
* **Tracing**: Integrated with Langfuse v4 using `@observe(name="assistant_chat")` and `@observe(name="tool_execution")`. Spans record inputs, outputs, models, latencies, and metadata.
* **Cost Accounting**: Token usage is read strictly from `response.usage_metadata`. Prices are calculated based on model-specific input/output rates per million tokens.

### CrewAI Multi-Agent Review
* Implemented in `agents/crew.py`.
* A sequential workflow coordinating three specialized agents:
  1. `Senior Research Analyst`: Conducts targeted web searches and compiles an initial briefing.
  2. `Tech Content Strategist`: Authors a structured executive report.
  3. `Editorial Quality Assurance Specialist`: Audits the report for word count, section completeness, and factual alignment, issuing a formal `VERDICT: APPROVED` or `VERDICT: REVISION NEEDED`.
* Reports are saved as versioned Markdown files in `reports/`.

---

## Verification & Testing

Run the full 16-point verification suite:

```bash
python test_full_capstone_verification.py
```

### Verified Capabilities:
1. Normal conversation generation
2. Web search tool execution
3. Code execution sandbox
4. PDF overview and page extraction
5. PDF document ingestion into ChromaDB
6. RAG knowledge base retrieval
7. Persistent memory recall across fresh Assistant instances
8. Prompt injection interception prior to LLM
9. Output safety and truncation guardrails
10. Cache miss -> LLM call -> cache population
11. Cache hit -> immediate return with 0 tokens consumed
12. Structured JSON logging to `app.log`
13. Langfuse v4 distributed trace dispatch and flush
14. Real-time token consumption and USD cost accounting
15. CrewAI multi-agent research and editorial audit execution
16. Model Context Protocol (MCP) architectural compatibility

---

## Known Limitations

* **Gemini Free-Tier Rate Limits**: The free tier is capped at 15 requests per minute (RPM). `assistant.py` includes automatic exponential backoff retry for 429 errors.
* **CrewAI Python Version Compatibility**: `crewai==1.15.18` relies on packages with binary dependencies optimized for Python 3.10–3.12. When run in Python 3.14 environments, `assistant.py` automatically bridges CrewAI execution to the Python 3.12 virtual environment.
