"""
Core AI Engineering Assistant Orchestration Layer (Week 10 Day 4 - Step 4)
Integrates:
- Tools (web_search, code_executor, read_pdf, read_pdf_page)
- ChromaDB RAG (vector_store, ingestion, chunker)
- Persistent Memory (PersistentMemory, memory_extractor)
- Production Layer:
    * Input & Output Guardrails (guardrails.py)
    * Two-Layer Exact & Semantic Caching (cache.py)
    * OpenTelemetry Langfuse v4 Tracing (tracer.py)
    * Structured JSON & Console Logging (logger.py)
    * Token & Cost Tracking (cost_tracker.py)
"""

import os
import sys
import time
import uuid
import json
import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from google import genai
from google.genai import types, errors

# Tool implementations
from tools.web_search import web_search
from tools.code_executor import execute_python
from tools.mcp_client import call_mcp_tool

try:
    from tools.pdf_reader import read_pdf, read_pdf_page
except ImportError:
    read_pdf = None
    read_pdf_page = None

# RAG implementations
from rag.vector_store import get_collection, search
from rag.ingestion import ingest_pdf, ingest_text_file

# Memory implementations
from memory.memory import PersistentMemory
from memory.memory_extractor import extract_facts_from_conversation

# Production Layer implementations
from guardrails import check_input, check_output
from logger import log
from tracer import langfuse, observe
from cost_tracker import tracker, CostTracker
from cache import app_cache, TwoLayerCache

# Rich Terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich import box

console = Console()


class Assistant:
    """
    Production-grade AI Engineering Assistant that orchestrates:
    - Conversational responses with Gemini
    - Autonomous tool execution
    - ChromaDB RAG retrieval
    - SQLite persistent memory (semantic + episodic)
    - Input & output boundary guardrails
    - Two-layer exact + semantic caching
    - OpenTelemetry distributed tracing via Langfuse v4
    - Real-time token usage and cost accounting
    - Structured JSON file and console logging
    """

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        memory_db_path: str = "assistant_memory.db",
        rag_collection: str = "knowledge_base",
        cache: Optional[TwoLayerCache] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.rag_collection = rag_collection

        # Initialize Persistent Memory (Semantic facts + Episodic turns)
        self.memory = PersistentMemory(db_path=memory_db_path)
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Initialize Active Document Tracking & Conversation/Task Context
        self.active_document: Optional[str] = None
        self.active_document_name: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize Production Layer components
        self.cache = cache or app_cache
        self.cost_tracker = cost_tracker or tracker

        self.tools = self._build_tools()

        log.info(
            "Assistant initialized successfully",
            extra={
                "model": self.model_name,
                "session_id": self.session_id,
                "rag_collection": self.rag_collection,
            },
        )

    def _build_tools(self) -> List[types.Tool]:
        """
        Defines tool contracts and JSON schemas for all assistant capabilities:
        - web_search
        - code_executor
        - read_pdf
        - read_pdf_page
        - search_knowledge_base
        """
        now = datetime.datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")
        current_year = now.year

        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="web_search",
                        description=(
                            f"Searches DuckDuckGo for live external web information, current events, breaking developments, and live documentation. "
                            f"Today's date is {current_date_str}. Use the current year ({current_year}) for recent or time-sensitive searches. "
                            f"DO NOT call this tool for general conceptual, historical, foundational, or programming questions "
                            f"(such as 'What is RAG?', 'Explain Python dictionaries', 'How does HTTP work?') that you can answer directly from internal knowledge. "
                            f"Call this tool ONLY when the user explicitly requests web searching or when fresh, current real-time external data is genuinely required."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(
                                    type=types.Type.STRING,
                                    description="Specific search query terms.",
                                ),
                            },
                            required=["query"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="code_executor",
                        description=(
                            "Executes standalone Python code in an isolated subprocess. "
                            "Use for mathematical calculations, data processing, sorting, or logic verification. "
                            "Must use print() to output results."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "code": types.Schema(
                                    type=types.Type.STRING,
                                    description="Valid executable Python code string.",
                                ),
                            },
                            required=["code"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="read_pdf",
                        description=(
                            "Extracts document metadata and the first 3000 characters of text from a local PDF file. "
                            "Use to obtain a high-level overview of a PDF document. "
                            "Do NOT invent or guess filenames. If an active document is loaded, use its canonical path or use search_knowledge_base."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "file_path": types.Schema(
                                    type=types.Type.STRING,
                                    description="Path to the local .pdf file.",
                                ),
                            },
                            required=["file_path"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="read_pdf_page",
                        description=(
                            "Extracts complete text from a single specific 1-indexed page of a local PDF document. "
                            "Do NOT invent or guess filenames. If an active document is loaded, use its canonical path."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "file_path": types.Schema(
                                    type=types.Type.STRING,
                                    description="Path to the local .pdf file.",
                                ),
                                "page_number": types.Schema(
                                    type=types.Type.INTEGER,
                                    description="1-indexed page number to inspect (e.g. 1 for first page).",
                                ),
                            },
                            required=["file_path", "page_number"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="search_knowledge_base",
                        description=(
                            "Searches the private ChromaDB vector database for indexed internal documents and notes. "
                            "Returns semantically relevant passages and similarity scores. "
                            "Use this as the primary retrieval tool for querying ingested documents."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "query": types.Schema(
                                    type=types.Type.STRING,
                                    description="The conceptual or semantic query to look up in the knowledge base.",
                                ),
                                "top_k": types.Schema(
                                    type=types.Type.INTEGER,
                                    description="Number of relevant documents to retrieve (default 3).",
                                ),
                            },
                            required=["query"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="mcp_explain_repository",
                        description=(
                            "Inspects and explains any file or folder/directory location using the Week 8 Model Context Protocol (MCP) server over stdio transport. "
                            "Works for both single files (e.g. 'main.py', 'rag/chunker.py') and folders/repositories "
                            "(e.g. '.', 'rag', 'tools', 'week 9', '../multi-agent-week9', 'week 8', or any absolute path). "
                            "Summarizes code purpose, detects potential syntax errors or anti-patterns, "
                            "and provides actionable recommendations and best practices."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "path": types.Schema(
                                    type=types.Type.STRING,
                                    description="Path or name of the file or folder to inspect (e.g. '.', 'rag', 'week 9', '../multi-agent-week9', 'main.py').",
                                ),
                            },
                            required=["path"],
                        ),
                    ),
                ]
            )
        ]

    @observe(name="tool_execution")
    def _call_tool(self, name: str, args: Dict[str, Any]) -> str:
        """
        Central tool dispatcher. Dispatches calls to appropriate tool functions.
        """
        try:
            if name == "web_search":
                query = args.get("query", "")
                return web_search(query=query)

            elif name in ("code_executor", "execute_python"):
                code = args.get("code", "")
                return execute_python(code=code)

            elif name in ("mcp_explain_repository", "explain_repository"):
                path = args.get("path", ".")
                return call_mcp_tool("explain_repository", {"path": path})

            elif name in ("mcp_calculate", "calculate"):
                expression = args.get("expression", "")
                return call_mcp_tool("calculate", {"expression": expression})

            elif name == "read_pdf":
                if read_pdf is None:
                    return "Error: PDF reading is unavailable because PyMuPDF (fitz) is not installed."
                file_path = str(args.get("file_path", "")).strip()
                # Active document fallback if file_path is missing, generic, or non-existent
                if self.active_document and (
                    not file_path
                    or not os.path.exists(file_path)
                    or file_path.lower() in ("the document", "document.pdf", "uploaded.pdf", "the_document.pdf", "pdf", "file")
                ):
                    file_path = self.active_document
                elif not os.path.exists(file_path) and not self.active_document:
                    return (
                        f"Error: File '{file_path}' does not exist and no active document is currently loaded. "
                        f"Please use '/load <path>' to load your document or use search_knowledge_base to query indexed documents."
                    )
                return read_pdf(file_path=file_path)

            elif name == "read_pdf_page":
                if read_pdf_page is None:
                    return "Error: PDF page reading is unavailable because PyMuPDF (fitz) is not installed."
                file_path = str(args.get("file_path", "")).strip()
                # Active document fallback if file_path is missing, generic, or non-existent
                if self.active_document and (
                    not file_path
                    or not os.path.exists(file_path)
                    or file_path.lower() in ("the document", "document.pdf", "uploaded.pdf", "the_document.pdf", "pdf", "file")
                ):
                    file_path = self.active_document
                elif not os.path.exists(file_path) and not self.active_document:
                    return (
                        f"Error: File '{file_path}' does not exist and no active document is currently loaded. "
                        f"Please use '/load <path>' to load your document or use search_knowledge_base to query indexed documents."
                    )
                page_number = int(args.get("page_number", 1))
                return read_pdf_page(file_path=file_path, page_number=page_number)

            elif name == "search_knowledge_base":
                query = args.get("query", "")
                top_k = int(args.get("top_k", 3))
                collection = get_collection(self.rag_collection)
                results = search(collection, query=query, top_k=top_k)

                if not results:
                    return f"No documents found in knowledge base matching query: '{query}'"

                formatted = []
                for i, r in enumerate(results, 1):
                    source = r["metadata"].get("source", "unknown")
                    page = r["metadata"].get("page", 1)
                    sim = round(r.get("similarity", 0.0), 3)
                    formatted.append(
                        f"[Result {i}] Source: {source} (Page {page}, Similarity: {sim})\nContent: {r['document']}"
                    )
                return "\n\n".join(formatted)

            else:
                return f"Error: Tool '{name}' is not recognized."

        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def _build_system_prompt(self) -> str:
        """
        Constructs the base system prompt including role, tool contracts,
        runtime date awareness, active loaded document, and persistent memory facts.
        """
        facts_block = self.memory.format_for_prompt()
        now = datetime.datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")
        current_year = now.year

        active_doc_block = ""
        if self.active_document:
            active_doc_block = (
                f"Active Document in Current Session:\n"
                f"- Canonical Path: {self.active_document}\n"
                f"- Filename: {self.active_document_name}\n"
                f"When the user refers to 'the document', 'the PDF', 'my uploaded document', or 'the file I just loaded', "
                f"use this active document path ('{self.active_document}') for tools like read_pdf / read_pdf_page, "
                f"or use search_knowledge_base to query its contents. Never invent or guess an alternative filename.\n\n"
            )
        else:
            active_doc_block = (
                "No active document is currently loaded in this session. "
                "Do NOT guess, invent, or hallucinate file names (such as 'Filler_Machine_Assignment.pdf', 'assignment.pdf', 'doc.pdf'). "
                "If the user asks about document contents, use search_knowledge_base to check indexed knowledge or ask the user to load the file via '/load <path>'.\n\n"
            )

        return (
            "You are an AI Engineering Assistant, a reliable and skilled technical assistant.\n\n"
            f"Current Runtime Date: {current_date_str} (Year: {current_year})\n\n"
            f"{facts_block}\n\n"
            f"{active_doc_block}"
            "Capabilities and Available Tools:\n"
            "- web_search: Search DuckDuckGo for live web information, current events, and fresh documentation.\n"
            "- code_executor: Run Python code in an isolated sandbox for math, data analysis, sorting, and script verification.\n"
            "- mcp_explain_repository: Inspect and explain a directory/repository, file inventory, and best practices via Week 8 MCP server.\n"
            "- read_pdf: Inspect local PDF files to obtain metadata and document overviews.\n"
            "- read_pdf_page: Read specific pages of a local PDF document in detail.\n"
            "- search_knowledge_base: Query the internal ChromaDB vector store for indexed documents and notes.\n\n"
            "Tool Selection and Operating Rules:\n"
            "1. Web Search Discipline: Avoid calling web_search for general conceptual, historical, foundational, or programming questions "
            "(e.g., 'What is RAG?', 'Explain Python dictionaries', 'How does TCP handshake work?'). Answer these directly from foundational knowledge. "
            "Call web_search ONLY when the user explicitly requests web search or when fresh, current real-time data is genuinely required.\n"
            f"2. Runtime Date Awareness: When interpreting time-sensitive terms ('latest', 'recent', 'current', 'today', 'this year', '2026'), "
            f"strictly use the current runtime date ({current_date_str}) and year ({current_year}). "
            f"Never assume or hardcode outdated years (such as 2024 or 2025) into search queries or answers.\n"
            "3. Document and RAG Discipline: Never fabricate, invent, or guess PDF filenames (e.g. do not invent 'Filler_Machine_Assignment.pdf'). "
            "If a document is loaded, use the active document path or search_knowledge_base. "
            "Do not make redundant or repetitive tool calls once sufficient context has been retrieved. If search_knowledge_base or read_pdf returned relevant content, synthesize your response without calling additional file tools.\n"
            "4. Personal Context: Use stored facts naturally without explicitly referencing 'my database', 'stored memory', or 'system records'.\n"
            "5. Tool Grounding: Ground all claims strictly on actual tool observations. Never fabricate tool outputs.\n"
            "6. Error Recovery: If a tool reports an error or returns empty results, do not loop through speculative calls; reason cleanly or inform the user.\n"
            "7. Clarity: Provide structured, concise, and clear answers.\n"
            "8. File and Folder Inspection via MCP: When the user asks to analyze, explain, or inspect any folder or file (e.g. 'rag', 'tools', 'week 9', 'week 8', or any relative/absolute path), invoke mcp_explain_repository with the requested path.\n"
        )

    def _update_memory(self, user_message: str, assistant_message: str) -> None:
        """
        Logs conversation turn to episodic memory and extracts/persists semantic facts.
        """
        # 1. Episodic memory logging
        self.memory.log_conversation(self.session_id, "user", user_message)
        self.memory.log_conversation(self.session_id, "assistant", assistant_message)

        # 2. Semantic memory fact extraction
        current_facts_str = self.memory.format_for_prompt()
        try:
            extracted_facts = extract_facts_from_conversation(user_message, current_facts_str)
        except Exception as e:
            log.warning(f"Memory fact extraction skipped due to API error: {e}")
            extracted_facts = []

        for fact in extracted_facts:
            if fact.action in ["store", "update"]:
                self.memory.store_fact(
                    key=fact.key,
                    value=fact.value,
                    category=fact.category,
                    confidence="high",
                    source="user_statement",
                )
                console.print(f"  [bold magenta]🧠 Memory Learned:[/bold magenta] [bold white]'{fact.key}'[/bold white] → [italic]'{fact.value}'[/italic] [dim]({fact.category})[/dim]")
            elif fact.action == "delete":
                deleted = self.memory.delete_fact(fact.key)
                if deleted:
                    console.print(f"  [bold red]🗑️ Memory Deleted:[/bold red] [bold white]'{fact.key}'[/bold white]")

    def load_document(self, file_path: str) -> str:
        """
        Ingests a PDF or plain text document into ChromaDB using existing Week 5 ingestion functions.
        """
        clean_path = file_path.strip("'\"")
        if not os.path.exists(clean_path):
            return f"Error: File not found at '{clean_path}'"

        ext = os.path.splitext(clean_path)[1].lower()
        if ext == ".pdf":
            try:
                chunks_count = ingest_pdf(clean_path, collection_name=self.rag_collection)
                self.active_document = os.path.abspath(clean_path)
                self.active_document_name = os.path.basename(clean_path)
                msg = (
                    f"Successfully ingested PDF '{os.path.basename(clean_path)}' "
                    f"({chunks_count} chunks) into knowledge base."
                )
                log.info("Document ingested (PDF)", extra={"file": clean_path, "chunks": chunks_count})
                return msg
            except Exception as e:
                log.error(f"Failed to ingest PDF: {e}", extra={"file": clean_path, "error": str(e)})
                return f"Error ingesting PDF '{clean_path}': {str(e)}"
        elif ext in (".txt", ".md", ".py", ".json", ".csv"):
            try:
                chunks_count = ingest_text_file(clean_path, collection_name=self.rag_collection)
                self.active_document = os.path.abspath(clean_path)
                self.active_document_name = os.path.basename(clean_path)
                msg = (
                    f"Successfully ingested text file '{os.path.basename(clean_path)}' "
                    f"({chunks_count} chunks) into knowledge base."
                )
                log.info("Document ingested (Text)", extra={"file": clean_path, "chunks": chunks_count})
                return msg
            except Exception as e:
                log.error(f"Failed to ingest text file: {e}", extra={"file": clean_path, "error": str(e)})
                return f"Error ingesting text file '{clean_path}': {str(e)}"
        else:
            return (
                f"Error: Unsupported file format '{ext}'. "
                f"Supported formats: .pdf, .txt, .md, .py, .json, .csv"
            )

    def show_memory(self) -> None:
        """
        Renders a Rich table of stored persistent memory using the Week 6 implementation.
        """
        self.memory.memory_report()

    def show_cost(self) -> None:
        """
        Renders an attractive Rich table of token consumption and costs recorded by the cost tracker.
        """
        summary = self.get_cost_summary()
        table = Table(
            title="💰 Real-Time Token Usage & Financial Cost Accounting",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Accounting Metric", style="cyan", no_wrap=True)
        table.add_column("Recorded Value", style="bold green")

        table.add_row("Total LLM Calls", f"{summary.get('total_calls', 0):,}")
        table.add_row("Input (Prompt) Tokens", f"{summary.get('total_input_tokens', 0):,}")
        table.add_row("Output (Completion) Tokens", f"{summary.get('total_output_tokens', 0):,}")
        table.add_row("Total Tokens Consumed", f"{summary.get('total_tokens', 0):,}")
        table.add_row("Estimated Spend (USD)", f"${summary.get('total_cost_usd', 0.0):.6f}")

        breakdown = summary.get("breakdown_by_model", {})
        if breakdown:
            for model_name, calls in breakdown.items():
                table.add_row(f"Model Breakdown ({model_name})", f"{calls} calls")

        console.print()
        console.print(table)
        console.print()

    def get_cost_summary(self) -> dict:
        """Returns total token consumption and dollar costs recorded by the cost tracker."""
        return self.cost_tracker.get_summary()

    def run_crew_review(self, topic: str) -> str:
        """
        Executes the Week 9 CrewAI multi-agent research & review team (Researcher, Writer, Reviewer).
        """
        try:
            from agents.crew import run_research_crew
            res = run_research_crew(topic)
            return (
                f"CrewAI multi-agent research run completed in {res['elapsed_time']}s.\n"
                f"Report saved to: {res['filepath']}\n\n"
                f"Audit Result:\n{res['output'][:500]}..."
            )
        except (ImportError, ModuleNotFoundError):
            import subprocess
            py312 = os.path.abspath("../multi-agent-week9/venv/Scripts/python.exe")
            if os.path.exists(py312):
                log.info(f"Delegating CrewAI execution to Python 3.12 environment", extra={"topic": topic})
                cmd = [py312, os.path.abspath("agents/crew.py"), topic]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    return f"CrewAI research team completed successfully!\nOutput:\n{result.stdout.strip()[-600:]}"
                else:
                    return f"CrewAI execution failed:\n{result.stderr.strip()}"
            return "Error: CrewAI is not installed in the current environment and Python 3.12 venv was not found."

    def _generate_with_retry(self, contents: Any, config: Any, max_retries: int = 3) -> Any:
        """Invokes generate_content with exponential backoff on 429 rate limit errors."""
        delay = 4.0
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            except errors.APIError as e:
                err_str = str(e)
                if getattr(e, "code", None) == 429 or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    log.warning(
                        f"Rate limit reached (429). Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...",
                        extra={"attempt": attempt + 1, "delay_s": delay},
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

    def flush(self) -> None:
        """Flushes buffered Langfuse traces to Cloud."""
        langfuse.flush()

    @observe(name="assistant_chat")
    def chat(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Production chat lifecycle:
        User input -> Input Guardrail -> Cache Lookup -> LLM / Tool Loop ->
        Output Guardrail -> Memory Update -> Cache Save -> Logging + Tracing + Cost -> Return.
        """
        start_time = time.time()
        log.info("Chat request received", extra={"user_input": user_input[:100]})

        # 1. Direct interactive command intercepts
        stripped = user_input.strip()
        if stripped.lower() == "memory":
            self.show_memory()
            return "Displayed memory state above."

        if stripped.lower().startswith(("load ", "/load ")):
            path = stripped.split(" ", 1)[1].strip()
            return self.load_document(path)

        if stripped.lower() == "cost":
            summary = self.get_cost_summary()
            return f"Cost Summary:\n{summary}"

        if stripped.lower() in ("reset", "/reset", "clear", "/clear"):
            self.conversation_history = []
            return "Conversation and task context reset."

        if stripped.lower().startswith(("crew ", "/crew ")):
            topic = stripped.split(" ", 1)[1].strip()
            return self.run_crew_review(topic)

        # 2. Input Guardrail
        input_check = check_input(user_input)
        if not input_check.is_valid:
            log.warning(
                "Blocked by input guardrail",
                extra={"reason": input_check.reason, "input": user_input[:80]},
            )
            langfuse.update_current_span(
                metadata={
                    "guardrail_blocked": True,
                    "guardrail_stage": "input",
                    "reason": input_check.reason,
                }
            )
            return f"[BLOCKED] {input_check.reason}"

        if input_check.warnings:
            log.warning("Input guardrail warnings", extra={"warnings": input_check.warnings})

        # 3. Cache Lookup
        cached_val, cache_source, score = self.cache.get(user_input)
        if cached_val is not None:
            latency = time.time() - start_time
            log.info(
                f"Cache hit ({cache_source})",
                extra={
                    "source": cache_source,
                    "score": round(score, 3),
                    "latency_s": round(latency, 3),
                },
            )
            langfuse.update_current_span(
                metadata={
                    "cache_hit": True,
                    "cache_source": cache_source,
                    "similarity_score": round(score, 3),
                    "latency_s": round(latency, 3),
                    "cost_usd": 0.0,
                }
            )
            return cached_val

        # 4. Cache Miss -> LLM / Tool Loop
        log.info("Cache miss, executing LLM tool loop", extra={"model": self.model_name})

        config = types.GenerateContentConfig(
            system_instruction=self._build_system_prompt(),
            tools=self.tools,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        # Build contents incorporating lightweight conversation/task history
        contents: List[Any] = []
        for turn in self.conversation_history[-6:]:
            contents.append(
                types.Content(
                    role=turn["role"],
                    parts=[types.Part.from_text(text=turn["text"])],
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )

        final_response = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_request_cost = 0.0

        tool_calls_history: Dict[str, int] = {}
        web_search_attempts = 0
        max_search_attempts = 3
        consecutive_tool_failures = 0
        loop_stopped_reason = ""
        last_tool_observation = ""
        should_break_loop = False

        try:
            for iteration in range(1, max_iterations + 1):
                if should_break_loop:
                    break

                response = self._generate_with_retry(
                    contents=contents,
                    config=config,
                )

                # Record token usage/cost for actual model call if metadata exists
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    p_tok = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    c_tok = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                    if p_tok or c_tok:
                        step_cost = self.cost_tracker.record_usage(
                            model=self.model_name,
                            prompt_tokens=p_tok,
                            completion_tokens=c_tok,
                        )
                        total_request_cost += step_cost
                        total_prompt_tokens += p_tok
                        total_completion_tokens += c_tok

                # Check if the model requested any tool calls
                if response.function_calls:
                    contents.append(response.candidates[0].content)

                    for function_call in response.function_calls:
                        fn_name = function_call.name
                        fn_args = dict(function_call.args) if function_call.args else {}

                        # Normalize argument signature
                        if fn_name == "web_search":
                            arg_sig = str(fn_args.get("query", "")).lower().strip()
                        else:
                            arg_sig = json.dumps(fn_args, sort_keys=True).lower()
                        call_key = f"{fn_name}:{arg_sig}"

                        # Check web search attempt limit
                        if fn_name == "web_search":
                            web_search_attempts += 1
                            if web_search_attempts > max_search_attempts:
                                loop_stopped_reason = "repeated tool call"
                                log.warning(
                                    "Tool loop terminated: repeated tool call",
                                    extra={
                                        "reason": "Max web search attempts reached",
                                        "attempts": web_search_attempts,
                                        "query": fn_args.get("query", ""),
                                    },
                                )
                                tool_result = (
                                    f"[Tool Guard] Maximum web search limit ({max_search_attempts} attempts) reached for this request. "
                                    f"Synthesize your final response now using previously gathered information."
                                )
                                contents.append(
                                    types.Part.from_function_response(
                                        name=fn_name,
                                        response={"result": tool_result},
                                    )
                                )
                                should_break_loop = True
                                break

                        # Check for repeated identical tool calls
                        call_count = tool_calls_history.get(call_key, 0) + 1
                        tool_calls_history[call_key] = call_count

                        if call_count >= 2:
                            loop_stopped_reason = "repeated tool call"
                            log.warning(
                                "Tool loop terminated: repeated tool call",
                                extra={
                                    "tool": fn_name,
                                    "tool_args": fn_args,
                                    "call_count": call_count,
                                },
                            )
                            tool_result = (
                                f"[Tool Guard] Tool '{fn_name}' was already called with identical arguments. "
                                f"To prevent an unconstructive loop, repeated calls are blocked. "
                                f"Formulate your final response immediately using the information already retrieved."
                            )
                            contents.append(
                                types.Part.from_function_response(
                                    name=fn_name,
                                    response={"result": tool_result},
                                )
                            )
                            should_break_loop = True
                            break

                        log.info(
                            f"Tool call requested: {fn_name}",
                            extra={"tool": fn_name, "tool_args": fn_args},
                        )
                        console.print(f"  [bold yellow]⚡ Tool Invocation:[/bold yellow] [bold cyan]{fn_name}[/bold cyan]([dim]{fn_args}[/dim])")
                        tool_result = self._call_tool(fn_name, fn_args)
                        last_tool_observation = str(tool_result)

                        # Check for repeated failure
                        is_error = (
                            str(tool_result).startswith("Error:")
                            or "Error executing" in str(tool_result)
                            or "attempt failed" in str(tool_result)
                        )
                        if is_error:
                            consecutive_tool_failures += 1
                        else:
                            consecutive_tool_failures = 0

                        if consecutive_tool_failures >= 3:
                            loop_stopped_reason = "tool failure"
                            log.warning(
                                "Tool loop terminated: tool failure",
                                extra={
                                    "tool": fn_name,
                                    "consecutive_failures": consecutive_tool_failures,
                                    "last_error": str(tool_result)[:100],
                                },
                            )
                            contents.append(
                                types.Part.from_function_response(
                                    name=fn_name,
                                    response={"result": str(tool_result)},
                                )
                            )
                            should_break_loop = True
                            break

                        log.info(
                            f"Tool executed: {fn_name}",
                            extra={"tool": fn_name, "result_len": len(str(tool_result))},
                        )
                        console.print(f"  [bold green]✔ Observation:[/bold green] [dim]{len(str(tool_result)):,} characters returned[/dim]")

                        contents.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": str(tool_result)},
                            )
                        )
                else:
                    final_response = response.text or ""
                    break

            if iteration >= max_iterations and not final_response:
                loop_stopped_reason = "maximum iterations"
                log.warning(
                    "Tool loop terminated: maximum iterations reached",
                    extra={"iterations": iteration, "max_iterations": max_iterations},
                )

            # Fallback response generation when loop stopped without model final text
            if not final_response:
                if loop_stopped_reason == "tool failure":
                    final_response = (
                        f"I encountered repeated errors while attempting to execute the required tools. "
                        f"Latest observation: {last_tool_observation[:200]}"
                    )
                elif loop_stopped_reason == "repeated tool call":
                    if last_tool_observation and "No relevant web search results found" in last_tool_observation:
                        final_response = (
                            f"I attempted to search for the requested information, but search queries yielded no results. "
                            f"Based on available verified knowledge, no further details could be retrieved."
                        )
                    else:
                        final_response = (
                            f"The search or tool operations completed without yielding new information. "
                            f"Available findings: {last_tool_observation[:300]}"
                        )
                elif loop_stopped_reason == "maximum iterations":
                    final_response = (
                        f"Reached maximum tool iterations ({max_iterations}) without reaching a final response. "
                        f"Observations collected: {last_tool_observation[:300]}"
                    )
                else:
                    final_response = "Unable to complete request with current tool observations."

            # 5. Output Guardrail
            output_check = check_output(final_response)
            if not output_check.is_valid:
                log.error("Blocked by output guardrail", extra={"reason": output_check.reason})
                langfuse.update_current_span(
                    metadata={
                        "output_guardrail_blocked": True,
                        "guardrail_stage": "output",
                        "reason": output_check.reason,
                    }
                )
                return f"[BLOCKED] {output_check.reason}"

            # 6. Memory update (only for valid, unblocked turns)
            try:
                self._update_memory(user_input, final_response)
            except Exception as e:
                log.warning(f"Memory update encountered an error: {e}", extra={"error": str(e)})

            # Update conversation/task context (separate from long-term memory)
            self.conversation_history.append({"role": "user", "text": user_input})
            self.conversation_history.append({"role": "model", "text": final_response})

            # 7. Cache population (only completed, non-error, non-blocked responses)
            self.cache.set(user_input, final_response)
            log.info("Response cached successfully", extra={"user_input": user_input[:80]})

            # 8. Logging + Tracing + Cost Tracking finalized
            latency = time.time() - start_time
            langfuse.update_current_span(
                metadata={
                    "model": self.model_name,
                    "latency_s": round(latency, 3),
                    "cost_usd": round(total_request_cost, 6),
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                }
            )
            log.info(
                "Chat request completed successfully",
                extra={
                    "latency_s": round(latency, 3),
                    "cost_usd": round(total_request_cost, 6),
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                },
            )

            return final_response

        except Exception as e:
            log.error(f"Chat request encountered an exception: {e}", extra={"error": str(e)})
            langfuse.update_current_span(metadata={"error": str(e)})
            raise

    def run(self):
        """
        Interactive REPL loop with rich formatting, Markdown rendering, and system panels.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d")
        console.print()
        console.print(
            Panel(
                Text.from_markup(
                    f"[bold bright_cyan]🤖 AI Engineering Assistant[/bold bright_cyan]  [dim]|[/dim]  [bold yellow]Production Ready v1.0[/bold yellow]\n"
                    f"[dim]Runtime Date: {now_str} • Engine: Gemini 3.1 Flash Lite • Session: {self.session_id}[/dim]\n\n"
                    f"[green]● Guardrails Active[/green]  [dim]•[/dim]  "
                    f"[green]● Two-Layer Cache Active[/green]  [dim]•[/dim]  "
                    f"[green]● Persistent Memory Active[/green]  [dim]•[/dim]  "
                    f"[green]● ChromaDB RAG Connected[/green]"
                ),
                box=box.ROUNDED,
                border_style="bright_blue",
                padding=(1, 2),
            )
        )

        cmd_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        cmd_table.add_column("Cmd", style="bold cyan")
        cmd_table.add_column("Desc", style="dim")
        cmd_table.add_column("Cmd2", style="bold cyan")
        cmd_table.add_column("Desc2", style="dim")
        cmd_table.add_row("/load <path>", "Ingest PDF/text into Vector Store", "/memory", "View long-term SQLite facts")
        cmd_table.add_row("/crew <topic>", "Launch CrewAI multi-agent team", "/cost", "Display token usage & dollar spend")
        cmd_table.add_row("/reset", "Clear conversation context", "/exit", "Flush Langfuse traces & quit")

        console.print(
            Panel(
                cmd_table,
                title="[bold yellow]⚡ Interactive Commands[/bold yellow]",
                title_align="left",
                border_style="blue",
                box=box.ROUNDED,
            )
        )
        console.print()

        while True:
            try:
                user_input = console.input("[bold bright_cyan]You ❯ [/bold bright_cyan]").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q", "/exit", "/quit"):
                    console.print("\n[dim]Flushing Langfuse telemetry and closing session...[/dim]")
                    self.flush()
                    summary = self.get_cost_summary()
                    console.print(
                        Panel(
                            f"[bold green]✔ Telemetry Flushed Successfully[/bold green]\n"
                            f"[dim]Session Calls:[/dim] [bold white]{summary.get('total_calls', 0)}[/bold white]  [dim]•[/dim]  "
                            f"[dim]Total Tokens:[/dim] [bold white]{summary.get('total_tokens', 0):,}[/bold white]  [dim]•[/dim]  "
                            f"[dim]Total Spend:[/dim] [bold green]${summary.get('total_cost_usd', 0.0):.6f}[/bold green]\n\n"
                            f"[bold cyan]Thank you for using AI Engineering Assistant. Goodbye! 👋[/bold cyan]",
                            title="👋 Session Ended",
                            border_style="green",
                            box=box.ROUNDED,
                        )
                    )
                    break

                if user_input.lower() in ("memory", "/memory"):
                    self.show_memory()
                    continue

                if user_input.lower() in ("cost", "/cost"):
                    self.show_cost()
                    continue

                if user_input.lower() in ("reset", "/reset", "clear", "/clear"):
                    self.conversation_history = []
                    console.print("[bold green]✔ Conversation and task context have been reset.[/bold green]\n")
                    continue

                if user_input.lower().startswith(("load ", "/load ")):
                    path = user_input.split(" ", 1)[1].strip()
                    with console.status(f"[bold cyan]Ingesting '{path}' into ChromaDB knowledge base...[/bold cyan]", spinner="dots"):
                        msg = self.load_document(path)
                    if msg.startswith("Error"):
                        console.print(Panel(f"[bold red]{msg}[/bold red]", border_style="red", box=box.ROUNDED))
                    else:
                        console.print(Panel(f"[bold green]{msg}[/bold green]", border_style="green", box=box.ROUNDED))
                    continue

                if user_input.lower().startswith(("crew ", "/crew ")):
                    topic = user_input.split(" ", 1)[1].strip()
                    console.print(f"[bold yellow]Launching CrewAI multi-agent research team on topic:[/bold yellow] [bold white]'{topic}'[/bold white]")
                    with console.status("[bold cyan]CrewAI agents (Researcher, Writer, Reviewer) working...[/bold cyan]", spinner="dots"):
                        crew_res = self.run_crew_review(topic)
                    console.print(
                        Panel(
                            Markdown(crew_res),
                            title="[bold yellow]👥 CrewAI Multi-Agent Research Output[/bold yellow]",
                            border_style="yellow",
                            box=box.ROUNDED,
                            padding=(1, 2),
                        )
                    )
                    continue

                # Standard chat turn with spinner & rich markdown rendering
                with console.status("[bold cyan]Thinking & executing tools...[/bold cyan]", spinner="dots"):
                    response = self.chat(user_input)

                doc_info = f" • Active Document: [bold white]{self.active_document_name}[/bold white]" if self.active_document_name else ""
                console.print()
                console.print(
                    Panel(
                        Markdown(response),
                        title="[bold cyan]🤖 AI Assistant[/bold cyan]",
                        title_align="left",
                        border_style="bright_blue",
                        box=box.ROUNDED,
                        padding=(1, 2),
                        subtitle=f"[dim]Model: {self.model_name}{doc_info}[/dim]",
                        subtitle_align="right",
                    )
                )
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Flushing telemetry...[/dim]")
                self.flush()
                console.print("[bold yellow]Session ended by user.[/bold yellow]\n")
                break
            except Exception as e:
                console.print(Panel(f"[bold red]Error:[/bold red] {e}", border_style="red", box=box.ROUNDED))


if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()
