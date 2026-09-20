"""
Core AI Engineering Assistant Orchestration Layer (Week 10 Day 4 - Step 3)
Integrates ChromaDB RAG, persistent semantic/episodic memory, and tool execution.
"""

import os
import sys
import uuid
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from google import genai
from google.genai import types

from tools.web_search import web_search
from tools.code_executor import execute_python

try:
    from tools.pdf_reader import read_pdf, read_pdf_page
except ImportError:
    read_pdf = None
    read_pdf_page = None

from rag.vector_store import get_collection, search
from rag.ingestion import ingest_pdf, ingest_text_file

from memory.memory import PersistentMemory
from memory.memory_extractor import extract_facts_from_conversation


class Assistant:
    """
    Core AI Engineering Assistant that orchestrates conversational responses,
    autonomous tool execution, ChromaDB RAG retrieval, and persistent memory.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        memory_db_path: str = "assistant_memory.db",
        rag_collection: str = "knowledge_base",
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

        self.tools = self._build_tools()

    def _build_tools(self) -> List[types.Tool]:
        """
        Defines tool contracts and JSON schemas for all assistant capabilities:
        - web_search
        - code_executor
        - read_pdf
        - read_pdf_page
        - search_knowledge_base
        """
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="web_search",
                        description=(
                            "Searches DuckDuckGo for live web information, current events, and documentation. "
                            "Returns numbered results with titles, URLs, and concise snippets."
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
                            "Use to obtain a high-level overview of a PDF document."
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
                            "Extracts complete text from a single specific 1-indexed page of a local PDF document."
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
                            "Returns semantically relevant passages and similarity scores."
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
                ]
            )
        ]

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

            elif name == "read_pdf":
                if read_pdf is None:
                    return "Error: PDF reading is unavailable because PyMuPDF (fitz) is not installed."
                file_path = args.get("file_path", "")
                return read_pdf(file_path=file_path)

            elif name == "read_pdf_page":
                if read_pdf_page is None:
                    return "Error: PDF page reading is unavailable because PyMuPDF (fitz) is not installed."
                file_path = args.get("file_path", "")
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
        and dynamically injected persistent memory facts.
        """
        facts_block = self.memory.format_for_prompt()

        return (
            "You are an AI Engineering Assistant, a reliable and skilled technical assistant.\n\n"
            f"{facts_block}\n\n"
            "Capabilities and Available Tools:\n"
            "- web_search: Search the live web for recent developments, documentation, and factual verification.\n"
            "- code_executor: Run Python code in an isolated sandbox for math, data analysis, and script verification.\n"
            "- read_pdf: Inspect local PDF files to obtain metadata and document overviews.\n"
            "- read_pdf_page: Read specific pages of a local PDF document in detail.\n"
            "- search_knowledge_base: Query the internal vector store for indexed documents and notes.\n\n"
            "Operating Rules:\n"
            "1. When answering questions, use any personal context and facts above to tailor your response.\n"
            "2. DO NOT explicitly reference 'my database', 'stored memory', or 'system records'. Respond naturally.\n"
            "3. Use tools whenever external facts, live data, calculations, or document inspections are required.\n"
            "4. Never fabricate or invent tool outputs. Ground all claims strictly on tool observations.\n"
            "5. If a tool reports an error or returns empty results, reason through alternative strategies or clearly inform the user.\n"
            "6. Provide clear, concise, and structured answers."
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
        extracted_facts = extract_facts_from_conversation(user_message, current_facts_str)

        for fact in extracted_facts:
            if fact.action in ["store", "update"]:
                self.memory.store_fact(
                    key=fact.key,
                    value=fact.value,
                    category=fact.category,
                    confidence="high",
                    source="user_statement",
                )
                print(f"  [Memory] Stored fact: '{fact.key}' -> '{fact.value}' ({fact.category})")
            elif fact.action == "delete":
                deleted = self.memory.delete_fact(fact.key)
                if deleted:
                    print(f"  [Memory] Deleted fact: '{fact.key}'")

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
                return (
                    f"Successfully ingested PDF '{os.path.basename(clean_path)}' "
                    f"({chunks_count} chunks) into knowledge base."
                )
            except Exception as e:
                return f"Error ingesting PDF '{clean_path}': {str(e)}"
        elif ext in (".txt", ".md", ".py", ".json", ".csv"):
            try:
                chunks_count = ingest_text_file(clean_path, collection_name=self.rag_collection)
                return (
                    f"Successfully ingested text file '{os.path.basename(clean_path)}' "
                    f"({chunks_count} chunks) into knowledge base."
                )
            except Exception as e:
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

    def chat(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Runs the agentic tool-use loop:
        user input -> LLM -> tool call? -> execute tool -> send result to LLM -> loop until final response.
        Enforces max_iterations protection against infinite tool loops.
        Updates persistent memory after completion.
        """
        # 1. Direct interactive command intercepts
        stripped = user_input.strip()
        if stripped.lower() == "memory":
            self.show_memory()
            return "Displayed memory state above."

        if stripped.lower().startswith(("load ", "/load ")):
            path = stripped.split(" ", 1)[1].strip()
            return self.load_document(path)

        config = types.GenerateContentConfig(
            system_instruction=self._build_system_prompt(),
            tools=self.tools,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        contents: List[Any] = [user_input]
        final_response = ""

        for iteration in range(1, max_iterations + 1):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            # Check if the model requested any tool calls
            if response.function_calls:
                contents.append(response.candidates[0].content)

                for function_call in response.function_calls:
                    fn_name = function_call.name
                    fn_args = dict(function_call.args) if function_call.args else {}

                    print(f"  [Tool Call] {fn_name}({fn_args})")
                    tool_result = self._call_tool(fn_name, fn_args)
                    print(f"  [Observation] {len(str(tool_result))} characters returned")

                    contents.append(
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": str(tool_result)},
                        )
                    )
            else:
                final_response = response.text or ""
                break

        if not final_response:
            final_response = (
                f"Warning: Reached maximum tool iterations ({max_iterations}) without reaching a final response."
            )

        # 2. Update persistent episodic and semantic memory
        try:
            self._update_memory(user_input, final_response)
        except Exception as e:
            print(f"  [Warning] Memory update encountered an error: {e}")

        return final_response

    def run(self):
        """
        Interactive REPL loop supporting chat, 'load <path>', 'memory', and 'exit'.
        """
        print("=== AI Engineering Assistant (RAG & Memory Enabled) ===")
        print("Commands: 'memory' to view facts, 'load <file_path>' to ingest, 'exit' to quit.\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                response = self.chat(user_input)
                print(f"\nAssistant:\n{response}")
            except KeyboardInterrupt:
                print("\nSession ended.")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()
