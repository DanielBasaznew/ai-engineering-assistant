"""
Core AI Engineering Assistant Orchestration Layer (Week 10 Day 4 - Step 2)
Implements tool schemas, dispatcher, system prompt, and agentic tool-use loop.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools.web_search import web_search
from tools.code_executor import execute_python

try:
    from tools.pdf_reader import read_pdf, read_pdf_page
except ImportError:
    read_pdf = None
    read_pdf_page = None


class Assistant:
    """
    Core AI Engineering Assistant that orchestrates conversational responses
    and autonomous tool execution using Gemini.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.tools = self._build_tools()

    def _build_tools(self) -> List[types.Tool]:
        """
        Defines tool contracts and JSON schemas for all assistant capabilities:
        - web_search
        - code_executor
        - read_pdf
        - read_pdf_page
        - search_knowledge_base (schema defined, RAG dispatch pending)
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

        NOTE: search_knowledge_base is defined in schemas but pending RAG integration.
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
                    return "Error: PDF reading is currently unavailable because PyMuPDF (fitz) is not installed."
                file_path = args.get("file_path", "")
                return read_pdf(file_path=file_path)

            elif name == "read_pdf_page":
                if read_pdf_page is None:
                    return "Error: PDF page reading is currently unavailable because PyMuPDF (fitz) is not installed."
                file_path = args.get("file_path", "")
                page_number = int(args.get("page_number", 1))
                return read_pdf_page(file_path=file_path, page_number=page_number)

            elif name == "search_knowledge_base":
                return (
                    "[PENDING] search_knowledge_base is defined but pending RAG integration. "
                    "Vector store retrieval will be wired in a subsequent step."
                )

            else:
                return f"Error: Tool '{name}' is not recognized."

        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def _build_system_prompt(self) -> str:
        """
        Constructs the base system prompt defining role, capabilities, and tool usage contracts.
        Memory injection will be added in a later step.
        """
        return (
            "You are an AI Engineering Assistant, a reliable and skilled technical assistant.\n\n"
            "Capabilities and Available Tools:\n"
            "- web_search: Search the live web for recent developments, documentation, and factual verification.\n"
            "- code_executor: Run Python code in an isolated sandbox for math, data analysis, and script verification.\n"
            "- read_pdf: Inspect local PDF files to obtain metadata and document overviews.\n"
            "- read_pdf_page: Read specific pages of a local PDF document in detail.\n"
            "- search_knowledge_base: Query the internal vector store for indexed documents.\n\n"
            "Operating Rules:\n"
            "1. Use tools whenever external facts, live data, calculations, or document inspections are required.\n"
            "2. Never fabricate or invent tool outputs. Ground all claims strictly on tool observations.\n"
            "3. If a tool reports an error or returns empty results, reason through alternative strategies or clearly inform the user.\n"
            "4. Provide clear, concise, and structured answers."
        )

    def chat(self, user_input: str, max_iterations: int = 5) -> str:
        """
        Runs the agentic tool-use loop:
        user input -> LLM -> tool call? -> execute tool -> send result to LLM -> loop until final response.
        Enforces max_iterations protection against infinite tool loops.
        """
        config = types.GenerateContentConfig(
            system_instruction=self._build_system_prompt(),
            tools=self.tools,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        contents: List[Any] = [user_input]

        for iteration in range(1, max_iterations + 1):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            # Check if the model requested any tool calls
            if response.function_calls:
                # Append the model's candidate content with function call(s)
                contents.append(response.candidates[0].content)

                for function_call in response.function_calls:
                    fn_name = function_call.name
                    fn_args = dict(function_call.args) if function_call.args else {}

                    print(f"  [Tool Call] {fn_name}({fn_args})")
                    tool_result = self._call_tool(fn_name, fn_args)
                    print(f"  [Observation] {len(str(tool_result))} characters returned")

                    # Append tool result back to contents
                    contents.append(
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": str(tool_result)},
                        )
                    )
            else:
                # Model produced final textual response
                return response.text or ""

        return (
            f"Warning: Reached maximum tool iterations ({max_iterations}) without reaching a final response."
        )


if __name__ == "__main__":
    print("=== Testing Assistant in Isolation ===")
    assistant = Assistant()

    print("\n--- Test 1: Direct Conversation ---")
    reply = assistant.chat("Hello! Who are you and what tools can you use?")
    print("Assistant:", reply)

    print("\n--- Test 2: Code Execution Tool ---")
    reply = assistant.chat("What is 123456 * 654321? Calculate it with python.")
    print("Assistant:", reply)

    print("\n--- Test 3: Web Search Tool ---")
    reply = assistant.chat("What is the latest major release of Python and when was it released? Search the web.")
    print("Assistant:", reply)
