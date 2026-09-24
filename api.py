"""
FastAPI Application Layer for AI Engineering Assistant (Production Ready).

Provides REST API endpoints wrapping the existing Assistant orchestration:
- GET  /health: Service health check
- POST /chat: Interactive chat with sliding-window IP rate limiting & input guardrail
- GET  /memory: Persistent semantic memory inspection
"""

import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure UTF-8 stdout on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# Import existing project components
from assistant import Assistant
from guardrails import check_input
from logger import log

# Curriculum alias
AIEngineeringAssistant = Assistant

# Rate Limiter Configuration: 10 requests per minute per IP (sliding window)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0

# In-memory storage for sliding window timestamps: ip -> list of float timestamps
_rate_limit_history: Dict[str, List[float]] = defaultdict(list)


def get_client_ip(request: Request) -> str:
    """Extracts client IP address, respecting reverse proxies (Railway, Vercel, etc.)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


def check_rate_limit(client_ip: str) -> None:
    """
    Enforces a 10 requests per minute per IP sliding-window rate limit.
    Raises HTTP 429 when exceeded.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    # Filter timestamps to keep only those within the current sliding window
    active_timestamps = [t for t in _rate_limit_history[client_ip] if t > window_start]

    if len(active_timestamps) >= RATE_LIMIT_REQUESTS:
        log.warning(
            "Rate limit exceeded (HTTP 429)",
            extra={"client_ip": client_ip, "requests_in_window": len(active_timestamps)},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 10 requests per minute allowed.",
        )

    active_timestamps.append(now)
    _rate_limit_history[client_ip] = active_timestamps


def reset_rate_limit() -> None:
    """Clears rate limit records (primarily for testing)."""
    _rate_limit_history.clear()


# --- Pydantic Schemas ---


class ChatRequest(BaseModel):
    """Input payload for POST /chat."""
    message: str = Field(..., description="User query or instruction for the assistant")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier")

    @model_validator(mode="before")
    @classmethod
    def populate_message(cls, values: Any) -> Any:
        """Allows flexible field names (message, prompt, query, user_input)."""
        if isinstance(values, dict):
            if not values.get("message"):
                for alt_key in ("prompt", "query", "user_input", "text"):
                    if values.get(alt_key):
                        values["message"] = values[alt_key]
                        break
        return values


class ChatResponse(BaseModel):
    """Structured response schema for POST /chat."""
    response: str = Field(..., description="Assistant response text or guardrail block message")
    status: str = Field(default="success", description="Status code ('success', 'blocked', 'error')")
    cost: Dict[str, Any] = Field(default_factory=dict, description="Token consumption & estimated dollar cost summary")
    cost_report: Dict[str, Any] = Field(default_factory=dict, description="Alias for cost accounting summary")


# --- FastAPI Application Initialization ---

app = FastAPI(
    title="AI Engineering Assistant API",
    description="Production FastAPI service layer for the AI Engineering Assistant.",
    version="1.0.0",
)

# Enable CORS for external frontends (e.g. Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Assistant instance using existing components
try:
    assistant = Assistant()
except Exception as e:
    assistant = None
    log.warning(f"Assistant could not be initialized at module import: {e}")


def get_assistant() -> Assistant:
    """Ensures an active Assistant singleton instance is available."""
    global assistant
    if assistant is None:
        assistant = Assistant()
    return assistant


# --- Endpoints ---


@app.get("/")
def root() -> Dict[str, str]:
    """Root info endpoint."""
    return {
        "service": "AI Engineering Assistant API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-engineering-assistant",
        "version": "1.0.0",
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat_request: ChatRequest, request: Request) -> ChatResponse:
    """
    Chat endpoint following the strict production flow:
    request -> IP rate limit check -> check_input() guardrail -> assistant.chat() -> cost report -> structured response.
    """
    # 1. IP rate limit check
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    # 2. check_input() guardrail check
    user_message = chat_request.message
    validation = check_input(user_message)
    current_assistant = get_assistant()

    if not validation.is_valid:
        log.warning(
            "Request blocked by input guardrail at API layer",
            extra={"reason": validation.reason, "client_ip": client_ip},
        )
        cost_summary = current_assistant.get_cost_summary()
        return ChatResponse(
            response=f"[BLOCKED] {validation.reason}",
            status="blocked",
            cost=cost_summary,
            cost_report=cost_summary,
        )

    # 3. assistant.chat()
    response_text = current_assistant.chat(user_message)

    # 4. cost report
    cost_summary = current_assistant.get_cost_summary()

    # 5. structured response
    status_str = "blocked" if response_text.startswith("[BLOCKED]") else "success"
    return ChatResponse(
        response=response_text,
        status=status_str,
        cost=cost_summary,
        cost_report=cost_summary,
    )


@app.get("/memory")
def get_memory() -> Dict[str, Any]:
    """Retrieves all stored facts from persistent semantic memory."""
    current_assistant = get_assistant()
    facts = current_assistant.memory.get_all_facts()
    return {
        "status": "ok",
        "facts": facts,
        "count": len(facts),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
