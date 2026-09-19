import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cache import app_cache
from cost_tracker import tracker
from guardrails import check_input, check_output
from langfuse import get_client, observe
from logger import log

load_dotenv()

langfuse = get_client()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.1-flash-lite"


@retry(
    retry=retry_if_exception_type(errors.ServerError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=True,
)
def _call_gemini_stream(prompt: str):
    """Initiates a streaming response from Gemini with backoff retry."""
    return client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
    )


@observe(name="gemini_stream_completion")
def stream_and_cache(prompt: str) -> str:
    """Streams response from Gemini, measures latency, records cost, and caches output."""
    start_time = time.time()
    accumulated_chunks = []
    first_token_time = None

    try:
        response_stream = _call_gemini_stream(prompt)

        for chunk in response_stream:
            chunk_text = chunk.text or ""
            if chunk_text:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                print(chunk_text, end="", flush=True)
                accumulated_chunks.append(chunk_text)

        print()  # Newline after stream finishes
        total_latency = time.time() - start_time
        full_response = "".join(accumulated_chunks)

        # Estimate tokens based on typical usage (~4 chars/token)
        est_input_tokens = max(1, len(prompt) // 4)
        est_output_tokens = max(1, len(full_response) // 4)

        call_cost = tracker.record_usage(
            model=MODEL_NAME,
            prompt_tokens=est_input_tokens,
            completion_tokens=est_output_tokens,
        )

        langfuse.update_current_span(
            metadata={
                "model": MODEL_NAME,
                "ttft_s": round(first_token_time or total_latency, 3),
                "total_latency_s": round(total_latency, 3),
                "cost_usd": round(call_cost, 6),
            }
        )

        log.info(
            f"Streaming completed: {MODEL_NAME}",
            extra={
                "ttft_s": round(first_token_time or total_latency, 3),
                "total_latency_s": round(total_latency, 3),
                "cost_usd": round(call_cost, 6),
            },
        )

        # Store in cache for future hits
        app_cache.set(prompt, full_response)
        return full_response

    except Exception as e:
        langfuse.update_current_span(metadata={"error": str(e)})
        log.error(f"Streaming failed: {e}", extra={"error": str(e)})
        raise


def cached_safe_chat(prompt: str):
    """End-to-end flow: input guardrail -> exact/semantic cache -> streaming LLM -> output guardrail."""
    # 1. Input Guardrail
    input_validation = check_input(prompt)
    if not input_validation.is_valid:
        log.warning("Blocked by input guardrail", extra={"reason": input_validation.reason})
        return f"[BLOCKED] {input_validation.reason}", "guardrail"

    # 2. Check Cache
    cached_val, source, score = app_cache.get(prompt)
    if cached_val:
        return cached_val, source

    # 3. Cache Miss -> Stream from LLM
    print(f"\n[STREAMING FROM {MODEL_NAME}]:")
    raw_response = stream_and_cache(prompt)

    # 4. Output Guardrail
    output_validation = check_output(raw_response)
    if not output_validation.is_valid:
        log.error("Blocked by output guardrail", extra={"reason": output_validation.reason})
        return f"[BLOCKED] {output_validation.reason}", "guardrail"

    return raw_response, "llm"


if __name__ == "__main__":
    print(f"--- Running Two-Layer Cache + Streaming Test ---")

    test_queries = [
        "What is retrieval augmented generation?",
        "What is retrieval augmented generation?",          # Exact repeat
        "Explain retrieval augmented generation to me.",    # Semantic paraphrase
        "What are the main functions of a car alternator?",  # Fresh question
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n==========================================")
        print(f"Query {i}: '{q}'")
        resp, src = cached_safe_chat(q)
        print(f"--> Result Source: [{src.upper()}]")
        if src != "llm":
            print(f"Cached Response preview:\n{resp[:150]}...")

    langfuse.flush()
    print("\n--- Final Cost Summary ---")
    print(tracker.get_summary())