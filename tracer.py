import time
from functools import wraps
from dotenv import load_dotenv
from langfuse import get_client, observe
from logger import log

# Load credentials from .env
load_dotenv()

# Initialize Langfuse client
langfuse = get_client()


def trace_llm_call(func):
    """Decorator — wraps an LLM execution with Langfuse v4 tracing and structured logging."""

    @wraps(func)
    @observe(name=func.__name__)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start

            # Update the active span with metadata in v4
            langfuse.update_current_span(
                metadata={"latency_s": round(latency, 3)},
            )

            log.info(
                f"{func.__name__} completed successfully",
                extra={"latency_s": round(latency, 3)},
            )
            return result

        except Exception as e:
            langfuse.update_current_span(
                metadata={"error": str(e)},
            )
            log.error(f"{func.__name__} failed: {e}", extra={"error": str(e)})
            raise

    return wrapper


if __name__ == "__main__":
    # Smoke test using a simulated LLM function
    @trace_llm_call
    def mock_generate_text(prompt: str) -> str:
        time.sleep(0.4)  # Simulate network / inference latency
        return f"Synthesized response for: '{prompt}'"

    print("Executing mock traced function...")
    output = mock_generate_text("Explain distributed tracing in 5 words.")
    print("Output:", output)

    # Flush background OpenTelemetry events to the cloud before exiting
    langfuse.flush()
    print("Traces flushed to Langfuse in trace section.")