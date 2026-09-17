import os
import time
from dotenv import load_dotenv
from google import genai

from cost_tracker import tracker
from langfuse import get_client, observe
from logger import log

load_dotenv()

# Initialize Langfuse client
langfuse = get_client()

# Initialize Google GenAI client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


@observe(name="gemini_generate_completion")
def generate_gemini_completion(
    prompt: str, model: str = "gemini-3.1-flash-lite"
) -> str:
    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        latency = time.time() - start_time
        text_output = response.text

        # Extract token usage from Gemini metadata
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0

        # Calculate cost
        call_cost = tracker.record_usage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Enrich active Langfuse span
        langfuse.update_current_span(
            metadata={
                "model": model,
                "latency_s": round(latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(call_cost, 6),
            }
        )

        log.info(
            f"Gemini call completed: {model}",
            extra={
                "model": model,
                "latency_s": round(latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(call_cost, 6),
            },
        )

        return text_output

    except Exception as e:
        langfuse.update_current_span(metadata={"error": str(e)})
        log.error(f"Gemini call failed: {e}", extra={"error": str(e)})
        raise


if __name__ == "__main__":
    test_prompt = "Explain in two sentences why software engineering observability matters."
    print("Sending prompt to Gemini...")

    reply = generate_gemini_completion(prompt=test_prompt)
    print("\nGemini Response:\n", reply)

    # Ensure telemetry is dispatched
    langfuse.flush()

    print("\n--- Session Cost Summary ---")
    print(tracker.get_summary())