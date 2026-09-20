from dataclasses import dataclass, field
from typing import Dict, Optional
from logger import log

# Pricing table: Cost per 1,000,000 tokens (USD)
# Standard format: {model_name: {"input": price_per_1M, "output": price_per_1M}}
MODEL_COSTS_PER_MILLION = {
    "gemini-3.1-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    # OpenAI & Groq fallbacks
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}

@dataclass
class CostTracker:
    """Tracks token consumption and running estimated dollar costs."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls_by_model: Dict[str, int] = field(default_factory=dict)

    def calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculates cost for a single invocation."""
        pricing = MODEL_COSTS_PER_MILLION.get(model.lower())

        if not pricing:
            log.warning(
                f"Model '{model}' not found in cost registry. Defaulting to $0.00."
            )
            return 0.0

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def record_usage(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Records token usage, updates aggregates, and logs the cost breakdown."""
        call_cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.total_cost_usd += call_cost
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1

        log.info(
            f"Usage recorded for {model}",
            extra={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "call_cost_usd": round(call_cost, 6),
                "total_cost_usd": round(self.total_cost_usd, 6),
            },
        )
        return call_cost

    def get_summary(self) -> dict:
        """Returns the current snapshot of all usage metrics."""
        return {
            "total_calls": sum(self.calls_by_model.values()),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "breakdown_by_model": self.calls_by_model,
        }


# Global tracker instance for easy import across modules
tracker = CostTracker()


if __name__ == "__main__":
    # Test tracking across multiple models
    print("Testing CostTracker...")

    # Simulated call 1: Llama on Groq
    c1 = tracker.record_usage(
        model="llama-3.3-70b-versatile",
        prompt_tokens=1250,
        completion_tokens=420,
    )
    print(f"Call 1 cost: ${c1:.6f}")

    # Simulated call 2: Gemini Flash
    c2 = tracker.record_usage(
        model="gemini-1.5-flash",
        prompt_tokens=3500,
        completion_tokens=850,
    )
    print(f"Call 2 cost: ${c2:.6f}")

    # Print overall summary
    summary = tracker.get_summary()
    print("\nTracker Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")