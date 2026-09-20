import json
import os
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class ExtractedFact(BaseModel):
    key: str = Field(
        description="A concise snake_case identifier for the fact (e.g., favorite_language, occupation)."
    )
    value: str = Field(
        description="The detailed value or statement (e.g., Python, Mechanical Engineer)."
    )
    category: str = Field(
        default="general",
        description="Category classification: 'preferences', 'background', 'goals', or 'general'."
    )
    action: Literal["store", "update", "delete", "none"] = Field(
        default="store",
        description="Action to take: 'store' for new facts, 'update' for changed facts, 'delete' for forgotten facts, or 'none'."
    )


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


SYSTEM_PROMPT = """You are an automated Memory Extraction Engine for an AI assistant.
Your job is to extract personal facts, preferences, background details, or direct instructions from the user's message.

Existing Stored Facts:
{existing_facts}

Rules:
1. Extract facts ONLY if the user explicitly stated something personal about themselves or their preferences.
2. Do NOT infer, assume, or fabricate facts from general questions.
3. If the user updates an existing fact (e.g., "I use Rust now"), set action to 'update'.
4. If the user asks you to forget something (e.g., "forget my age"), set action to 'delete'.
5. If no personal facts were stated, return an empty 'facts' list.
"""


def extract_facts_from_conversation(
    user_message: str,
    existing_facts_formatted: str = "No existing facts."
) -> list[ExtractedFact]:
    """
    Analyzes a user message using Gemini API and returns extracted ExtractedFact objects.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[Warning] GEMINI_API_KEY missing. Skipping automatic memory extraction.")
        return []

    try:
        client = genai.Client(api_key=api_key)
        prompt = SYSTEM_PROMPT.format(existing_facts=existing_facts_formatted)

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=f"{prompt}\n\nUser Message: {user_message}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.0
            )
        )

        result = ExtractionResult.model_validate_json(response.text)
        return [f for f in result.facts if f.action != "none"]

    except Exception as e:
        print(f"[Warning] Memory extraction encountered an error ({e}). Skipping turn.")
        return []


if __name__ == "__main__":
    print("--- Testing Memory Extraction Engine ---")

    test_messages = [
        # 1. Multiple facts in one sentence
        "I live in Amsterdam and I'm highly allergic to peanuts.",
        
        # 2. Ambiguous/Future preference
        "I'm thinking about learning Go sometime next year if I have the time.",
        
        # 3. Fact about someone else (Should NOT be stored as the user's fact)
        "My brother is a doctor and he works crazy hours.",
        
        # 4. Temporary state (Should it store this long-term?)
        "I'm super exhausted today from lack of sleep.",
        
        # 5. Implicit/Indirect fact
        "I've been driving my Tesla for three years now.",
        
        # 6. System/Procedural instruction
        "Please always address me as 'Captain' from now on.",
        
        # 7. Contradiction/Update in a single breath
        "I used to hate coffee, but now I drink three cups a day.",
        
        # 8. Negative preference
        "Make sure you never suggest horror movies to me, I absolutely despise them.",
        
        # 9. Generic conversational noise + technical question
        "Wow, the weather is really nice today. Anyway, how do I write a Python loop?",
        
        # 10. Complex Update/Correction
        "Actually, scratch what I said earlier, I don't live in Amsterdam, I moved to Rotterdam."
    ]

    current_facts_str = "No existing facts."

    for msg in test_messages:
        print(f"\nUser Said: '{msg}'")
        extracted = extract_facts_from_conversation(msg, current_facts_str)
        if not extracted:
            print("  -> No personal facts extracted.")
        else:
            for fact in extracted:
                print(f"  -> Extracted: key='{fact.key}', value='{fact.value}', action='{fact.action}'")
