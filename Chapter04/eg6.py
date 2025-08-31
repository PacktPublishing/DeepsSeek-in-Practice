"""
eg6.py — Strategies for Robustness and Special Cases with DeepSeek structured output.

This example shows:
1) Handling reasoning models that output both "reasoning_content" (thinking) and "content" (answer).
2) A Retry-Repair loop to recover from occasional invalid JSON.
3) A defensive schema pattern for the 2025-style possessive/apostrophe quirk.
4) A lightweight "which model when" cheat sheet.

WHY these patterns:
- Reasoning models "think before they speak" and return extra tokens you must NEVER reuse in follow-up turns.
- Even with structured generation, malformed JSON can occur; having the model repair its own output is simple and effective.
- Possessive field names (user’s_name) can provoke subtle key-matching failures; use aliases or rename fields to avoid landmines.
"""

import os
from typing import Optional, Type

import instructor
from openai import OpenAI
from pydantic import AliasChoices, BaseModel, Field

# Model selection cheat sheet (choose the right tool for the job):
MODEL_HINTS = {
    # Fast, cheap, reliable for simple extraction
    "simple_extraction": "deepseek-chat",
    # When you want code + metadata jointly
    "code_plus_metadata": "deepseek-coder",
    # Complex extraction with an audit trail of the model's thinking
    "complex_reasoning": "deepseek-reasoner",
    # Constrained JSON via Fireworks-hosted R1 (separate provider/config)
    "constrained_json": "fireworks/deepseek-r1",
}


# Wrap OpenAI client with Instructor, pointing at DeepSeek's API.
# MD_JSON mode is recommended for reasoning models that produce "thinking" + "answer".
openai_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
client = instructor.from_openai(openai_client, mode=instructor.Mode.MD_JSON)


# --------------- Schemas ---------------


class DateInfo(BaseModel):
    """Small, clear schema to demo extraction; minimal so the model can succeed easily."""

    month: str = Field(description="Month name or number, e.g., 'March' or '03'")
    day: int = Field(ge=1, le=31, description="Day of month (1-31)")
    year: Optional[int] = Field(default=None, description="4-digit year if present")


class Contact(BaseModel):
    """
    Defensive schema for the possessive/apostrophe pitfall.
    We normalize field names (no apostrophes) but accept multiple variants at validation time.
    This lets the parser succeed even if the model returns user’s_name / user's_name / users_name.
    """

    user_name: str = Field(
        description="Normalized user name (schema avoids apostrophes)",
        validation_alias=AliasChoices(
            "user_name", "user’s_name", "user's_name", "users_name"
        ),
    )
    company_address: str = Field(
        description="Normalized address (schema avoids apostrophes)",
        validation_alias=AliasChoices(
            "company_address",
            "company’s_address",
            "company's_address",
            "companys_address",
        ),
    )


# --------------- Retry-Repair Loop ---------------


def extract_with_retry(
    prompt: str,
    model_class: Type[BaseModel],
    max_retries: int = 2,
    model: str = MODEL_HINTS["simple_extraction"],
) -> BaseModel:
    """
    Make the model fix its own malformed JSON by reflecting the error back.
    WHY: LLMs are excellent at self-correction when you show them the exact failure.
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=model_class,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                # Feed the error back in a crisp way; ask for corrected JSON only.
                prompt = (
                    f"The previous JSON was invalid:\n{str(e)}\n\n"
                    f"Please fix and return valid JSON for: {prompt}"
                )
            else:
                raise


# --------------- Demo ---------------

if __name__ == "__main__":
    # 1) Reasoning models: capture both the parsed result and the raw completion
    # NOTE: The raw completion can contain message.reasoning_content (the chain-of-thought).
    # You can log it to understand model behavior, but DO NOT feed it back into the conversation.
    date, raw = client.chat.completions.create_with_completion(
        model=MODEL_HINTS["complex_reasoning"],
        messages=[
            {
                "role": "user",
                "content": "Extract the date: The meeting is on March 15th",
            }
        ],
        response_model=DateInfo,
    )

    print("Reasoning (do not reuse in future turns):")
    print(
        getattr(
            raw.choices[0].message,
            "reasoning_content",
            "(no reasoning_content available)",
        )
    )

    print("\nModel answer content (the final, user-facing output):")
    print(raw.choices[0].message.content)

    print("\nParsed struct:")
    print(date.model_dump())

    # 2) Retry-Repair loop demo + possessive/apostrophe robustness:
    # The schema avoids apostrophes in field names (best practice),
    # but also accepts multiple possessive variants via validation_alias.
    contact_prompt = """Extract contact JSON with fields "user_name" and "company_address".
The text: user's_name is Pat O’Neil; company’s_address is 221B Baker St, London."""
    contact = extract_with_retry(
        prompt=contact_prompt,
        model_class=Contact,
        max_retries=2,
        model=MODEL_HINTS["simple_extraction"],
    )
    print("\nContact (robust to possessive naming quirks):")
    print(contact.model_dump())
