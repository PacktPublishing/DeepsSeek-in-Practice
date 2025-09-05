import json
import os
from typing import Literal
import pickle

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from garminconnect import Garmin
from loguru import logger
from openai import OpenAI

from utils import (
    SYSTEM_PROMPT,
    DailySummary,
    HealthSummaryRequest,
    get_daily_summary_prompt,
    get_garmin_client,
)

load_dotenv(".envrc", override=True)

assert os.environ["DEEPSEEK_API_KEY"] is not None, "DEEPSEEK_API_KEY is not set"


def llm(
    messages: list[dict], model: str, response_format: dict | None = None
) -> tuple[dict, str | None]:
    """Call DeepSeek LLM API with messages.

    Args:
        messages: List of chat messages with role and content
        model: Model name (deepseek-chat or deepseek-reasoner)
        response_format: Optional response format specification

    Returns:
        Tuple of (parsed JSON response, reasoning content if available)
    """
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0.0,
    )
    message = response.choices[0].message

    if hasattr(message, "reasoning_content"):
        reasoning_content = message.reasoning_content
    else:
        reasoning_content = None

    logger.info(f"LLM response generated successfully using {model}")
    return json.loads(message.content), reasoning_content


def get_daily_summary(
    garmin: Garmin,
    date: str,
    model: Literal["deepseek-chat", "deepseek-reasoner"],
    verbose: bool = False,
) -> DailySummary:
    """Generate AI-powered daily health summary for a specific date.

    Args:
        garmin: Authenticated Garmin client instance
        date: Date string in YYYY-MM-DD format
        model: AI model to use for analysis
        verbose: Whether to print detailed output (default: False)

    Returns:
        Daily health summary with insights and recommendations
    """

    if not garmin.username:
        logger.warning("Using test Garmin account, loading prompt from file")

        prompt = pickle.load(open("daily_summary_prompt.pkl", "rb"))
    else:
        prompt = get_daily_summary_prompt(garmin, date)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response, reasoning = llm(messages, model, {"type": "json_object"})
    health_summary = DailySummary.model_validate(response)

    logger.info(f"Daily summary generated successfully for {date}")
    return health_summary


app = FastAPI(title="Garmin Health Summary API")


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


# FastAPI endpoints
@app.post("/health-summary", response_model=DailySummary)
async def get_health_summary(
    request: HealthSummaryRequest,
    garmin_email: str = Header(..., description="Garmin email address"),
    garmin_password: str = Header(..., description="Garmin password"),
) -> DailySummary:
    """Get daily health summary for a specific date.

    Args:
        request: Health summary request with date and model
        garmin_email: Garmin account email from header
        garmin_password: Garmin account password from header

    Returns:
        Daily health summary with AI-generated insights
    """
    try:
        garmin = get_garmin_client(garmin_email, garmin_password)
        summary = get_daily_summary(garmin, request.date, request.model)
        logger.info(
            f"Health summary API request completed successfully for {request.date}"
        )
        return summary
    except ValueError as e:
        logger.error(f"Invalid date format provided: {request.date}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Failed to generate health summary for {request.date}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")


if __name__ == "__main__":
    import os
    from pprint import pprint

    garmin = get_garmin_client(
        email=os.environ["GARMIN_EMAIL"],
        password=os.environ["GARMIN_PASSWORD"],
    )
    summary = get_daily_summary(garmin, "2025-09-05", "deepseek-chat")
    pprint(summary)
