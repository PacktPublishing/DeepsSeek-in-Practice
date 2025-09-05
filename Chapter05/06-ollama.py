import os
import pickle
from shutil import which

from dotenv import load_dotenv
from garminconnect import Garmin
from ollama import chat
from loguru import logger

from utils import (
    SYSTEM_PROMPT,
    DailySummary,
    get_daily_summary_prompt,
    get_garmin_client,
)

load_dotenv(".envrc", override=True)


def check_ollama_installed() -> None:
    """Check if 'ollama' CLI is available in PATH."""
    if not which("ollama"):
        raise RuntimeError("ollama CLI is not installed")


def get_daily_summary(
    garmin: Garmin,
    date: str,
) -> DailySummary:
    """Generate AI-powered daily health summary for a specific date.

    Args:
        garmin: Authenticated Garmin client instance
        date: Date string in YYYY-MM-DD format

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

    response = chat(
        messages=messages,
        model="deepseek-r1:1.5b",
        format=DailySummary.model_json_schema(),
        options={
            "temperature": 0.0,
        },
    )

    return DailySummary.model_validate_json(response.message.content)


if __name__ == "__main__":
    check_ollama_installed()
    garmin = get_garmin_client(
        email=os.environ["GARMIN_EMAIL"],
        password=os.environ["GARMIN_PASSWORD"],
    )
    summary = get_daily_summary(garmin, "2025-08-20")
    print(summary)
