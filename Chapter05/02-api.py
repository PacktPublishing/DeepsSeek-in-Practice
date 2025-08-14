import datetime
import json
import os
from enum import Enum
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from garminconnect import Garmin
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(".envrc", override=True)


# AI prompting setup
TRAINING_DAY_EXAMPLE = {
    "day_type": "training",
    "title": "Strong training day with elevated activity across all metrics.",
    "emoji": "💪",
    "observation": "Exercise minutes doubled your baseline with 95 minutes of activity, supported by 18,500 steps. Despite the high training load, body battery started at a solid 85, indicating good recovery from yesterday.",
    "recommendation": "Consider an active recovery or rest day tomorrow to allow adaptation from today's effort. Prioritize sleep tonight to maintain your body battery levels and support muscle recovery.",
}

HIGH_STRESS_DAY_EXAMPLE = {
    "day_type": "high_stress",
    "emoji": "😫",
    "title": "Elevated stress and poor recovery despite minimal physical activity.",
    "observation": "Stress levels jumped 46% above baseline while sleep dropped to just 6.1 hours, resulting in a low body battery start of 45. Exercise and movement were minimal, suggesting stress is from non-physical sources.",
    "recommendation": "Focus on stress management techniques and aim for 8+ hours of sleep tonight. Consider light exercise like walking or yoga tomorrow, as gentle movement can help regulate stress levels.",
}

EXAMPLES_STR = "\n\n\n".join(
    [json.dumps(ex) for ex in [TRAINING_DAY_EXAMPLE, HIGH_STRESS_DAY_EXAMPLE]]
)


# Pydantic models
class DayType(str, Enum):
    TRAINING = "training"
    ACTIVE_RECOVERY = "active_recovery"
    REST = "rest"
    HIGH_STRESS = "high_stress"
    BALANCED = "balanced"


class HealthSummaryRequest(BaseModel):
    date: str = Field(
        default_factory=lambda: datetime.date.today().isoformat(),
        description="Date in YYYY-MM-DD format, defaults to today",
        example=datetime.date.today().isoformat(),
    )
    model: Literal["deepseek-chat", "deepseek-reasoner"] = Field(
        default="deepseek-chat", description="AI model to use"
    )


class DailySummary(BaseModel):
    day_type: DayType = Field(
        ...,
        description="Classification of the day based on activity and recovery metrics",
    )
    title: str = Field(..., description="One sentence summary of the day")
    emoji: str = Field(..., description="Emoji to represent the day type")
    observation: str = Field(
        ...,
        description="Two sentence observation about key metrics and patterns",
    )
    recommendation: str = Field(
        ...,
        description="Two sentence actionable recommendation for tomorrow",
    )


SYSTEM_PROMPT = f"""
Instructions:
* You will be given a summary of the user's health and fitness data for today, in comparison to the past 7 days.
* Your goal is to generate a summary that will be shown in the user's smart watch. 
* Keep things short, but also interesting to the user. 
* Your summary should include a type of day, a title, some observations and recommendations for the user. 
* Your summary should be in JSON format. Only output the JSON, no other text.

---JSON SCHEMA---
{DailySummary.model_json_schema()}
---END JSON SCHEMA---

---EXAMPLE JSON OUTPUTS---
{EXAMPLES_STR}
---END EXAMPLE JSON OUTPUTS---
"""

app = FastAPI(title="Garmin Health Summary API")


@lru_cache(maxsize=1)
def get_garmin_client(email: str, password: str) -> Garmin:
    """Initialize and cache Garmin connection.

    Args:
        email: Garmin account email
        password: Garmin account password

    Returns:
        Authenticated Garmin client instance
    """

    try:
        garmin = Garmin(email=email, password=password, is_cn=False)
        garmin.login()
        logger.info("Successfully authenticated with Garmin")
        return garmin
    except Exception as e:
        logger.error(f"Failed to authenticate with Garmin: {e}")
        raise HTTPException(status_code=401, detail=f"Could not login to Garmin: {e}")


# Health data functions
def get_daily_health_summary(
    api: Any, start: datetime.date, end: datetime.date
) -> list[dict[str, Any]]:
    """Get daily health summary for a date range.

    Args:
        api: Garmin API client instance
        start: Start date for data collection
        end: End date for data collection

    Returns:
        List of daily health metrics dictionaries with fields:
        resting_heart_rate, exercise_minutes, stress_level, sleep_hours, steps, body_battery_final
    """

    def dstr(d: datetime.date) -> str:
        """Convert date to ISO string format.

        Args:
            d: Date to convert

        Returns:
            ISO formatted date string
        """
        return d.isoformat()

    def daterange(a: datetime.date, b: datetime.date):
        """Generate date range from start to end inclusive.

        Args:
            a: Start date
            b: End date

        Yields:
            Each date in the range
        """
        for i in range((b - a).days + 1):
            yield a + datetime.timedelta(days=i)

    out: list[dict[str, Any]] = []

    for day in daterange(start, end):
        s = dstr(day)
        day_of_week = day.strftime("%A")
        summary = api.get_user_summary(s) or {}
        rhr = summary.get("restingHeartRate")
        steps = summary.get("totalSteps")
        stress_level = summary.get("averageStressLevel")
        body_battery_final = summary.get("bodyBatteryMostRecentValue") or summary.get(
            "mostRecentBodyBattery"
        )
        exercise_minutes = (summary.get("moderateIntensityMinutes") or 0) + (
            summary.get("vigorousIntensityMinutes") or 0
        )
        sleep_seconds = summary.get("sleepingSeconds")
        sleep_hours = round(sleep_seconds / 3600, 2)
        body_battery_start = summary.get("bodyBatteryAtWakeTime")
        total_distance_meters = summary.get("totalDistanceMeters")

        out.append(
            {
                "date": s,
                "day_of_week": day_of_week,
                "resting_heart_rate": rhr,
                "exercise_minutes": exercise_minutes,
                "stress_level": stress_level,
                "sleep_hours": sleep_hours,
                "steps": steps,
                "total_distance_meters": total_distance_meters,
                "body_battery_start_day": body_battery_start,
                "body_battery_end_day": body_battery_final,
            }
        )

    return out


def detect_trend(values, pct_threshold=5):
    """Detect trend direction in values comparing recent vs earlier periods.

    Args:
        values: List of numeric values
        pct_threshold: Percentage threshold for trend detection (default: 5)

    Returns:
        Trend direction: "up", "down", or "flat"
    """
    recent, earlier = np.mean(values[-3:]), np.mean(values[:3])
    return (
        "up"
        if recent > earlier * (1 + pct_threshold / 100)
        else ("down" if recent < earlier * (1 - pct_threshold / 100) else "flat")
    )


def build_llm_context_md(
    summary_for_today: list[dict], summary_for_past_7_days: list[dict]
) -> str:
    """Creates a Markdown-formatted context string for LLM analysis.

    Args:
        summary_for_today: Single day health summary data
        summary_for_past_7_days: Seven days of historical health data

    Returns:
        Markdown formatted string with metrics comparison and trends
    """
    assert len(summary_for_today) == 1, "Expected 1 day of summary"
    today = summary_for_today[0]

    metrics = [key for key in today.keys() if key not in ["date", "day_of_week"]]
    lines = [
        f"# Daily Metrics Summary for {today['date']} ({today['day_of_week']})",
        "_Note: All comparisons use the **previous 7 days only**, excluding today._",
        "",
    ]

    better_is_lower = [
        "resting_heart_rate",
        "stress_level",
    ]

    for metric in metrics:
        today_val = today[metric]
        past_vals = [
            day[metric] for day in summary_for_past_7_days if day[metric] is not None
        ]
        avg_7d = sum(past_vals) / len(past_vals) if past_vals else 0
        delta_pct = ((today_val - avg_7d) / avg_7d * 100) if avg_7d else 0
        trend_dir = detect_trend(past_vals)
        arrow = "↑" if trend_dir == "up" else ("↓" if trend_dir == "down" else "→")

        lines.append(
            f"## {metric.replace('_', ' ').title()}\n"
            f"- Today's value ({today['date']}): {today_val}\n"
            f"- 7-day baseline average (excluding today): {avg_7d:.2f}\n"
            f"- Percent change vs. baseline: {delta_pct:+.1f}%\n"
            f"- Trend over previous 7 days: {trend_dir} {arrow}\n"
            f"- Better is lower: {metric in better_is_lower}\n"
        )

    return "\n".join(lines)


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
        model=model, messages=messages, response_format=response_format, temperature=0.0
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
    date_for_summary = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    summary_in_date = get_daily_health_summary(
        garmin, date_for_summary, date_for_summary
    )

    past_period_start = date_for_summary - datetime.timedelta(days=7)
    past_period_end = date_for_summary - datetime.timedelta(days=1)
    summary_in_past_period = get_daily_health_summary(
        garmin, past_period_start, past_period_end
    )

    prompt = build_llm_context_md(summary_in_date, summary_in_past_period)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response, reasoning = llm(messages, model, {"type": "json_object"})
    health_summary = DailySummary.model_validate(response)

    logger.info(f"Daily summary generated successfully for {date}")
    return health_summary


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


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Status dictionary indicating API health
    """
    return {"status": "healthy"}


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
