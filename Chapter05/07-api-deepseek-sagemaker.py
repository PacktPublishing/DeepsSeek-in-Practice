import pickle
import sagemaker
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from garminconnect import Garmin
from loguru import logger
from pydantic import BaseModel

from utils import (
    SYSTEM_PROMPT,
    DailySummary,
    HealthSummaryRequestAWS,
    get_daily_summary_prompt,
    get_garmin_client,
)

load_dotenv(".envrc", override=True)

# Replace with your own endpoint name
ENDPOINT_NAME = "DeepSeek-R1-Distill-Qwen-14B-2025-08-24-08-52-31-391-endpoint"


def get_aws_llm(endpoint_name: str) -> sagemaker.Predictor:
    """Initialize AWS SageMaker predictor for model endpoint.

    Args:
        endpoint_name: Name of the SageMaker endpoint

    Returns:
        SageMaker predictor instance
    """
    sess = sagemaker.Session()

    return sagemaker.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sess,
        serializer=sagemaker.serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer(),
    )


def llm(
    messages: list[dict],
    endpoint_name: str,
    response_model: BaseModel,
) -> BaseModel:
    """Call DeepSeek AWS Sagemaker endpoint with messages.

    Args:
        messages: List of chat messages with role and content
        endpoint_name: Name of the SageMaker endpoint
        response_model: Pydantic model for response validation

    Returns:
        Validated response model instance
    """

    client = get_aws_llm(endpoint_name=endpoint_name)

    response = client.predict(
        {
            "messages": messages,
            "temperature": 0.01,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            },
            "extra_body": {"guided_decoding_backend": "xgrammar"},
        }
    )

    return response_model.model_validate_json(
        response["choices"][0]["message"]["reasoning_content"]
    )


def get_daily_summary(
    garmin: Garmin,
    date: str,
    endpoint_name: str,
) -> DailySummary:
    """Generate AI-powered daily health summary for a specific date.

    Args:
        garmin: Authenticated Garmin client instance
        date: Date string in YYYY-MM-DD format
        endpoint_name: AWS Sagemaker endpoint name

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
    response = llm(messages, endpoint_name, response_model=DailySummary)

    logger.info(f"Daily summary generated successfully for {date}")
    return response


app = FastAPI(title="Garmin Health Summary API")


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


# FastAPI endpoints
@app.post("/health-summary", response_model=DailySummary)
async def get_health_summary(
    request: HealthSummaryRequestAWS,
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
        summary = get_daily_summary(garmin, request.date, ENDPOINT_NAME)
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
    summary = get_daily_summary(
        garmin,
        "2025-08-24",
        endpoint_name=ENDPOINT_NAME,
    )
    pprint(summary)
