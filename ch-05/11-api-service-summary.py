# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi[standard]",
#     "uvicorn[standard]",
#     "requests",
#     "markdownify",
#     "loguru",
# ]
# ///

import json
from fastapi.responses import RedirectResponse
import uvicorn
import requests
from markdownify import markdownify as md
import os
from pydantic import BaseModel
from fastapi import FastAPI
from loguru import logger
from fastapi import HTTPException


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-r1-0528"
APP = FastAPI(title="Website Summarizer", version="0.1.0")


class ArticleSummary(BaseModel):
    title: str
    summary: str
    author: str
    date: str
    topics: list[str]

    model_config = {
        "json_schema_extra": {
            "description": "A summary of an article from a website.",
            "examples": [
                {
                    "title": "Example Article",
                    "summary": "This is a summary of the example article.",
                    "author": "John Doe",
                    "date": "2023-10-01",
                    "topics": ["example", "article", "summary"],
                }
            ],
        }
    }


def read_website(url: str) -> str:
    """Reads the content of a website and returns it as text.

    Args:
        url: The URL of the website to read.
    """
    try:
        html_content = requests.get(url).text
        markdown_content = md(html_content)

        return markdown_content
    except Exception as e:
        raise RuntimeError(f"Failed to read website {url}: {e}")


def summarize_website_from(url: str) -> ArticleSummary:
    logger.info(f"Summarizing website: {url}")
    markdown_content = read_website(url)
    logger.info(f"Got markdown for website: {url}")

    PROMPT = f"Summarize the followig article:\n\n{markdown_content}\n\n"

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT,
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "website_summary",
                    "strict": True,
                    "schema": ArticleSummary.model_json_schema(),
                },
            },
        },
    )
    logger.success(f"Got summary for website: {url}")
    data = response.json()
    summary_str = data["choices"][0]["message"]["content"]
    summary_dict = json.loads(summary_str)
    return ArticleSummary(**summary_dict)


@APP.get("/summarize", response_model=ArticleSummary)
def summarize_website(url: str) -> ArticleSummary:
    try:
        return summarize_website_from(url)
    except Exception as e:
        logger.error(f"Error summarizing website {url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@APP.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    logger.info("Starting API service...")
    port = 8000
    url = f"http://localhost:{port}/docs"
    logger.info(f"Visit {url} to access the API documentation.")
    uvicorn.run(APP, host="0.0.0.0", port=port, log_level="info")
