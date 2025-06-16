# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi[standard]",
#     "openai",
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
import os
from pydantic import BaseModel
from fastapi import FastAPI
from loguru import logger
from fastapi import HTTPException
from enum import Enum


from openai import OpenAI


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-r1-0528"
OPENAI_CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
APP = FastAPI(title="Note Classifier", version="0.1.0")


class Categories(str, Enum):
    personal = "personal"
    work = "work"
    idea = "idea"
    todo = "todo"
    other = "other"


class NoteCategory(BaseModel):
    category: Categories

    model_config = {
        "json_schema_extra": {
            "description": "Classifies a note into one of the predefined categories.",
            "examples": [{"category": "idea"}],
        }
    }


def classify(note: str) -> NoteCategory:
    PROMPT = f"Classify the following note into one of the categories:\n {note}"

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
                    "name": "note_category",
                    "strict": True,
                    "schema": NoteCategory.model_json_schema(),
                },
            },
        },
    )
    logger.success(f"Got summary for website: {url}")
    data = response.json()
    note_cat_str = data["choices"][0]["message"]["content"]
    note_cat_args = json.loads(note_cat_str)
    return NoteCategory(**note_cat_args)


class ClassificationRequest(BaseModel):
    note: str

    model_config = {
        "json_schema_extra": {
            "description": "Request body for classifying a note.",
            "examples": [{"note": "This is a note about my work project."}],
        }
    }


@APP.post("/classify", response_model=NoteCategory)
def classify_endpoint(request: ClassificationRequest) -> NoteCategory:
    try:
        return classify(request.note)
    except Exception as e:
        logger.error(f"Error classifying note: {e}")
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
