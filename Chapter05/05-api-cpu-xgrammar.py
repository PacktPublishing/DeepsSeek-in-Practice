import torch
import transformers
import xgrammar as xgr
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from garminconnect import Garmin
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils import (
    SYSTEM_PROMPT,
    DailySummary,
    HealthSummaryRequest,
    get_daily_summary_prompt,
    get_garmin_client,
)


def get_device(force_cpu: bool = False):
    if force_cpu is True:
        device = "cpu"
        logger.warning("Forcing CPU")
        return device

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    return device


MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=get_device(
        force_cpu=True
    ),  # you can set force_cpu=False to use GPU or MPS on mac
)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
MAX_NEW_TOKENS = 1024
transformers.set_seed(42)


def get_daily_summary(
    garmin: Garmin,
    date: str,
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

    prompt = get_daily_summary_prompt(garmin, date)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    texts = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = TOKENIZER(texts, return_tensors="pt").to(MODEL.device)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        TOKENIZER, vocab_size=CONFIG.vocab_size
    )
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_json_schema(DailySummary)
    xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    generated_ids = MODEL.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        logits_processor=[xgr_logits_processor],
        do_sample=True,
        temperature=0.01,
        top_p=0.95,
        top_k=50,
    )
    generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    model_response = TOKENIZER.decode(generated_ids, skip_special_tokens=True)
    return DailySummary.model_validate_json(model_response)


app = FastAPI(title="Garmin Health Summary API")


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
        summary = get_daily_summary(garmin, request.date)
        logger.info(
            f"Health summary API request completed successfully for {request.date}"  # noqa: E501
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
