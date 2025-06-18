from dataclasses import dataclass
from typing import Any, Dict, Optional

from constants import (
    GEMMA_1B_MODEL_BASE,
    GEMMA_4B_MODEL_BASE,
    GEMMA_12B_MODEL_BASE,
    HF_MODEL_REPO_1B,
    HF_MODEL_REPO_4B,
    HF_MODEL_REPO_12B,
    MAX_SEQ_LENGTH_1B,
    MAX_SEQ_LENGTH_4B,
    MAX_SEQ_LENGTH_12B,
    MODEL_OUTPUT_DIR_1B,
    MODEL_OUTPUT_DIR_4B,
    MODEL_OUTPUT_DIR_12B,
)


@dataclass
class ModelConfig:
    """Configuration for a specific model variant."""

    base_model_id: str
    max_seq_length: int
    output_dir: str
    hf_repo: str
    # Special tokenizer kwargs for 1B model
    tokenizer_kwargs: Optional[Dict[str, Any]] = None


# Model configurations
MODEL_CONFIGS = {
    "1b": ModelConfig(
        base_model_id=GEMMA_1B_MODEL_BASE,
        max_seq_length=MAX_SEQ_LENGTH_1B,
        output_dir=MODEL_OUTPUT_DIR_1B,
        hf_repo=HF_MODEL_REPO_1B,
        tokenizer_kwargs={
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "additional_special_tokens": ["<bos>"],
        },
    ),
    "4b": ModelConfig(
        base_model_id=GEMMA_4B_MODEL_BASE,
        max_seq_length=MAX_SEQ_LENGTH_4B,
        output_dir=MODEL_OUTPUT_DIR_4B,
        hf_repo=HF_MODEL_REPO_4B,
        tokenizer_kwargs=None,  # 4b model uses defaults
    ),
    "12b": ModelConfig(
        base_model_id=GEMMA_12B_MODEL_BASE,
        max_seq_length=MAX_SEQ_LENGTH_12B,
        output_dir=MODEL_OUTPUT_DIR_12B,
        hf_repo=HF_MODEL_REPO_12B,
        tokenizer_kwargs=None,  # 12b model uses defaults
    ),
}
