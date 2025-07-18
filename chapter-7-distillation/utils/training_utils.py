import unsloth  # noqa
import logging
from typing import Any, Dict, Optional

from datasets.arrow_dataset import Dataset
from .models import ModelConfig
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from constants import MODEL_TURN_MARKER, TRAINING_CONFIG, USER_TURN_MARKER

logger = logging.getLogger(__name__)


def create_trainer(
    model: Any,
    tokenizer: Any,
    train_data: Dataset,
    val_data: Dataset,
    config: ModelConfig,
    training_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create and configure the SFTTrainer.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_data: Training dataset
        val_data: Validation dataset
        config: Model configuration
        training_config: Optional custom training configuration

    Returns:
        Configured SFTTrainer instance
    """
    # Use custom training config if provided, otherwise use default
    train_cfg = training_config if training_config else TRAINING_CONFIG

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=SFTConfig(
            dataset_text_field="text", output_dir=config.output_dir, **train_cfg
        ),
    )

    # Train only on the assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part=USER_TURN_MARKER,
        response_part=MODEL_TURN_MARKER,
    )

    return trainer


def save_and_push_model(model: Any, tokenizer: Any, config: ModelConfig) -> None:
    """Save model locally and push to Hugging Face Hub.

    Args:
        model: The trained model
        tokenizer: The tokenizer
        config: Model configuration
    """
    # Save the model
    logger.info(f"Saving model to {config.output_dir}/final")
    model.save_pretrained(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")

    # Push the model to the hub
    logger.info(f"Pushing model to {config.hf_repo}")
    model.push_to_hub(config.hf_repo)
