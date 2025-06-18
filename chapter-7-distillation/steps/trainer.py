import logging
from contextlib import contextmanager
from typing import Any, Tuple

import polars as pl
import torch
from constants import (
    FINETUNE_ATTENTION_MODULES,
    FINETUNE_LANGUAGE_LAYERS,
    FINETUNE_MLP_MODULES,
    FINETUNE_VISION_LAYERS,
    FULL_FINETUNING,
    LOAD_IN_4BIT,
    LOAD_IN_8BIT,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_RANDOM_STATE,
    LORA_RANK,
)
from transformers import AutoTokenizer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from utils.custom_exceptions import ModelConfigError, TrainingError
from utils.dataset_utils import prepare_training_datasets
from utils.logging_utils import log_training_info
from utils.models import MODEL_CONFIGS, ModelConfig
from utils.training_utils import create_trainer, save_and_push_model
from utils.visualization_utils import create_training_visualization
from zenml import get_step_context, step
from zenml.types import HTMLString

logger = logging.getLogger(__name__)


def get_model_config(model_size: str) -> ModelConfig:
    """Get the model configuration for the specified size."""
    if model_size not in MODEL_CONFIGS:
        raise ModelConfigError(
            f"Invalid model size: '{model_size}'. Must be one of {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_size]


def create_tokenizer(config: ModelConfig) -> AutoTokenizer:
    """Create and configure tokenizer based on model configuration."""
    if config.tokenizer_kwargs:
        # Special tokenizer handling for 1B model
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_id, cache_dir=config.output_dir, **config.tokenizer_kwargs
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Standard tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)

    return tokenizer


def setup_model_and_tokenizer(config: ModelConfig) -> Tuple[Any, Any]:
    """Set up model and tokenizer with LoRA adapters.

    Args:
        config: Model configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(
        f"Loading model {config.base_model_id} with max sequence length {config.max_seq_length}"
    )

    # Load model with the appropriate configuration
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.base_model_id,
        max_seq_length=config.max_seq_length,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        full_finetuning=FULL_FINETUNING,
    )

    # Set up the Gemma3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    # Configure LoRA adapters
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=FINETUNE_VISION_LAYERS,
        finetune_language_layers=FINETUNE_LANGUAGE_LAYERS,
        finetune_attention_modules=FINETUNE_ATTENTION_MODULES,
        finetune_mlp_modules=FINETUNE_MLP_MODULES,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        random_state=LORA_RANDOM_STATE,
    )

    return model, tokenizer


@contextmanager
def GPUMonitor(task_name: str = "Task"):
    """Context manager for monitoring GPU memory usage during a task.

    Args:
        task_name: Name of the task being monitored

    Yields:
        dict: Dictionary containing GPU info that can be updated
    """
    gpu_info = {"available": False, "start_memory": 0, "max_memory": 0, "name": "N/A"}

    if torch.cuda.is_available():
        gpu_info["available"] = True
        gpu_stats = torch.cuda.get_device_properties(0)
        gpu_info["name"] = gpu_stats.name
        gpu_info["max_memory"] = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        gpu_info["start_memory"] = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        logger.info(
            f"GPU = {gpu_info['name']}. Max memory = {gpu_info['max_memory']} GB."
        )
        logger.info(
            f"{gpu_info['start_memory']} GB of memory reserved before {task_name}."
        )

    try:
        yield gpu_info
    finally:
        if gpu_info["available"]:
            used_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            used_memory_for_task = round(used_memory - gpu_info["start_memory"], 3)
            logger.info(f"Peak reserved memory = {used_memory} GB")
            logger.info(
                f"Peak reserved memory for {task_name} = {used_memory_for_task} GB"
            )


@step(enable_cache=False)
def finetune_model(
    train_dataset: pl.DataFrame,
    val_dataset: pl.DataFrame,
    model_size: str = "4b",
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
) -> HTMLString:
    """Finetune the selected model.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_size: Model size to use ('1b', '4b', or '12b')
        lora_rank: Rank for LoRA adapters
        lora_alpha: Alpha for LoRA adapters

    Returns:
        HTML visualization of training completion
    """
    # Get model configuration
    config = get_model_config(model_size)

    # Set up model and tokenizer with custom LoRA parameters if needed
    if lora_rank != 8 or lora_alpha != 8:
        # If custom LoRA parameters, we need to do it manually
        logger.info(f"Loading model {config.base_model_id} with custom LoRA parameters")

        # Create tokenizer
        tokenizer = create_tokenizer(config)

        # Load model
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.base_model_id,
            max_seq_length=config.max_seq_length,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            full_finetuning=FULL_FINETUNING,
        )

        # Set up the Gemma3 chat template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

        # Configure LoRA adapters with custom parameters
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=FINETUNE_VISION_LAYERS,
            finetune_language_layers=FINETUNE_LANGUAGE_LAYERS,
            finetune_attention_modules=FINETUNE_ATTENTION_MODULES,
            finetune_mlp_modules=FINETUNE_MLP_MODULES,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            random_state=LORA_RANDOM_STATE,
        )
    else:
        # Use default setup
        model, tokenizer = setup_model_and_tokenizer(config)

    # Prepare datasets
    train_data, val_data = prepare_training_datasets(
        train_dataset, val_dataset, tokenizer
    )

    # Get training config from step context if available
    training_config = None
    try:
        context = get_step_context()
        logger.info(f"Step context available: {context}")
        logger.info(f"Step name: {context.step_run.name}")

        # Try to access extra configuration from the step run configuration
        if hasattr(context.step_run, "config") and hasattr(
            context.step_run.config, "extra"
        ):
            extra_config = context.step_run.config.extra
            logger.info(f"Extra config found: {extra_config}")

            if isinstance(extra_config, dict) and "training_config" in extra_config:
                training_config = extra_config["training_config"]
                logger.info(
                    f"Using custom training configuration from YAML: {training_config}"
                )
            else:
                logger.info("No training_config found in extra")
        else:
            # Try alternate method - check step configuration
            logger.info("Checking for step configuration in context...")
            logger.info(f"Context attributes: {dir(context)}")
            if hasattr(context, "step_run"):
                logger.info(f"Step run attributes: {dir(context.step_run)}")
                if hasattr(context.step_run, "config"):
                    logger.info(f"Step run config: {context.step_run.config}")
                    logger.info(
                        f"Step run config type: {type(context.step_run.config)}"
                    )
                    logger.info(
                        f"Step run config attributes: {dir(context.step_run.config)}"
                    )
    except Exception as e:
        # If we can't get context (e.g., running outside ZenML), use defaults
        logger.info(f"Could not get step context: {e}")
        import traceback

        logger.info(f"Traceback: {traceback.format_exc()}")
        pass

    # Create trainer
    trainer = create_trainer(
        model, tokenizer, train_data, val_data, config, training_config
    )

    # Log training information
    log_training_info(trainer, train_data, training_config)

    # Train with GPU monitoring
    with GPUMonitor("training") as gpu_info:
        logger.info("Starting training")
        try:
            trainer_stats = trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            # Check if it's a GPU memory error
            gpu_memory_error = "CUDA out of memory" in str(e) or "OOM" in str(e)
            raise TrainingError(
                f"Training failed: {str(e)}", gpu_memory_error=gpu_memory_error
            ) from e

        if gpu_info["available"]:
            logger.info(
                f"Training took {trainer_stats.metrics['train_runtime']} seconds"
            )

    # Save and push model
    save_and_push_model(model, tokenizer, config)

    # Get final loss from trainer stats if available
    final_loss = None
    if (
        trainer_stats
        and hasattr(trainer_stats, "metrics")
        and "train_loss" in trainer_stats.metrics
    ):
        final_loss = trainer_stats.metrics["train_loss"]

    # Create visualization
    viz = create_training_visualization(model_size, final_loss)

    return viz
