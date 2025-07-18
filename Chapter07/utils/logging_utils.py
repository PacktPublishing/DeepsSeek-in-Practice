import logging
import os
from typing import Any, Dict, Optional

from constants import BATCH_SIZE, GRAD_ACCUMULATION, TRAINING_CONFIG
from datasets.arrow_dataset import Dataset
from transformers import logging as transformers_logging

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", quiet: bool = False) -> None:
    """Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        quiet: If True, reduce verbosity even more
    """

    # Note: Unsloth compile-time environment variables are already set at import time

    # Configure Python logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if quiet:
        # In quiet mode, be more aggressive about reducing output
        transformers_logging.set_verbosity_error()  # Only errors
        os.environ["TQDM_MININTERVAL"] = (
            "10"  # Update progress bar at most every 10 seconds
        )

        # Set all loggers to WARNING or higher
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("unsloth").setLevel(logging.WARNING)
        logging.getLogger("tqdm").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)
    else:
        # Normal mode
        transformers_logging.set_verbosity_warning()  # Only show warnings and errors
        transformers_logging.disable_progress_bar()  # Disable default transformers progress bar

        # Set environment variable to control tqdm behavior
        os.environ["TQDM_MININTERVAL"] = (
            "5"  # Update progress bar at most every 5 seconds
        )

        # Reduce verbosity of specific libraries
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers.trainer").setLevel(
            logging.INFO
        )  # Keep trainer info
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("unsloth").setLevel(logging.INFO)
        logging.getLogger("tqdm").setLevel(logging.WARNING)  # Reduce tqdm verbosity
        logging.getLogger("accelerate").setLevel(
            logging.WARNING
        )  # Reduce accelerate verbosity


def log_training_info(
    trainer: Any, train_data: Dataset, training_config: Optional[Dict[str, Any]] = None
) -> None:
    """Log training information and sample data.

    Args:
        trainer: The configured trainer
        train_data: Training dataset
        training_config: Optional custom training configuration
    """
    # Use provided config or default
    train_cfg = training_config if training_config else TRAINING_CONFIG

    # Calculate total steps for user reference
    if len(train_data) > 0:
        batch_size = train_cfg.get("per_device_train_batch_size", BATCH_SIZE)
        grad_accum = train_cfg.get("gradient_accumulation_steps", GRAD_ACCUMULATION)
        num_epochs = train_cfg.get("num_train_epochs", 2)

        total_batch_size = batch_size * grad_accum
        total_steps = (len(train_data) // total_batch_size) * num_epochs

        logger.info(
            f"🔄 Training on {len(train_data)} examples over {num_epochs} epochs"
        )
        logger.info(
            f"🧮 Batch size per device = {batch_size}, Gradient accumulation = {grad_accum}, Total batch size = {total_batch_size}"
        )
        logger.info(
            f"📊 Total steps ≈ {total_steps}, Logging every {train_cfg.get('logging_steps', 20)} steps"
        )

        # Log key hyperparameters
        logger.info(
            f"📊 Learning rate: {train_cfg.get('learning_rate', 2e-4)}, Warmup: {train_cfg.get('warmup_steps', 5)} steps, Scheduler: {train_cfg.get('lr_scheduler_type', 'linear')}"
        )
