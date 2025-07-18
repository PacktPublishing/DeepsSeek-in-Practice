import logging
import os
from typing import Any, Dict, Optional, TypedDict

from .models import MODEL_CONFIGS

logger = logging.getLogger(__name__)


def handle_exception(e: Exception) -> None:
    """Handle custom exceptions with helpful error messages and suggestions.

    Args:
        e: The exception to handle
    """
    if isinstance(e, (ModelConfigError, DatasetError, ModelLoadError, TrainingError)):
        logger.error(f"\n❌ {type(e).__name__}: {str(e)}")

        # Log additional context if available
        if hasattr(e, "file_path") and e.file_path:
            logger.error(f"   File: {e.file_path}")
        if hasattr(e, "model_path") and e.model_path:
            logger.error(f"   Model path: {e.model_path}")
        if hasattr(e, "model_size") and e.model_size:
            logger.error(f"   Model size: {e.model_size}")
        if hasattr(e, "epoch") and e.epoch is not None:
            logger.error(f"   Epoch: {e.epoch}")
        if hasattr(e, "batch") and e.batch is not None:
            logger.error(f"   Batch: {e.batch}")

        # Show suggestions if available
        if hasattr(e, "suggestions") and e.suggestions:
            logger.error("\n💡 Suggestions:")
            for suggestion in e.suggestions:
                logger.error(f"   • {suggestion}")
    else:
        # For other exceptions, just log normally
        logger.error(f"Error: {str(e)}")

    # Always show the full traceback in debug mode
    if logger.isEnabledFor(logging.DEBUG):
        logger.exception("Full traceback:")


class ModelConfigError(Exception):
    """Raised when there's an issue with model configuration."""

    def __init__(
        self,
        message: str,
        model_size: Optional[str] = None,
        config_key: Optional[str] = None,
    ):
        super().__init__(message)
        self.model_size = model_size
        self.config_key = config_key
        self.suggestions = []

        if model_size and model_size not in MODEL_CONFIGS:
            available = ", ".join(MODEL_CONFIGS.keys())
            self.suggestions.append(f"Available model sizes: {available}")


class DatasetError(Exception):
    """Raised when there's an issue with dataset loading or processing."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
    ):
        super().__init__(message)
        self.file_path = file_path
        self.expected_format = expected_format
        self.actual_format = actual_format
        self.suggestions = []

        if file_path and not os.path.exists(file_path):
            self.suggestions.append(
                f"File not found. Run 'python download_hf_rationales_dataset.py' to download the dataset."
            )


class ModelLoadError(Exception):
    """Raised when there's an issue loading a model."""

    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        model_size: Optional[str] = None,
        is_training_required: bool = False,
    ):
        super().__init__(message)
        self.model_path = model_path
        self.model_size = model_size
        self.is_training_required = is_training_required
        self.suggestions = []

        if is_training_required:
            self.suggestions.append(
                "Run without --eval_only flag to train the model first"
            )
        if model_path and not os.path.exists(model_path):
            self.suggestions.append(
                f"Model not found at {model_path}. Check if training completed successfully."
            )


class TrainingError(Exception):
    """Raised when there's an issue during training."""

    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        gpu_memory_error: bool = False,
    ):
        super().__init__(message)
        self.epoch = epoch
        self.batch = batch
        self.gpu_memory_error = gpu_memory_error
        self.suggestions = []

        if gpu_memory_error:
            self.suggestions.append("Try reducing batch_size or max_seq_length")
            self.suggestions.append(
                "Consider using gradient checkpointing or a smaller model"
            )


class EvaluationError(TypedDict):
    """Structure for evaluation error information."""

    item: Dict[str, Any]
    true_label: str
    pred_label: str
    response: str
