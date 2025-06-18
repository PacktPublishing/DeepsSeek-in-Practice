import logging
from pathlib import Path
from typing import Any, Tuple

import polars as pl
from constants import DEFAULT_DATASET_ID, DEFAULT_OUTPUT_DIR
from datasets import load_dataset
from datasets.arrow_dataset import Dataset

from .custom_exceptions import DatasetError
from .io_utils import write_jsonl_file

logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format file size in a human-readable format"""
    for unit in ["bytes", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def download_and_save_dataset(
    dataset_id: str = DEFAULT_DATASET_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    create_full_file: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Download a dataset from Hugging Face Hub and save each split as a JSONL file.

    Args:
        dataset_id: The Hugging Face Hub dataset ID
        output_dir: Directory to save the JSONL files
        create_full_file: Whether to create a combined file with all splits
        overwrite: If True, overwrite existing files and reload the dataset
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing_files = list(output_path.glob("*_data_with_rationales.jsonl"))
    full_file = output_path / "full_data_with_rationales.jsonl"
    files_exist = bool(existing_files) or full_file.exists()

    if files_exist and not overwrite:
        logger.info(
            f"Dataset files already exist in {output_dir}. "
            f"Set overwrite=True to reload and overwrite."
        )
        return

    try:
        dataset = load_dataset(dataset_id)
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise DatasetError(f"Failed to load dataset '{dataset_id}': {str(e)}") from e

    all_examples = []
    total_examples = 0

    for split_name, split_dataset in dataset.items():
        output_file = output_path / f"{split_name}_data_with_rationales.jsonl"
        examples = [example for example in split_dataset]

        write_jsonl_file(examples, output_file)

        if create_full_file:
            all_examples.extend(examples)
            total_examples += len(examples)

        file_size = output_file.stat().st_size
        readable_size = format_file_size(file_size)
        logger.info(
            f"Saved {len(examples)} examples to {output_file} ({readable_size})"
        )

    if create_full_file and all_examples:
        full_output_file = output_path / "full_data_with_rationales.jsonl"
        write_jsonl_file(all_examples, full_output_file)

        file_size = full_output_file.stat().st_size
        readable_size = format_file_size(file_size)
        logger.info(f"Combined dataset saved to {full_output_file} ({readable_size})")


def prepare_training_datasets(
    train_df: pl.DataFrame, val_df: pl.DataFrame, tokenizer: Any
) -> Tuple[Dataset, Dataset]:
    """Prepare training datasets and handle tokenization issues.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_data = Dataset.from_polars(train_df)
    val_data = Dataset.from_polars(val_df)

    # Check if data needs chat template reapplied
    sample_idx = min(1, len(train_data) - 1)
    if isinstance(train_data[sample_idx]["text"], int):
        logger.warning(
            "Found integer values in text field. Reapplying chat template..."
        )

        # Reapply chat template correctly
        def fix_chat_template(examples):
            if "conversations" in examples:
                texts = tokenizer.apply_chat_template(
                    examples["conversations"], tokenize=False
                )
                return {"text": texts}
            return examples

        train_data = train_data.map(fix_chat_template, batched=True)
        val_data = val_data.map(fix_chat_template, batched=True)

    return train_data, val_data
