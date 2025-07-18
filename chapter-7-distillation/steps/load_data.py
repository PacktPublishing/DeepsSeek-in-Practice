import json
import logging
import random
import sys
from typing import Annotated, Optional, Tuple

import polars as pl
from zenml import step
from zenml.types import HTMLString

sys.path.append("..")
from constants import TEST_DATA_PATH
from utils.custom_exceptions import DatasetError
from utils.dataset_utils import download_and_save_dataset
from utils.unsloth_utils import load_datasets
from utils.visualization_utils import create_data_loading_visualization

from steps.trainer import create_tokenizer, get_model_config

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def load_data(
    model_size: str = "12b",
    dataset_id: str = "zenml/cuad-deepseek",
    output_dir: str = "data/processed_cuad/hf_dataset",
    max_samples: Optional[int] = None,
    tokenize: bool = False,
    filter_none_labels: bool = True,
) -> Tuple[
    Annotated[pl.DataFrame, "train_dataset"],
    Annotated[pl.DataFrame, "val_dataset"],
    Annotated[HTMLString, "data_loading_viz"],
]:
    """Load the dataset from the Hugging Face Hub.

    The first time you run this step it will take a while to serialize and
    upload the dataset if you are using a cloud artifact store, but subsequent
    runs will be much faster thanks to ZenML's caching.

    Args:
        model_size: Model size to use ('1b', '4b', or '12b')
        dataset_id: HF Hub dataset ID
        output_dir: Directory to save the dataset
        max_samples: Maximum number of samples to use
        tokenize: Whether to tokenize the dataset
        filter_none_labels: Whether to filter out examples with 'NONE' classification
    """
    # Get model configuration
    config = get_model_config(model_size)

    # Create tokenizer
    tokenizer = create_tokenizer(config)

    # Download and prepare the dataset
    download_and_save_dataset(dataset_id=dataset_id, output_dir=output_dir)
    train_dataset, val_dataset = load_datasets(
        max_samples, tokenizer, tokenize, filter_none_labels
    )

    # Create visualization
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    viz = create_data_loading_visualization(train_size, val_size, filter_none_labels)

    return train_dataset.to_polars(), val_dataset.to_polars(), viz


@step(enable_cache=True)
def load_test_data(
    num_samples: Optional[int] = 50,
    exclude_none: bool = True,
    randomize: bool = False,
    random_seed: Optional[int] = None,
) -> Annotated[pl.DataFrame, "test_dataset"]:
    """Load test data for evaluation.

    Args:
        num_samples: Number of test samples to load
        exclude_none: Whether to exclude examples labeled as "NONE"
        randomize: Whether to randomize the dataset before sampling
        random_seed: Random seed for reproducible sampling (None for random)

    Returns:
        DataFrame containing test data
    """
    # Load all test data first
    logger.info(f"Loading test data from {TEST_DATA_PATH}")
    all_test_items = []
    try:
        with open(TEST_DATA_PATH, "r") as f:
            for i, line in enumerate(f):
                try:
                    all_test_items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetError(
                        f"Invalid JSON on line {i + 1} in test data: {str(e)}",
                        file_path=str(TEST_DATA_PATH),
                        expected_format="JSONL",
                        actual_format="Invalid JSON",
                    ) from e
    except FileNotFoundError as e:
        raise DatasetError(
            f"Test data file not found: {TEST_DATA_PATH}", file_path=str(TEST_DATA_PATH)
        ) from e
    except Exception as e:
        if not isinstance(e, DatasetError):
            raise DatasetError(
                f"Error loading test data: {str(e)}", file_path=str(TEST_DATA_PATH)
            ) from e
        raise

    logger.info(f"Loaded {len(all_test_items)} total test samples from file")

    # Set random seed if provided
    if randomize and random_seed is not None:
        random.seed(random_seed)
        logger.info(f"Set random seed to {random_seed}")

    # Randomize the dataset if requested
    if randomize:
        random.shuffle(all_test_items)
        logger.info("Randomized test data order")

    # Filter out NONE examples if requested
    if exclude_none:
        non_none_items = [item for item in all_test_items if item["label"] != "NONE"]
        none_count = len(all_test_items) - len(non_none_items)
        logger.info(
            f"Found {len(non_none_items)} non-NONE examples (excluded {none_count} NONE examples)"
        )
        items_to_sample = non_none_items
    else:
        items_to_sample = all_test_items

    # Sample the requested number of items
    if num_samples and num_samples < len(items_to_sample):
        evaluation_items = items_to_sample[:num_samples]
        logger.info(f"Selected {num_samples} samples for evaluation")
    else:
        evaluation_items = items_to_sample
        if num_samples and num_samples > len(items_to_sample):
            logger.warning(
                f"Requested {num_samples} samples but only {len(items_to_sample)} available "
                f"{'after filtering NONE' if exclude_none else ''}"
            )

    logger.info(f"Final evaluation dataset contains {len(evaluation_items)} samples")

    return pl.DataFrame(evaluation_items)
