import json
import logging
import os
import sys
from typing import Tuple

from datasets import Dataset

sys.path.append("..")
from constants import DATA_DIR, LEGAL_LABEL_SCHEMA, TRAIN_DATA_PATH, VAL_DATA_PATH

# Default logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_dataset_from_file(
    jsonl_path,
    max_samples=None,
    tokenizer=None,
    tokenize=False,
    filter_none_labels=False,
) -> Dataset:
    """Process JSONL data with enhanced instructional format using Gemma3 template.

    Args:
        jsonl_path: Path to the JSONL file
        max_samples: Maximum number of samples to process (None for all)
        tokenizer: The tokenizer to use for applying chat templates
        tokenize: Whether to tokenize when applying chat template (default: False)
        filter_none_labels: Whether to filter out examples with 'NONE' classification

    Returns:
        A Dataset object with the processed data
    """
    data = []
    none_count = 0
    total_count = 0

    logger.info(f"Loading data from {jsonl_path}")
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            total_count += 1

            item = json.loads(line)

            # Skip NONE examples if filter_none_labels is True
            if filter_none_labels and item.get("label", "").strip().upper() == "NONE":
                none_count += 1
                continue

            if max_samples and len(data) >= max_samples:
                break

            # Enhanced input with detailed instructions and schema
            # Update valid_labels based on filter_none_labels
            valid_labels = (
                [l for l in LEGAL_LABEL_SCHEMA if l != "NONE"]
                if filter_none_labels
                else LEGAL_LABEL_SCHEMA
            )

            input_json = json.dumps(
                {
                    "task": "classify_legal_clause",
                    "instructions": 'Analyze the legal clause and provide a detailed rationale for why it belongs to a specific category, then classify it according to the provided schema. IMPORTANT: First explain your reasoning thoroughly, then provide the label. Your output must be valid JSON. Example format: {"rationale": "This clause describes... because...", "label": "Termination for Convenience"}',
                    "schema": {
                        "rationale": "Detailed explanation of why the clause belongs to this category",
                        "label": "The classification category from the list of valid labels",
                    },
                    "valid_labels": valid_labels,
                    "inputs": {
                        "clause": item["clause"].strip(),
                        "clause_with_context": item["clause_with_context"].strip(),
                        "contract_type": item.get("contract_type", "").strip()
                        if item.get("contract_type")
                        else "",
                    },
                },
                ensure_ascii=False,
            )

            # Reordered output to prioritize reasoning before classification
            output_json = json.dumps(
                {
                    "rationale": item["rationale"].strip(),
                    "label": item["label"].strip() if item["label"] else "NONE",
                },
                ensure_ascii=False,
            )

            # Store conversations in format for Gemma3 chat template
            data.append(
                {
                    "conversations": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": input_json}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": output_json}],
                        },
                    ]
                }
            )

    if filter_none_labels and none_count > 0:
        logger.info(
            f"Prepared {len(data)} samples from {jsonl_path} "
            f"(filtered {none_count} NONE examples from {total_count} total)"
        )
    else:
        logger.info(f"Prepared {len(data)} samples from {jsonl_path}")

    # Convert to Dataset
    dataset = Dataset.from_list(data)

    # Apply the Gemma3 chat template
    if tokenizer:
        logger.info("Applying Gemma3 chat template to dataset")

        def apply_chat_template(examples):
            texts = tokenizer.apply_chat_template(
                examples["conversations"], tokenize=tokenize
            )
            return {"text": texts}

        dataset = dataset.map(apply_chat_template, batched=True)

    return dataset


def load_datasets(
    max_samples=None,
    tokenizer=None,
    tokenize=False,
    filter_none_labels=False,
) -> Tuple[Dataset, Dataset]:
    """Load train and validation datasets from pre-split files.

    Args:
        max_samples: Maximum number of samples to load from each dataset (None for all)
        tokenizer: The tokenizer to use for applying chat templates
        tokenize: Whether to tokenize when applying chat template (default: False)
        filter_none_labels: Whether to filter out examples with 'NONE' classification

    Returns:
        train_dataset, val_dataset: The prepared datasets
    """
    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory not found: {DATA_DIR}")
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Check if the data files exist
    if not os.path.exists(TRAIN_DATA_PATH):
        logger.error(f"Training data file not found: {TRAIN_DATA_PATH}")
        raise FileNotFoundError(f"Training data file not found: {TRAIN_DATA_PATH}")

    if not os.path.exists(VAL_DATA_PATH):
        logger.error(f"Validation data file not found: {VAL_DATA_PATH}")
        raise FileNotFoundError(f"Validation data file not found: {VAL_DATA_PATH}")

    # Load the datasets
    train_dataset = prepare_dataset_from_file(
        TRAIN_DATA_PATH,
        max_samples=max_samples,
        tokenizer=tokenizer,
        tokenize=tokenize,
        filter_none_labels=filter_none_labels,
    )

    val_dataset = prepare_dataset_from_file(
        VAL_DATA_PATH,
        max_samples=max_samples,
        tokenizer=tokenizer,
        tokenize=tokenize,
        filter_none_labels=filter_none_labels,
    )

    logger.info(
        f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples"
    )

    return train_dataset, val_dataset


def test_input_json() -> str:
    """Return a test input JSON for inference."""
    return json.dumps(
        {
            "task": "classify_legal_clause",
            "instructions": 'Analyze the legal clause and provide a detailed rationale for why it belongs to a specific category, then classify it according to the provided schema. IMPORTANT: First explain your reasoning thoroughly, then provide the label. Your output must be valid JSON. Example format: {"rationale": "This clause describes... because...", "label": "Termination for Convenience"}',
            "schema": {
                "rationale": "Detailed explanation of why the clause belongs to this category",
                "label": "The classification category from the list of valid labels",
            },
            "valid_labels": LEGAL_LABEL_SCHEMA,
            "inputs": {
                "clause": "Company may terminate this Agreement at any time.",
                "clause_with_context": "Section 8. Termination. Company may terminate this Agreement at any time. Upon termination, all rights granted herein shall cease.",
                "contract_type": "SERVICE AGREEMENT",
            },
        },
        ensure_ascii=False,
    )
