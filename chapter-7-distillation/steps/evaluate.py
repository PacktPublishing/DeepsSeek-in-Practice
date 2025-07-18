import logging
from datetime import datetime
from typing import Annotated, Any, Dict, Tuple

import polars as pl
from constants import LEGAL_LABEL_SCHEMA, LOAD_IN_4BIT, LOAD_IN_8BIT, TEST_DATA_PATH
from peft import PeftModel
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from utils.custom_exceptions import ModelLoadError
from utils.evaluation_utils import (
    calculate_detailed_metrics,
    create_base_prompt_json,
    extract_prediction,
    log_error_breakdown,
)
from utils.visualization_utils import EvaluationResults, create_evaluation_visualization
from zenml import step
from zenml.types import HTMLString

from steps.trainer import get_model_config

logger = logging.getLogger(__name__)


def create_prompt(item: Dict) -> str:
    """Create a prompt from a test item."""
    input_json = create_base_prompt_json(item, LEGAL_LABEL_SCHEMA)

    # Format as Gemma3 messages
    messages = [{"role": "user", "content": [{"type": "text", "text": input_json}]}]
    return messages


def run_inference(model, tokenizer, messages, verbose=False) -> str:
    """
    Run inference with a model on a test item.

    Args:
        model: The model to use
        tokenizer: The tokenizer
        messages: The input messages
        verbose: Whether to print the generated text

    Returns:
        The model's response
    """
    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    streamer = TextStreamer(tokenizer, skip_prompt=True) if verbose else None

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,  # Lower temperature for more deterministic outputs
        top_p=0.95,
        top_k=64,
        streamer=streamer,
    )

    result = tokenizer.batch_decode(outputs)[0]

    # Extract just the generated portion
    response_start = len(text)
    response = result[response_start:].strip()

    return response


@step(enable_cache=False)
def evaluate_model(
    test_dataset: pl.DataFrame,
    model_size: str = "4b",
    verbose: bool = False,
    use_local_model: bool = True,
) -> Tuple[
    Annotated[Dict[str, Any], "evaluation_results"],
    Annotated[HTMLString, "evaluation_viz"],
]:
    """Evaluate a fine-tuned model on test data.

    Args:
        test_dataset: Test dataset to evaluate on
        model_size: Model size to use ('1b', '4b', or '12b')
        verbose: Whether to print detailed outputs
        use_local_model: If True, load from local path; if False, load from HF Hub

    Returns:
        Tuple of evaluation results and visualization
    """
    # Get model configuration
    config = get_model_config(model_size)

    # Determine model path
    if use_local_model:
        model_path = f"{config.output_dir}/final"
        logger.info(f"Loading LoRA adapters from local path: {model_path}")
    else:
        model_path = config.hf_repo
        logger.info(f"Loading LoRA adapters from HF Hub: {model_path}")

    logger.info("Model Configuration:")
    logger.info(f"  Base model: {config.base_model_id}")
    logger.info(f"  Model size: {model_size}")
    logger.info(f"  Max sequence length: {config.max_seq_length}")
    logger.info(f"  4-bit loading: {LOAD_IN_4BIT}")
    logger.info(f"  8-bit loading: {LOAD_IN_8BIT}")

    logger.info(f"Loading base model from {config.base_model_id}...")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_id,
        max_seq_length=config.max_seq_length,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )

    logger.info("✓ Base model loaded successfully")

    # Enable faster inference
    FastLanguageModel.for_inference(model)
    logger.info("✓ Enabled faster inference mode")

    # Set up chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    logger.info("✓ Chat template configured")

    # Load adapters
    logger.info(f"Loading LoRA adapters from {model_path}...")
    try:
        model = PeftModel.from_pretrained(model, model_path)
        logger.info(f"✓ LoRA adapters loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"✗ Failed to load LoRA adapters: {str(e)}")
        raise ModelLoadError(
            f"Cannot load model adapters from {model_path}",
            model_path=model_path,
            model_size=model_size,
            is_training_required=use_local_model,
        ) from e

    # Convert polars DataFrame to list of dicts for evaluation
    evaluation_items = test_dataset.to_dicts()

    logger.info(f"Starting evaluation on {len(evaluation_items)} test examples")

    # Evaluate model
    true_labels = []
    pred_labels = []
    correct_count = 0
    errors = []

    # For non-NONE metrics (even if we're already filtering)
    non_none_true_labels = []
    non_none_pred_labels = []
    non_none_correct_count = 0
    non_none_total = 0

    # Evaluate each item
    for i, item in enumerate(evaluation_items):
        # Log progress periodically
        if i % 10 == 0:
            logger.info(f"Processing example {i + 1}/{len(evaluation_items)}...")

        true_label = item["label"]
        true_labels.append(true_label)

        # Log what we're processing (debug level)
        logger.debug(f"\nProcessing item {i + 1}:")
        logger.debug(f"  True label: {true_label}")
        logger.debug(f"  Clause preview: {item['clause'][:100]}...")

        messages = create_prompt(item)

        # Run inference
        response = run_inference(model, tokenizer, messages, verbose)

        # Log the raw response (debug level)
        logger.debug(f"  Model response: {response[:200]}...")

        # Extract prediction
        pred_label = extract_prediction(response, log_failures=True)

        if pred_label:
            pred_labels.append(pred_label)
            logger.debug(f"  Predicted label: {pred_label}")

            if pred_label == true_label:
                correct_count += 1
                logger.debug(f"  Result: ✓ CORRECT")
            else:
                logger.debug(f"  Result: ✗ INCORRECT")
                errors.append(
                    {
                        "item": item,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "response": response[:200] + "..."
                        if len(response) > 200
                        else response,
                    }
                )
                # Log failed prediction details
                logger.warning(
                    f"Misprediction - Expected: {true_label}, Got: {pred_label}, "
                    f"Clause: {item['clause'][:100]}..."
                )

            # Track non-NONE metrics separately
            if true_label != "NONE":
                non_none_true_labels.append(true_label)
                non_none_pred_labels.append(pred_label)
                non_none_total += 1
                if pred_label == true_label:
                    non_none_correct_count += 1
        else:
            # If we couldn't parse a prediction, count it as an error
            pred_labels.append("PARSING_ERROR")
            logger.debug(f"  Result: ✗ PARSING ERROR")
            logger.error(
                f"Failed to parse prediction for item {i + 1}. "
                f"True label: {true_label}, Response: {response[:200]}..."
            )
            errors.append(
                {
                    "item": item,
                    "true_label": true_label,
                    "pred_label": "PARSING_ERROR",
                    "response": response[:200] + "..."
                    if len(response) > 200
                    else response,
                }
            )

            # Track for non-NONE metrics
            if true_label != "NONE":
                non_none_true_labels.append(true_label)
                non_none_pred_labels.append("PARSING_ERROR")
                non_none_total += 1

        # Log running accuracy every 50 items
        if (i + 1) % 50 == 0:
            current_accuracy = correct_count / (i + 1)
            logger.info(
                f"Progress: {i + 1}/{len(evaluation_items)} - Current accuracy: {current_accuracy:.2%}"
            )

    # Calculate overall metrics
    accuracy = correct_count / len(evaluation_items) if evaluation_items else 0

    # Calculate non-NONE metrics
    non_none_accuracy = (
        non_none_correct_count / non_none_total if non_none_total > 0 else 0
    )

    # Display evaluation results summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 50)

    parsing_error_count = pred_labels.count("PARSING_ERROR")
    parsing_error_pct = (
        parsing_error_count / len(evaluation_items) * 100 if evaluation_items else 0
    )

    logger.info("\nEVALUATION RESULTS SUMMARY:")
    logger.info(
        f"  Overall Accuracy: {accuracy:.2%} ({correct_count}/{len(evaluation_items)} correct)"
    )

    if non_none_total > 0:
        logger.info(
            f"  Non-NONE Accuracy: {non_none_accuracy:.2%} ({non_none_correct_count}/{non_none_total} correct)"
        )

    logger.info(
        f"  Parsing Errors: {parsing_error_count} ({parsing_error_pct:.1f}% of total)"
    )
    logger.info(
        f"  Total Errors: {len(errors)} (including mispredictions and parsing errors)"
    )

    # Log error breakdown if there are errors
    log_error_breakdown(errors)

    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(
        true_labels, pred_labels, LEGAL_LABEL_SCHEMA
    )

    # Log detailed metrics
    logger.info("\nDETAILED METRICS:")
    logger.info(f"  Precision (weighted): {detailed_metrics['overall_precision']:.2%}")
    logger.info(f"  Recall (weighted): {detailed_metrics['overall_recall']:.2%}")
    logger.info(f"  F1 Score (weighted): {detailed_metrics['overall_f1']:.2%}")

    results: EvaluationResults = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(evaluation_items),
        "errors": errors,
        "non_none_accuracy": non_none_accuracy,
        "non_none_correct_count": non_none_correct_count,
        "non_none_total_count": non_none_total,
        # Add new metrics
        "precision": detailed_metrics["overall_precision"],
        "recall": detailed_metrics["overall_recall"],
        "f1_score": detailed_metrics["overall_f1"],
        "per_class_precision": detailed_metrics["per_class_precision"],
        "per_class_recall": detailed_metrics["per_class_recall"],
        "per_class_f1": detailed_metrics["per_class_f1"],
        "confusion_matrix": detailed_metrics["confusion_matrix"],
        "classification_report": detailed_metrics["classification_report"],
    }

    # Add metadata
    metadata = {
        "model_size": model_size,
        "model_path": model_path,
        "model_display_name": "Gemma 3 1B"
        if model_size == "1b"
        else ("Gemma 3 4B" if model_size == "4b" else "Gemma 3 12B"),
        "base_model": config.base_model_id,
        "test_data_path": str(TEST_DATA_PATH),
        "use_local_model": use_local_model,
        "evaluation_timestamp": datetime.now().isoformat(),
        "excluded_none": len(test_dataset) - len(evaluation_items),
    }

    # Combine results with metadata
    full_results = {**results, "metadata": metadata}

    # Create visualization with metadata
    viz = create_evaluation_visualization(full_results)

    # Return as plain dict for ZenML compatibility
    return dict(full_results), viz
