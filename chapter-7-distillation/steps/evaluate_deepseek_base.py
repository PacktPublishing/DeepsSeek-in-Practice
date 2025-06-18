import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Annotated, Any, Dict, Optional, Tuple

# Suppress litellm verbose logging
import litellm
import polars as pl
from constants import LEGAL_LABEL_SCHEMA, OPEN_ROUTER_EVALS_BASE_MODEL_NAME
from litellm import completion
from pydantic import BaseModel, Field
from utils.evaluation_utils import (
    calculate_detailed_metrics,
    create_base_prompt_json,
    extract_prediction,
    log_error_breakdown,
)
from utils.visualization_utils import create_evaluation_visualization
from zenml import step
from zenml.types import HTMLString

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class LegalClauseClassification(BaseModel):
    """Schema for legal clause classification response."""

    rationale: str = Field(
        description="Detailed explanation of why the clause belongs to this category"
    )
    label: str = Field(
        description="The classification category from the list of valid labels"
    )


def create_prompt(item: Dict) -> str:
    """Create a prompt from a test item."""
    # Remove NONE from the valid labels
    valid_labels = [label for label in LEGAL_LABEL_SCHEMA if label != "NONE"]

    return create_base_prompt_json(item, valid_labels)


def run_inference(item: Dict) -> Tuple[Dict, Optional[str], str]:
    """
    Run inference with DeepSeek model on a test item.

    Args:
        item: The test item

    Returns:
        Tuple of (item, predicted_label, response)
    """
    prompt = create_prompt(item)

    try:
        # Make API call with JSON response format
        response = completion(
            model=OPEN_ROUTER_EVALS_BASE_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,  # Lower temperature for more deterministic outputs
            max_tokens=1024,
            response_format={"type": "json_object"},  # Use simple JSON object format
        )

        # Extract the response content
        response_text = response.choices[0].message.content

        # Debug log to see the raw response
        if not response_text:
            logger.warning("Received empty response from API")
        else:
            logger.debug(f"Raw API response: {response_text[:300]}...")

        # Extract prediction
        pred_label = extract_prediction(response_text)

        if not pred_label:
            logger.debug(
                f"Failed to extract label from response: {response_text[:200]}..."
            )
        elif pred_label not in LEGAL_LABEL_SCHEMA:
            logger.warning(
                f"Model predicted invalid label '{pred_label}' not in schema. "
                f"Valid labels: {', '.join(LEGAL_LABEL_SCHEMA[:5])}..."
            )
            # Optionally, you could set pred_label to None here to treat as parsing error
            # pred_label = None

        return item, pred_label, response_text

    except Exception as e:
        error_msg = f"Error during inference: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Failed item: {item.get('clause', '')[:100]}...")
        return item, None, error_msg


@step(enable_cache=False)
def evaluate_deepseek_base(
    test_dataset: pl.DataFrame,
    max_workers: int = 5,
    verbose: bool = False,
) -> Tuple[
    Annotated[Dict[str, Any], "deepseek_evaluation_results"],
    Annotated[HTMLString, "deepseek_evaluation_viz"],
]:
    """Evaluate DeepSeek base model on test data.

    Args:
        test_dataset: Test dataset to evaluate on
        max_workers: Maximum number of concurrent API calls
        verbose: Whether to print detailed outputs

    Returns:
        Tuple of evaluation results and visualization
    """
    # Convert polars DataFrame to list of dicts for evaluation
    evaluation_items = test_dataset.to_dicts()
    total_items = len(evaluation_items)

    logger.info(f"Starting DeepSeek evaluation on {total_items} test examples")
    logger.info(f"Using model: {OPEN_ROUTER_EVALS_BASE_MODEL_NAME}")
    logger.info(f"Max concurrent workers: {max_workers}")

    # Initialize results
    results = {
        "accuracy": 0.0,
        "correct_count": 0,
        "total_count": total_items,
        "errors": [],
        "non_none_accuracy": 0.0,
        "non_none_correct_count": 0,
        "non_none_total_count": 0,
    }

    # Track results in order
    all_predictions = [None] * total_items
    true_labels = []
    pred_labels = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(run_inference, item): i
            for i, item in enumerate(evaluation_items)
        }

        # Process completed futures
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            item, pred_label, response = future.result()

            completed += 1

            # Log progress every 10 samples
            if completed % 10 == 0 or completed == total_items:
                logger.info(f"Progress: {completed}/{total_items} samples evaluated...")

            true_label = item["label"]

            # Store prediction in order
            prediction_data = {
                "item": item,
                "true_label": true_label,
                "pred_label": pred_label,
                "response": response[:500] + "..." if len(response) > 500 else response,
            }
            all_predictions[index] = prediction_data

            if pred_label:
                if pred_label == true_label:
                    results["correct_count"] += 1
                else:
                    results["errors"].append(
                        {
                            "item": item,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "response": response[:200] + "..."
                            if len(response) > 200
                            else response,
                        }
                    )

                # Track non-NONE metrics
                if true_label != "NONE":
                    results["non_none_total_count"] += 1
                    if pred_label == true_label:
                        results["non_none_correct_count"] += 1
            else:
                # Parsing error
                results["errors"].append(
                    {
                        "item": item,
                        "true_label": true_label,
                        "pred_label": "PARSING_ERROR",
                        "response": response[:200] + "..."
                        if len(response) > 200
                        else response,
                    }
                )

                if true_label != "NONE":
                    results["non_none_total_count"] += 1

                # Log parsing error with response details
                logger.warning(
                    f"Failed to parse prediction for item {index + 1}. "
                    f"True label: {true_label}. "
                    f"Response: {response[:200]}..."
                )
                logger.debug(f"Clause snippet: {item.get('clause', '')[:100]}...")

    # Collect labels in order for metric calculation
    for prediction_data in all_predictions:
        if prediction_data:
            true_labels.append(prediction_data["true_label"])
            pred_labels.append(
                prediction_data["pred_label"]
                if prediction_data["pred_label"]
                else "PARSING_ERROR"
            )

    # Calculate accuracy
    results["accuracy"] = (
        results["correct_count"] / results["total_count"]
        if results["total_count"] > 0
        else 0
    )
    results["non_none_accuracy"] = (
        results["non_none_correct_count"] / results["non_none_total_count"]
        if results["non_none_total_count"] > 0
        else 0
    )

    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(
        true_labels, pred_labels, LEGAL_LABEL_SCHEMA
    )

    # Display evaluation results summary
    logger.info("\n" + "=" * 50)
    logger.info("DEEPSEEK EVALUATION COMPLETED")
    logger.info("=" * 50)

    parsing_error_count = sum(
        1 for err in results["errors"] if err["pred_label"] == "PARSING_ERROR"
    )
    parsing_error_pct = (
        parsing_error_count / results["total_count"] * 100
        if results["total_count"]
        else 0
    )

    logger.info("\nEVALUATION RESULTS SUMMARY:")
    logger.info(
        f"  Overall Accuracy: {results['accuracy']:.2%} ({results['correct_count']}/{results['total_count']} correct)"
    )

    if results["non_none_total_count"] > 0:
        logger.info(
            f"  Non-NONE Accuracy: {results['non_none_accuracy']:.2%} ({results['non_none_correct_count']}/{results['non_none_total_count']} correct)"
        )

    logger.info(
        f"  Parsing Errors: {parsing_error_count} ({parsing_error_pct:.1f}% of total)"
    )
    logger.info(
        f"  Total Errors: {len(results['errors'])} (including mispredictions and parsing errors)"
    )

    # Log error breakdown if there are errors
    log_error_breakdown(results["errors"], title="Error breakdown (top 10)")

    # Log detailed metrics
    logger.info("\nDETAILED METRICS:")
    logger.info(f"  Precision (weighted): {detailed_metrics['overall_precision']:.2%}")
    logger.info(f"  Recall (weighted): {detailed_metrics['overall_recall']:.2%}")
    logger.info(f"  F1 Score (weighted): {detailed_metrics['overall_f1']:.2%}")

    # Add metadata
    metadata = {
        "model_size": "DeepSeek Base",
        "model_path": OPEN_ROUTER_EVALS_BASE_MODEL_NAME,
        "model_display_name": "DeepSeek R1 (Base Model)",
        "base_model": "deepseek-r1-0528",
        "test_data_path": "Provided by pipeline",
        "use_local_model": False,
        "evaluation_timestamp": datetime.now().isoformat(),
        "max_workers": max_workers,
    }

    # Add detailed metrics to results
    results.update(
        {
            "precision": detailed_metrics["overall_precision"],
            "recall": detailed_metrics["overall_recall"],
            "f1_score": detailed_metrics["overall_f1"],
            "per_class_precision": detailed_metrics["per_class_precision"],
            "per_class_recall": detailed_metrics["per_class_recall"],
            "per_class_f1": detailed_metrics["per_class_f1"],
            "confusion_matrix": detailed_metrics["confusion_matrix"],
            "classification_report": detailed_metrics["classification_report"],
        }
    )

    # Combine results with metadata
    full_results = {**results, "metadata": metadata}

    # Create visualization with metadata
    viz = create_evaluation_visualization(full_results)

    # Return as plain dict for ZenML compatibility
    return dict(full_results), viz
