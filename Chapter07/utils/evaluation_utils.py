"""Common utilities for model evaluation."""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def extract_prediction(response: str, log_failures: bool = True) -> Optional[str]:
    """
    Extract the predicted label from the model's response.

    Args:
        response: The model's text response
        log_failures: Whether to log failed parsing attempts

    Returns:
        The extracted label or None if parsing failed
    """
    try:
        # First, try to remove markdown code blocks if present
        if "```json" in response:
            # Extract content between ```json and ```
            start_marker = "```json"
            end_marker = "```"
            start_idx = response.find(start_marker) + len(start_marker)
            end_idx = response.find(end_marker, start_idx)
            if end_idx > start_idx:
                response = response[start_idx:end_idx].strip()
        elif "```" in response and "{" in response:
            # Handle cases where it's just ``` without json
            start_idx = response.find("```") + 3
            end_idx = response.find("```", start_idx)
            if end_idx > start_idx:
                response = response[start_idx:end_idx].strip()

        # Now find the JSON part in the response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed_json = json.loads(json_str)

            if "label" in parsed_json:
                return parsed_json["label"].strip()
            else:
                if log_failures:
                    logger.debug(f"No 'label' field in parsed JSON: {parsed_json}")
        else:
            if log_failures:
                logger.debug(
                    f"No JSON structure found in response: {response[:200]}..."
                )
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        if log_failures:
            logger.debug(f"Failed to parse response: {str(e)}")
            logger.debug(f"Response was: {response[:500]}...")

    return None


def get_error_breakdown(errors: List[Dict]) -> Dict[str, int]:
    """
    Get a breakdown of error types from a list of errors.

    Args:
        errors: List of error dictionaries with 'true_label' and 'pred_label' keys

    Returns:
        Dictionary mapping error types to counts
    """
    error_by_type = {}
    for error in errors:
        key = f"{error['true_label']} -> {error['pred_label']}"
        error_by_type[key] = error_by_type.get(key, 0) + 1
    return error_by_type


def log_error_breakdown(
    errors: List[Dict], title: str = "Error breakdown", top_n: int = 10
):
    """
    Log a breakdown of the most common error types.

    Args:
        errors: List of error dictionaries
        title: Title for the error breakdown
        top_n: Number of top error types to show
    """
    if not errors:
        return

    logger.info(f"\nTotal errors: {len(errors)}")
    error_by_type = get_error_breakdown(errors)

    logger.info(f"{title}:")
    for error_type, count in sorted(
        error_by_type.items(), key=lambda x: x[1], reverse=True
    )[:top_n]:
        logger.info(f"  {error_type}: {count}")


def create_base_prompt_json(item: Dict, valid_labels: List[str]) -> str:
    """
    Create the base JSON prompt for legal clause classification.

    Args:
        item: Test item containing clause data
        valid_labels: List of valid classification labels

    Returns:
        JSON string for the prompt
    """
    return json.dumps(
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


def calculate_detailed_metrics(
    true_labels: List[str], pred_labels: List[str], label_schema: List[str]
) -> Dict[str, Any]:
    """Calculate precision, recall, F1, and confusion matrix for predictions.

    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        label_schema: List of all possible labels

    Returns:
        Dictionary containing detailed metrics
    """
    # Filter out parsing errors for metric calculation
    valid_indices = [i for i, pred in enumerate(pred_labels) if pred != "PARSING_ERROR"]
    valid_true = [true_labels[i] for i in valid_indices]
    valid_pred = [pred_labels[i] for i in valid_indices]

    # Calculate metrics only on valid predictions
    if valid_true and valid_pred:
        # Overall metrics
        overall_precision = precision_score(
            valid_true,
            valid_pred,
            labels=label_schema,
            average="weighted",
            zero_division=0,
        )
        overall_recall = recall_score(
            valid_true,
            valid_pred,
            labels=label_schema,
            average="weighted",
            zero_division=0,
        )
        overall_f1 = f1_score(
            valid_true,
            valid_pred,
            labels=label_schema,
            average="weighted",
            zero_division=0,
        )

        # Per-class metrics
        per_class_precision = precision_score(
            valid_true, valid_pred, labels=label_schema, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            valid_true, valid_pred, labels=label_schema, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            valid_true, valid_pred, labels=label_schema, average=None, zero_division=0
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(valid_true, valid_pred, labels=label_schema)

        # Classification report
        class_report = classification_report(
            valid_true,
            valid_pred,
            labels=label_schema,
            output_dict=True,
            zero_division=0,
        )
    else:
        # Return zeros if no valid predictions
        overall_precision = overall_recall = overall_f1 = 0.0
        per_class_precision = np.zeros(len(label_schema))
        per_class_recall = np.zeros(len(label_schema))
        per_class_f1 = np.zeros(len(label_schema))
        conf_matrix = np.zeros((len(label_schema), len(label_schema)))
        class_report = {}

    return {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "per_class_precision": {
            label: float(prec) for label, prec in zip(label_schema, per_class_precision)
        },
        "per_class_recall": {
            label: float(rec) for label, rec in zip(label_schema, per_class_recall)
        },
        "per_class_f1": {
            label: float(f1) for label, f1 in zip(label_schema, per_class_f1)
        },
        "confusion_matrix": conf_matrix.tolist(),  # Convert to list for JSON serialization
        "classification_report": class_report,
        "valid_predictions_count": len(valid_true),
        "parsing_errors_count": len(pred_labels) - len(valid_true),
    }
