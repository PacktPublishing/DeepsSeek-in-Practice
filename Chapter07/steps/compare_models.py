import logging
from typing import Annotated, Any, Dict, Tuple

from utils.visualization_utils import create_model_comparison_visualization
from zenml import step
from zenml.types import HTMLString

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def compare_models(
    finetuned_results: Dict[str, Any],
    deepseek_results: Dict[str, Any],
) -> Tuple[
    Annotated[Dict[str, Any], "comparison_results"],
    Annotated[HTMLString, "comparison_viz"],
]:
    """Compare evaluation results between fine-tuned model and DeepSeek base model.

    Args:
        finetuned_results: Evaluation results from fine-tuned model
        deepseek_results: Evaluation results from DeepSeek base model

    Returns:
        Tuple of comparison results and visualization
    """
    logger.info("Comparing fine-tuned model with DeepSeek base model")

    # Extract key metrics
    ft_accuracy = finetuned_results["accuracy"]
    ds_accuracy = deepseek_results["accuracy"]

    ft_non_none_accuracy = finetuned_results["non_none_accuracy"]
    ds_non_none_accuracy = deepseek_results["non_none_accuracy"]

    # Extract new metrics
    ft_precision = finetuned_results.get("precision", 0)
    ds_precision = deepseek_results.get("precision", 0)

    ft_recall = finetuned_results.get("recall", 0)
    ds_recall = deepseek_results.get("recall", 0)

    ft_f1 = finetuned_results.get("f1_score", 0)
    ds_f1 = deepseek_results.get("f1_score", 0)

    # Calculate improvements
    overall_improvement = ft_accuracy - ds_accuracy
    overall_improvement_pct = (
        (overall_improvement / ds_accuracy * 100) if ds_accuracy > 0 else 0
    )

    non_none_improvement = ft_non_none_accuracy - ds_non_none_accuracy
    non_none_improvement_pct = (
        (non_none_improvement / ds_non_none_accuracy * 100)
        if ds_non_none_accuracy > 0
        else 0
    )

    # Analyze error patterns
    ft_errors = finetuned_results.get("errors", [])
    ds_errors = deepseek_results.get("errors", [])

    # Count parsing errors
    ft_parsing_errors = sum(
        1 for err in ft_errors if err["pred_label"] == "PARSING_ERROR"
    )
    ds_parsing_errors = sum(
        1 for err in ds_errors if err["pred_label"] == "PARSING_ERROR"
    )

    # Calculate clean accuracy (excluding parsing errors)
    total_samples = finetuned_results["total_count"]
    ft_clean_correct = finetuned_results[
        "correct_count"
    ]  # Fine-tuned has no parsing errors
    ft_clean_total = total_samples  # All samples are valid for fine-tuned
    ft_clean_accuracy = ft_clean_correct / ft_clean_total if ft_clean_total > 0 else 0

    # For DeepSeek, exclude parsing errors from total
    ds_clean_total = total_samples - ds_parsing_errors
    # Calculate correct predictions excluding parsing errors
    # Parsing errors are always incorrect, so we don't need to adjust correct count
    ds_clean_correct = deepseek_results["correct_count"]
    ds_clean_accuracy = ds_clean_correct / ds_clean_total if ds_clean_total > 0 else 0

    # Calculate clean improvement
    clean_improvement = ft_clean_accuracy - ds_clean_accuracy
    clean_improvement_pct = (
        (clean_improvement / ds_clean_accuracy * 100) if ds_clean_accuracy > 0 else 0
    )

    # Create error breakdown for both models
    def get_error_breakdown(errors):
        error_by_type = {}
        for error in errors:
            key = f"{error['true_label']} -> {error['pred_label']}"
            error_by_type[key] = error_by_type.get(key, 0) + 1
        return error_by_type

    ft_error_breakdown = get_error_breakdown(ft_errors)
    ds_error_breakdown = get_error_breakdown(ds_errors)

    # Log comparison results
    logger.info("\n" + "=" * 50)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 50)

    logger.info("\nOVERALL ACCURACY:")
    logger.info(f"  Fine-tuned Model: {ft_accuracy:.2%}")
    logger.info(f"  DeepSeek Base:    {ds_accuracy:.2%}")
    logger.info(
        f"  Improvement:      {overall_improvement:+.2%} ({overall_improvement_pct:+.1f}%)"
    )

    logger.info("\nNON-NONE ACCURACY:")
    logger.info(f"  Fine-tuned Model: {ft_non_none_accuracy:.2%}")
    logger.info(f"  DeepSeek Base:    {ds_non_none_accuracy:.2%}")
    logger.info(
        f"  Improvement:      {non_none_improvement:+.2%} ({non_none_improvement_pct:+.1f}%)"
    )

    logger.info("\nPRECISION:")
    logger.info(f"  Fine-tuned Model: {ft_precision:.2%}")
    logger.info(f"  DeepSeek Base:    {ds_precision:.2%}")
    logger.info(f"  Improvement:      {ft_precision - ds_precision:+.2%}")

    logger.info("\nRECALL:")
    logger.info(f"  Fine-tuned Model: {ft_recall:.2%}")
    logger.info(f"  DeepSeek Base:    {ds_recall:.2%}")
    logger.info(f"  Improvement:      {ft_recall - ds_recall:+.2%}")

    logger.info("\nF1 SCORE:")
    logger.info(f"  Fine-tuned Model: {ft_f1:.2%}")
    logger.info(f"  DeepSeek Base:    {ds_f1:.2%}")
    logger.info(f"  Improvement:      {ft_f1 - ds_f1:+.2%}")

    logger.info("\nPARSING ERRORS:")
    logger.info(f"  Fine-tuned Model: {ft_parsing_errors}")
    logger.info(f"  DeepSeek Base:    {ds_parsing_errors}")
    logger.info(f"  Reduction:        {ds_parsing_errors - ft_parsing_errors}")

    logger.info("\nCLEAN ACCURACY (excluding parsing errors):")
    logger.info(
        f"  Fine-tuned Model: {ft_clean_accuracy:.2%} ({ft_clean_correct}/{ft_clean_total})"
    )
    logger.info(
        f"  DeepSeek Base:    {ds_clean_accuracy:.2%} ({ds_clean_correct}/{ds_clean_total})"
    )
    logger.info(
        f"  Improvement:      {clean_improvement:+.2%} ({clean_improvement_pct:+.1f}%)"
    )

    # Find common and unique errors
    ft_error_types = set(ft_error_breakdown.keys())
    ds_error_types = set(ds_error_breakdown.keys())

    common_errors = ft_error_types & ds_error_types
    ft_unique_errors = ft_error_types - ds_error_types
    ds_unique_errors = ds_error_types - ft_error_types

    logger.info(f"\nERROR ANALYSIS:")
    logger.info(f"  Common error types: {len(common_errors)}")
    logger.info(f"  Unique to fine-tuned: {len(ft_unique_errors)}")
    logger.info(f"  Unique to DeepSeek: {len(ds_unique_errors)}")

    # Create comparison results
    comparison_results = {
        "finetuned_accuracy": ft_accuracy,
        "deepseek_accuracy": ds_accuracy,
        "overall_improvement": overall_improvement,
        "overall_improvement_pct": overall_improvement_pct,
        "finetuned_non_none_accuracy": ft_non_none_accuracy,
        "deepseek_non_none_accuracy": ds_non_none_accuracy,
        "non_none_improvement": non_none_improvement,
        "non_none_improvement_pct": non_none_improvement_pct,
        # Clean accuracy metrics (excluding parsing errors)
        "finetuned_clean_accuracy": ft_clean_accuracy,
        "deepseek_clean_accuracy": ds_clean_accuracy,
        "finetuned_clean_correct": ft_clean_correct,
        "deepseek_clean_correct": ds_clean_correct,
        "finetuned_clean_total": ft_clean_total,
        "deepseek_clean_total": ds_clean_total,
        "clean_improvement": clean_improvement,
        "clean_improvement_pct": clean_improvement_pct,
        # New metrics
        "finetuned_precision": ft_precision,
        "deepseek_precision": ds_precision,
        "precision_improvement": ft_precision - ds_precision,
        "finetuned_recall": ft_recall,
        "deepseek_recall": ds_recall,
        "recall_improvement": ft_recall - ds_recall,
        "finetuned_f1": ft_f1,
        "deepseek_f1": ds_f1,
        "f1_improvement": ft_f1 - ds_f1,
        # Existing metrics
        "finetuned_correct": finetuned_results["correct_count"],
        "deepseek_correct": deepseek_results["correct_count"],
        "total_samples": finetuned_results["total_count"],
        "finetuned_parsing_errors": ft_parsing_errors,
        "deepseek_parsing_errors": ds_parsing_errors,
        "parsing_error_reduction": ds_parsing_errors - ft_parsing_errors,
        "finetuned_total_errors": len(ft_errors),
        "deepseek_total_errors": len(ds_errors),
        "common_error_types": len(common_errors),
        "finetuned_unique_error_types": len(ft_unique_errors),
        "deepseek_unique_error_types": len(ds_unique_errors),
        "finetuned_error_breakdown": ft_error_breakdown,
        "deepseek_error_breakdown": ds_error_breakdown,
        # Additional data from results
        "finetuned_confusion_matrix": finetuned_results.get("confusion_matrix", []),
        "deepseek_confusion_matrix": deepseek_results.get("confusion_matrix", []),
        "finetuned_metadata": finetuned_results.get("metadata", {}),
        "deepseek_metadata": deepseek_results.get("metadata", {}),
    }

    # Create visualization
    viz = create_model_comparison_visualization(comparison_results)

    return comparison_results, viz
