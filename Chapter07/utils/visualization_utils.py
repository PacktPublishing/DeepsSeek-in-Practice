import base64
import io
import logging
from typing import Any, Dict, List, Optional, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from zenml.types import HTMLString

# Use non-interactive backend
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def create_confusion_matrix_plot(
    confusion_matrix: List[List[int]],
    labels: List[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (12, 10),
    cmap: str = "Blues",
) -> str:
    """Create a confusion matrix visualization and return as base64 string.

    Args:
        confusion_matrix: The confusion matrix as a list of lists
        labels: List of class labels
        title: Title for the plot
        figsize: Figure size
        cmap: Color map

    Returns:
        Base64 encoded image string
    """
    # Convert to numpy array
    cm = np.array(confusion_matrix)

    # Create figure
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "Count"},
    )

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return image_base64


def create_metrics_bar_chart(
    metrics: Dict[str, float],
    title: str = "Performance Metrics",
    figsize: tuple = (10, 6),
    color: str = "#3b82f6",
) -> str:
    """Create a bar chart for metrics and return as base64 string.

    Args:
        metrics: Dictionary of metric names and values
        title: Title for the plot
        figsize: Figure size
        color: Bar color

    Returns:
        Base64 encoded image string
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Prepare data
    labels = list(metrics.keys())
    values = list(metrics.values())

    # Create bar chart
    bars = plt.bar(labels, values, color=color)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2%}",
            ha="center",
            va="bottom",
        )

    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1.1)  # Set y-axis from 0 to 1.1 to leave room for labels

    # Add grid for better readability
    plt.grid(axis="y", alpha=0.3)

    # Rotate x labels if needed
    if len(labels) > 5:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return image_base64


class EvaluationResults(TypedDict):
    """Structure for model evaluation results."""

    accuracy: float
    correct_count: int
    total_count: int
    errors: List[Dict[str, Any]]
    non_none_accuracy: float
    non_none_correct_count: int
    non_none_total_count: int
    # New metrics
    precision: float
    recall: float
    f1_score: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]


def create_evaluation_visualization(results: Dict[str, Any]) -> HTMLString:
    """Create an HTML visualization for evaluation results.

    Args:
        results: Dictionary containing evaluation results

    Returns:
        HTMLString with visualization
    """
    accuracy = results["accuracy"]
    correct = results["correct_count"]
    total = results["total_count"]
    non_none_accuracy = results.get("non_none_accuracy", 0)
    non_none_correct = results.get("non_none_correct_count", 0)
    non_none_total = results.get("non_none_total_count", 0)

    # Extract new metrics
    precision = results.get("precision", 0)
    recall = results.get("recall", 0)
    f1_score = results.get("f1_score", 0)
    confusion_matrix_data = results.get("confusion_matrix", [])

    # Create confusion matrix plot if available
    confusion_matrix_img = ""
    if confusion_matrix_data:
        from constants import LEGAL_LABEL_SCHEMA

        confusion_matrix_img = create_confusion_matrix_plot(
            confusion_matrix_data,
            LEGAL_LABEL_SCHEMA,
            title="Confusion Matrix - Model Predictions",
        )

    # Create metrics bar chart
    metrics_chart_img = ""
    if precision or recall or f1_score:
        metrics_dict = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "Accuracy": accuracy,
        }
        metrics_chart_img = create_metrics_bar_chart(
            metrics_dict, title="Overall Performance Metrics"
        )

    # Extract metadata if available
    metadata = results.get("metadata", {})

    # Determine color based on accuracy
    if accuracy >= 0.8:
        color = "#4ade80"  # green-400
        status = "Excellent"
    elif accuracy >= 0.6:
        color = "#facc15"  # yellow-400
        status = "Good"
    else:
        color = "#f87171"  # red-400
        status = "Needs Improvement"

    # Create HTML dashboard
    html_content = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h2 style="text-align: center; color: #1f2937; margin-bottom: 30px;">Model Evaluation Results</h2>
        
        <div style="background: {color}20; border: 2px solid {
        color
    }; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
            <h3 style="margin: 0 0 10px 0; color: #1f2937;">Overall Performance: {
        status
    }</h3>
            <div style="font-size: 48px; font-weight: bold; color: {
        color
    }; margin: 10px 0;">
                {accuracy:.1%}
            </div>
            <p style="margin: 0; color: #4b5563; font-size: 18px;">
                {correct} / {total} correct predictions
            </p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px;">
            <div style="background: #f3f4f6; border-radius: 8px; padding: 20px;">
                <h4 style="margin: 0 0 15px 0; color: #1f2937;">Overall Metrics</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Accuracy:</span>
                    <strong>{accuracy:.2%}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Correct:</span>
                    <strong>{correct}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Total:</span>
                    <strong>{total}</strong>
                </div>
            </div>
            
            <div style="background: #f3f4f6; border-radius: 8px; padding: 20px;">
                <h4 style="margin: 0 0 15px 0; color: #1f2937;">Performance Metrics</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Precision:</span>
                    <strong>{precision:.2%}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Recall:</span>
                    <strong>{recall:.2%}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>F1 Score:</span>
                    <strong>{f1_score:.2%}</strong>
                </div>
            </div>
            
            <div style="background: #f3f4f6; border-radius: 8px; padding: 20px;">
                <h4 style="margin: 0 0 15px 0; color: #1f2937;">Non-NONE Metrics</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Accuracy:</span>
                    <strong>{non_none_accuracy:.2%}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Correct:</span>
                    <strong>{non_none_correct}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Total:</span>
                    <strong>{non_none_total}</strong>
                </div>
            </div>
        </div>
        
        {
        f'''
        <div style="margin-bottom: 30px;">
            <h3 style="text-align: center; color: #1f2937; margin-bottom: 20px;">Performance Metrics Visualization</h3>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{metrics_chart_img}" style="max-width: 100%; height: auto;">
            </div>
        </div>
        '''
        if metrics_chart_img
        else ""
    }
        
        {
        f'''
        <div style="margin-bottom: 30px;">
            <h3 style="text-align: center; color: #1f2937; margin-bottom: 20px;">Confusion Matrix</h3>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{confusion_matrix_img}" style="max-width: 100%; height: auto;">
            </div>
        </div>
        '''
        if confusion_matrix_img
        else ""
    }
        
        {
        f'''
        <div style="margin-top: 20px; background: #e5e7eb; border-radius: 8px; padding: 20px;">
            <h4 style="margin: 0 0 15px 0; color: #1f2937;">Evaluation Details</h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #4b5563;">Model:</span>
                <strong style="color: #1f2937;">{
            metadata.get("model_display_name", "Unknown")
        }</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #4b5563;">Model Path:</span>
                <strong style="color: #1f2937; font-size: 12px; word-break: break-all;">{
            metadata.get("model_path", "Unknown")
        }</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #4b5563;">Test Data:</span>
                <strong style="color: #1f2937; font-size: 12px;">{
            metadata.get("test_data_path", "Unknown")
        }</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #4b5563;">Evaluation Time:</span>
                <strong style="color: #1f2937; font-size: 12px;">{
            metadata.get("evaluation_timestamp", "Unknown")
        }</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #4b5563;">Model Source:</span>
                <strong style="color: #1f2937;">{
            "OpenRouter API"
            if "openrouter" in metadata.get("model_path", "").lower()
            else (
                "Local" if metadata.get("use_local_model", True) else "Hugging Face Hub"
            )
        }</strong>
            </div>
        </div>
        '''
        if metadata
        else ""
    }
        
        <div style="margin-top: 30px; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px;">
            <p style="margin: 0; color: #92400e;">
                <strong>Note:</strong> Non-NONE metrics exclude examples labeled as "NONE" from calculations.
                This is useful when the model was trained without NONE examples.
            </p>
        </div>
    </div>
    """

    return HTMLString(html_content)


def create_data_loading_visualization(
    train_size: int, val_size: int, filter_none: bool
) -> HTMLString:
    """Create an HTML visualization for data loading statistics.

    Args:
        train_size: Number of training examples
        val_size: Number of validation examples
        filter_none: Whether NONE examples were filtered

    Returns:
        HTMLString with visualization
    """
    total_size = train_size + val_size
    train_pct = (train_size / total_size * 100) if total_size > 0 else 0
    val_pct = (val_size / total_size * 100) if total_size > 0 else 0

    html_content = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="text-align: center; color: #1f2937; margin-bottom: 30px;">Dataset Loading Summary</h2>
        
        <div style="background: #f3f4f6; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 20px 0; color: #1f2937;">Dataset Statistics</h3>
            
            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #4b5563;">Training Set:</span>
                    <strong style="color: #1f2937;">{train_size:,} examples ({train_pct:.1f}%)</strong>
                </div>
                <div style="background: #e5e7eb; border-radius: 4px; height: 24px; overflow: hidden;">
                    <div style="background: #3b82f6; height: 100%; width: {train_pct}%;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #4b5563;">Validation Set:</span>
                    <strong style="color: #1f2937;">{val_size:,} examples ({val_pct:.1f}%)</strong>
                </div>
                <div style="background: #e5e7eb; border-radius: 4px; height: 24px; overflow: hidden;">
                    <div style="background: #10b981; height: 100%; width: {val_pct}%;"></div>
                </div>
            </div>
            
            <div style="border-top: 1px solid #d1d5db; margin-top: 20px; padding-top: 15px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #4b5563;">Total Examples:</span>
                    <strong style="color: #1f2937; font-size: 18px;">{total_size:,}</strong>
                </div>
            </div>
        </div>
        
        {"<div style='background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px;'><p style='margin: 0; color: #92400e;'><strong>Note:</strong> NONE examples were filtered out from training and validation sets.</p></div>" if filter_none else ""}
    </div>
    """

    return HTMLString(html_content)


def create_training_visualization(
    model_size: str, final_loss: Optional[float] = None
) -> HTMLString:
    """Create an HTML visualization for training completion.

    Args:
        model_size: Model size used for training
        final_loss: Final training loss if available

    Returns:
        HTMLString with visualization
    """
    if model_size == "1b":
        model_display = "Gemma 3 1B"
    elif model_size == "4b":
        model_display = "Gemma 3 4B"
    else:  # 12b
        model_display = "Gemma 3 12B"

    html_content = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="text-align: center; color: #1f2937; margin-bottom: 30px;">Training Completed</h2>
        
        <div style="background: #10b98120; border: 2px solid #10b981; border-radius: 12px; padding: 30px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 10px;">✅</div>
            <h3 style="margin: 0 0 10px 0; color: #1f2937;">Model Successfully Fine-tuned</h3>
            <p style="margin: 0; color: #4b5563; font-size: 18px;">
                {model_display} model trained and saved
            </p>
            {f'<p style="margin: 10px 0 0 0; color: #6b7280;">Final Loss: {final_loss:.4f}</p>' if final_loss else ""}
        </div>
        
        <div style="margin-top: 20px; background: #f3f4f6; border-radius: 8px; padding: 20px;">
            <h4 style="margin: 0 0 15px 0; color: #1f2937;">Model Details</h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="color: #4b5563;">Base Model:</span>
                <strong style="color: #1f2937;">{model_display}</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #4b5563;">Fine-tuning Method:</span>
                <strong style="color: #1f2937;">LoRA (4-bit)</strong>
            </div>
        </div>
    </div>
    """

    return HTMLString(html_content)


def create_model_comparison_visualization(
    comparison_results: Dict[str, Any],
) -> HTMLString:
    """Create an HTML visualization comparing fine-tuned model with DeepSeek base model.

    Args:
        comparison_results: Dictionary containing comparison metrics

    Returns:
        HTMLString with comparison visualization
    """
    # Extract metrics
    ft_accuracy = comparison_results["finetuned_accuracy"]
    ds_accuracy = comparison_results["deepseek_accuracy"]
    overall_improvement = comparison_results["overall_improvement"]
    overall_improvement_pct = comparison_results["overall_improvement_pct"]

    ft_non_none_accuracy = comparison_results["finetuned_non_none_accuracy"]
    ds_non_none_accuracy = comparison_results["deepseek_non_none_accuracy"]
    non_none_improvement = comparison_results["non_none_improvement"]
    non_none_improvement_pct = comparison_results["non_none_improvement_pct"]

    # Extract clean accuracy metrics
    ft_clean_accuracy = comparison_results.get("finetuned_clean_accuracy", ft_accuracy)
    ds_clean_accuracy = comparison_results.get("deepseek_clean_accuracy", ds_accuracy)
    clean_improvement = comparison_results.get("clean_improvement", 0)
    clean_improvement_pct = comparison_results.get("clean_improvement_pct", 0)

    # Extract new metrics
    ft_precision = comparison_results.get("finetuned_precision", 0)
    ds_precision = comparison_results.get("deepseek_precision", 0)
    ft_recall = comparison_results.get("finetuned_recall", 0)
    ds_recall = comparison_results.get("deepseek_recall", 0)
    ft_f1 = comparison_results.get("finetuned_f1", 0)
    ds_f1 = comparison_results.get("deepseek_f1", 0)

    total_samples = comparison_results["total_samples"]

    # Create comparison metrics chart
    comparison_metrics_img = ""
    if ft_precision or ds_precision:
        # Create grouped bar chart data
        plt.figure(figsize=(12, 6))
        metrics = ["Accuracy", "Clean Accuracy", "Precision", "Recall", "F1 Score"]
        ft_values = [ft_accuracy, ft_clean_accuracy, ft_precision, ft_recall, ft_f1]
        ds_values = [ds_accuracy, ds_clean_accuracy, ds_precision, ds_recall, ds_f1]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = plt.bar(
            x - width / 2, ft_values, width, label="Fine-tuned", color="#3b82f6"
        )
        bars2 = plt.bar(
            x + width / 2, ds_values, width, label="DeepSeek", color="#6b7280"
        )

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.ylabel("Score")
        plt.title("Model Performance Comparison")
        plt.xticks(x, metrics)
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        comparison_metrics_img = base64.b64encode(buffer.read()).decode()
        plt.close()

    # Determine improvement status and color
    if overall_improvement > 0:
        improvement_color = "#10b981"  # green-500
        improvement_icon = "📈"
        improvement_text = "Improvement"
    elif overall_improvement < 0:
        improvement_color = "#ef4444"  # red-500
        improvement_icon = "📉"
        improvement_text = "Regression"
    else:
        improvement_color = "#6b7280"  # gray-500
        improvement_icon = "➡️"
        improvement_text = "No Change"

    # Create bar chart heights (as percentages)
    max_accuracy = max(ft_accuracy, ds_accuracy, 1.0)
    ft_bar_height = ft_accuracy / max_accuracy * 100
    ds_bar_height = ds_accuracy / max_accuracy * 100

    ft_nn_bar_height = ft_non_none_accuracy / max_accuracy * 100
    ds_nn_bar_height = ds_non_none_accuracy / max_accuracy * 100

    # Extract metadata
    ft_metadata = comparison_results.get("finetuned_metadata", {})
    ds_metadata = comparison_results.get("deepseek_metadata", {})

    html_content = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px;">
        <h2 style="text-align: center; color: #1f2937; margin-bottom: 30px;">Model Performance Comparison</h2>
        
        <!-- Overall Improvement Summary -->
        <div style="background: {improvement_color}20; border: 2px solid {
        improvement_color
    }; border-radius: 12px; padding: 20px; margin-bottom: 30px; text-align: center;">
            <h3 style="margin: 0 0 10px 0; color: #1f2937;">Overall {improvement_text} {
        improvement_icon
    }</h3>
            <div style="font-size: 36px; font-weight: bold; color: {
        improvement_color
    };">
                {overall_improvement:+.1%}
            </div>
            <p style="margin: 5px 0 0 0; color: #4b5563; font-size: 16px;">
                ({overall_improvement_pct:+.1f}% relative improvement)
            </p>
        </div>
        
        <!-- Side-by-side Comparison -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
            <!-- Fine-tuned Model -->
            <div style="background: #f3f4f6; border-radius: 12px; padding: 20px;">
                <h3 style="margin: 0 0 20px 0; color: #1f2937; text-align: center;">Fine-tuned Model</h3>
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px;">
                        {ft_metadata.get("model_display_name", "Fine-tuned Model")}
                    </div>
                    <div style="font-size: 36px; font-weight: bold; color: #3b82f6;">
                        {ft_accuracy:.1%}
                    </div>
                    <div style="color: #4b5563;">
                        {comparison_results["finetuned_correct"]} / {
        total_samples
    } correct
                    </div>
                </div>
                
                <!-- Additional metrics -->
                <div style="background: white; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;">
                        <div>
                            <div style="font-size: 12px; color: #6b7280;">Precision</div>
                            <div style="font-size: 18px; font-weight: bold; color: #3b82f6;">{ft_precision:.1%}</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #6b7280;">Recall</div>
                            <div style="font-size: 18px; font-weight: bold; color: #3b82f6;">{ft_recall:.1%}</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #6b7280;">F1 Score</div>
                            <div style="font-size: 18px; font-weight: bold; color: #3b82f6;">{ft_f1:.1%}</div>
                        </div>
                    </div>
                </div>
                
                <!-- Non-NONE accuracy -->
                <div style="background: white; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px;">Non-NONE Accuracy</div>
                    <div style="font-size: 24px; font-weight: bold; color: #3b82f6;">
                        {ft_non_none_accuracy:.1%}
                    </div>
                </div>
                
                <!-- Clean accuracy -->
                <div style="background: white; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px;">Clean Accuracy (excl. parsing)</div>
                    <div style="font-size: 24px; font-weight: bold; color: #3b82f6;">
                        {ft_clean_accuracy:.1%}
                    </div>
                </div>
            </div>
            
            <!-- DeepSeek Base Model -->
            <div style="background: #f3f4f6; border-radius: 12px; padding: 20px;">
                <h3 style="margin: 0 0 20px 0; color: #1f2937; text-align: center;">DeepSeek Base</h3>
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px;">
                        {ds_metadata.get("model_display_name", "DeepSeek R1")}
                    </div>
                    <div style="font-size: 36px; font-weight: bold; color: #6b7280;">
                        {ds_accuracy:.1%}
                    </div>
                    <div style="color: #4b5563;">
                        {comparison_results["deepseek_correct"]} / {
        total_samples
    } correct
                    </div>
                </div>
                
                <!-- Additional metrics -->
                <div style="background: white; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;">
                        <div>
                            <div style="font-size: 12px; color: #6b7280;">Precision</div>
                            <div style="font-size: 18px; font-weight: bold; color: #6b7280;">{ds_precision:.1%}</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #6b7280;">Recall</div>
                            <div style="font-size: 18px; font-weight: bold; color: #6b7280;">{ds_recall:.1%}</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #6b7280;">F1 Score</div>
                            <div style="font-size: 18px; font-weight: bold; color: #6b7280;">{ds_f1:.1%}</div>
                        </div>
                    </div>
                </div>
                
                <!-- Non-NONE accuracy -->
                <div style="background: white; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px;">Non-NONE Accuracy</div>
                    <div style="font-size: 24px; font-weight: bold; color: #6b7280;">
                        {ds_non_none_accuracy:.1%}
                    </div>
                </div>
                
                <!-- Clean accuracy -->
                <div style="background: white; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px;">Clean Accuracy (excl. parsing)</div>
                    <div style="font-size: 24px; font-weight: bold; color: #6b7280;">
                        {ds_clean_accuracy:.1%}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Visual Comparison Bars -->
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
            <h4 style="margin: 0 0 20px 0; color: #1f2937;">Accuracy Comparison</h4>
            
            <!-- Overall Accuracy Bars -->
            <div style="margin-bottom: 25px;">
                <div style="font-size: 14px; color: #6b7280; margin-bottom: 10px;">Overall Accuracy</div>
                <div style="display: flex; align-items: flex-end; height: 120px; gap: 20px; position: relative;">
                    <div style="flex: 1; text-align: center; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%;">
                        <div style="font-weight: bold; color: #3b82f6; margin-bottom: 5px;">
                            {ft_accuracy:.1%}
                        </div>
                        <div style="background: #3b82f6; height: {
        ft_bar_height
    }%; width: 80%; border-radius: 4px 4px 0 0; min-height: 5px;">
                        </div>
                        <div style="margin-top: 5px; font-size: 12px; color: #6b7280;">Fine-tuned</div>
                    </div>
                    <div style="flex: 1; text-align: center; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%;">
                        <div style="font-weight: bold; color: #6b7280; margin-bottom: 5px;">
                            {ds_accuracy:.1%}
                        </div>
                        <div style="background: #6b7280; height: {
        ds_bar_height
    }%; width: 80%; border-radius: 4px 4px 0 0; min-height: 5px;">
                        </div>
                        <div style="margin-top: 5px; font-size: 12px; color: #6b7280;">DeepSeek</div>
                    </div>
                </div>
            </div>
            
            <!-- Non-NONE Accuracy Bars -->
            <div>
                <div style="font-size: 14px; color: #6b7280; margin-bottom: 10px;">Non-NONE Accuracy</div>
                <div style="display: flex; align-items: flex-end; height: 120px; gap: 20px; position: relative;">
                    <div style="flex: 1; text-align: center; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%;">
                        <div style="font-weight: bold; color: #3b82f6; margin-bottom: 5px;">
                            {ft_non_none_accuracy:.1%}
                        </div>
                        <div style="background: #3b82f6; height: {
        ft_nn_bar_height
    }%; width: 80%; border-radius: 4px 4px 0 0; min-height: 5px;">
                        </div>
                        <div style="margin-top: 5px; font-size: 12px; color: #6b7280;">Fine-tuned</div>
                    </div>
                    <div style="flex: 1; text-align: center; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%;">
                        <div style="font-weight: bold; color: #6b7280; margin-bottom: 5px;">
                            {ds_non_none_accuracy:.1%}
                        </div>
                        <div style="background: #6b7280; height: {
        ds_nn_bar_height
    }%; width: 80%; border-radius: 4px 4px 0 0; min-height: 5px;">
                        </div>
                        <div style="margin-top: 5px; font-size: 12px; color: #6b7280;">DeepSeek</div>
                    </div>
                </div>
            </div>
        </div>
        
        {
        f'''
        <!-- Performance Metrics Comparison Chart -->
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
            <h4 style="margin: 0 0 20px 0; color: #1f2937; text-align: center;">Performance Metrics Comparison</h4>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{comparison_metrics_img}" style="max-width: 100%; height: auto;">
            </div>
        </div>
        '''
        if comparison_metrics_img
        else ""
    }
        
        <!-- Error Analysis -->
        <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
            <h4 style="margin: 0 0 15px 0; color: #92400e;">Error Analysis</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                <div>
                    <div style="font-size: 14px; color: #92400e; margin-bottom: 5px;">Parsing Errors</div>
                    <div style="font-size: 20px; font-weight: bold; color: #d97706;">
                        FT: {comparison_results["finetuned_parsing_errors"]} | DS: {
        comparison_results["deepseek_parsing_errors"]
    }
                    </div>
                    <div style="font-size: 12px; color: #92400e;">
                        Reduction: {comparison_results["parsing_error_reduction"]}
                    </div>
                </div>
                <div>
                    <div style="font-size: 14px; color: #92400e; margin-bottom: 5px;">Total Errors</div>
                    <div style="font-size: 20px; font-weight: bold; color: #d97706;">
                        FT: {comparison_results["finetuned_total_errors"]} | DS: {
        comparison_results["deepseek_total_errors"]
    }
                    </div>
                </div>
                <div>
                    <div style="font-size: 14px; color: #92400e; margin-bottom: 5px;">Error Type Analysis</div>
                    <div style="font-size: 14px; color: #92400e;">
                        Common: {comparison_results["common_error_types"]}<br>
                        FT Unique: {
        comparison_results["finetuned_unique_error_types"]
    }<br>
                        DS Unique: {comparison_results["deepseek_unique_error_types"]}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Improvement Summary -->
        <div style="background: #e5e7eb; border-radius: 8px; padding: 20px;">
            <h4 style="margin: 0 0 15px 0; color: #1f2937;">Summary</h4>
            <ul style="margin: 0; padding-left: 20px; color: #4b5563;">
                <li>Overall accuracy improved by <strong>{overall_improvement:+.1%}</strong> ({overall_improvement_pct:+.1f}% relative)</li>
                <li>Clean accuracy (excl. parsing errors) improved by <strong>{clean_improvement:+.1%}</strong> ({clean_improvement_pct:+.1f}% relative)</li>
                <li>Precision improved by <strong>{comparison_results.get("precision_improvement", 0):+.1%}</strong></li>
                <li>Recall improved by <strong>{comparison_results.get("recall_improvement", 0):+.1%}</strong></li>
                <li>F1 Score improved by <strong>{comparison_results.get("f1_improvement", 0):+.1%}</strong></li>
                <li>Non-NONE accuracy improved by <strong>{non_none_improvement:+.1%}</strong> ({non_none_improvement_pct:+.1f}% relative)</li>
                <li>Parsing errors reduced by <strong>{
        comparison_results["parsing_error_reduction"]
    }</strong></li>
                <li>Evaluated on <strong>{total_samples}</strong> test samples</li>
            </ul>
        </div>
    </div>
    """

    return HTMLString(html_content)
