from typing import Optional

from steps.compare_models import compare_models
from steps.evaluate import evaluate_model
from steps.evaluate_deepseek_base import evaluate_deepseek_base
from steps.load_data import load_test_data
from zenml import pipeline


@pipeline
def evaluation_pipeline(
    num_samples: Optional[int] = 50,
    exclude_none: bool = True,
    randomize: bool = False,
    random_seed: Optional[int] = None,
    model_size: str = "4b",
    use_local_model: bool = True,
    deepseek_max_workers: int = 5,
):
    """Pipeline to evaluate a fine-tuned model and compare with DeepSeek base model.

    Args:
        num_samples: Number of test samples to evaluate
        exclude_none: Whether to exclude examples labeled as "NONE" from evaluation
        randomize: Whether to randomize the dataset before sampling
        random_seed: Random seed for reproducible sampling (None for random)
        model_size: Model size to use ('1b', '4b', or '12b')
        use_local_model: If True, load from local path; if False, load from HF Hub
        deepseek_max_workers: Maximum number of concurrent API calls for DeepSeek evaluation
    """
    # Load test data
    test_dataset = load_test_data(
        num_samples=num_samples,
        exclude_none=exclude_none,
        randomize=randomize,
        random_seed=random_seed,
    )

    # Evaluate fine-tuned model
    finetuned_results, finetuned_viz = evaluate_model(
        test_dataset=test_dataset,
        model_size=model_size,
        use_local_model=use_local_model,
    )

    # Evaluate DeepSeek base model
    deepseek_results, deepseek_viz = evaluate_deepseek_base(
        test_dataset=test_dataset,
        max_workers=deepseek_max_workers,
    )

    # Compare models
    comparison_results, comparison_viz = compare_models(
        finetuned_results=finetuned_results,
        deepseek_results=deepseek_results,
    )

    return finetuned_results, deepseek_results, comparison_results
