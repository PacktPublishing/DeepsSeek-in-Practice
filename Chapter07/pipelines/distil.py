from typing import Optional

from steps.load_data import load_data
from steps.trainer import finetune_model
from zenml import pipeline


@pipeline
def distill_finetuning(
    max_samples: Optional[int] = None,
    model_size: str = "4b",
    filter_none_labels: bool = True,
):
    """Pipeline to load the dataset and finetune the model.

    Args:
        max_samples: Maximum number of samples to use for training
        model_size: Model size to use ('1b', '4b', or '12b')
        filter_none_labels: Whether to filter out examples with 'NONE' classification from training
    """
    # For ZenML validation, we need to explicitly handle None values
    # by only passing parameters when they're not None
    train_kwargs = {}
    if max_samples is not None:
        train_kwargs["max_samples"] = max_samples

    # Pass model_size and filter_none_labels to load_data
    train_dataset, val_dataset, data_viz = load_data(
        model_size=model_size, filter_none_labels=filter_none_labels, **train_kwargs
    )

    # Train the model
    finetune_model(train_dataset, val_dataset, model_size=model_size)
