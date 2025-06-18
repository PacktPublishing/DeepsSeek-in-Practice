import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

# IMPORTANT: Set unsloth environment variables BEFORE importing any modules
# that might use unsloth. This must happen at module level.
if "--quiet" in sys.argv:
    os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "0")
    os.environ.setdefault("UNSLOTH_COMPILE_DEBUG", "0")
else:
    os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")
    os.environ.setdefault("UNSLOTH_COMPILE_DEBUG", "0")

# Now safe to import other modules
from pipelines.distil import distill_finetuning
from pipelines.evaluate import evaluation_pipeline
from utils.custom_exceptions import handle_exception
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all CLI options
            for training and evaluation pipelines.
    """
    parser = argparse.ArgumentParser(
        description="Run legal document classification training or evaluation pipelines"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["1b", "4b", "12b"],
        default="12b",
        help="Model size to use (default: 12b)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum training samples",
    )
    parser.add_argument(
        "--filter-none-labels",
        action="store_true",
        default=True,
        help="Filter out 'NONE' classifications from training (default: True)",
    )
    parser.add_argument(
        "--include-none-labels",
        action="store_true",
        help="Include 'NONE' classifications in training (overrides filter-none-labels)",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Skip training, only evaluate"
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=50,
        help="Number of evaluation samples (default: 50)",
    )
    parser.add_argument(
        "--exclude_none",
        action="store_true",
        default=True,
        help="Exclude 'NONE' labels from evaluation (default: True)",
    )
    parser.add_argument(
        "--include_none_eval",
        action="store_true",
        help="Include 'NONE' labels in evaluation (overrides exclude_none)",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the dataset before sampling",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (requires --randomize)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (deprecated, use --training-config or --eval-config)",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        help="Path to YAML config file for training pipeline",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        help="Path to YAML config file for evaluation pipeline",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser


def run_evaluation(args: argparse.Namespace, config_path: Optional[str] = None) -> Any:
    """Run the evaluation pipeline with given arguments.

    Args:
        args: Parsed command line arguments containing evaluation parameters.
        config_path: Path to the evaluation config file (if any).

    Returns:
        Any: Results from the evaluation pipeline execution.
    """
    logger.info("Starting evaluation pipeline...")

    if config_path:
        logger.info(f"Using evaluation configuration from: {config_path}")
        return evaluation_pipeline.with_options(
            config_path=config_path,
            run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )()
    else:
        # Handle the logic: if include_none_eval is set, don't exclude
        exclude_none = args.exclude_none and not args.include_none_eval
        return evaluation_pipeline(
            num_samples=args.eval_samples,
            exclude_none=exclude_none,
            randomize=args.randomize,
            random_seed=args.random_seed,
            model_size=args.model_size,
            use_local_model=True,
        )


def run_training(args: argparse.Namespace, config_path: Optional[str] = None) -> Any:
    """Run the training pipeline with given arguments.

    Args:
        args: Parsed command line arguments containing training parameters
            such as max_samples, model_size, and filter_none_labels.
        config_path: Path to the training config file (if any).

    Returns:
        Any: Results from the training pipeline execution.
    """
    logger.info("Starting training pipeline...")

    if config_path:
        logger.info(f"Using training configuration from: {config_path}")
        return distill_finetuning.with_options(config_path=config_path)()
    else:
        # Handle the logic: if include-none-labels is set, don't filter
        filter_none = args.filter_none_labels and not args.include_none_labels
        return distill_finetuning(
            max_samples=args.max_samples,
            model_size=args.model_size,
            filter_none_labels=filter_none,
        )


def run_post_training_evaluation(
    args: argparse.Namespace, eval_config_path: Optional[str] = None
) -> Optional[Any]:
    """Run evaluation after training completion.

    This function attempts to run evaluation after training has finished.
    If evaluation fails, it logs helpful instructions for manual execution
    rather than failing the entire pipeline.

    Args:
        args: Parsed command line arguments containing evaluation parameters.
        eval_config_path: Path to the evaluation config file (if any).

    Returns:
        Optional[Any]: Results from evaluation pipeline if successful, None if failed.
    """
    try:
        logger.info("Running post-training evaluation...")
        return run_evaluation(args, config_path=eval_config_path)
    except Exception as e:
        logger.error(f"Post-training evaluation failed: {e}")
        logger.info(
            "Training completed successfully. Run evaluation manually with:\n"
            f"  python run.py --eval_only --model_size {args.model_size} --eval_samples 500"
        )
        return None


def main() -> None:
    """Main entry point for the training/evaluation pipeline.

    Parses command line arguments, configures logging, and orchestrates
    the execution of either evaluation-only or training+evaluation workflows.

    Exits with status code 1 if any unhandled exception occurs.
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(args.log_level, args.quiet)

    # Determine which config files to use
    training_config = args.training_config or args.config
    eval_config = args.eval_config or args.config

    try:
        if args.eval_only:
            # Evaluation only
            run_evaluation(args, config_path=eval_config)
            logger.info("Evaluation completed successfully")

        else:
            # Training + optional evaluation
            run_training(args, config_path=training_config)
            logger.info("Training completed successfully")

            # Run evaluation after training
            run_post_training_evaluation(args, eval_config_path=eval_config)

    except Exception as e:
        handle_exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
