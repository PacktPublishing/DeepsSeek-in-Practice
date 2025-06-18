# DeepSeek Book Code - Legal Document Classification

This repository contains the code examples for the "DeepSeek Deep Dive" book, focusing on fine-tuning and deploying language models for legal clause classification using the CUAD (Contract Understanding Atticus Dataset).

## Overview

This project demonstrates how to:
1. Generate explanatory rationales for legal classifications using DeepSeek LLMs
2. Fine-tune smaller language models (Gemma 3) using knowledge distillation
3. Evaluate model performance on legal document classification tasks

## Prerequisites

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENROUTER_API_KEY="your_openrouter_api_key"
export HF_API_KEY="your_huggingface_api_key"  # Optional, for dataset uploads
```

### Hardware Requirements

- **1B model**: 8GB+ GPU memory
- **4B model**: 16GB+ GPU memory  
- **12B model**: 24GB+ GPU memory (recommended: 32GB+)

## Project Structure

```
book_code/
├── configs/                          # Configuration files for pipelines
│   ├── config_12b_training.yaml      # 12B model training config
│   ├── config_12b_evaluation.yaml    # 12B model evaluation config
│   └── README.md                     # Config documentation
├── pipelines/                        # ZenML pipeline definitions
│   ├── distil.py                     # Training/fine-tuning pipeline
│   └── evaluate.py                   # Evaluation pipeline
├── steps/                            # Pipeline step implementations
│   ├── load_data.py                  # Data loading and preprocessing
│   ├── trainer.py                    # Model fine-tuning logic
│   └── evaluate.py                   # Model evaluation metrics
├── utils/                            # Utility functions
│   ├── dataset_utils.py              # Dataset handling utilities
│   ├── logging_utils.py              # Logging configuration
│   └── training_utils.py             # Training helpers
└── run.py                            # Main entry point
```

## Pipelines

### 1. Training Pipeline (`distill_finetuning`)

Fine-tunes a Gemma 3 model on legal clause classification using knowledge distillation from DeepSeek rationales.

**Features:**
- Loads pre-labeled data with DeepSeek-generated explanations
- Fine-tunes using LoRA (Low-Rank Adaptation) for efficiency
- Supports multiple model sizes (1B, 4B, 12B)
- Automatically saves fine-tuned models

### 2. Evaluation Pipeline (`evaluation_pipeline`)

Evaluates the fine-tuned model's performance on legal document classification.

**Features:**
- Compares local model predictions with DeepSeek API (i.e. the oracle model that
  we distilled from)
- Calculates precision, recall, and F1 scores
- Generates confusion matrices
- Supports randomized sampling for consistent evaluation

## Quick Start

### Basic Training

```bash
# Train the default 12B model
python run.py

# Train a smaller 4B model
python run.py --model_size 4b

# Train with limited samples for testing
python run.py --model_size 1b --max_samples 100
```

### Basic Evaluation

```bash
# Evaluate the most recently trained model
python run.py --eval_only

# Evaluate with specific settings
python run.py --eval_only --eval_samples 100 --randomize --random_seed 42

# Evaluate a specific model size
python run.py --eval_only --model_size 4b --eval_samples 50
```

### Using Configuration Files

For production runs, use configuration files for reproducible settings:

```bash
# Train with optimized 12B settings
python run.py --training-config configs/config_12b_training.yaml

# Evaluate with predefined settings
python run.py --eval_only --eval-config configs/config_12b_evaluation.yaml

# Train and evaluate with separate configs
python run.py \
  --training-config configs/config_12b_training.yaml \
  --eval-config configs/config_12b_evaluation.yaml
```

## Configuration Options

### Training Options

- `--model_size`: Model size to use (1b, 4b, 12b)
- `--max_samples`: Limit training samples (default: all)
- `--filter-none-labels`: Exclude NONE classifications from training
- `--quiet`: Reduce logging verbosity

### Evaluation Options

- `--eval_samples`: Number of samples to evaluate (default: 50)
- `--exclude_none`: Exclude NONE labels from evaluation
- `--randomize`: Randomize dataset before sampling
- `--random_seed`: Set seed for reproducible sampling

See `configs/README.md` for detailed configuration file documentation.

## Dataset

The project uses the CUAD dataset with DeepSeek-generated rationales, available at:
- Hugging Face: [zenml/cuad-deepseek](https://huggingface.co/datasets/zenml/cuad-deepseek)

The dataset includes:
- Legal contract excerpts
- Binary classification labels for various clause types
- DeepSeek-generated explanations for each classification

## Model Outputs

Fine-tuned models are saved to:
- `gemma3-legal-classifier-1b/` (1B model)
- `gemma3-legal-classifier-4b/` (4B model)
- `gemma3-legal-classifier-12b/` (12B model)

## Tips for Best Results

1. **Memory Management**: For 12B models, use the provided config files which include optimized batch sizes and gradient accumulation settings.

2. **Evaluation Consistency**: Use `--randomize` with a fixed `--random_seed` for comparable evaluation runs.

3. **Excluding NONE Labels**: Both training and evaluation exclude NONE labels by default, as they represent the absence of a clause type.

4. **GPU Selection**: The code automatically selects available GPUs. For multi-GPU systems, set `CUDA_VISIBLE_DEVICES=0` to use a specific GPU.

## Troubleshooting

### Out of Memory Errors

- Reduce batch size in the configuration file
- Increase gradient accumulation steps to maintain effective batch size
- Use a smaller model size

### Slow Training

- Ensure you're using a GPU: check with `nvidia-smi`
- Disable logging for faster training: use `--quiet` flag
- Reduce the number of training samples with `--max_samples`

### API Rate Limits

- The evaluation pipeline includes rate limiting for DeepSeek API calls
- Adjust `deepseek_max_workers` in the evaluation config if needed
