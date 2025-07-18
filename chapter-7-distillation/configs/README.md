# Configuration Files

This directory contains YAML configuration files for training and evaluating legal document classification models. Each pipeline has its own dedicated configuration file.

## Available Configurations

### Training Pipeline Configurations

- **`config_12b_training.yaml`** - Optimized configuration for training the 12B Gemma model with the `distill_finetuning` pipeline
  - Uses lower learning rate (1e-4) for stability
  - Reduced batch size (1) with increased gradient accumulation (8)
  - Larger LoRA rank (16) and alpha (32) for better capacity
  - Cosine learning rate scheduler for smooth decay
  - Extended warmup steps (100) for stability

### Evaluation Pipeline Configurations

- **`config_12b_evaluation.yaml`** - Configuration for evaluating the 12B model with the `evaluation_pipeline`
  - Evaluates 100 samples with randomization
  - Seeds set for reproducible results
  - Concurrent DeepSeek API calls limited to 5
  - Excludes NONE labels from evaluation

## Usage Examples

### Training with Configuration

```bash
# Train 12B model with optimized settings
python run.py --training-config configs/config_12b_training.yaml

# Or using the deprecated --config flag
python run.py --config configs/config_12b_training.yaml
```

### Evaluation with Configuration

```bash
# Evaluate 12B model with config settings
python run.py --eval_only --eval-config configs/config_12b_evaluation.yaml

# Or using the deprecated --config flag
python run.py --eval_only --config configs/config_12b_evaluation.yaml
```

### Training + Evaluation with Separate Configs

```bash
# Train and then evaluate with different configurations
python run.py \
  --training-config configs/config_12b_training.yaml \
  --eval-config configs/config_12b_evaluation.yaml
```

### Mixing Config Files and CLI Arguments

Command-line arguments can override specific parameters from config files:

```bash
# Use config but override sample count
python run.py --eval_only \
  --eval-config configs/config_12b_evaluation.yaml \
  --eval_samples 200

# Use training config but override model size
python run.py \
  --training-config configs/config_12b_training.yaml \
  --model_size 4b
```

## Configuration Structure

### Pipeline Parameters
- Located under `parameters:` section
- Define inputs to the entire pipeline
- Must match the pipeline function signature

### Step Parameters
- Located under `steps:` → `[step_name]:` → `parameters:`
- Configure individual pipeline steps
- Override default step parameters

### Training Configuration
- Located under `steps:` → `finetune_model:` → `extra:` → `training_config:`
- Controls model training hyperparameters
- Passed to the trainer via ZenML's step context

## Creating New Configurations

When creating configurations for different model sizes:

1. Copy an existing config file as a template
2. Adjust the following key parameters:
   - `model_size`: "1b", "4b", or "12b"
   - `learning_rate`: Smaller for larger models
   - `per_device_train_batch_size`: Smaller for larger models
   - `gradient_accumulation_steps`: Increase to maintain effective batch size
   - `lora_rank` and `lora_alpha`: Increase for larger models

## Key Differences for 12B Model

The 12B configuration includes several optimizations compared to smaller models:

1. **Memory Management**
   - Batch size reduced to 1 (from 2)
   - Gradient accumulation increased to 8 (from 4)
   - Maintains effective batch size of 8

2. **Learning Dynamics**
   - Lower learning rate (1e-4 vs 2e-4)
   - Cosine scheduler instead of linear
   - Extended warmup (100 vs 5 steps)

3. **Model Capacity**
   - Increased LoRA rank (16 vs 8)
   - Proportional alpha (32 vs 8)

These adjustments help the larger model train more stably and effectively.