# Default constants
DEFAULT_DATASET_ID = "zenml/cuad-deepseek"
DEFAULT_OUTPUT_DIR = "data/processed_cuad/hf_dataset"

# Model base constants
GEMMA_1B_MODEL_BASE = "unsloth/gemma-3-1b-it-bnb-4bit"
GEMMA_4B_MODEL_BASE = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
GEMMA_12B_MODEL_BASE = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"

# Output directories
MODEL_OUTPUT_DIR_1B = "./gemma3-1b-legal-classifier"
MODEL_OUTPUT_DIR_4B = "./gemma3-4b-legal-classifier"
MODEL_OUTPUT_DIR_12B = "./gemma3-12b-legal-classifier"

# HF model repositories
HF_MODEL_REPO_1B = "zenml/deepseek-cuad-gemma-3-1b-it-bnb-4bit"
HF_MODEL_REPO_4B = "zenml/deepseek-cuad-gemma-3-4b-it-bnb-4bit"
HF_MODEL_REPO_12B = "zenml/deepseek-cuad-gemma-3-12b-it-bnb-4bit"

# Sequence lengths
MAX_SEQ_LENGTH_1B = 1024
MAX_SEQ_LENGTH_4B = 2048
MAX_SEQ_LENGTH_12B = 4096

# Training parameters
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4

# Model loading configuration
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
FULL_FINETUNING = False

# LoRA configuration constants
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0
LORA_BIAS = "none"
LORA_RANDOM_STATE = 3407

# Fine-tuning layers configuration
FINETUNE_VISION_LAYERS = False  # Turn off for just text
FINETUNE_LANGUAGE_LAYERS = True  # Should leave on
FINETUNE_ATTENTION_MODULES = True  # Attention good for training
FINETUNE_MLP_MODULES = True  # Should leave on always

# Training configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUMULATION,
    "warmup_steps": 5,
    "num_train_epochs": 2,
    "learning_rate": 2e-4,
    "logging_steps": 20,  # Log loss/lr every 20 steps
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": LORA_RANDOM_STATE,
    "report_to": "none",
    "logging_first_step": True,  # Log the first step
    "logging_nan_inf_filter": True,  # Filter out nan/inf from logs
}

# Chat template markers
USER_TURN_MARKER = "<start_of_turn>user\n"
MODEL_TURN_MARKER = "<start_of_turn>model\n"

# Data paths
DATA_DIR = "data/processed_cuad/hf_dataset"
TRAIN_DATA_PATH = f"{DATA_DIR}/train_data_with_rationales.jsonl"
VAL_DATA_PATH = f"{DATA_DIR}/validation_data_with_rationales.jsonl"
TEST_DATA_PATH = f"{DATA_DIR}/test_data_with_rationales.jsonl"

# OpenRouter model for base evaluation comparisons
OPEN_ROUTER_EVALS_BASE_MODEL_NAME = "openrouter/deepseek/deepseek-r1-0528"

# The legal clause categories from CUAD
LEGAL_LABEL_SCHEMA = [
    "Anti-Assignment",
    "Audit Rights",
    "Cap on Liability",
    "Change of Control",
    "Competitive Restriction Exception",
    "Covenant Not to Sue",
    "Effective Date",
    "Exclusivity",
    "Expiration Date",
    "Governing Law",
    "Insurance",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "License Grant",
    "Liquidated Damages",
    "Minimum Commitment",
    "Most Favored Nation",
    "Non-Compete",
    "Non-Disparagement",
    "Non-Solicit of Customers",
    "Non-Solicit of Employees",
    "Non-Transferable License",
    "Notice to Terminate Renewal",
    "Post-Termination Services",
    "Price Restriction",
    "Renewal Term",
    "Revenue/Profit Sharing",
    "Right of First Refusal",
    "Source Code Escrow",
    "Termination for Convenience",
    "Third Party Beneficiary",
    "Uncapped Liability",
    "Volume Restriction",
    "Warranty Duration",
    "Affiliate License-Licensee",
    "Affiliate License-Licensor",
    "Irrevocable License",
    "NONE",
]
