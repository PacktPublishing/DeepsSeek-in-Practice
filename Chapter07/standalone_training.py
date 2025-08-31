"""Standalone Gemma‑3 LoRA fine‑tuning script (no ZenML).

Key features:
- Loads a Hugging Face dataset and ensures train/val/test splits
- Builds Gemma‑3 chat-formatted training texts (user JSON ↔ assistant JSON)
- Uses Unsloth FastModel in 4-bit (by default) with LoRA adapters
- Trains with TRL's SFTTrainer on assistant responses only
- Saves adapters locally and (optionally) pushes to Hugging Face Hub
- Optional quick sanity evaluation on the test split

Example:
    python standalone_training.py \
      --dataset-id zenml/cuad-deepseek \
      --model-size 12b \
      --filter-none-labels \
      --val-size 0.1 --test-size 0.1 \
      --output-dir ./gemma3-12b-legal-classifier \
      --num-train-epochs 2 \
      --per-device-train-batch-size 1 \
      --gradient-accumulation-steps 8 \
      --learning-rate 1e-4 \
      --warmup-steps 100 --lr-scheduler-type cosine \
      --lora-rank 16 --lora-alpha 32 \
      --push-to-hub --hf-repo zenml/deepseek-cuad-gemma-3-12b-it-bnb-4bit \
      --eval-after-train --eval-max-samples 100 --quiet
"""

# ---------------------------------------------------------
# 1) Set Unsloth env *before* any import that might use it
# ---------------------------------------------------------
import os, sys

if "--quiet" in sys.argv:
    os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "0")
    os.environ.setdefault("UNSLOTH_COMPILE_DEBUG", "0")
else:
    os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")
    os.environ.setdefault("UNSLOTH_COMPILE_DEBUG", "0")

# --------------------------
# 2) Standard library imports
# --------------------------
import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --------------------------
# 3) Third-party dependencies
# --------------------------
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import logging as hf_logging
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch

# -----------------------------------------------
# 4) Standalone defaults (formerly constants.py)
#    No intra-repo imports; safe to run standalone.
# -----------------------------------------------

# Default dataset
DEFAULT_DATASET_ID = "zenml/cuad-deepseek"

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

# Training parameters (used to seed TRAINING_CONFIG defaults)
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4

# Model loading configuration
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
FULL_FINETUNING = False

# LoRA configuration defaults
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0
LORA_BIAS = "none"
LORA_RANDOM_STATE = 3407

# Fine-tuning layers configuration
FINETUNE_VISION_LAYERS = False  # Text-only
FINETUNE_LANGUAGE_LAYERS = True
FINETUNE_ATTENTION_MODULES = True
FINETUNE_MLP_MODULES = True

# Training configuration (defaults consumed by argparse)
TRAINING_CONFIG = {
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUMULATION,
    "warmup_steps": 5,
    "num_train_epochs": 2,
    "learning_rate": 2e-4,
    "logging_steps": 20,           # Log loss/lr every 20 steps
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": LORA_RANDOM_STATE,
    "report_to": "none",
    "logging_first_step": True,    # Log the first step
    "logging_nan_inf_filter": True # Filter out nan/inf from logs
}

# Chat template markers
USER_TURN_MARKER = "<start_of_turn>user\n"
MODEL_TURN_MARKER = "<start_of_turn>model\n"

# Legal clause categories (CUAD)
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

# -------------------------
# 5) Logging configuration
# -------------------------
LOGGER = logging.getLogger("standalone_ft")


def setup_logging(log_level: str = "INFO", quiet: bool = False) -> None:
    """Configure Python & HF logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if quiet:
        hf_logging.set_verbosity_error()
        os.environ["TQDM_MININTERVAL"] = "10"
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("unsloth").setLevel(logging.WARNING)
        logging.getLogger("accelerate").setLevel(logging.ERROR)
    else:
        hf_logging.set_verbosity_warning()
        os.environ["TQDM_MININTERVAL"] = "5"
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("unsloth").setLevel(logging.INFO)
        logging.getLogger("accelerate").setLevel(logging.WARNING)


# ---------------------------------
# 6) Model/LORA configuration types
# ---------------------------------
@dataclass
class ModelConfig:
    base_model_id: str
    max_seq_length: int
    output_dir: str
    hf_repo: Optional[str] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class LoraParams:
    r: int = LORA_RANK
    alpha: int = LORA_ALPHA
    dropout: float = float(LORA_DROPOUT)
    bias: str = LORA_BIAS
    random_state: int = LORA_RANDOM_STATE


def get_model_config(
    model_size: str, output_dir: Optional[str], hf_repo: Optional[str]
) -> ModelConfig:
    """Return a ModelConfig populated from repo constants, with optional overrides."""
    size = model_size.lower().strip()
    if size not in {"1b", "4b", "12b"}:
        raise ValueError(
            f"Invalid --model-size '{model_size}'. Expected one of: 1b, 4b, 12b"
        )

    if size == "1b":
        cfg = ModelConfig(
            base_model_id=GEMMA_1B_MODEL_BASE,
            max_seq_length=MAX_SEQ_LENGTH_1B,
            output_dir=output_dir or MODEL_OUTPUT_DIR_1B,
            hf_repo=hf_repo or HF_MODEL_REPO_1B,
            tokenizer_kwargs={
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "additional_special_tokens": ["<bos>"],
            },
        )
    elif size == "4b":
        cfg = ModelConfig(
            base_model_id=GEMMA_4B_MODEL_BASE,
            max_seq_length=MAX_SEQ_LENGTH_4B,
            output_dir=output_dir or MODEL_OUTPUT_DIR_4B,
            hf_repo=hf_repo or HF_MODEL_REPO_4B,
            tokenizer_kwargs=None,
        )
    else:  # 12b
        cfg = ModelConfig(
            base_model_id=GEMMA_12B_MODEL_BASE,
            max_seq_length=MAX_SEQ_LENGTH_12B,
            output_dir=output_dir or MODEL_OUTPUT_DIR_12B,
            hf_repo=hf_repo or HF_MODEL_REPO_12B,
            tokenizer_kwargs=None,
        )
    return cfg


def create_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Create tokenizer; handle 1B special tokens."""
    if cfg.tokenizer_kwargs:
        tok = AutoTokenizer.from_pretrained(
            cfg.base_model_id, cache_dir=cfg.output_dir, **cfg.tokenizer_kwargs
        )
        # Unify padding behavior for chat
        if (
            getattr(tok, "pad_token", None) is None
            and getattr(tok, "eos_token", None) is not None
        ):
            tok.pad_token = tok.eos_token
    else:
        tok = AutoTokenizer.from_pretrained(cfg.base_model_id)
    return tok


def setup_model_and_tokenizer(
    cfg: ModelConfig,
    lora: LoraParams,
    load_in_4bit: bool,
    load_in_8bit: bool,
    full_finetuning: bool,
) -> Tuple[Any, PreTrainedTokenizerBase]:
    """Load FastModel, apply chat template, and attach LoRA adapters."""
    LOGGER.info("Loading base model: %s", cfg.base_model_id)
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=bool(load_in_4bit),
        load_in_8bit=bool(load_in_8bit),
        full_finetuning=bool(full_finetuning),
    )

    # Ensure Gemma‑3 chat template is set on tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # Attach LoRA adapters (text-only modules)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=bool(FINETUNE_VISION_LAYERS),
        finetune_language_layers=bool(FINETUNE_LANGUAGE_LAYERS),
        finetune_attention_modules=bool(FINETUNE_ATTENTION_MODULES),
        finetune_mlp_modules=bool(FINETUNE_MLP_MODULES),
        r=int(lora.r),
        lora_alpha=int(lora.alpha),
        lora_dropout=float(lora.dropout),
        bias=str(lora.bias),
        random_state=int(lora.random_state),
    )
    LOGGER.info("✓ Model + LoRA ready (r=%d, alpha=%d)", lora.r, lora.alpha)
    return model, tokenizer


# -------------------
# 7) Data preparation
# -------------------
def load_hf_dataset(dataset_id: str) -> DatasetDict:
    try:
        return load_dataset(dataset_id)  # type: ignore[no-any-return]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_id}': {type(e).__name__}: {e}"
        ) from e


def resolve_splits(
    ds: DatasetDict, val_size: float, test_size: float, seed: int
) -> DatasetDict:
    """Ensure we have train/validation/test splits."""
    if not (
        0.0 <= val_size < 1.0 and 0.0 <= test_size < 1.0 and val_size + test_size < 1.0
    ):
        raise ValueError(
            f"Invalid val/test sizes: val={val_size}, test={test_size} (must sum to < 1.0)"
        )

    keys = set(ds.keys())
    # Already good
    if {"train", "validation", "test"}.issubset(keys):
        return DatasetDict(
            train=ds["train"], validation=ds["validation"], test=ds["test"]
        )

    # If only train present
    if (
        keys == {"train"}
        or keys.issuperset({"train"})
        and "validation" not in keys
        and "test" not in keys
    ):
        # First carve out test from train
        first = ds["train"].train_test_split(test_size=test_size, seed=seed)
        # Now carve out validation from the remaining train
        remaining = first["train"]
        # The fraction of validation relative to remaining
        relative_val = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
        second = remaining.train_test_split(test_size=relative_val, seed=seed)
        return DatasetDict(
            train=second["train"], validation=second["test"], test=first["test"]
        )

    # If train + validation but no test
    if "train" in keys and "validation" in keys and "test" not in keys:
        first = ds["train"].train_test_split(test_size=test_size, seed=seed)
        return DatasetDict(
            train=first["train"], validation=ds["validation"], test=first["test"]
        )

    # If train + test but no validation
    if "train" in keys and "test" in keys and "validation" not in keys:
        rel_val = val_size
        split = ds["train"].train_test_split(test_size=rel_val, seed=seed)
        return DatasetDict(
            train=split["train"], validation=split["test"], test=ds["test"]
        )

    # Heuristics: if only 'validation' and 'test', create train by combining (rare)
    if "train" not in keys and "validation" in keys and "test" in keys:
        concat = Dataset.from_dict({})  # empty placeholder
        # Fallback: use validation as train (small datasets)
        LOGGER.warning("Dataset lacks a 'train' split; using 'validation' as train.")
        return DatasetDict(
            train=ds["validation"], validation=ds["validation"], test=ds["test"]
        )

    raise RuntimeError(f"Cannot resolve splits from keys: {sorted(keys)}")


def limit_and_filter(
    ds: DatasetDict, max_samples: Optional[int], filter_none: bool
) -> DatasetDict:
    """Apply per-split NONE filtering and optional subsetting."""
    out: Dict[str, Dataset] = {}
    for name in ["train", "validation", "test"]:
        if name not in ds:
            continue
        split = ds[name]
        if filter_none and "label" in split.column_names:
            split = split.filter(
                lambda x: (x.get("label", "").strip().upper() != "NONE")
            )
        if max_samples is not None and max_samples > 0 and len(split) > max_samples:
            split = split.select(range(max_samples))
        out[name] = split
        LOGGER.info(
            "Split %-10s → %5d examples%s",
            name,
            len(split),
            " (filtered NONE)" if filter_none else "",
        )
    return DatasetDict(**out)


def _build_prompt_json(item: Dict[str, Any], valid_labels: List[str]) -> str:
    """Create the base JSON prompt."""
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
                "clause": (item.get("clause") or "").strip(),
                "clause_with_context": (item.get("clause_with_context") or "").strip(),
                "contract_type": (item.get("contract_type") or "").strip(),
            },
        },
        ensure_ascii=False,
    )


def to_conversations(
    ds: Dataset, include_none: bool, label_schema: List[str]
) -> Dataset:
    """Add a 'conversations' column in Gemma‑3 chat shape (user JSON / assistant JSON)."""
    required = {"clause", "clause_with_context", "label"}
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise RuntimeError(
            f"Dataset missing required columns: {missing}. "
            "Expected at least: clause, clause_with_context, label (+ optional rationale, contract_type)."
        )

    valid_labels = (
        label_schema if include_none else [l for l in label_schema if l != "NONE"]
    )

    def _map(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        convs: List[List[Dict[str, Any]]] = []
        n = len(batch["clause"])
        for i in range(n):
            item = {k: batch.get(k, [None] * n)[i] for k in batch.keys()}
            # Build user (prompt JSON)
            input_json = _build_prompt_json(item, valid_labels)
            # Build assistant (rationale + label) – rationale may be missing
            output_json = json.dumps(
                {
                    "rationale": (item.get("rationale") or "").strip(),
                    "label": (item.get("label") or "NONE").strip(),
                },
                ensure_ascii=False,
            )
            convs.append(
                [
                    {"role": "user", "content": [{"type": "text", "text": input_json}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": output_json}],
                    },
                ]
            )
        return {"conversations": convs}

    return ds.map(
        _map,
        batched=True,
        remove_columns=[c for c in ds.column_names if c != "conversations"],
    )


def apply_chat_template_to_ds(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    tokenize: bool = False,
) -> Dataset:
    """Apply tokenizer.apply_chat_template over 'conversations' → add 'text'."""

    def _apply(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = tokenizer.apply_chat_template(batch["conversations"], tokenize=tokenize)
        return {"text": texts}

    return ds.map(_apply, batched=True)


# ---------------
# 8) Trainer utils
# ---------------
def create_trainer(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    train: Dataset,
    val: Dataset,
    cfg: ModelConfig,
    sft_kwargs: Dict[str, Any],
) -> SFTTrainer:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        eval_dataset=val,
        args=SFTConfig(
            dataset_text_field="text", output_dir=cfg.output_dir, **sft_kwargs
        ),
    )
    # Train only on assistant responses (Gemma‑3 markers from constants.py)
    trainer = train_on_responses_only(
        trainer, instruction_part=USER_TURN_MARKER, response_part=MODEL_TURN_MARKER
    )
    return trainer


def _gpu_summary(prefix: str = "Task") -> None:
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        used = torch.cuda.max_memory_reserved() / (1024**3)
        LOGGER.info(
            "GPU: %s | Cap: %.2f GB | Peak reserved after %s: %.2f GB",
            dev.name,
            dev.total_memory / (1024**3),
            prefix,
            used,
        )


def train_and_log(trainer: SFTTrainer) -> Any:
    LOGGER.info("Starting training ...")
    try:
        out = trainer.train()
        LOGGER.info("Training finished. Stats: %s", getattr(out, "metrics", {}))
    finally:
        _gpu_summary("training")
    return out


# -----------------------------
# 9) Save locally & push (opt.)
# -----------------------------
def save_and_maybe_push(
    model: Any, tokenizer: PreTrainedTokenizerBase, cfg: ModelConfig, push_to_hub: bool
) -> None:
    save_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    LOGGER.info("Saving adapters & tokenizer to: %s", save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    if push_to_hub:
        repo = cfg.hf_repo
        if not repo:
            raise RuntimeError(
                "--push-to-hub was set, but no --hf-repo was provided and no default repo is configured."
            )
        LOGGER.info("Pushing LoRA adapters to Hub repo: %s", repo)
        try:
            model.push_to_hub(repo)
            tokenizer.push_to_hub(repo)
            LOGGER.info("✓ Pushed to Hub: %s", repo)
        except Exception as e:
            raise RuntimeError(
                f"Failed to push to Hub: {type(e).__name__}: {e}\n"
                "Tip: run `huggingface-cli login` or set HF_TOKEN."
            ) from e


# -------------------------------
# 10) Optional quick sanity eval
# -------------------------------
def extract_prediction(response: str, log_failures: bool = True) -> Optional[str]:
    """Robustly pull a 'label' from a JSON-looking response string."""
    try:
        # Strip fenced code blocks
        if "```json" in response:
            s, e = (
                response.find("```json") + len("```json"),
                response.find("```", response.find("```json") + 7),
            )
            if e > s:
                response = response[s:e].strip()
        elif "```" in response and "{" in response:
            s, e = (
                response.find("```") + 3,
                response.find("```", response.find("```") + 3),
            )
            if e > s:
                response = response[s:e].strip()

        js = response[response.find("{") : response.rfind("}") + 1]
        if js and js.startswith("{") and js.endswith("}"):
            parsed = json.loads(js)
            if isinstance(parsed, dict) and "label" in parsed:
                return str(parsed["label"]).strip()
            if log_failures:
                LOGGER.debug("No 'label' in parsed JSON: %s", parsed)
        elif log_failures:
            LOGGER.debug(
                "No JSON structure found in response (head): %s", response[:200]
            )
    except Exception as e:
        if log_failures:
            LOGGER.debug("extract_prediction failed: %s | head=%s", e, response[:300])
    return None


def quick_eval(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    test_ds: Dataset,
    max_samples: int = 100,
    exclude_none: bool = True,
) -> Dict[str, Any]:
    """Tiny post-train eval: generate JSON, parse 'label', compute simple accuracies."""
    if len(test_ds) == 0:
        LOGGER.warning("No test examples available for quick evaluation.")
        return {
            "accuracy": 0.0,
            "count": 0,
            "non_none_accuracy": 0.0,
            "non_none_count": 0,
        }

    items: List[Dict[str, Any]] = [
        test_ds[i] for i in range(min(max_samples, len(test_ds)))
    ]
    if exclude_none:
        items = [
            it for it in items if (it.get("label") or "").strip().upper() != "NONE"
        ]

    valid_labels = (
        [l for l in LEGAL_LABEL_SCHEMA if l != "NONE"]
        if exclude_none
        else LEGAL_LABEL_SCHEMA
    )

    correct = 0
    non_none_total = 0
    non_none_correct = 0

    for idx, it in enumerate(items, start=1):
        prompt_json = _build_prompt_json(it, valid_labels)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt_json}]}
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        try:
            outputs = model.generate(
                **inputs, max_new_tokens=1024, temperature=0.1, top_p=0.95, top_k=64
            )
            decoded = tokenizer.batch_decode(outputs)[0]
            response = decoded[len(text) :].strip()
        except Exception as e:
            response = f"GENERATION_ERROR: {type(e).__name__}: {e}"

        pred = extract_prediction(response, log_failures=False)
        true = (it.get("label") or "").strip()

        if pred is not None and pred == true:
            correct += 1

        if true != "NONE":
            non_none_total += 1
            if pred is not None and pred == true:
                non_none_correct += 1

        if idx % 10 == 0 or idx == len(items):
            LOGGER.info("Quick eval progress: %d/%d", idx, len(items))

    total = len(items)
    acc = (correct / total) if total else 0.0
    nn_acc = (non_none_correct / non_none_total) if non_none_total else 0.0
    LOGGER.info(
        "Quick eval: accuracy=%.3f (%d/%d), non-NONE=%.3f (%d/%d)",
        acc,
        correct,
        total,
        nn_acc,
        non_none_correct,
        non_none_total,
    )
    return {
        "accuracy": acc,
        "count": total,
        "non_none_accuracy": nn_acc,
        "non_none_count": non_none_total,
    }


# ---------------
# 11) CLI parsing
# ---------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone Gemma‑3 LoRA fine‑tuning (no ZenML)."
    )

    # Data
    p.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="HF dataset ID (e.g., zenml/cuad-deepseek)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit per split (after filtering)",
    )
    p.add_argument(
        "--filter-none-labels",
        action="store_true",
        help="Drop training examples with label == 'NONE'",
    )
    p.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split size if not present",
    )
    p.add_argument(
        "--test-size", type=float, default=0.1, help="Test split size if not present"
    )
    p.add_argument(
        "--seed", type=int, default=LORA_RANDOM_STATE, help="Random seed for splits"
    )

    # Model
    p.add_argument("--model-size", type=str, choices=["1b", "4b", "12b"], default="12b")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override default output dir for this model size",
    )
    p.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Override default HF repo for this model size",
    )
    p.add_argument("--load-in-4bit", action="store_true", default=LOAD_IN_4BIT)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--load-in-8bit", action="store_true", default=LOAD_IN_8BIT)
    p.add_argument("--full-finetuning", action="store_true", default=FULL_FINETUNING)

    # LoRA
    p.add_argument("--lora-rank", type=int, default=LORA_RANK)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=float(LORA_DROPOUT))
    p.add_argument(
        "--lora-bias", type=str, default=LORA_BIAS, choices=["none", "lora_only", "all"]
    )

    # Training (TRL/SFT)
    p.add_argument(
        "--num-train-epochs",
        type=int,
        default=TRAINING_CONFIG.get("num_train_epochs", 2),
    )
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=TRAINING_CONFIG.get("per_device_train_batch_size", 2),
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=TRAINING_CONFIG.get("gradient_accumulation_steps", 4),
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=TRAINING_CONFIG.get("learning_rate", 2e-4),
    )
    p.add_argument(
        "--warmup-steps", type=int, default=TRAINING_CONFIG.get("warmup_steps", 5)
    )
    p.add_argument(
        "--lr-scheduler-type",
        type=str,
        default=TRAINING_CONFIG.get("lr_scheduler_type", "linear"),
    )
    p.add_argument(
        "--weight-decay", type=float, default=TRAINING_CONFIG.get("weight_decay", 0.01)
    )
    p.add_argument(
        "--optim", type=str, default=TRAINING_CONFIG.get("optim", "adamw_8bit")
    )
    p.add_argument(
        "--logging-steps", type=int, default=TRAINING_CONFIG.get("logging_steps", 20)
    )
    # Optional SFT args (None by default)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--save-total-limit", type=int, default=None)
    p.add_argument("--load-best-model-at-end", action="store_true", default=False)

    # I/O and Hub
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After training, push adapters & tokenizer to HF Hub",
    )

    # Ops
    p.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # Eval (optional)
    p.add_argument(
        "--eval-after-train",
        action="store_true",
        help="Run quick sanity eval on test split",
    )
    p.add_argument("--eval-max-samples", type=int, default=100)
    p.add_argument(
        "--exclude-none-eval",
        action="store_true",
        help="Exclude NONE labels from quick eval",
    )

    return p.parse_args()


# ---------------
# 12) Orchestration
# ---------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, args.quiet)

    # Compose config objects
    cfg = get_model_config(args.model_size, args.output_dir, args.hf_repo)
    lora = LoraParams(
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        bias=args.lora_bias,
        random_state=args.seed,
    )

    # Load model + tokenizer first so we can use the exact tokenizer for chat templating
    model, tokenizer = setup_model_and_tokenizer(
        cfg=cfg,
        lora=lora,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
    )

    # Load & prepare data
    raw = load_hf_dataset(args.dataset_id)
    splits = resolve_splits(
        raw, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )

    # Keep a raw copy of test split for quick_eval (with/without NONE filtering later)
    raw_test = splits["test"]

    # Apply training-time filtering/limiting (per split)
    filtered = limit_and_filter(
        splits, max_samples=args.max_samples, filter_none=args.filter_none_labels
    )

    # Build chat 'conversations' and then 'text' for SFTTrainer
    include_none_in_valid_labels = not args.filter_none_labels
    train_conv = to_conversations(
        filtered["train"], include_none_in_valid_labels, LEGAL_LABEL_SCHEMA
    )
    val_conv = to_conversations(
        filtered["validation"], include_none_in_valid_labels, LEGAL_LABEL_SCHEMA
    )

    # Apply chat template to produce 'text' column
    train_text = apply_chat_template_to_ds(train_conv, tokenizer, tokenize=False)
    val_text = apply_chat_template_to_ds(val_conv, tokenizer, tokenize=False)

    # Prepare SFT config kwargs
    sft_kwargs: Dict[str, Any] = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        optim=args.optim,
        seed=args.seed,
        report_to="none",
    )
    # Optional knobs if provided
    if args.save_steps is not None:
        sft_kwargs["save_steps"] = args.save_steps
    if args.eval_steps is not None:
        sft_kwargs["eval_steps"] = args.eval_steps
    if args.save_total_limit is not None:
        sft_kwargs["save_total_limit"] = args.save_total_limit
    if args.load_best_model_at_end:
        sft_kwargs["load_best_model_at_end"] = True
    # Note: eval_strategy can be implicitly enabled if eval_steps/save_steps set; TRL handles None defaults.

    # Build trainer and train
    trainer = create_trainer(model, tokenizer, train_text, val_text, cfg, sft_kwargs)
    train_and_log(trainer)

    # Save locally and maybe push
    save_and_maybe_push(model, tokenizer, cfg, push_to_hub=args.push_to_hub)

    # Optional quick sanity eval
    if args.eval_after_train:
        LOGGER.info("Running quick sanity evaluation on the test split...")
        # Respect user's eval NONE preference separately from training filter
        test_for_eval = raw_test
        res = quick_eval(
            model=model,
            tokenizer=tokenizer,
            test_ds=test_for_eval,
            max_samples=max(1, int(args.eval_max_samples)),
            exclude_none=bool(args.exclude_none_eval),
        )
        LOGGER.info("Quick eval summary: %s", res)
    else:
        LOGGER.info("Skipping quick eval. To enable, pass --eval-after-train.")

    LOGGER.info(
        "Done. Artifacts saved under: %s", os.path.join(cfg.output_dir, "final")
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOGGER.error("Fatal error: %s: %s", type(e).__name__, e)
        raise
