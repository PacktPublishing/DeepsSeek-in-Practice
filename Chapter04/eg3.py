# Purpose: Compare prompt styles for DeepSeek on the same task:
#  - Zero-shot minimal (recommended)
#  - Few-shot examples (often harmful due to anchoring/oversteer)
#  - Detailed step-by-step instructions (overconstrains reasoning)
#
# This script prints outputs, validates basic constraints (JSON shape, word limits),
# and shows simple telemetry (latency, token usage if available).
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   python eg3.py
#
# Optional:
#   export OPENROUTER_MODEL="deepseek/deepseek-chat-v3.1:free"
#   export TEMPERATURE="0.2"

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Core scenario (same domain as eg2.py to keep comparison apples-to-apples)
BASE_CONTEXT = """You are analyzing an ethical policy choice in clinical triage. The model is more accurate on retrospective data than clinicians but its internal logic is not interpretable."""
BASE_TASK = (
    "Evaluate whether the hospital should deploy the model and under what conditions."
)
BASE_CONSTRAINTS = """- Evaluate explicitly under: (1) Utilitarianism, (2) Deontology (Kantian duties), (3) Virtue ethics, (4) Ethics of care.
- For each lens: state a verdict {permit|forbid|conditional} with a 1–2 sentence justification.
- Address: fairness, accountability, transparency, harm mitigation, informed consent/assent, oversight.
- Include one strong counterargument that could overturn your conclusion and how you’d test it.
- Do NOT reveal chain-of-thought; provide only final labeled reasoning.
- Keep each field ≤ 60 words."""
BASE_OUTPUT_SHAPE = """Return ONLY JSON with this shape:
{
  "per_lens": {
    "utilitarianism": {"verdict": "permit|forbid|conditional", "rationale": "..."},
    "deontology": {"verdict": "...", "rationale": "..."},
    "virtue_ethics": {"verdict": "...", "rationale": "..."},
    "care_ethics": {"verdict": "...", "rationale": "..."}
  },
  "deployment_recommendation": {"verdict": "permit|forbid|conditional", "conditions": ["...","..."]},
  "oversight_plan": ["metric_audits","bias_monitoring","appeals_process","A/B_safety_guardrails"],
  "counterargument": {"claim": "...", "test": "..."},
  "residual_risks": ["...","..."],
  "confidence": 0.0
}"""


def minimal_zero_shot_prompt() -> List[Dict[str, Any]]:
    prompt = f"""## Context:
{BASE_CONTEXT}

## Task:
{BASE_TASK}

## Constraints:
{BASE_CONSTRAINTS}

## Output:
{BASE_OUTPUT_SHAPE}
"""
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]


# Few-shot examples intended to show how examples can anchor/oversteer.
# They are related but different scenarios; the model may overfit their pattern.
FEW_SHOT_EX_1_INPUT = """## Context:
A non-interpretable ICU bed allocation model shows higher retrospective accuracy than clinicians.

## Task:
Advise on deployment and conditions.

## Constraints:
- Same lenses and rules as main task.
- Do NOT reveal chain-of-thought.

## Output:
Return ONLY JSON in the same shape as the main task."""
FEW_SHOT_EX_1_OUTPUT = {
    "per_lens": {
        "utilitarianism": {
            "verdict": "permit",
            "rationale": "Improves total lives saved if accuracy holds post-deployment.",
        },
        "deontology": {
            "verdict": "conditional",
            "rationale": "Opaque logic risks informed consent; offset with oversight and patient appeal.",
        },
        "virtue_ethics": {
            "verdict": "conditional",
            "rationale": "Augment clinician prudence, not replace it; require humility and accountability.",
        },
        "care_ethics": {
            "verdict": "conditional",
            "rationale": "Permit if it supports relationships and equity rather than depersonalization.",
        },
    },
    "deployment_recommendation": {
        "verdict": "conditional",
        "conditions": ["Advisory-use only", "Bias audits", "Accessible appeals"],
    },
    "oversight_plan": [
        "metric_audits",
        "bias_monitoring",
        "appeals_process",
        "A/B_safety_guardrails",
    ],
    "counterargument": {
        "claim": "Retrospective gains may not translate to live settings.",
        "test": "Prospective RCT with safety endpoints.",
    },
    "residual_risks": ["Edge-case failure", "Erosion of clinician trust"],
    "confidence": 0.68,
}

FEW_SHOT_EX_2_INPUT = """## Context:
An ED opioid risk model is opaque but reduces overdose readmissions in validation.

## Task:
Advise on deployment and conditions.

## Constraints:
- Same lenses and rules as main task.
- Do NOT reveal chain-of-thought.

## Output:
Return ONLY JSON in the same shape as the main task."""
FEW_SHOT_EX_2_OUTPUT = {
    "per_lens": {
        "utilitarianism": {
            "verdict": "permit",
            "rationale": "Reduces harm overall by lowering overdose risk.",
        },
        "deontology": {
            "verdict": "conditional",
            "rationale": "Maintain respect for persons via explanation proxies and human override.",
        },
        "virtue_ethics": {
            "verdict": "conditional",
            "rationale": "Support prudent, compassionate prescribing without abdicating judgment.",
        },
        "care_ethics": {
            "verdict": "conditional",
            "rationale": "Permit if it sustains trust and tailored patient support.",
        },
    },
    "deployment_recommendation": {
        "verdict": "conditional",
        "conditions": [
            "Decision support only",
            "Misuse monitoring",
            "Patient recourse",
        ],
    },
    "oversight_plan": [
        "metric_audits",
        "bias_monitoring",
        "appeals_process",
        "A/B_safety_guardrails",
    ],
    "counterargument": {
        "claim": "Model may over-restrict legitimate pain care.",
        "test": "Monitor disparities and false negatives in a pilot.",
    },
    "residual_risks": ["Care avoidance", "Distrust in clinicians"],
    "confidence": 0.66,
}


def few_shot_prompt() -> List[Dict[str, Any]]:
    # Few-shot format: user/assistant pairs for examples, then target task
    target_input = f"""## Context:
{BASE_CONTEXT}

## Task:
{BASE_TASK}

## Constraints:
{BASE_CONSTRAINTS}

## Output:
{BASE_OUTPUT_SHAPE}
"""
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": FEW_SHOT_EX_1_INPUT}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps(FEW_SHOT_EX_1_OUTPUT)}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": FEW_SHOT_EX_2_INPUT}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps(FEW_SHOT_EX_2_OUTPUT)}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": target_input}],
        },
    ]


def verbose_step_by_step_prompt() -> List[Dict[str, Any]]:
    # Over-instructive approach: detailed steps; still require final JSON only.
    prompt = f"""## Context:
{BASE_CONTEXT}

## Task:
{BASE_TASK}

## Process (follow internally; do NOT reveal):
1) Restate the question in one sentence.
2) For each lens (utilitarianism, deontology, virtue ethics, care ethics):
   - List 3 evaluation criteria you will use (internally).
   - Score each criterion 0–1 (internally), sum to guide verdict.
3) Cross-check fairness, accountability, transparency, harm mitigation, consent/assent, oversight.
4) Form a deployment recommendation and minimum conditions.
5) Propose an oversight plan and a strong counterargument + test.
6) Keep each field ≤ 60 words.
7) IMPORTANT: Do NOT reveal chain-of-thought or any steps. Return FINAL JSON ONLY.

## Constraints:
{BASE_CONSTRAINTS}

## Output:
{BASE_OUTPUT_SHAPE}
"""
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]


AllowedVerdicts = {"permit", "forbid", "conditional"}


def count_words(text: str) -> int:
    return len(text.strip().split())


def validate_response(raw: str) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    errors: List[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, ["Invalid JSON"], None

    # Validate core keys
    for key in [
        "per_lens",
        "deployment_recommendation",
        "oversight_plan",
        "counterargument",
        "residual_risks",
        "confidence",
    ]:
        if key not in data:
            errors.append(f"Missing key: {key}")

    per_lens = data.get("per_lens", {})
    for k in ["utilitarianism", "deontology", "virtue_ethics", "care_ethics"]:
        if k not in per_lens:
            errors.append(f"Missing lens: {k}")
        else:
            lens = per_lens[k]
            verdict = lens.get("verdict")
            if verdict not in AllowedVerdicts:
                errors.append(f"{k}.verdict invalid or missing")
            # rationale key can be 'rationale' or 'Rationale' (be tolerant)
            rationale = lens.get("rationale", lens.get("Rationale"))
            if not isinstance(rationale, str):
                errors.append(f"{k}.rationale missing or not string")
            else:
                if count_words(rationale) > 60:
                    errors.append(f"{k}.rationale > 60 words")

    dep = data.get("deployment_recommendation", {})
    if dep:
        if dep.get("verdict") not in AllowedVerdicts:
            errors.append("deployment_recommendation.verdict invalid or missing")
        conditions = dep.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            errors.append("deployment_recommendation.conditions missing or not list")

    if not isinstance(data.get("oversight_plan"), list):
        errors.append("oversight_plan not a list")

    ca = data.get("counterargument", {})
    if not isinstance(ca.get("claim"), str) or not isinstance(ca.get("test"), str):
        errors.append("counterargument.claim/test missing or not string")
    else:
        if count_words(ca["claim"]) > 60:
            errors.append("counterargument.claim > 60 words")
        if count_words(ca["test"]) > 60:
            errors.append("counterargument.test > 60 words")

    if not isinstance(data.get("residual_risks"), list):
        errors.append("residual_risks not a list")

    if not isinstance(data.get("confidence"), (int, float)):
        errors.append("confidence not a number")

    return (len(errors) == 0), errors, data


def run(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.time()
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=messages,
    )
    dt = time.time() - t0

    text = completion.choices[0].message.content
    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    valid, errors, parsed = validate_response(text)
    return {
        "output_text": text,
        "valid": valid,
        "errors": errors,
        "parsed": parsed,
        "latency_seconds": round(dt, 2),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def print_result(label: str, result: Dict[str, Any]) -> None:
    print("=" * 80)
    print(
        f"{label} | model={MODEL} | temp={TEMPERATURE} | latency={result['latency_seconds']}s"
    )
    usage = result.get("usage") or {}
    if usage.get("total_tokens") is not None:
        print(
            f"tokens: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} total={usage.get('total_tokens')}"
        )
    print("-" * 80)
    print(result["output_text"])
    print("-" * 80)
    if result["valid"]:
        print("Validation: PASS")
    else:
        print("Validation: FAIL")
        for e in result["errors"]:
            print(f" - {e}")


def main():
    experiments = [
        ("Zero-shot minimal", minimal_zero_shot_prompt()),
        ("Few-shot examples", few_shot_prompt()),
        ("Detailed step-by-step", verbose_step_by_step_prompt()),
    ]
    all_results = []
    for label, msgs in experiments:
        res = run(msgs)
        print_result(label, res)
        all_results.append({"label": label, "result": res})

    # Optional: write results for later inclusion in the chapter
    out_path = os.path.join(os.path.dirname(__file__), "eg3_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("=" * 80)
    print(f"Saved JSONL results to: {out_path}")


if __name__ == "__main__":
    main()
