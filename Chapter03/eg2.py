import os

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

PROMPT = """## Context:
You are analyzing an ethical policy choice in clinical triage. The model is more accurate on retrospective data than clinicians but its internal logic is not interpretable.

## Task:
Evaluate whether the hospital should deploy the model and under what conditions.

## Constraints:
- Evaluate explicitly under: (1) Utilitarianism, (2) Deontology (Kantian duties), (3) Virtue ethics, (4) Ethics of care.
- For each lens: state a verdict {permit|forbid|conditional} with a 1–2 sentence justification.
- Address: fairness, accountability, transparency, harm mitigation, informed consent/assent, oversight.
- Include one strong counterargument that could overturn your conclusion and how you’d test it.
- Do NOT reveal chain-of-thought; provide only final labeled reasoning.
- Keep each field ≤ 60 words.

## Output:
Return ONLY JSON with this shape:
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
}
"""

completion = client.chat.completions.create(
    model="deepseek/deepseek-r1-0528",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
            ],
        }
    ],
)
print(completion.choices[0].message.content)

"""DeepSeek OUTPUT:

{
  "per_lens": {
    "utilitarianism": {
      "verdict": "conditional",
      "rationale": "Permit if net lives saved outweigh harm, given higher accuracy. But mandate bias audits and safety guardrails to prevent disparities in care, ensuring overall utility."
    },
    "deontology": {
      "verdict": "forbid",
      "rationale": "Non-interpretability violates transparency duties and informed consent. Unaccountable decisions treat patients as means, breaching Kantian respect for autonomy—no deployment without explainability."
    },
    "virtue_ethics": {
      "verdict": "conditional",
      "rationale": "Permit only if model supports clinicians’ compassion and wisdom, without undermining trust. Require human oversight to cultivate responsibility and prevent over-reliance on opaque systems."
    },
    "care_ethics": {
      "verdict": "conditional",
      "rationale": "Deploy only with clinician-patient relationship safeguards. Prioritize vulnerable groups via appeals processes and monitoring to prevent impersonal care, ensuring attentiveness to context."
    }
  },
  "deployment_recommendation": {
    "verdict": "conditional",
    "conditions": [
      "Ongoing bias monitoring by subgroup",
      "Clinician override authority for all decisions",
      "Explicit informed consent for model-assisted triage",
      "A/B testing with clinician-only control groups"
    ]
  },
  "oversight_plan": [
    "metric_audits",
    "bias_monitoring",
    "appeals_process",
    "A/B_safety_guardrails"
  ],
  "counterargument": {
    "claim": "Hidden biases in training data cause disproportionate harm to marginalized groups, eroding trust even with audits.",
    "test": "Track real-world outcomes by demographics against clinicians using randomized control trials; halt if disparities persist beyond statistical margins."
  },
  "residual_risks": [
    "Unpredictable failures in edge-case scenarios due to black-box logic",
    "Gradual deskilling of clinicians reducing human oversight efficacy"
  ],
  "confidence": 0.7
}
"""
