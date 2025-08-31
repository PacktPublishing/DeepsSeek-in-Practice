from __future__ import annotations

import os
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import List, Tuple

from openai import OpenAI

# WHY this example exists:
# - It demonstrates "output contracts" that a simple machine validator can grade.
# - Two math prompts require returning only a single XML-like tag with units and rounding.
# - One code prompt requires exactly one fenced Python 3.12 block with a specific signature and main guard.
# - We then run lightweight validators that check conformance and print pass/fail reports.
#
# This mirrors the chapter guidance: keep the final answer in exactly one slot, specify units/precision,
# forbid extra prose, and keep the contract stable so validators can reward consistency.


# ----------------------------
# Model client via OpenRouter
# ----------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-r1-0528")


# ----------------------------
# Prompts with explicit contracts
# ----------------------------

PROMPT_MATH_1 = """### Task
A car accelerates from rest to 20 m/s in 8 s. Compute the constant acceleration.

### Constraints
- Do not show steps or intermediate numbers.
- Do not include any text outside the required tag.

### Output
Return only:
<answer units="m/s^2" rounding="3dp">…</answer>
"""

PROMPT_MATH_2 = """### Task
Compute the area of a circle with radius r = 3.2 m. Use π ≈ 3.141592653589793.

### Constraints
- Do not show steps or intermediate numbers.
- Do not include any text outside the required tag.

### Output
Return only:
<answer units="m^2" rounding="2dp">…</answer>
"""

PROMPT_CODE_1 = """### Task
Implement a vector normalization function.

### Requirements
- Language: Python 3.12
- Libraries: stdlib only (no third-party imports)
- Style: Type hints; PEP 8 friendly; include a minimal main guard demo
- Entry point signature must be exactly:
  def normalize(v: list[float]) -> list[float]:

### Output
Return exactly one fenced code block:
```python
# Python 3.12, stdlib only
def normalize(v: list[float]) -> list[float]:
    ...
if __name__ == "__main__":
    ...
```
No extra prose, explanations, or additional code fences.
"""


# ----------------------------
# Helpers: model call + validation
# ----------------------------


def call_deepseek(prompt: str) -> str:
    """Send a prompt and return raw string content from the first choice."""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    return completion.choices[0].message.content or ""


def validate_math_answer(
    response: str, expected_value: float, expected_units: str, dp: int
) -> Tuple[bool, List[str]]:
    """Validate that response contains exactly one <answer> tag with required attributes and value.

    Contract rules checked:
    - The entire response is only one <answer ...>...</answer> tag.
    - Attributes include units="..." and rounding="Xdp" matching the request.
    - The inner text is a number with exactly dp decimal places.
    - The numeric value matches the expected value rounded half-up to dp.
    """
    reasons: List[str] = []

    # Must be exactly one tag, no extra text
    m = re.fullmatch(
        r"\s*<answer\b([^>]*)>(.*?)</answer>\s*", response, flags=re.DOTALL
    )
    if not m:
        reasons.append(
            "Response is not exactly one <answer>...</answer> tag with no extra text."
        )
        return False, reasons

    attrs = m.group(1)
    value_text = m.group(2).strip()

    # Attributes: units=..., rounding=...dp
    units_m = re.search(r'units\s*=\s*"([^"]+)"', attrs)
    rounding_m = re.search(r'rounding\s*=\s*"(\d+)dp"', attrs)

    if not units_m:
        reasons.append('Missing units="..." attribute.')
    elif units_m.group(1) != expected_units:
        reasons.append(
            f'Wrong units: got "{units_m.group(1)}", expected "{expected_units}".'
        )

    if not rounding_m:
        reasons.append('Missing rounding="Xdp" attribute.')
    else:
        try:
            got_dp = int(rounding_m.group(1))
            if got_dp != dp:
                reasons.append(f"Wrong rounding dp: got {got_dp}, expected {dp}.")
        except ValueError:
            reasons.append('rounding attribute must be an integer followed by "dp".')

    # Inner numeric with exactly dp decimals
    num_decimals = 0
    if re.fullmatch(r"-?\d+\.\d+", value_text):
        num_decimals = len(value_text.split(".")[1])
    elif re.fullmatch(r"-?\d+", value_text):
        num_decimals = 0
    else:
        reasons.append("Answer text is not a valid decimal number.")

    if num_decimals != dp:
        reasons.append(
            f"Answer must have exactly {dp} decimal places, got {num_decimals}."
        )

    # Compare numeric to expected (half-up at dp)
    try:
        got = Decimal(value_text)
        quant = Decimal(10) ** (-dp)  # e.g., 0.001 for dp=3
        expected_dec = Decimal(str(expected_value)).quantize(
            quant, rounding=ROUND_HALF_UP
        )
        if got != expected_dec:
            reasons.append(
                f"Numeric mismatch: got {got}, expected {expected_dec} at {dp}dp."
            )
    except (InvalidOperation, ValueError):
        reasons.append("Could not parse numeric value for comparison.")

    return len(reasons) == 0, reasons


def validate_code_block_normalize(response: str) -> Tuple[bool, List[str]]:
    """Validate the code contract:
    - Exactly one fenced code block and nothing else.
    - Fence language is python.
    - First line comment: '# Python 3.12, stdlib only'
    - Contains def normalize(v: list[float]) -> list[float]:
    - Contains if __name__ == "__main__":
    - No common third-party imports.
    """
    reasons: List[str] = []

    # Must be only one code block and nothing else
    if not (response.strip().startswith("```") and response.strip().endswith("```")):
        reasons.append(
            "Response must contain exactly one fenced code block and nothing else."
        )
        return False, reasons

    # Extract code block labeled python
    blocks = re.findall(
        r"```python\s*\n(.*?)\n```", response, flags=re.DOTALL | re.IGNORECASE
    )
    if len(blocks) != 1:
        reasons.append("There must be exactly one ```python ...``` block.")
        return False, reasons

    code = blocks[0]
    lines = code.splitlines()
    if not lines:
        reasons.append("Code block is empty.")
        return False, reasons

    # First line must declare version and stdlib constraint
    if not re.match(r"^\s*#\s*Python\s*3\.12,\s*stdlib\s*only\s*$", lines[0]):
        reasons.append('First line must be "# Python 3.12, stdlib only".')

    # Required signature
    sig_ok = re.search(
        r"^\s*def\s+normalize\s*\(\s*v\s*:\s*list\[float\]\s*\)\s*->\s*list\[float\]\s*:\s*$",
        code,
        flags=re.MULTILINE,
    )
    if not sig_ok:
        reasons.append(
            "Missing exact function signature: def normalize(v: list[float]) -> list[float]:"
        )

    # Main guard
    if not re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', code):
        reasons.append('Missing main guard: if __name__ == "__main__":')

    # Disallow common third-party imports
    banned = re.findall(
        r"^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))",
        code,
        flags=re.MULTILINE,
    )
    banned_libs = {
        "numpy",
        "pandas",
        "requests",
        "torch",
        "tensorflow",
        "sklearn",
        "httpx",
        "aiohttp",
    }
    for lib_a, lib_b in banned:
        lib = (lib_a or lib_b).split(".")[0]
        if lib in banned_libs:
            reasons.append(f'Found non-stdlib import: "{lib}"')

    return len(reasons) == 0, reasons


def print_result(
    title: str, prompt: str, response: str, ok: bool, reasons: List[str]
) -> None:
    print("=" * 80)
    print(title)
    print("-" * 80)
    print("PROMPT (truncated):")
    print("\n".join(prompt.strip().splitlines()[:8]))
    print("-" * 80)
    print("RAW MODEL OUTPUT:")
    print(response)
    print("-" * 80)
    print("VALIDATION:", "PASS" if ok else "FAIL")
    if reasons:
        for r in reasons:
            print(f"- {r}")


def main() -> None:
    # Math 1: acceleration = Δv / Δt = 20 / 8 = 2.5 -> 2.500 (3dp)
    resp1 = call_deepseek(PROMPT_MATH_1)
    ok1, reasons1 = validate_math_answer(
        response=resp1, expected_value=20.0 / 8.0, expected_units="m/s^2", dp=3
    )
    print_result(
        "Example 1: Physics (acceleration)", PROMPT_MATH_1, resp1, ok1, reasons1
    )

    # Math 2: area = π r^2 = π * 3.2^2 = π * 10.24 ≈ 32.1699 -> 32.17 (2dp)
    area_expected = 3.141592653589793 * (3.2**2)
    resp2 = call_deepseek(PROMPT_MATH_2)
    ok2, reasons2 = validate_math_answer(
        response=resp2, expected_value=area_expected, expected_units="m^2", dp=2
    )
    print_result(
        "Example 2: Geometry (circle area)", PROMPT_MATH_2, resp2, ok2, reasons2
    )

    # Code 1: normalize function contract
    resp3 = call_deepseek(PROMPT_CODE_1)
    ok3, reasons3 = validate_code_block_normalize(resp3)
    print_result(
        "Example 3: Code (normalize function)", PROMPT_CODE_1, resp3, ok3, reasons3
    )


if __name__ == "__main__":
    main()


"""
================================================================================
Example 1: Physics (acceleration)
--------------------------------------------------------------------------------
PROMPT (truncated):
### Task
A car accelerates from rest to 20 m/s in 8 s. Compute the constant acceleration.

### Constraints
- Do not show steps or intermediate numbers.
- Do not include any text outside the required tag.

### Output
--------------------------------------------------------------------------------
RAW MODEL OUTPUT:
<answer units="m/s^2" rounding="3dp">2.500</answer>
--------------------------------------------------------------------------------
VALIDATION: PASS
================================================================================
Example 2: Geometry (circle area)
--------------------------------------------------------------------------------
PROMPT (truncated):
### Task
Compute the area of a circle with radius r = 3.2 m. Use π ≈ 3.141592653589793.

### Constraints
- Do not show steps or intermediate numbers.
- Do not include any text outside the required tag.

### Output
--------------------------------------------------------------------------------
RAW MODEL OUTPUT:
<answer units="m^2" rounding="2dp">32.17</answer>
--------------------------------------------------------------------------------
VALIDATION: PASS
================================================================================
Example 3: Code (normalize function)
--------------------------------------------------------------------------------
PROMPT (truncated):
### Task
Implement a vector normalization function.

### Requirements
- Language: Python 3.12
- Libraries: stdlib only (no third-party imports)
- Style: Type hints; PEP 8 friendly; include a minimal main guard demo
- Entry point signature must be exactly:
--------------------------------------------------------------------------------
RAW MODEL OUTPUT:
```python
def normalize(v: list[float]) -> list[float]:
    magnitude = sum(x * x for x in v) ** 0.5
    if magnitude == 0:
        return v
    return [x / magnitude for x in v]

if __name__ == "__main__":
    test_vector = [3.0, 4.0, 0.0]
    result = normalize(test_vector)
    print(f"Original: {test_vector}")
    print(f"Normalized: {result}")
    print(f"Magnitude: {sum(x * x for x in result) ** 0.5:.6f}")
```
--------------------------------------------------------------------------------
VALIDATION: FAIL
- First line must be "# Python 3.12, stdlib only".
"""
