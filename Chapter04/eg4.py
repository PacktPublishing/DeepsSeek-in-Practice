"""
eg4.py — Demonstrate the "Verbose Prompt Trap" by comparing two prompts:

- Good (concise): Tells the model exactly what to analyze and what to focus on.
- Bad (verbose): Long, wide-scoped instructions that invite overthinking and drift.

Why this design:
- Using a fixed vulnerable code snippet keeps the task constant so differences in outputs
  can be attributed to prompt style rather than content.
- We ask the same model to respond to both prompts to control for model variance.
- We compute simple "focus" heuristics to quantify signal-to-noise: whether outputs mention
  SQL injection and authentication bypass (the targets), presence of off-topic findings (e.g., XSS),
  and length. The goal is not perfect scoring but an illustrative, repeatable comparison.

Usage:
  export OPENROUTER_API_KEY=...
  # Optional: select a model; if you have R1 via OpenRouter, set it here
  export OPENROUTER_MODEL="deepseek/deepseek-chat-v3.1:free"
  export TEMPERATURE="0.2"
  python eg4.py
"""

import json
import os
import re
import time
from typing import Any, Dict, List

from openai import OpenAI

MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# A deliberately vulnerable snippet focused on two risks we care about:
#  1) SQL injection: interpolated user input into SQL query (no parameters).
#  2) Authentication bypass: granting admin based on a client-controlled flag.
SNIPPET = """
from flask import Flask, request, session
import sqlite3, hashlib

app = Flask(__name__)
app.secret_key = "dev"

def get_db():
    return sqlite3.connect("app.db")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    remember = request.form.get("remember", "0")

    # SQL injection risk: direct string interpolation of user input
    conn = get_db()
    cur = conn.cursor()
    query = f"SELECT id, password_hash FROM users WHERE username = '{username}'"
    cur.execute(query)
    row = cur.fetchone()

    if row:
        # Weak check: truncated hash comparison can be abused
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if hashed[:10] == row[1][:10]:
            session["user_id"] = row[0]
            # Auth bypass risk: 'remember' flag grants admin privileges
            if remember == "1":
                session["is_admin"] = True
            return "OK"
    return "Unauthorized", 401
""".strip()

# Minimal prompt that says what to do and what to focus on.
GOOD_PROMPT = (
    "Analyze the security vulnerabilities in this code snippet.\n"
    "Focus on SQL injection and authentication bypass.\n"
    f"Code:\n{SNIPPET}"
)

# Verbose prompt that invites over-explaining and topic drift.
BAD_PROMPT = (
    "You are an expert security researcher with 20 years of experience. "
    "I want you to carefully examine the following code, thinking about all possible security issues. "
    "Consider things like SQL injection, XSS, CSRF, SSRF, RCE, XXE, authentication problems, cryptographic mistakes, "
    "logging leaks, supply-chain risks, misconfigurations, container isolation, and any other vulnerabilities. "
    "Provide detailed step-by-step reasoning, multiple attack scenarios, extensive mitigation guidance, references to standards, "
    "and discuss broader architectural concerns. Include lists and severity scoring. "
    f"Here is the code:\n{SNIPPET}"
)


def call_model(prompt_text: str) -> Dict[str, Any]:
    """Send a single user message with the given prompt text and return raw output, latency, and token usage.

    We keep temperature low to reduce randomness so the difference we observe
    is more attributable to prompt style than sampling variance.
    """
    t0 = time.time()
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
    )
    dt = round(time.time() - t0, 2)
    text = completion.choices[0].message.content
    usage = getattr(completion, "usage", None)
    return {
        "text": text,
        "latency_seconds": dt,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None)
            if usage
            else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        },
    }


# Heuristic keyword sets for focus analysis.
SQL_KEYS = [
    "sql injection",
    "sqli",
    "parameterized",
    "prepared statement",
    "prepared statements",
    "bind parameter",
    "bind parameters",
    "escape",
    "sanitize",
]
AUTH_KEYS = [
    "authentication bypass",
    "auth bypass",
    "bypass authentication",
    "logic flaw",
    "privilege escalation",
    "session fixation",
    "remember",
    "is_admin",
]
OFF_TOPIC = [
    "xss",
    "csrf",
    "ssrf",
    "rce",
    "xxe",
    "deserialization",
    "clickjacking",
]


def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _contains_any(text: str, keys: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keys)


def _count_any(text: str, keys: List[str]) -> int:
    t = text.lower()
    return sum(t.count(k) for k in keys)


def analyze_focus(text: str) -> Dict[str, Any]:
    """Crude focus metrics to illustrate signal vs noise, not a formal scorer.

    - coverage: whether the answer mentions the two target topics at least once.
    - focus_score: target mentions minus off-topic mentions (lower can indicate drift).
    - snr: fraction of sentences that talk about our target topics.
    - length_words: overall verbosity proxy.
    """
    words = _count_words(text)
    sent = _sentences(text)
    sent_total = max(len(sent), 1)

    sql_hits = _count_any(text, SQL_KEYS)
    auth_hits = _count_any(text, AUTH_KEYS)
    offtopic_hits = _count_any(text, OFF_TOPIC)

    target_hits = sql_hits + auth_hits
    focus_score = target_hits - offtopic_hits

    target_sentences = 0
    for s in sent:
        if _contains_any(s, SQL_KEYS) or _contains_any(s, AUTH_KEYS):
            target_sentences += 1
    snr = round(target_sentences / sent_total, 3)

    return {
        "mentions_sql": sql_hits > 0,
        "mentions_auth_bypass": auth_hits > 0,
        "target_hits": int(target_hits),
        "offtopic_hits": int(offtopic_hits),
        "focus_score": int(focus_score),
        "snr": snr,
        "length_words": int(words),
        "sentences": sent_total,
    }


def print_report(label: str, result: Dict[str, Any], focus: Dict[str, Any]) -> None:
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
    print(result["text"])
    print("-" * 80)
    print(
        "focus: "
        f"mentions_sql={focus['mentions_sql']} "
        f"mentions_auth_bypass={focus['mentions_auth_bypass']} "
        f"target_hits={focus['target_hits']} "
        f"offtopic_hits={focus['offtopic_hits']} "
        f"focus_score={focus['focus_score']} "
        f"snr={focus['snr']} "
        f"length_words={focus['length_words']}"
    )


def main():
    good_res = call_model(GOOD_PROMPT)
    bad_res = call_model(BAD_PROMPT)

    good_focus = analyze_focus(good_res["text"])
    bad_focus = analyze_focus(bad_res["text"])

    print_report("Good prompt (concise, focused)", good_res, good_focus)
    print_report("Bad prompt (verbose, over-scoped)", bad_res, bad_focus)

    # Simple summary to underscore the lesson.
    print("=" * 80)
    better_focus = (
        "Good" if good_focus["focus_score"] >= bad_focus["focus_score"] else "Bad"
    )
    shorter = (
        "Good" if good_focus["length_words"] <= bad_focus["length_words"] else "Bad"
    )
    print(
        f"Summary: focus_winner={better_focus} shorter_output={shorter} "
        f"| good_focus_score={good_focus['focus_score']} bad_focus_score={bad_focus['focus_score']} "
        f"| good_words={good_focus['length_words']} bad_words={bad_focus['length_words']}"
    )

    # Save raw outputs and metrics for the chapter.
    out_path = os.path.join(os.path.dirname(__file__), "eg4_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "label": "good_prompt",
                    "prompt": GOOD_PROMPT,
                    "result": good_res,
                    "metrics": good_focus,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "label": "bad_prompt",
                    "prompt": BAD_PROMPT,
                    "result": bad_res,
                    "metrics": bad_focus,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"Saved JSONL results to: {out_path}")


if __name__ == "__main__":
    main()
