"""
Task definitions — easy / medium / hard.
"""

from __future__ import annotations
import random


def _select_basic(emails: list, rng: random.Random) -> list:
    urgent = [e for e in emails if e["ground_truth"]["priority"] == "urgent"]
    spam   = [e for e in emails if e["ground_truth"]["priority"] == "spam"]
    rest   = [e for e in emails if e["ground_truth"]["priority"] not in ("urgent", "spam")]
    chosen: list = []
    chosen += rng.sample(urgent, min(1, len(urgent)))
    chosen += rng.sample(spam,   min(1, len(spam)))
    chosen += rng.sample(rest, min(5 - len(chosen), len(rest)))
    rng.shuffle(chosen)
    return chosen[:5]


def _select_full(emails: list, rng: random.Random) -> list:
    by_p: dict = {}
    for e in emails:
        by_p.setdefault(e["ground_truth"]["priority"], []).append(e)
    chosen: list = []
    for p in ("urgent", "high", "normal", "low", "spam"):
        chosen += rng.sample(by_p.get(p, []), min(1, len(by_p.get(p, []))))
    used = {e["id"] for e in chosen}
    rest = [e for e in emails if e["id"] not in used]
    chosen += rng.sample(rest, min(8 - len(chosen), len(rest)))
    rng.shuffle(chosen)
    return chosen[:8]


def _select_hard(emails: list, rng: random.Random) -> list:
    by_p: dict = {}
    for e in emails:
        by_p.setdefault(e["ground_truth"]["priority"], []).append(e)
    chosen: list = []
    for p, n in [("urgent", 2), ("spam", 2), ("high", 2), ("normal", 2), ("low", 2)]:
        chosen += rng.sample(by_p.get(p, []), min(n, len(by_p.get(p, []))))
    used = {e["id"] for e in chosen}
    rest = [e for e in emails if e["id"] not in used]
    chosen += rng.sample(rest, min(10 - len(chosen), len(rest)))
    rng.shuffle(chosen)
    return chosen[:10]


TASKS: dict = {
    "basic_triage": {
        "display_name": "Basic Email Triage",
        "difficulty": "easy",
        "description": (
            "You are an email triage assistant. For each email in your inbox, "
            "assign the correct priority: urgent, high, normal, low, or spam. "
            "Triage one email per step. Scored on priority accuracy only."
        ),
        "email_count": 5,
        "requires_label": False,
        "requires_reply_flag": False,
        "requires_summary": False,
        "reward_weights": {"priority": 1.0},
        "email_selector": _select_basic,
    },
    "full_triage": {
        "display_name": "Full Email Triage",
        "difficulty": "medium",
        "description": (
            "Triage 8 emails per episode. For each email assign: "
            "(1) priority (urgent/high/normal/low/spam), "
            "(2) category label (billing/support/infrastructure/sales/hr/other), "
            "(3) reply_needed flag (true/false). All three fields are scored."
        ),
        "email_count": 8,
        "requires_label": True,
        "requires_reply_flag": True,
        "requires_summary": False,
        "reward_weights": {"priority": 0.50, "label": 0.30, "reply_needed": 0.20},
        "email_selector": _select_full,
    },
    "triage_and_summarize": {
        "display_name": "Triage and Summarize",
        "difficulty": "hard",
        "description": (
            "Triage 10 emails per episode. For each email assign: "
            "(1) priority, (2) label, (3) reply_needed, AND "
            "(4) write a concise 1-sentence executive summary (10-40 words). "
            "All four dimensions are scored."
        ),
        "email_count": 10,
        "requires_label": True,
        "requires_reply_flag": True,
        "requires_summary": True,
        "reward_weights": {"priority": 0.40, "label": 0.20, "reply_needed": 0.20, "summary": 0.20},
        "email_selector": _select_hard,
    },
}


def get_task(task_name: str) -> dict:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Valid: {list(TASKS.keys())}")
    return TASKS[task_name]
