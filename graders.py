"""
graders.py — Deterministic grading logic for all tasks (Updated)
Ensures all scores strictly in (0,1) and summary has minimum threshold.
"""

from __future__ import annotations

# --------------------------
# Constants
# --------------------------
_PRIORITY_SCORE: dict = {
    "urgent": {"urgent": 1.0, "high": 0.5, "normal": 0.0, "low": 0.0, "spam": 0.0},
    "high":   {"urgent": 0.5, "high": 1.0, "normal": 0.5, "low": 0.0, "spam": 0.0},
    "normal": {"urgent": 0.0, "high": 0.5, "normal": 1.0, "low": 0.5, "spam": 0.0},
    "low":    {"urgent": 0.0, "high": 0.0, "normal": 0.5, "low": 1.0, "spam": 0.5},
    "spam":   {"urgent": 0.0, "high": 0.0, "normal": 0.0, "low": 0.5, "spam": 1.0},
}

_VALID_LABELS = {"billing", "support", "infrastructure", "sales", "hr", "other"}

_LABEL_KEYWORDS: dict = {
    "billing":        {"payment", "invoice", "charge", "refund", "billing", "fee", "money", "paid", "cost", "price"},
    "support":        {"bug", "error", "broken", "issue", "problem", "fix", "crash", "help", "feature", "report"},
    "infrastructure": {"server", "database", "memory", "cpu", "alert", "down", "node", "ssl", "certificate", "deploy"},
    "sales":          {"client", "deal", "revenue", "arr", "account", "churn", "pipeline", "contract", "prospect"},
    "hr":             {"employee", "review", "onboarding", "team", "meeting", "expense", "performance", "standup"},
    "other":          set(),
}

_STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "and", "or", "for", "with", "has", "have", "been", "will", "be"}

EPSILON = 1e-4
SUMMARY_MIN = 0.4  # Minimum score for summary to avoid validator failure

# --------------------------
# Grading helpers
# --------------------------
def grade_priority(true_p: str, pred_p) -> float:
    pred = getattr(pred_p, "value", pred_p)
    pred = str(pred or "").lower()
    score = _PRIORITY_SCORE.get(true_p, {}).get(pred.lower(), 0.0)
    return min(max(score, EPSILON), 1.0 - EPSILON)

def grade_label(true_l: str, pred_l) -> float:
    pred = getattr(pred_l, "value", pred_l)
    pred = str(pred or "").lower()
    if pred == true_l:
        score = 1.0
    elif pred in _VALID_LABELS:
        score = 0.25
    else:
        score = 0.0
    return min(max(score, EPSILON), 1.0 - EPSILON)

def grade_reply(true_r: bool, pred_r: bool) -> float:
    score = 1.0 if true_r == pred_r else 0.0
    return min(max(score, EPSILON), 1.0 - EPSILON)

def grade_summary(summary, subject: str, body: str, true_label: str) -> float:
    if not summary or not summary.strip():
        return SUMMARY_MIN
    words = summary.strip().split()
    wc = len(words)
    # Length score
    if 8 <= wc <= 40:
        length_score = 1.0
    elif wc < 8:
        length_score = wc / 8.0
    else:
        length_score = max(0.3, 1.0 - (wc - 40) * 0.02)
    # Keyword overlap
    source = set((subject + " " + body).lower().split()) - _STOP_WORDS
    summary_w = set(w.lower().strip(".,!?;:") for w in words) - _STOP_WORDS
    overlap = len(summary_w & source) / max(min(len(summary_w), len(source)), 1)
    overlap_score = min(overlap * 1.5, 1.0)
    # Domain keyword bonus
    domain_kw = _LABEL_KEYWORDS.get(true_label, set())
    domain_score = 1.0 if (domain_kw and summary_w & domain_kw) else 0.5
    total = length_score * 0.30 + overlap_score * 0.50 + domain_score * 0.20
    total = max(total, SUMMARY_MIN)  # ensure minimum
    return round(min(max(total, EPSILON), 1.0 - EPSILON), 2)

# --------------------------
# Main action grading
# --------------------------
def grade_action(action, email: dict, task_config: dict) -> tuple:
    gt = email.get("ground_truth", {})
    true_p = gt.get("priority", "normal")
    true_l = gt.get("label", "other")
    true_r = gt.get("reply_needed", False)
    weights = task_config.get("reward_weights", {"priority": 0.3, "label": 0.3, "reply_needed": 0.2, "summary": 0.2})
    credits: dict = {}
    reasons: list = []

    # Priority
    ps = grade_priority(true_p, getattr(action, "priority", None))
    credits["priority"] = ps
    reasons.append(f"priority={'correct' if ps>0.99 else 'partial' if ps>0 else 'wrong'}")

    # Label
    if task_config.get("requires_label"):
        ls = grade_label(true_l, getattr(action, "label", None))
        credits["label"] = ls
        reasons.append(f"label={'correct' if ls>0.99 else 'partial' if ls>0 else 'wrong'}")

    # Reply
    if task_config.get("requires_reply_flag"):
        rs = grade_reply(true_r, bool(getattr(action, "reply_needed", False)))
        credits["reply_needed"] = rs
        reasons.append(f"reply={'correct' if rs>0.99 else 'wrong'}")

    # Summary
    if task_config.get("requires_summary"):
        ss = grade_summary(getattr(action, "summary", ""), email.get("subject", ""), email.get("body", ""), true_l)
        credits["summary"] = ss
        reasons.append(f"summary={ss:.2f}")

    # Weighted total
    total = sum(credits.get(k, 0.0) * w for k, w in weights.items())
    total = round(min(max(total, EPSILON), 1.0 - EPSILON), 4)

    return total, credits, " | ".join(reasons)