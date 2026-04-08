"""
inference.py — Email Triage OpenEnv Baseline Inference Script
=============================================================

Required env vars:
  API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model id      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
  ENV_URL        Env base URL  (default: http://localhost:7860)

Stdout format:
  [START] task=<t> env=email-triage-env model=<m>
  [STEP]  step=<n> action=<a> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations
import json, os, sys, time
from typing import List, Optional

import httpx
from openai import OpenAI
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-key"
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK          = "email-triage-env"
TASKS              = ["basic_triage", "full_triage", "triage_and_summarize"]
MAX_STEPS          = 12
TEMPERATURE        = 0.2
MAX_TOKENS         = 400
SUCCESS_THRESHOLD  = 0.5
REQUEST_TIMEOUT    = 60

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert email triage assistant inside an automated system.

Respond with ONLY a valid JSON object — no markdown, no explanation.

JSON schema:
{
  "email_id": "<exactly as given>",
  "priority": "<urgent|high|normal|low|spam>",
  "label": "<billing|support|infrastructure|sales|hr|other>",
  "reply_needed": <true|false>,
  "summary": "<one sentence 10-40 words — only for triage_and_summarize task, else null>"
}

Priority rules:
  urgent  — system down, security breach, payment failure, needs action NOW
  high    — customer complaint, churn risk, time-sensitive (within 24h)
  normal  — routine requests, invoices, meetings, standard tickets
  low     — newsletters, FYIs, non-urgent reminders
  spam    — unsolicited, scams, phishing, prize notifications

Output ONLY the JSON. Nothing else."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error.replace("\n", " ") if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rstr}", flush=True)

# ---------------------------------------------------------------------------
# Env HTTP client
# ---------------------------------------------------------------------------
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.http = httpx.Client(timeout=REQUEST_TIMEOUT)

    def reset(self, task: str, seed: int = 42) -> dict:
        r = self.http.post(f"{self.base_url}/reset", json={"task": task, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = self.http.post(f"{self.base_url}/step", json=action)
        r.raise_for_status()
        return r.json()

    def close(self):
        self.http.close()

# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
def call_model(client: OpenAI, email: dict, task_name: str, step: int, total: int) -> dict:
    import re
    needs_summary = task_name == "triage_and_summarize"

    user_prompt = f"""Task: {task_name} | Step {step}/{total}

Email:
ID: {email["id"]}
From: {email["sender"]}
Subject: {email["subject"]}
Body: {email["body"]}

IMPORTANT:
Return ONLY valid JSON. No text before/after.

summary: {"required (10-40 words)" if needs_summary else "null"}
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (resp.choices[0].message.content or "").strip()
        print("[DEBUG MODEL OUTPUT]", text, flush=True)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise ValueError("No JSON found")

        if "email_id" not in parsed:
            parsed["email_id"] = email["id"]

        return parsed

    except Exception as exc:
        print(f"[DEBUG] model error: {exc}", flush=True)
        body = email["body"].lower()
        subject = email["subject"].lower()

        if "invoice" in body or "payment" in body:
            priority = "high"
            label = "billing"
        elif "server" in body or "down" in body:
            priority = "urgent"
            label = "infrastructure"
        elif "unsubscribe" in body or "offer" in body:
            priority = "low"
            label = "other"
        else:
            priority = "normal"
            label = "support"

        return {
            "email_id": email["id"],
            "priority": priority,
            "label": label,
            "reply_needed": True,
            "summary": "Automated fallback summary." if needs_summary else None
        }

# ---------------------------------------------------------------------------
# Single task episode
# ---------------------------------------------------------------------------
def run_task(task_name: str, env_client: EnvClient, llm_client: OpenAI) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_client.reset(task=task_name, seed=42)
        obs = result["observation"]
        emails = {e["id"]: e for e in obs["emails"]}
        remaining = set(obs["remaining_ids"])
        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done or not remaining:
                break

            email_id = next(iter(remaining))
            email = emails[email_id]
            action = call_model(llm_client, email, task_name, step, obs["total_emails"])

            error_msg = None
            try:
                step_res = env_client.step(action)
                reward = float(step_res.get("reward", 0.0))
                done = step_res.get("done", False)
                new_obs = step_res.get("observation", {})
                remaining = set(new_obs.get("remaining_ids", []))
            except Exception as e:
                reward, error_msg, done = 0.0, str(e), True

            rewards.append(reward)
            steps_taken = step
            action_str = f"triage(id={action.get('email_id','?')},priority={action.get('priority','?')},label={action.get('label','?')})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        score = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] episode error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI()
env_client = EnvClient(ENV_URL)  # shared instance

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
async def reset_env():
    """Reset OpenEnv environment for all tasks."""
    try:
        res = {task: env_client.reset(task=task) for task in TASKS}
        return {"status": "OK", "details": res}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[DEBUG] env={ENV_URL} model={MODEL_NAME}", flush=True)
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores: List[float] = []

    try:
        for task_name in TASKS:
            print(f"\n[DEBUG] ===== {task_name} =====", flush=True)
            s = run_task(task_name, env_client, llm_client)
            scores.append(s)
            time.sleep(1)
    finally:
        env_client.close()

    print(f"\n[DEBUG] ===== Summary =====", flush=True)
    for t, s in zip(TASKS, scores):
        print(f"[DEBUG]   {t}: {s:.3f}", flush=True)
    print(f"[DEBUG]   mean: {sum(scores)/len(scores):.3f}", flush=True)

def run_api():
    import uvicorn
    uvicorn.run("inference:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()