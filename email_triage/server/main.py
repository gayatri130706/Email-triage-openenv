"""
FastAPI server — exposes Email Triage env over HTTP.

POST /reset   → ResetResult
POST /step    → StepResult
GET  /state   → dict
GET  /health  → {"status": "ok"}
GET  /tasks   → available tasks
GET  /docs    → Swagger UI
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import EmailTriageEnv
from env.models import ResetResult, StepResult, TriageAction
from env.tasks import TASKS

app = FastAPI(
    title="Email Triage OpenEnv",
    description="OpenEnv environment for AI email triage agents.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env: Optional[EmailTriageEnv] = None


class ResetRequest(BaseModel):
    task: str = "basic_triage"
    seed: Optional[int] = 42


@app.get("/health")
def health():
    return {"status": "ok", "service": "email-triage-env"}


@app.get("/tasks")
def list_tasks():
    return {
        name: {
            "display_name": cfg["display_name"],
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
            "email_count": cfg["email_count"],
            "requires_label": cfg["requires_label"],
            "requires_reply_flag": cfg["requires_reply_flag"],
            "requires_summary": cfg["requires_summary"],
            "reward_weights": cfg["reward_weights"],
        }
        for name, cfg in TASKS.items()
    }


@app.post("/reset", response_model=ResetResult)
def reset(req: ResetRequest = ResetRequest()) -> ResetResult:
    global _env
    if req.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{req.task}'. Valid: {list(TASKS.keys())}")
    _env = EmailTriageEnv()
    return _env.reset(task_name=req.task, seed=req.seed)


@app.post("/step", response_model=StepResult)
def step(action: TriageAction) -> StepResult:
    if _env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    try:
        return _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state() -> dict:
    if _env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    return _env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
