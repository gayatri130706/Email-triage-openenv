"""
Core Email Triage environment implementing the OpenEnv interface.
  reset(task_name, seed) → ResetResult
  step(TriageAction)     → StepResult
  state()                → dict
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional

from .email_data import EMAILS, strip_ground_truth
from .graders import grade_action
from .models import (
    Email, EmailObservation, EmailReward,
    ResetResult, StepResult, TriageAction,
)
from .tasks import get_task


class EmailTriageEnv:

    def __init__(self) -> None:
        self._task_name: str = "basic_triage"
        self._task_config: dict = {}
        self._emails_full: List[dict] = []
        self._emails_obs: List[Email] = []
        self._actions: List[TriageAction] = []
        self._rewards: List[float] = []
        self._current_step: int = 0
        self._done: bool = False
        self._seed: Optional[int] = None

    # ------------------------------------------------------------------ reset
    def reset(self, task_name: str = "basic_triage", seed: Optional[int] = 42) -> ResetResult:
        self._task_name = task_name
        self._task_config = get_task(task_name)
        self._seed = seed
        rng = random.Random(seed)
        selector = self._task_config["email_selector"]
        self._emails_full = selector(EMAILS, rng)
        self._emails_obs = [Email(**strip_ground_truth(e)) for e in self._emails_full]
        self._actions = []
        self._rewards = []
        self._current_step = 0
        self._done = False
        return ResetResult(observation=self._build_obs(), done=False, reward=0.0, info={})

    # ------------------------------------------------------------------ step
    def step(self, action: TriageAction) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done. Call reset() to start a new episode.")

        email_full = next((e for e in self._emails_full if e["id"] == action.email_id), None)

        if email_full is None:
            reward_val = 0.0
            detail = EmailReward(value=0.0, reason=f"Invalid email_id '{action.email_id}'", partial_credits={})
        else:
            reward_val, credits, reason = grade_action(action, email_full, self._task_config)
            detail = EmailReward(value=reward_val, reason=reason, partial_credits=credits)

        self._actions.append(action)
        self._rewards.append(reward_val)
        self._current_step += 1

        if self._current_step >= len(self._emails_full):
            self._done = True

        return StepResult(
            observation=self._build_obs(),
            reward=reward_val,
            done=self._done,
            info={
                "step": self._current_step,
                "cumulative_reward": round(sum(self._rewards), 4),
                "email_id_graded": action.email_id,
            },
            reward_detail=detail,
        )

    # ------------------------------------------------------------------ state
    def state(self) -> Dict[str, Any]:
        triaged = [a.email_id for a in self._actions]
        remaining = [e["id"] for e in self._emails_full if e["id"] not in set(triaged)]
        return {
            "task_name": self._task_name,
            "difficulty": self._task_config.get("difficulty"),
            "current_step": self._current_step,
            "total_emails": len(self._emails_full),
            "done": self._done,
            "seed": self._seed,
            "rewards": self._rewards,
            "cumulative_reward": round(sum(self._rewards), 4),
            "mean_reward": round(sum(self._rewards) / len(self._rewards), 4) if self._rewards else 0.0,
            "triaged_ids": triaged,
            "remaining_ids": remaining,
            "actions": [a.model_dump() for a in self._actions],
        }

    # ------------------------------------------------------------------ helper
    def _build_obs(self) -> EmailObservation:
        triaged = [a.email_id for a in self._actions]
        remaining = [e["id"] for e in self._emails_full if e["id"] not in set(triaged)]
        return EmailObservation(
            emails=self._emails_obs,
            current_step=self._current_step,
            total_emails=len(self._emails_full),
            task_name=self._task_name,
            task_description=self._task_config["description"],
            triaged_ids=triaged,
            remaining_ids=remaining,
            inbox_metadata={
                "difficulty": self._task_config.get("difficulty"),
                "requires_label": self._task_config.get("requires_label", False),
                "requires_reply_flag": self._task_config.get("requires_reply_flag", False),
                "requires_summary": self._task_config.get("requires_summary", False),
                "reward_weights": self._task_config.get("reward_weights", {}),
            },
        )
