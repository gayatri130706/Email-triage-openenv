"""
Pydantic models for the Email Triage OpenEnv environment.
Implements typed Observation, Action, and Reward per the OpenEnv spec.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class EmailPriority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SPAM = "spam"


class EmailLabel(str, Enum):
    BILLING = "billing"
    SUPPORT = "support"
    INFRASTRUCTURE = "infrastructure"
    SALES = "sales"
    HR = "hr"
    OTHER = "other"


class Email(BaseModel):
    id: str = Field(..., description="Unique email identifier")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    body: str = Field(..., description="Full email body text")
    timestamp: str = Field(..., description="ISO-8601 received timestamp")
    labels_existing: List[str] = Field(default_factory=list)


class EmailObservation(BaseModel):
    emails: List[Email] = Field(..., description="All emails the agent must triage")
    current_step: int = Field(..., description="Steps completed (0 at reset)")
    total_emails: int = Field(..., description="Total emails this episode")
    task_name: str = Field(..., description="Active task identifier")
    task_description: str = Field(..., description="Natural language goal")
    triaged_ids: List[str] = Field(default_factory=list)
    remaining_ids: List[str] = Field(default_factory=list)
    inbox_metadata: Dict[str, Any] = Field(default_factory=dict)


class TriageAction(BaseModel):
    email_id: str = Field(..., description="ID of email being triaged")
    priority: EmailPriority = Field(..., description="urgent|high|normal|low|spam")
    label: Optional[EmailLabel] = Field(None, description="billing|support|infrastructure|sales|hr|other")
    reply_needed: bool = Field(False, description="Does this email require a reply?")
    summary: Optional[str] = Field(None, description="1-sentence executive summary (hard task only)")


class EmailReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Step reward in [0,1]")
    reason: str = Field(..., description="Human-readable explanation")
    partial_credits: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: EmailObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    reward_detail: EmailReward


class ResetResult(BaseModel):
    observation: EmailObservation
    done: bool = False
    reward: float = 0.0
    info: Dict[str, Any] = Field(default_factory=dict)
