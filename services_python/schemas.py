"""
Pydantic Schemas for API Validation
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from services_python.architecture_config import validate_architecture_config

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ImportError:
    from dataclasses import dataclass
    from dataclasses import field as Field

    BaseModel = None


class JobType(StrEnum):
    FINE_TUNE = "fine_tune"
    PRETRAIN = "pretrain"
    EVAL = "eval"
    PREPROCESS = "preprocess"


class PriorityTier(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


if BaseModel:

    class JobCreateRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        job_type: JobType = Field(default=JobType.FINE_TUNE)
        base_model: str = Field(
            default="distribai-small", min_length=1, max_length=256, pattern=r"^[a-zA-Z0-9/_-]+$"
        )
        model_name: str | None = Field(default=None, max_length=256)
        dataset_ref: str = Field(default="", max_length=512)
        description: str = Field(default="", max_length=1024)
        steps: int = Field(default=100, ge=1, le=1000000)
        batch_size: int = Field(default=32, ge=1, le=2048)
        priority: int = Field(default=0, ge=0, le=100)
        priority_tier: PriorityTier = Field(default=PriorityTier.P1)
        hparams: dict = Field(default_factory=dict)
        submitter_id: str = Field(default="distribai", max_length=64)
        org: str = Field(default="DistribAI", max_length=64)
        deadline_seconds: int = Field(default=600, ge=60, le=86400)
        max_attempts: int = Field(default=3, ge=1, le=10)
        steps_per_task: int | None = Field(default=None, ge=1, le=10000)
        batch_blob_url: str | None = Field(default=None, max_length=512)
        weight_blob_url: str | None = Field(default=None, max_length=512)
        script_package_b64: str | None = Field(default=None, max_length=8_000_000)
        script_content: str | None = Field(default=None, max_length=2_000_000)
        requirements: str | None = Field(default=None, max_length=16_384)
        architecture_config: dict[str, Any] | None = Field(default=None)

        @field_validator("architecture_config")
        @classmethod
        def validate_architecture(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
            return validate_architecture_config(v) if v is not None else None

        @field_validator("hparams")
        @classmethod
        def validate_hparams(cls, v: dict) -> dict:
            if not isinstance(v, dict):
                raise ValueError("hparams must be a dictionary")
            if "lr" in v:
                lr = v["lr"]
                if not (1e-6 <= float(lr) <= 1.0):
                    raise ValueError("learning rate must be between 1e-6 and 1.0")
            if "epochs" in v:
                epochs = v["epochs"]
                if not (1 <= int(epochs) <= 1000):
                    raise ValueError("epochs must be between 1 and 1000")
            return v

        @field_validator("dataset_ref", "batch_blob_url", "weight_blob_url")
        @classmethod
        def validate_s3_refs(cls, v: str | None) -> str | None:
            if v is None:
                return v
            if v == "":
                return v
            if ".." in v or "~" in v:
                raise ValueError("Invalid characters in S3 reference")
            if v.startswith("s3://"):
                parts = v[5:].split("/", 1)
                if len(parts) < 1 or not parts[0]:
                    raise ValueError("Invalid S3 URI format")
            return v

    class VoteRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        job_id: str = Field(min_length=4, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
        credits: int = Field(ge=1, le=1000000)

        @field_validator("job_id")
        @classmethod
        def validate_job_id(cls, v: str) -> str:
            v = v.replace("/", "_").replace("\\", "_")
            if not v or v.startswith("_"):
                raise ValueError("Invalid job ID")
            return v

    class GovernanceVoteCreateRequest(BaseModel):
        """Body for opening a governance voting round (admin API).

        Distinct from :class:`VoteRequest`, which spends credits on a job
        priority boost — the two shapes were previously conflated, making
        the admin create endpoint reject every valid payload.
        """

        model_config = ConfigDict(extra="forbid")
        proposer: str = Field(min_length=1, max_length=64)
        title: str = Field(min_length=1, max_length=256)
        vote_type: str = Field(default="job_priority", max_length=64)
        description: str = Field(default="", max_length=4096)
        options: list[str] = Field(default_factory=lambda: ["yes", "no"])

        @field_validator("options")
        @classmethod
        def validate_options(cls, v: list[str]) -> list[str]:
            if not 2 <= len(v) <= 16:
                raise ValueError("options must contain between 2 and 16 entries")
            if any(not isinstance(item, str) or not item.strip() or len(item) > 128 for item in v):
                raise ValueError("each option must be a non-empty string of at most 128 chars")
            if len({item.strip() for item in v}) != len(v):
                raise ValueError("options must be unique")
            return v

    class NodeRegisterRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        node_id: str | None = Field(default=None, max_length=64)
        invite_code: str | None = Field(default=None, max_length=64)
        public_key: str = Field(default="", max_length=512)
        os: str = Field(default="unknown", max_length=32)
        gpu_model: str = Field(default="unknown", max_length=64)
        driver_version: str = Field(default="", max_length=32)
        challenge_id: str | None = Field(default=None, max_length=128)
        nonce: str | None = Field(default=None, max_length=128)
        vram_mb: int = Field(default=0, ge=0, le=2_000_000)
        cpu_cores: int | None = Field(default=None, ge=1, le=4096)
        ram_gb: float | None = Field(default=None, ge=0.0, le=1_000_000.0)

    class JobResponse(BaseModel):
        model_config = ConfigDict(extra="ignore")
        job_id: str
        queue_position: int | None = None
        current_votes: int = 0
        estimated_start_hours: float | None = None
        status: str = "queued"

    class VoteResponse(BaseModel):
        model_config = ConfigDict(extra="ignore")
        vote_id: str
        credits_deducted: int
        job_new_vote_total: int
        job_new_queue_position: int | None = None
        your_new_balance: float

    class CreditBalanceResponse(BaseModel):
        model_config = ConfigDict(extra="ignore")
        confirmed: float
        pending: float
        lifetime_earned: float
        lifetime_votes_cast: float
        multipliers: dict | None = None
else:
    from dataclasses import dataclass

    @dataclass
    class JobCreateRequest:
        job_type: str = "fine_tune"
        base_model: str = "distribai-small"
        model_name: str | None = None
        dataset_ref: str = ""
        description: str = ""
        steps: int = 100
        batch_size: int = 32
        priority: int = 0
        priority_tier: str = "P1"
        hparams: dict = Field(default_factory=dict)
        submitter_id: str = "distribai"
        org: str = "DistribAI"
        deadline_seconds: int = 600
        max_attempts: int = 3
        steps_per_task: int | None = None
        batch_blob_url: str | None = None
        weight_blob_url: str | None = None
        script_package_b64: str | None = None
        script_content: str | None = None
        requirements: str | None = None
        architecture_config: dict[str, Any] | None = None

    @dataclass
    class VoteRequest:
        job_id: str = ""
        credits: int = 0

    @dataclass
    class GovernanceVoteCreateRequest:
        proposer: str = ""
        title: str = ""
        vote_type: str = "job_priority"
        description: str = ""
        options: list[str] = Field(default_factory=lambda: ["yes", "no"])

    @dataclass
    class NodeRegisterRequest:
        node_id: str | None = None
        invite_code: str | None = None
        public_key: str = ""
        os: str = "unknown"
        gpu_model: str = "unknown"
        driver_version: str = ""
        challenge_id: str | None = None
        nonce: str | None = None
        vram_mb: int = 0
        cpu_cores: int | None = None
        ram_gb: float | None = None

    @dataclass
    class JobResponse:
        job_id: str = ""
        queue_position: int | None = None
        current_votes: int = 0
        estimated_start_hours: float | None = None
        status: str = "queued"

    @dataclass
    class VoteResponse:
        vote_id: str = ""
        credits_deducted: int = 0
        job_new_vote_total: int = 0
        job_new_queue_position: int | None = None
        your_new_balance: float = 0.0

    @dataclass
    class CreditBalanceResponse:
        confirmed: float = 0.0
        pending: float = 0.0
        lifetime_earned: float = 0.0
        lifetime_votes_cast: float = 0.0
        multipliers: dict | None = None


def validate_job_create(data: dict) -> tuple[bool, str | None, Any | None]:
    try:
        if BaseModel:
            req = JobCreateRequest(**data)
            return True, None, req
        else:
            req = JobCreateRequest(**data)
            if req.steps < 1 or req.steps > 1000000:
                return False, "steps must be between 1 and 1000000", None
            if req.batch_size < 1 or req.batch_size > 2048:
                return False, "batch_size must be between 1 and 2048", None
            if len(req.base_model) > 256:
                return False, "base_model too long", None
            if req.priority < 0 or req.priority > 100:
                return False, "priority must be between 0 and 100", None
            return True, None, req
    except Exception as e:
        return False, str(e), None


def validate_vote(data: dict) -> tuple[bool, str | None, Any | None]:
    try:
        if BaseModel:
            req = VoteRequest(**data)
            return True, None, req
        else:
            req = VoteRequest(**data)
            if req.credits < 1:
                return False, "credits must be positive", None
            if not req.job_id:
                return False, "job_id required", None
            return True, None, req
    except Exception as e:
        return False, str(e), None


def validate_governance_vote_create(data: dict) -> tuple[bool, str | None, Any | None]:
    """Validate the admin governance-vote creation body."""
    try:
        req = GovernanceVoteCreateRequest(**data)
        if not BaseModel:
            if not req.proposer:
                return False, "proposer required", None
            if not req.title:
                return False, "title required", None
        return True, None, req
    except Exception as e:
        return False, str(e), None


def validate_node_register(data: dict) -> tuple[bool, str | None, Any | None]:
    try:
        if BaseModel:
            req = NodeRegisterRequest(**data)
            return True, None, req
        else:
            req = NodeRegisterRequest(**data)
            return True, None, req
    except Exception as e:
        return False, str(e), None
