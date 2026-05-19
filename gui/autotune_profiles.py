"""Autotune profile definitions shared by GUI benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutotuneProfile:
    key: str
    title: str
    description: str
    min_ctx: int
    ctx_values: str
    tasks: str
    task_ids: str
    max_tokens: int
    real_context_chars: int
    request_timeout: int
    task_hard_timeout: int
    task_fail_timeout: int
    allow_ctx_above_16k: bool
    batch_min: int
    batch_max: int
    batch_step: int
    ubatch_min: int
    ubatch_max: int
    ubatch_step: int
    button_label: str

    @property
    def lane_summary(self) -> str:
        task_part = self.tasks if not self.task_ids else f"{self.tasks}:{self.task_ids}"
        reuse_part = "repo-snapshot, no-reuse, no-prime, thinking on"
        return f"ctx={self.ctx_values}, tasks={task_part}, max_tokens={self.max_tokens}, {reuse_part}"


ACTIVE_PROMPT_PROFILE = AutotuneProfile(
    key="active-ctx12k",
    title="Active <16K Prompt",
    description="Current Qwen3.6/RDNA4 cold-first prompt-heavy target.",
    min_ctx=12288,
    ctx_values="12288",
    tasks="quick",
    task_ids="triage_diff,review_bug",
    max_tokens=120,
    real_context_chars=21872,
    request_timeout=180,
    task_hard_timeout=60,
    task_fail_timeout=60,
    allow_ctx_above_16k=False,
    batch_min=4096,
    batch_max=8192,
    batch_step=2048,
    ubatch_min=1024,
    ubatch_max=3072,
    ubatch_step=512,
    button_label="Run Auto-tune <16K",
)

ARCHIVAL_32K_PROFILE = AutotuneProfile(
    key="archival-ctx32k",
    title="Archival 32K Probe",
    description="Large-context reference lane kept for comparison, not the current default target.",
    min_ctx=32768,
    ctx_values="32768",
    tasks="v2-mini",
    task_ids="",
    max_tokens=120,
    real_context_chars=21872,
    request_timeout=120,
    task_hard_timeout=60,
    task_fail_timeout=60,
    allow_ctx_above_16k=True,
    batch_min=2048,
    batch_max=8192,
    batch_step=2048,
    ubatch_min=128,
    ubatch_max=1024,
    ubatch_step=128,
    button_label="Run Auto-tune 32K",
)

AUTOTUNE_PROFILES = (ACTIVE_PROMPT_PROFILE, ARCHIVAL_32K_PROFILE)
AUTOTUNE_PROFILE_BY_KEY = {profile.key: profile for profile in AUTOTUNE_PROFILES}
DEFAULT_AUTOTUNE_PROFILE_KEY = ACTIVE_PROMPT_PROFILE.key


def autotune_profile_by_key(key: str) -> AutotuneProfile:
    return AUTOTUNE_PROFILE_BY_KEY.get(key, ACTIVE_PROMPT_PROFILE)