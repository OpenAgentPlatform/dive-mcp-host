"""Skill data models.

Ref: https://agentskills.io/specification
"""

from pathlib import Path

from pydantic import BaseModel, Field


class SkillMeta(BaseModel):
    """Definition of skills frontmatter."""

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
    )
    description: str = Field(min_length=1, max_length=1024)
    license: str | None = None
    compatibility: str | None = Field(default=None, min_length=1, max_length=500)
    metadata: dict[str, str] | None = None
    allowed_tools: str | None = Field(default=None)


class Skill(BaseModel):
    """Contnet loaded from SKILL.md."""

    meta: SkillMeta
    content: str
    base_dir: Path
