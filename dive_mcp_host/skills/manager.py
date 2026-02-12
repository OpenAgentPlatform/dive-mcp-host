"""Skill manager for reading and managing skills."""

# ruff: noqa: PLR2004

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import frontmatter
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from dive_mcp_host.env import DIVE_SKILL_DIR
from dive_mcp_host.skills.models import Skill, SkillMeta

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class _DiveSkillInput(BaseModel):
    """Input schema for the dive_skill tool."""

    skill_name: str = Field(description="Name of the skill to load.")


class SkillManager:
    """Manager for reading and listing skills from the skill directory."""

    def __init__(self, skill_dir: Path = DIVE_SKILL_DIR) -> None:
        """Initialize the skill manager.

        Args:
            skill_dir: Path to the directory containing skill folders.
        """
        self._skill_dir = skill_dir
        self._skills_cache: dict[str, Skill] = {}
        self.refresh()

    def refresh(self) -> None:
        """Reload skills."""
        if not self._skill_dir.exists():
            logger.warning("skill dir not found: %s", self._skill_dir)
            self._skills_cache = {}

        result: dict[str, Skill] = {}
        try:
            for entry in sorted(self._skill_dir.iterdir()):
                if not entry.is_dir():
                    continue

                skill = self._load_skill(entry.name)

                if skill is None:
                    logger.warning(
                        "no SKILL.md found in %s", self._skill_dir / entry.name
                    )
                    continue

                if skill.meta.name in result:
                    logger.warning(
                        "Found duplicate skill names, will shadow previous skill: %s",
                        skill.meta.name,
                    )

                result[skill.meta.name] = skill

        except Exception:
            logger.exception("error when loading skill")

        logger.debug("refresh found %s skills under: %s", len(result), self._skill_dir)
        self._skills_cache = result

    @property
    def skill_dir(self) -> Path:
        """Return the skill directory path."""
        return self._skill_dir

    def list_skills(self) -> list[Skill]:
        """List all installed skills."""
        return list(self._skills_cache.values())

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills_cache.get(name)

    def _load_skill(self, entry: str) -> Skill | None:
        """Load a skill from its directory."""
        skill_file = self._skill_dir / entry / "SKILL.md"
        logger.debug("load skill: %s", skill_file)

        if not skill_file.exists():
            logger.warning("skill file not found: %s", skill_file)
            return None

        try:
            with skill_file.open("r") as f:
                post = frontmatter.load(f)
            return Skill(
                meta=SkillMeta.model_validate(post.metadata),
                content=post.content,
                base_dir=skill_file,
            )
        except Exception:
            logger.exception("Failed to load skill: %s", skill_file)
            return None

    def get_tools(self, skill_names: list[str] | None = None) -> list[BaseTool]:
        """Transform skills into tools.

        Args:
            skill_names: Optional list of skill names to include.
                         If None, all skills are included.
        """
        if skill_names is not None:
            skills = [
                s for name in skill_names if (s := self.get_skill(name)) is not None
            ]
        else:
            skills = self.list_skills()

        base_desc = (
            "Load a skill to get detailed instructions for a specific task.\n"
            "Skills provide specialized knowledge and step-by-step guidance.\n"
            "Use this when a task matches an available skill's description."
        )
        if not skills:
            description = base_desc + "\n\nNo skills are currently available."
        else:
            lines = ["<available_skills>"]
            for skill in skills:
                lines.append("  <skill>")
                lines.append(f"    <name>{skill.meta.name}</name>")
                if skill.meta.description:
                    lines.append(
                        f"    <description>{skill.meta.description}</description>"
                    )
                lines.append("  </skill>")
            lines.append("</available_skills>")
            description = (
                base_desc
                + "\nOnly the skills listed here are available:\n"
                + "\n".join(lines)
            )

        skills_dict = {s.meta.name: s for s in skills}

        def read_skill_content(skill_name: str) -> str:
            """Read a skill's content."""
            skill = skills_dict.get(skill_name)

            if skill is None:
                if skills_dict:
                    available = ", ".join(skills_dict)
                    return (
                        f"Error: Skill '{skill_name}' not found. "
                        f"Available skills: {available}"
                    )
                return (
                    f"Error: Skill '{skill_name}' not found. No skills are installed."
                )

            content = skill.content
            if len(content) > 100000:
                content = content[:100000] + "\n... (truncated)"

            return f"""
## Skill: {skill_name}

**Base directory**: {skill.base_dir}

{content}"""

        return [
            StructuredTool.from_function(
                func=read_skill_content,
                name="dive_skill",
                description=description,
                args_schema=_DiveSkillInput,
            )
        ]
