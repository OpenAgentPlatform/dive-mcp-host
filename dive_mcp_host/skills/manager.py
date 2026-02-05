"""Skill manager for reading and managing skills."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

from dive_mcp_host.skills.models import Skill

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def get_skill_manager() -> "SkillManager":
    """Get the default SkillManager instance using DIVE_SKILL_DIR."""
    from dive_mcp_host.env import DIVE_SKILL_DIR

    return SkillManager(DIVE_SKILL_DIR)


class SkillManager:
    """Manager for reading and listing skills from the skill directory."""

    def __init__(self, skill_dir: Path) -> None:
        """Initialize the skill manager.

        Args:
            skill_dir: Path to the directory containing skill folders.
        """
        self._skill_dir = skill_dir

    @property
    def skill_dir(self) -> Path:
        """Return the skill directory path."""
        return self._skill_dir

    def list_skills(self) -> list[Skill]:
        """List all installed skills.

        Returns:
            List of Skill objects for all valid installed skills.
        """
        if not self._skill_dir.exists():
            return []

        skills: list[Skill] = []
        try:
            for entry in sorted(self._skill_dir.iterdir()):
                if not entry.is_dir():
                    continue

                skill = self._load_skill(entry.name)
                if skill is not None:
                    skills.append(skill)
        except OSError:
            pass

        return skills

    def get_skill(self, skill_name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            skill_name: Name of the skill directory.

        Returns:
            Skill object if found and valid, None otherwise.
        """
        return self._load_skill(skill_name)

    def get_skill_content(self, skill_name: str) -> str | None:
        """Get the body content of a skill (without frontmatter).

        Args:
            skill_name: Name of the skill directory.

        Returns:
            The skill's markdown content (body only), or None if not found.
        """
        skill_path = self._skill_dir / skill_name / "SKILL.md"

        if not skill_path.exists() or not skill_path.is_file():
            return None

        try:
            post = frontmatter.load(skill_path)
            return post.content
        except Exception:  # noqa: BLE001
            return None

    def _load_skill(self, skill_name: str) -> Skill | None:
        """Load a skill from its directory.

        Args:
            skill_name: Name of the skill directory.

        Returns:
            Skill object if valid, None otherwise.
        """
        skill_file = self._skill_dir / skill_name / "SKILL.md"

        if not skill_file.exists():
            return None

        try:
            post = frontmatter.load(skill_file)
            metadata = dict(post.metadata) if post.metadata else {}

            return Skill(
                name=metadata.get("name", skill_name),
                description=metadata.get("description", ""),
                license=metadata.get("license"),
                compatibility=metadata.get("compatibility"),
                metadata=metadata.get("metadata"),
                allowed_tools=metadata.get("allowed_tools"),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to load skill: %s", skill_name)
            return None

    def get_tools(self) -> list["BaseTool"]:
        """Get tools provided by skills.

        Returns:
            List of tools from installed skills.
        """
        from dive_mcp_host.skills.tools import create_dive_skill_tool

        return [create_dive_skill_tool(self)]
