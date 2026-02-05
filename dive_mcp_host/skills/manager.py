"""Skill manager for reading and managing skills."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import frontmatter

from dive_mcp_host.env import DIVE_SKILL_DIR
from dive_mcp_host.skills.models import Skill, SkillMeta
from dive_mcp_host.skills.tools import create_dive_skill_tool

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


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

    def get_tools(self) -> list[BaseTool]:
        """Transform skills into tools."""
        return [create_dive_skill_tool(self)]
