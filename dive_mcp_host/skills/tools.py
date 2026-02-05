"""Skill tools for LangChain agents.

Provides tools for reading and installing skills.
"""

# ruff: noqa: PLR2004, PLR0911

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolArg, tool
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig  # noqa: TC002

from dive_mcp_host.skills.manager import SkillManager, get_skill_manager
from dive_mcp_host.skills.models import Skill

logger = logging.getLogger(__name__)


def _build_dive_skill_description(skills: list[Skill]) -> str:
    """Build the dynamic tool description listing available skills."""
    base = (
        "Load a skill to get detailed instructions for a specific task.\n"
        "Skills provide specialized knowledge and step-by-step guidance.\n"
        "Use this when a task matches an available skill's description."
    )
    if not skills:
        return base + "\n\nNo skills are currently available."

    lines = ["<available_skills>"]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{skill.name}</name>")
        if skill.description:
            lines.append(f"    <description>{skill.description}</description>")
        lines.append("  </skill>")
    lines.append("</available_skills>")

    return base + "\nOnly the skills listed here are available:\n" + "\n".join(lines)


class _DiveSkillInput(BaseModel):
    """Input schema for the dive_skill tool."""

    skill_name: str = Field(description="Name of the skill to load.")


def create_dive_skill_tool(manager: SkillManager) -> BaseTool:
    """Create the dive_skill tool with a dynamic description of available skills.

    Args:
        manager: The SkillManager instance to use for reading skills.

    Returns:
        A BaseTool instance named 'dive_skill'.
    """
    from langchain_core.tools import StructuredTool

    skills = manager.list_skills()
    description = _build_dive_skill_description(skills)

    def read_skill_content(skill_name: str) -> str:
        """Read a skill's content using the bound SkillManager."""
        content = manager.get_skill_content(skill_name)

        if content is None:
            installed = manager.list_skills()
            if installed:
                available = ", ".join(s.name for s in installed)
                return f"Error: Skill '{skill_name}' not found. Available skills: {available}"
            return f"Error: Skill '{skill_name}' not found. No skills are installed."

        if len(content) > 100000:
            content = content[:100000] + "\n... (truncated)"

        base_dir = str(manager.skill_dir / skill_name)
        return f"## Skill: {skill_name}\n\n**Base directory**: {base_dir}\n\n{content}"

    return StructuredTool.from_function(
        func=read_skill_content,
        name="dive_skill",
        description=description,
        args_schema=_DiveSkillInput,
    )


@tool(
    description="""Install a skill from a local directory containing a SKILL.md file.

Copies the entire skill directory (SKILL.md and all accompanying files such as
scripts, templates, etc.) into the skill directory.

To install a skill from a remote source (e.g., GitHub), first clone or download
the repository to a temporary directory using git/bash, then use this tool to
install from the local path.

Will refuse to overwrite an existing skill unless overwrite=True.
Validates skill_name against path traversal characters (/, \\, ..).

Example:
  dive_install_skill_from_path(
    skill_name="code-review",
    skill_path="/tmp/skills-repo/skills/code-review",
  )
"""
)
async def dive_install_skill_from_path(
    skill_name: Annotated[
        str,
        Field(description="Directory name for the skill (e.g., 'code-review')."),
    ],
    skill_path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to the skill directory containing SKILL.md "
                "(e.g., '/tmp/skills-repo/skills/code-review')."
            )
        ),
    ],
    overwrite: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to overwrite an existing skill.",
        ),
    ] = False,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Install a skill by copying its entire directory from a local path."""
    from dive_mcp_host.mcp_installer_plugin.tools.common import (
        _check_aborted,
        _ensure_config,
        _get_abort_signal,
    )

    config = _ensure_config(config)
    abort_signal = _get_abort_signal(config)

    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Validate skill_name against path traversal
    if "/" in skill_name or "\\" in skill_name or ".." in skill_name:
        return "Error: Invalid skill name. Must not contain '/', '\\', or '..'."

    if not skill_name.strip():
        return "Error: Skill name must not be empty."

    source = Path(skill_path)

    # Accept both a directory containing SKILL.md and a direct SKILL.md file path
    if source.is_file() and source.name == "SKILL.md":
        source = source.parent

    if not source.is_dir():
        return f"Error: Source path '{skill_path}' is not a directory."

    skill_md = source / "SKILL.md"
    if not skill_md.exists():
        return f"Error: No SKILL.md found in '{skill_path}'."

    manager = get_skill_manager()
    target_dir = manager.skill_dir / skill_name

    if target_dir.exists() and not overwrite:
        return (
            f"Error: Skill '{skill_name}' already exists. "
            "Set overwrite=True to replace it."
        )

    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source, target_dir)
        return f"Successfully installed skill '{skill_name}' from '{source}'."
    except OSError as e:
        return f"Error installing skill '{skill_name}': {e}"
