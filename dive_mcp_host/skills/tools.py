"""Skill tools for LangChain agents.

Provides tools for installing skills.
"""

# ruff: noqa: PLR0911

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from langgraph.pregel.main import ensure_config
from pydantic import Field

from dive_mcp_host.host.agents.agent_factory import get_abort_signal, get_skill_manager
from dive_mcp_host.internal_tools.tools.common import (
    check_aborted,
)

logger = logging.getLogger(__name__)


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
    config = ensure_config(config)
    abort_signal = get_abort_signal(config)
    skill_manager = get_skill_manager(config)

    if not skill_manager:
        return "Error: SkillManager not loaded"

    if check_aborted(abort_signal):
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

    target_dir = skill_manager.skill_dir / skill_name
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
