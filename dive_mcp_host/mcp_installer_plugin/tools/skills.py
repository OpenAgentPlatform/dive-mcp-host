"""Skill management tools for reading, searching, and installing skills.

Skills are directories containing a SKILL.md file with YAML frontmatter
(name, description) and markdown instructions.
"""

# ruff: noqa: PLR2004, PLR0911

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from pathlib import Path

import yaml
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from pydantic import BaseModel, Field

from dive_mcp_host.mcp_installer_plugin.tools.common import (
    _check_aborted,
    _ensure_config,
    _get_abort_signal,
)

logger = logging.getLogger(__name__)


def _get_skill_dir() -> Path:
    """Return the skill directory path from environment configuration."""
    from dive_mcp_host.env import DIVE_SKILL_DIR

    return DIVE_SKILL_DIR


def _parse_skill_frontmatter(content: str) -> dict[str, str]:
    """Parse YAML frontmatter from SKILL.md content.

    Expects content starting with '---' delimiter, followed by YAML,
    and closed with another '---' delimiter.

    Returns:
        Dict with parsed frontmatter fields (e.g. name, description).
        Returns empty dict if frontmatter is missing or invalid.
    """
    if not content.startswith("---"):
        return {}

    end_index = content.find("---", 3)
    if end_index == -1:
        return {}

    frontmatter_text = content[3:end_index].strip()
    try:
        parsed = yaml.safe_load(frontmatter_text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except yaml.YAMLError:
        return {}


def _get_installed_skills() -> list[dict[str, str]]:
    """Scan the skill directory and return metadata for all installed skills.

    Returns:
        List of dicts with 'dir_name', 'name', and 'description' keys.
        Optional frontmatter fields are included only when present.
    """
    skill_dir = _get_skill_dir()

    if not skill_dir.exists():
        return []

    skills: list[dict[str, str]] = []
    try:
        for entry in sorted(skill_dir.iterdir()):
            if not entry.is_dir():
                continue

            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue

            try:
                content = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue

            frontmatter = _parse_skill_frontmatter(content)
            skill_info: dict[str, str] = {
                "dir_name": entry.name,
                "name": frontmatter.get("name", entry.name),
                "description": frontmatter.get("description", ""),
            }
            for key in ("license", "compatibility", "metadata", "allowed_tools"):
                val = frontmatter.get(key)
                if val is not None:
                    skill_info[key] = val
            skills.append(skill_info)
    except OSError:
        pass

    return skills


def _build_dive_skill_description(
    skills: list[dict[str, str]],
) -> str:
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
        lines.append(f"    <name>{skill['dir_name']}</name>")
        if skill["description"]:
            lines.append(f"    <description>{skill['description']}</description>")
        lines.append("  </skill>")
    lines.append("</available_skills>")

    return base + "\nOnly the skills listed here are available:\n" + "\n".join(lines)


class _DiveSkillInput(BaseModel):
    """Input schema for the dive_skill tool."""

    skill_name: str = Field(description="Name of the skill to load.")


def _read_skill_content(skill_name: str) -> str:
    """Read a skill's SKILL.md file content."""
    skill_dir = _get_skill_dir()
    skill_path = skill_dir / skill_name / "SKILL.md"

    try:
        if not skill_path.exists():
            # List available skills in error message
            installed = _get_installed_skills()
            if installed:
                available = ", ".join(s["dir_name"] for s in installed)
                return (
                    f"Error: Skill '{skill_name}' not found. "
                    f"Available skills: {available}"
                )
            return f"Error: Skill '{skill_name}' not found. No skills are installed."
        if not skill_path.is_file():
            return f"Error: SKILL.md is not a file for skill '{skill_name}'."

        content = skill_path.read_text(encoding="utf-8")
        if len(content) > 100000:
            content = content[:100000] + "\n... (truncated)"

        base_dir = str(skill_dir / skill_name)
        return f"## Skill: {skill_name}\n\n**Base directory**: {base_dir}\n\n{content}"

    except OSError as e:
        return f"Error reading skill '{skill_name}': {e}"


def create_dive_skill_tool() -> BaseTool:
    """Create the dive_skill tool with a dynamic description of available skills.

    Scans the skill directory for installed skills and builds a tool whose
    description lists all available skills. The tool reads the full SKILL.md
    content for a given skill name.

    Returns:
        A BaseTool instance named 'dive_skill'.
    """
    from langchain_core.tools import StructuredTool

    skills = _get_installed_skills()
    description = _build_dive_skill_description(skills)

    return StructuredTool.from_function(
        func=lambda skill_name: _read_skill_content(skill_name),
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
    from pathlib import Path

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

    skill_dir = _get_skill_dir()
    target_dir = skill_dir / skill_name

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
