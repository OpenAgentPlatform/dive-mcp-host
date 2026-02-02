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
from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

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


def _to_github_raw_url(url: str) -> str:
    """Convert a GitHub blob URL to a raw.githubusercontent.com URL.

    Converts URLs like:
      https://github.com/user/repo/blob/main/path/SKILL.md
    To:
      https://raw.githubusercontent.com/user/repo/main/path/SKILL.md

    Non-GitHub URLs are returned unchanged.
    """
    import re

    match = re.match(r"https?://github\.com/([^/]+/[^/]+)/blob/(.+)", url)
    if match:
        return f"https://raw.githubusercontent.com/{match.group(1)}/{match.group(2)}"
    return url


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


@tool(
    description="""Read the full content of an installed skill.

Use this to retrieve the SKILL.md file for a given skill name.
Returns the complete content including frontmatter and instructions.
Truncates content at 100,000 characters if the file is very large.

Example:
  read_skill(skill_name="code-review")
"""
)
async def read_skill(
    skill_name: Annotated[
        str,
        Field(description="Name of the skill directory to read."),
    ],
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Read a skill's SKILL.md file."""
    config = _ensure_config(config)
    abort_signal = _get_abort_signal(config)

    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    skill_dir = _get_skill_dir()
    skill_path = skill_dir / skill_name / "SKILL.md"

    try:
        if not skill_path.exists():
            return f"Error: Skill '{skill_name}' not found."
        if not skill_path.is_file():
            return f"Error: SKILL.md is not a file for skill '{skill_name}'."

        content = skill_path.read_text(encoding="utf-8")
        if len(content) > 100000:
            return content[:100000] + "\n... (truncated)"
        return content

    except OSError as e:
        return f"Error reading skill '{skill_name}': {e}"


@tool(
    description="""Search for installed skills.

Lists all installed skills with their name and description from frontmatter.
Optionally filter by a case-insensitive query that matches against
skill name or description.

Returns a formatted list of matching skills, or a message if none are found.

Example:
  search_skills()  # List all skills
  search_skills(query="git")  # Filter skills matching "git"
"""
)
async def search_skills(
    query: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional case-insensitive filter on name or description.",
        ),
    ] = None,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Search installed skills."""
    config = _ensure_config(config)
    abort_signal = _get_abort_signal(config)

    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    skill_dir = _get_skill_dir()

    if not skill_dir.exists():
        skill_dir.mkdir(parents=True, exist_ok=True)
        return "No skills installed."

    results: list[str] = []
    query_lower = query.lower() if query else None

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
            name = frontmatter.get("name", entry.name)
            description = frontmatter.get("description", "")

            if query_lower and (
                query_lower not in name.lower()
                and query_lower not in description.lower()
                and query_lower not in entry.name.lower()
            ):
                continue

            results.append(
                f"- {entry.name}: {name} - {description}"
                if description
                else f"- {entry.name}: {name}"
            )

    except OSError as e:
        return f"Error searching skills: {e}"

    if not results:
        if query:
            return f"No skills found matching '{query}'."
        return "No skills installed."

    return f"Installed skills ({len(results)}):\n" + "\n".join(results)


@tool(
    description="""Get the guidelines and prompt for installing a skill.

Call this tool BEFORE installing any skill. It returns the required format
and rules for creating a valid SKILL.md file, including frontmatter structure
and content expectations.

Example:
  dive_skill_install_guide()
"""
)
async def dive_skill_install_guide(
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Return skill installation guidelines."""
    config = _ensure_config(config)
    abort_signal = _get_abort_signal(config)

    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    return """\
## Skill Installation Guide

A skill is a SKILL.md file stored in its own directory under the skill directory.

### SKILL.md Format

The file MUST start with YAML frontmatter enclosed by `---` delimiters:

```markdown
---
name: Human-Readable Skill Name
description: A short summary of what this skill does.
---

## Instructions

The body of the skill in markdown. This is the actual prompt/instructions
that will be used when the skill is invoked.
```

### Rules

1. **Frontmatter is required**: Must contain at least `name` and `description`.
2. **`name`**: A human-readable display name for the skill.
3. **`description`**: A concise summary (one or two sentences).
4. **Body**: The markdown content after the frontmatter. This is the skill's \
actual instructions/prompt.
5. **`skill_name`** (directory name): Use lowercase kebab-case \
(e.g., `code-review`, `git-commit`). Must not contain `/`, `\\`, or `..`.
6. **Overwrite**: Existing skills will NOT be overwritten unless you explicitly \
set `overwrite=True`.

### Available Install Methods (in order of preference)

1. **`dive_install_skill_from_path`**: Copy a local SKILL.md file by path. \
Use this when the user already has a SKILL.md file on disk.
2. **`dive_install_skill_from_url`**: Download a SKILL.md file from a URL. \
Use this when the user provides a link to a SKILL.md file.
3. **`dive_install_skill_from_content`**: Provide name, description, and content \
directly. Use this only as a last resort when no file or URL is available.
"""


@tool(
    description="""Install a new skill by creating a SKILL.md file.

Creates a skill directory with a SKILL.md file containing the provided content.
The content should include YAML frontmatter with at least a description field.

Will refuse to overwrite an existing skill unless overwrite=True.
Validates skill_name against path traversal characters (/, \\, ..).

Example:
  dive_install_skill_from_content(
    skill_name="code-review",
    description="Reviews code for best practices",
    content="Detailed instructions for code review...",
  )
"""
)
async def dive_install_skill_from_content(
    skill_name: Annotated[
        str,
        Field(description="Directory name for the skill (e.g., 'code-review')."),
    ],
    description: Annotated[
        str,
        Field(description="Short description of what the skill does."),
    ],
    content: Annotated[
        str,
        Field(
            description=(
                "The full markdown content for the skill"
                " instructions (body after frontmatter)."
            )
        ),
    ],
    display_name: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional human-readable display name. Defaults to skill_name.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to overwrite an existing skill.",
        ),
    ] = False,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Install a skill by creating its directory and SKILL.md file."""
    config = _ensure_config(config)
    abort_signal = _get_abort_signal(config)

    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Validate skill_name against path traversal
    if "/" in skill_name or "\\" in skill_name or ".." in skill_name:
        return "Error: Invalid skill name. Must not contain '/', '\\', or '..'."

    if not skill_name.strip():
        return "Error: Skill name must not be empty."

    skill_dir = _get_skill_dir()
    target_dir = skill_dir / skill_name
    skill_file = target_dir / "SKILL.md"

    if skill_file.exists() and not overwrite:
        return (
            f"Error: Skill '{skill_name}' already exists. "
            "Set overwrite=True to replace it."
        )

    # Build SKILL.md content with frontmatter
    name = display_name or skill_name
    frontmatter = yaml.dump(
        {"name": name, "description": description},
        default_flow_style=False,
        allow_unicode=True,
    ).strip()
    full_content = f"---\n{frontmatter}\n---\n\n{content}"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(full_content, encoding="utf-8")
        return f"Successfully installed skill '{skill_name}'."
    except OSError as e:
        return f"Error installing skill '{skill_name}': {e}"


@tool(
    description="""Install a skill from a local SKILL.md file path.

Copies the SKILL.md file from the given path into the skill directory.
Only requires the file path and a skill name.

Will refuse to overwrite an existing skill unless overwrite=True.
Validates skill_name against path traversal characters (/, \\, ..).

Example:
  dive_install_skill_from_path(
    skill_name="code-review",
    skill_path="/home/user/my-skills/SKILL.md",
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
        Field(description="Absolute path to the SKILL.md file to copy."),
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
    """Install a skill by copying a SKILL.md file from a local path."""
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

    if not source.exists():
        return f"Error: Source file '{skill_path}' not found."
    if not source.is_file():
        return f"Error: Source path '{skill_path}' is not a file."

    skill_dir = _get_skill_dir()
    target_dir = skill_dir / skill_name
    target_file = target_dir / "SKILL.md"

    if target_file.exists() and not overwrite:
        return (
            f"Error: Skill '{skill_name}' already exists. "
            "Set overwrite=True to replace it."
        )

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_file)
        return f"Successfully installed skill '{skill_name}' from '{skill_path}'."
    except OSError as e:
        return f"Error installing skill '{skill_name}': {e}"


@tool(
    description="""Install a skill by downloading a SKILL.md file from a URL.

Fetches the SKILL.md content from the given URL and saves it into the skill directory.
Only requires the URL and a skill name.

Will refuse to overwrite an existing skill unless overwrite=True.
Validates skill_name against path traversal characters (/, \\, ..).

Example:
  dive_install_skill_from_url(
    skill_name="code-review",
    url="https://example.com/skills/code-review/SKILL.md",
  )
"""
)
async def dive_install_skill_from_url(
    skill_name: Annotated[
        str,
        Field(description="Directory name for the skill (e.g., 'code-review')."),
    ],
    url: Annotated[
        str,
        Field(description="URL to download the SKILL.md file from."),
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
    """Install a skill by downloading a SKILL.md file from a URL."""
    import httpx

    config = _ensure_config(config)
    abort_signal = _get_abort_signal(config)

    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Validate skill_name against path traversal
    if "/" in skill_name or "\\" in skill_name or ".." in skill_name:
        return "Error: Invalid skill name. Must not contain '/', '\\', or '..'."

    if not skill_name.strip():
        return "Error: Skill name must not be empty."

    # Convert GitHub blob URLs to raw content URLs
    url = _to_github_raw_url(url)

    skill_dir = _get_skill_dir()
    target_dir = skill_dir / skill_name
    target_file = target_dir / "SKILL.md"

    if target_file.exists() and not overwrite:
        return (
            f"Error: Skill '{skill_name}' already exists. "
            "Set overwrite=True to replace it."
        )

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.text
    except httpx.HTTPStatusError as e:
        return f"Error downloading skill from '{url}': HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        return f"Error downloading skill from '{url}': {e}"

    if not content.strip():
        return f"Error: Downloaded content from '{url}' is empty."

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content, encoding="utf-8")
        return f"Successfully installed skill '{skill_name}' from '{url}'."
    except OSError as e:
        return f"Error installing skill '{skill_name}': {e}"
