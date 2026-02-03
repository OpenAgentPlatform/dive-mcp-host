"""Tests for the skill management tools."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.graph.state import RunnableConfig

from dive_mcp_host.mcp_installer_plugin.tools.skills import (
    _parse_skill_frontmatter,
    create_dive_skill_tool,
    dive_install_skill_from_path,
)

MOCK_TARGET = "dive_mcp_host.mcp_installer_plugin.tools.skills._get_skill_dir"


def _make_config() -> RunnableConfig:
    """Create a minimal test config."""
    return {
        "configurable": {
            "stream_writer": lambda _: None,
        }
    }


def _write_skill(skill_dir: Path, name: str, content: str) -> None:
    """Write a SKILL.md into a skill directory."""
    d = skill_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(content, encoding="utf-8")


VALID_SKILL = """\
---
name: Code Review
description: Reviews code for best practices
---

## Instructions

Review the code carefully.
"""

SKILL_NO_FRONTMATTER = """\
## Instructions

Just some instructions without frontmatter.
"""

SKILL_INVALID_YAML = """\
---
name: [invalid
description: : bad yaml {{
---

Body content.
"""

GIT_COMMIT_SKILL = (
    "---\nname: Git Commit\ndescription: Helps write commit messages\n---\n\nContent."
)


class TestParseSkillFrontmatter:
    """Tests for _parse_skill_frontmatter."""

    def test_valid_frontmatter(self) -> None:
        """Test parsing valid YAML frontmatter."""
        result = _parse_skill_frontmatter(VALID_SKILL)
        assert result["name"] == "Code Review"
        assert result["description"] == "Reviews code for best practices"

    def test_missing_frontmatter(self) -> None:
        """Test parsing content without frontmatter."""
        result = _parse_skill_frontmatter(SKILL_NO_FRONTMATTER)
        assert result == {}

    def test_invalid_yaml(self) -> None:
        """Test parsing invalid YAML frontmatter."""
        result = _parse_skill_frontmatter(SKILL_INVALID_YAML)
        assert result == {}

    def test_empty_content(self) -> None:
        """Test parsing empty content."""
        result = _parse_skill_frontmatter("")
        assert result == {}

    def test_no_closing_delimiter(self) -> None:
        """Test parsing frontmatter without closing delimiter."""
        result = _parse_skill_frontmatter("---\nname: test\n")
        assert result == {}

    def test_non_dict_frontmatter(self) -> None:
        """Test parsing frontmatter that is not a dict."""
        content = "---\n- item1\n- item2\n---\n"
        result = _parse_skill_frontmatter(content)
        assert result == {}


class TestDiveSkill:
    """Tests for the dive_skill tool created by create_dive_skill_tool."""

    def test_load_existing_skill(self) -> None:
        """Test loading an existing skill returns its content."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "code-review", VALID_SKILL)

            with patch(MOCK_TARGET, return_value=skill_dir):
                tool = create_dive_skill_tool()
                result = tool.invoke({"skill_name": "code-review"})

            assert "Code Review" in result
            assert "Review the code carefully" in result

    def test_load_nonexistent_skill(self) -> None:
        """Test loading a nonexistent skill returns error with available list."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "code-review", VALID_SKILL)

            with patch(MOCK_TARGET, return_value=skill_dir):
                tool = create_dive_skill_tool()
                result = tool.invoke({"skill_name": "nonexistent"})

            assert "Error" in result
            assert "not found" in result
            assert "code-review" in result

    def test_load_nonexistent_skill_no_skills_installed(self) -> None:
        """Test loading a nonexistent skill when no skills are installed."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                tool = create_dive_skill_tool()
                result = tool.invoke({"skill_name": "nonexistent"})

            assert "Error" in result
            assert "not found" in result
            assert "No skills are installed" in result

    def test_dynamic_description_includes_skills(self) -> None:
        """Test that the tool description lists installed skill names."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "code-review", VALID_SKILL)
            _write_skill(skill_dir, "git-commit", GIT_COMMIT_SKILL)

            with patch(MOCK_TARGET, return_value=skill_dir):
                tool = create_dive_skill_tool()

            assert "code-review" in tool.description
            assert "git-commit" in tool.description
            assert "<available_skills>" in tool.description
            assert "Reviews code for best practices" in tool.description
            assert "Helps write commit messages" in tool.description

    def test_dynamic_description_empty_directory(self) -> None:
        """Test that an empty skill directory produces the no-skills message."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                tool = create_dive_skill_tool()

            assert "No skills are currently available" in tool.description
            assert "<available_skills>" not in tool.description

    def test_dynamic_description_nonexistent_directory(self) -> None:
        """Test that a nonexistent skill directory produces the no-skills message."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "nonexistent"

            with patch(MOCK_TARGET, return_value=skill_dir):
                tool = create_dive_skill_tool()

            assert "No skills are currently available" in tool.description

    def test_truncates_large_content(self) -> None:
        """Test that large skill content is truncated."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            large_content = "---\nname: Big\ndescription: big\n---\n" + "x" * 200000
            _write_skill(skill_dir, "big-skill", large_content)

            with patch(MOCK_TARGET, return_value=skill_dir):
                tool = create_dive_skill_tool()
                result = tool.invoke({"skill_name": "big-skill"})

            assert len(result) <= 100100
            assert "truncated" in result

    def test_tool_name(self) -> None:
        """Test that the tool is named dive_skill."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                tool = create_dive_skill_tool()

            assert tool.name == "dive_skill"


class TestInstallSkill:
    """Tests for the dive_install_skill_from_path tool."""

    @pytest.mark.asyncio
    async def test_install_from_directory(self) -> None:
        """Test installing a skill from a directory copies all files."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "installed"
            skill_dir.mkdir()
            source_dir = Path(tmp) / "source" / "my-skill"
            source_dir.mkdir(parents=True)
            (source_dir / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")
            scripts_dir = source_dir / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "run.py").write_text("print('hi')", encoding="utf-8")

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await dive_install_skill_from_path.arun(
                    tool_input={
                        "skill_name": "my-skill",
                        "skill_path": str(source_dir),
                    },
                    config=_make_config(),
                )

            assert "Successfully installed" in result
            assert (skill_dir / "my-skill" / "SKILL.md").exists()
            assert (skill_dir / "my-skill" / "scripts" / "run.py").exists()

    @pytest.mark.asyncio
    async def test_install_from_skill_md_path(self) -> None:
        """Test installing when given a SKILL.md file path directly."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "installed"
            skill_dir.mkdir()
            source_dir = Path(tmp) / "source" / "my-skill"
            source_dir.mkdir(parents=True)
            (source_dir / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")
            (source_dir / "helper.py").write_text("pass", encoding="utf-8")

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await dive_install_skill_from_path.arun(
                    tool_input={
                        "skill_name": "my-skill",
                        "skill_path": str(source_dir / "SKILL.md"),
                    },
                    config=_make_config(),
                )

            assert "Successfully installed" in result
            assert (skill_dir / "my-skill" / "SKILL.md").exists()
            assert (skill_dir / "my-skill" / "helper.py").exists()

    @pytest.mark.asyncio
    async def test_install_reject_overwrite_without_flag(self) -> None:
        """Test that overwrite is rejected without the flag."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "installed"
            skill_dir.mkdir()
            _write_skill(skill_dir, "existing-skill", VALID_SKILL)
            source_dir = Path(tmp) / "source" / "existing-skill"
            source_dir.mkdir(parents=True)
            (source_dir / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await dive_install_skill_from_path.arun(
                    tool_input={
                        "skill_name": "existing-skill",
                        "skill_path": str(source_dir),
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "already exists" in result

    @pytest.mark.asyncio
    async def test_install_overwrite_with_flag(self) -> None:
        """Test overwriting an existing skill with the flag."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "installed"
            skill_dir.mkdir()
            _write_skill(skill_dir, "existing-skill", VALID_SKILL)
            source_dir = Path(tmp) / "source" / "existing-skill"
            source_dir.mkdir(parents=True)
            new_content = "---\nname: Updated\ndescription: New\n---\n\nNew content."
            (source_dir / "SKILL.md").write_text(new_content, encoding="utf-8")

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await dive_install_skill_from_path.arun(
                    tool_input={
                        "skill_name": "existing-skill",
                        "skill_path": str(source_dir),
                        "overwrite": True,
                    },
                    config=_make_config(),
                )

            assert "Successfully installed" in result
            content = (skill_dir / "existing-skill" / "SKILL.md").read_text()
            assert "New content." in content

    @pytest.mark.asyncio
    async def test_install_no_skill_md(self) -> None:
        """Test that a directory without SKILL.md is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "installed"
            skill_dir.mkdir()
            source_dir = Path(tmp) / "source" / "bad-skill"
            source_dir.mkdir(parents=True)
            (source_dir / "README.md").write_text("no skill here", encoding="utf-8")

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await dive_install_skill_from_path.arun(
                    tool_input={
                        "skill_name": "bad-skill",
                        "skill_path": str(source_dir),
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "No SKILL.md" in result

    @pytest.mark.asyncio
    async def test_install_path_traversal_slash(self) -> None:
        """Test that path traversal with .. is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await dive_install_skill_from_path.arun(
                    tool_input={
                        "skill_name": "../escape",
                        "skill_path": "/tmp",
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "Invalid skill name" in result

    @pytest.mark.asyncio
    async def test_install_path_traversal_backslash(self) -> None:
        """Test that path traversal with backslash is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await dive_install_skill_from_path.arun(
                    tool_input={
                        "skill_name": "..\\escape",
                        "skill_path": "/tmp",
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "Invalid skill name" in result

    @pytest.mark.asyncio
    async def test_install_empty_name(self) -> None:
        """Test that empty skill name is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await dive_install_skill_from_path.arun(
                    tool_input={
                        "skill_name": "  ",
                        "skill_path": "/tmp",
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "empty" in result
