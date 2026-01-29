"""Tests for the skill management tools."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.graph.state import RunnableConfig

from dive_mcp_host.mcp_installer_plugin.tools.skills import (
    _parse_skill_frontmatter,
    install_skill,
    read_skill,
    search_skills,
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


class TestReadSkill:
    """Tests for the read_skill tool."""

    @pytest.mark.asyncio
    async def test_read_existing_skill(self) -> None:
        """Test reading an existing skill."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "code-review", VALID_SKILL)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await read_skill.arun(
                    tool_input={"skill_name": "code-review"},
                    config=_make_config(),
                )

            assert "Code Review" in result
            assert "Review the code carefully" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_skill(self) -> None:
        """Test reading a skill that does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await read_skill.arun(
                    tool_input={"skill_name": "nonexistent"},
                    config=_make_config(),
                )

            assert "Error" in result
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_read_truncates_large_content(self) -> None:
        """Test that large skill content is truncated."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            large_content = "---\nname: Big\ndescription: big\n---\n" + "x" * 200000
            _write_skill(skill_dir, "big-skill", large_content)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await read_skill.arun(
                    tool_input={"skill_name": "big-skill"},
                    config=_make_config(),
                )

            assert len(result) <= 100100
            assert "truncated" in result


class TestSearchSkills:
    """Tests for the search_skills tool."""

    @pytest.mark.asyncio
    async def test_search_all_skills(self) -> None:
        """Test listing all installed skills."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "code-review", VALID_SKILL)
            _write_skill(
                skill_dir,
                "git-commit",
                GIT_COMMIT_SKILL,
            )

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await search_skills.arun(
                    tool_input={},
                    config=_make_config(),
                )

            assert "code-review" in result
            assert "git-commit" in result
            assert "2" in result

    @pytest.mark.asyncio
    async def test_search_with_query_filter(self) -> None:
        """Test filtering skills by query."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "code-review", VALID_SKILL)
            _write_skill(
                skill_dir,
                "git-commit",
                GIT_COMMIT_SKILL,
            )

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await search_skills.arun(
                    tool_input={"query": "git"},
                    config=_make_config(),
                )

            assert "git-commit" in result
            assert "code-review" not in result

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self) -> None:
        """Test that search is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "code-review", VALID_SKILL)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await search_skills.arun(
                    tool_input={"query": "CODE"},
                    config=_make_config(),
                )

            assert "code-review" in result

    @pytest.mark.asyncio
    async def test_search_empty_directory(self) -> None:
        """Test searching in an empty skill directory."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await search_skills.arun(
                    tool_input={},
                    config=_make_config(),
                )

            assert "No skills installed" in result

    @pytest.mark.asyncio
    async def test_search_nonexistent_directory(self) -> None:
        """Test searching when skill directory does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "nonexistent"

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await search_skills.arun(
                    tool_input={},
                    config=_make_config(),
                )

            assert "No skills installed" in result

    @pytest.mark.asyncio
    async def test_search_no_matching_query(self) -> None:
        """Test searching with a query that matches nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "code-review", VALID_SKILL)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await search_skills.arun(
                    tool_input={"query": "zzz-nonexistent"},
                    config=_make_config(),
                )

            assert "No skills found matching" in result


class TestInstallSkill:
    """Tests for the install_skill tool."""

    @pytest.mark.asyncio
    async def test_install_new_skill(self) -> None:
        """Test installing a new skill."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await install_skill.arun(
                    tool_input={
                        "skill_name": "my-skill",
                        "description": "A test skill",
                        "content": "Do the thing.",
                    },
                    config=_make_config(),
                )

            assert "Successfully installed" in result
            skill_file = skill_dir / "my-skill" / "SKILL.md"
            assert skill_file.exists()

            content = skill_file.read_text(encoding="utf-8")
            assert "A test skill" in content
            assert "Do the thing." in content

    @pytest.mark.asyncio
    async def test_install_with_display_name(self) -> None:
        """Test installing a skill with a display name."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await install_skill.arun(
                    tool_input={
                        "skill_name": "my-skill",
                        "description": "A test skill",
                        "content": "Instructions here.",
                        "display_name": "My Awesome Skill",
                    },
                    config=_make_config(),
                )

            assert "Successfully installed" in result
            path = skill_dir / "my-skill" / "SKILL.md"
            content = path.read_text()
            assert "My Awesome Skill" in content

    @pytest.mark.asyncio
    async def test_install_reject_overwrite_without_flag(
        self,
    ) -> None:
        """Test that overwrite is rejected without the flag."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "existing-skill", VALID_SKILL)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await install_skill.arun(
                    tool_input={
                        "skill_name": "existing-skill",
                        "description": "New description",
                        "content": "New content.",
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "already exists" in result

    @pytest.mark.asyncio
    async def test_install_overwrite_with_flag(self) -> None:
        """Test overwriting an existing skill with the flag."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            _write_skill(skill_dir, "existing-skill", VALID_SKILL)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await install_skill.arun(
                    tool_input={
                        "skill_name": "existing-skill",
                        "description": "Updated description",
                        "content": "Updated content.",
                        "overwrite": True,
                    },
                    config=_make_config(),
                )

            assert "Successfully installed" in result
            path = skill_dir / "existing-skill" / "SKILL.md"
            content = path.read_text()
            assert "Updated description" in content
            assert "Updated content." in content

    @pytest.mark.asyncio
    async def test_install_path_traversal_slash(self) -> None:
        """Test that path traversal with .. is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await install_skill.arun(
                    tool_input={
                        "skill_name": "../escape",
                        "description": "Bad skill",
                        "content": "Evil content.",
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "Invalid skill name" in result

    @pytest.mark.asyncio
    async def test_install_path_traversal_backslash(
        self,
    ) -> None:
        """Test that path traversal with backslash is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await install_skill.arun(
                    tool_input={
                        "skill_name": "..\\escape",
                        "description": "Bad skill",
                        "content": "Evil content.",
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "Invalid skill name" in result

    @pytest.mark.asyncio
    async def test_install_path_traversal_forward_slash(
        self,
    ) -> None:
        """Test that forward slash in name is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)

            with patch(MOCK_TARGET, return_value=skill_dir):
                result = await install_skill.arun(
                    tool_input={
                        "skill_name": "foo/bar",
                        "description": "Bad skill",
                        "content": "Evil content.",
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
                result = await install_skill.arun(
                    tool_input={
                        "skill_name": "  ",
                        "description": "Bad skill",
                        "content": "Content.",
                    },
                    config=_make_config(),
                )

            assert "Error" in result
            assert "empty" in result
