"""Tests for the skill management tools."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from langgraph.graph.state import RunnableConfig

from dive_mcp_host.host.agents.agent_factory import ConfigurableKey
from dive_mcp_host.skills.manager import SkillManager
from dive_mcp_host.skills.tools import dive_install_skill_from_path


def _make_config(skill_manager: SkillManager) -> RunnableConfig:
    """Create a minimal test config."""
    return {
        "configurable": {
            ConfigurableKey.STREAM_WRITER: lambda _: None,
            ConfigurableKey.SKILL_MANAGER: skill_manager,
        }
    }


def _write_skill(skill_dir: Path, name: str, content: str) -> None:
    """Write a SKILL.md into a skill directory."""
    d = skill_dir / name
    d.mkdir(parents=True, exist_ok=True)
    skill_path = d / "SKILL.md"
    skill_path.write_text(content, encoding="utf-8")


VALID_SKILL = """\
---
name: code-review
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
    "---\nname: git-commit\ndescription: Helps write commit messages\n---\n\nContent."
)


@pytest.fixture
def temp_resource() -> Generator[tuple[SkillManager, Path]]:
    """Returns a tuple containing a skill manager and a extra temp dir."""
    with (
        tempfile.TemporaryDirectory() as manager_tmp,
        tempfile.TemporaryDirectory() as tmp_path,
    ):
        try:
            yield (SkillManager(Path(manager_tmp)), Path(tmp_path))
        finally:
            pass


class TestDiveSkill:
    """Tests for the dive_skill tool created by SkillManager.get_tools."""

    def test_load_existing_skill(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test loading an existing skill returns its content."""
        manager, _ = temp_resource
        _write_skill(manager.skill_dir, "code-review", VALID_SKILL)

        manager.refresh()
        tool = manager.get_tools()[0]
        result = tool.invoke({"skill_name": "code-review"})

        assert "## Instructions" in result
        assert "Review the code carefully" in result

    def test_load_nonexistent_skill(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test loading a nonexistent skill returns error with available list."""
        manager, _ = temp_resource
        _write_skill(manager.skill_dir, "code-review", VALID_SKILL)

        manager.refresh()
        tool = manager.get_tools()[0]
        result = tool.invoke({"skill_name": "nonexistent"})

        assert "Error" in result
        assert "not found" in result
        assert "code-review" in result

    def test_load_nonexistent_skill_no_skills_installed(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test loading a nonexistent skill when no skills are installed."""
        manager, _ = temp_resource
        tool = manager.get_tools()[0]
        result = tool.invoke({"skill_name": "nonexistent"})

        assert "Error" in result
        assert "not found" in result
        assert "No skills are installed" in result

    def test_dynamic_description_includes_skills(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test that the tool description lists installed skill names."""
        manager, _ = temp_resource
        _write_skill(manager.skill_dir, "code-review", VALID_SKILL)
        _write_skill(manager.skill_dir, "git-commit", GIT_COMMIT_SKILL)

        manager.refresh()
        tool = manager.get_tools()[0]

        assert "code-review" in tool.description
        assert "git-commit" in tool.description
        assert "<available_skills>" in tool.description
        assert "Reviews code for best practices" in tool.description
        assert "Helps write commit messages" in tool.description

    def test_dynamic_description_empty_directory(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test that an empty skill directory produces the no-skills message."""
        manager, _ = temp_resource
        tool = manager.get_tools()[0]

        assert "No skills are currently available" in tool.description
        assert "<available_skills>" not in tool.description

    def test_dynamic_description_nonexistent_directory(self) -> None:
        """Test that a nonexistent skill directory produces the no-skills message."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "nonexistent"

            manager = SkillManager(skill_dir)
            tool = manager.get_tools()[0]

            assert "No skills are currently available" in tool.description

    def test_truncates_large_content(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test that large skill content is truncated."""
        manager, _ = temp_resource
        large_content = "---\nname: big-skill\ndescription: big\n---\n" + "x" * 200000
        _write_skill(manager.skill_dir, "big-skill", large_content)

        manager.refresh()
        tool = manager.get_tools()[0]
        result = tool.invoke({"skill_name": "big-skill"})

        assert len(result) <= 100100
        assert "truncated" in result

    def test_tool_name(self, temp_resource: tuple[SkillManager, Path]) -> None:
        """Test that the tool is named dive_skill."""
        manager, _ = temp_resource
        tool = manager.get_tools()[0]
        assert tool.name == "dive_skill"


class TestInstallSkill:
    """Tests for the dive_install_skill_from_path tool."""

    @pytest.mark.asyncio
    async def test_install_from_directory(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test installing a skill from a directory copies all files."""
        manager, source_dir = temp_resource

        (source_dir / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")
        scripts_dir = source_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "run.py").write_text("print('hi')", encoding="utf-8")

        result = await dive_install_skill_from_path.arun(
            tool_input={
                "skill_name": "my-skill",
                "skill_path": str(source_dir),
            },
            config=_make_config(manager),
        )

        assert "Successfully installed" in result
        assert (manager.skill_dir / "my-skill" / "SKILL.md").exists()
        assert (manager.skill_dir / "my-skill" / "scripts" / "run.py").exists()

    @pytest.mark.asyncio
    async def test_install_from_skill_md_path(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test installing when given a SKILL.md file path directly."""
        manager, source_dir = temp_resource

        (source_dir / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")
        (source_dir / "helper.py").write_text("pass", encoding="utf-8")

        result = await dive_install_skill_from_path.arun(
            tool_input={
                "skill_name": "my-skill",
                "skill_path": str(source_dir / "SKILL.md"),
            },
            config=_make_config(manager),
        )

        assert "Successfully installed" in result
        assert (manager.skill_dir / "my-skill" / "SKILL.md").exists()
        assert (manager.skill_dir / "my-skill" / "helper.py").exists()

    @pytest.mark.asyncio
    async def test_install_reject_overwrite_without_flag(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test that overwrite is rejected without the flag."""
        manager, source_dir = temp_resource

        _write_skill(manager.skill_dir, "existing-skill", VALID_SKILL)
        (source_dir / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")

        result = await dive_install_skill_from_path.arun(
            tool_input={
                "skill_name": "existing-skill",
                "skill_path": str(source_dir),
            },
            config=_make_config(manager),
        )

        assert "Error" in result
        assert "already exists" in result

    @pytest.mark.asyncio
    async def test_install_overwrite_with_flag(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test overwriting an existing skill with the flag."""
        manager, source_dir = temp_resource

        _write_skill(manager.skill_dir, "existing-skill", VALID_SKILL)
        new_content = "---\nname: updated\ndescription: New\n---\n\nNew content."
        (source_dir / "SKILL.md").write_text(new_content, encoding="utf-8")

        result = await dive_install_skill_from_path.arun(
            tool_input={
                "skill_name": "existing-skill",
                "skill_path": str(source_dir),
                "overwrite": True,
            },
            config=_make_config(manager),
        )

        assert "Successfully installed" in result
        content = (manager.skill_dir / "existing-skill" / "SKILL.md").read_text()
        assert "New content." in content

    @pytest.mark.asyncio
    async def test_install_no_skill_md(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test that a directory without SKILL.md is rejected."""
        manager, source_dir = temp_resource

        (source_dir / "README.md").write_text("no skill here", encoding="utf-8")

        result = await dive_install_skill_from_path.arun(
            tool_input={
                "skill_name": "bad-skill",
                "skill_path": str(source_dir),
            },
            config=_make_config(manager),
        )

        assert "Error" in result
        assert "No SKILL.md" in result

    @pytest.mark.asyncio
    async def test_install_path_traversal_slash(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test that path traversal with .. is rejected."""
        manager, source_dir = temp_resource

        result = await dive_install_skill_from_path.arun(
            tool_input={
                "skill_name": "../escape",
                "skill_path": str(source_dir),
            },
            config=_make_config(manager),
        )

        assert "Error" in result
        assert "Invalid skill name" in result

    @pytest.mark.asyncio
    async def test_install_path_traversal_backslash(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test that path traversal with backslash is rejected."""
        manager, source_dir = temp_resource

        result = await dive_install_skill_from_path.arun(
            tool_input={
                "skill_name": "..\\escape",
                "skill_path": str(source_dir),
            },
            config=_make_config(manager),
        )

        assert "Error" in result
        assert "Invalid skill name" in result

    @pytest.mark.asyncio
    async def test_install_empty_name(
        self, temp_resource: tuple[SkillManager, Path]
    ) -> None:
        """Test that empty skill name is rejected."""
        manager, source_dir = temp_resource

        result = await dive_install_skill_from_path.arun(
            tool_input={
                "skill_name": "  ",
                "skill_path": str(source_dir),
            },
            config=_make_config(manager),
        )

        assert "Error" in result
        assert "empty" in result
