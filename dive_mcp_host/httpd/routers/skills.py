from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.models import DataResult
from dive_mcp_host.httpd.server import DiveHostAPI

skills = APIRouter(tags=["skills"])


class Skill(BaseModel):
    """Definition of skills frontmatter.

    Ref: https://agentskills.io/specification
    """

    # NOTE: Temporary placement, should be defined where SkillManager is created
    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
    )
    description: str = Field(min_length=1, max_length=1024)
    license: str | None = None
    compatibility: str | None = Field(default=None, min_length=1, max_length=500)
    metadata: dict[str, str] | None = None
    allowed_tools: str | None = Field(default=None)


@skills.get("/")
async def list_skills(
    app: DiveHostAPI = Depends(get_app),
) -> DataResult[list[Skill]]:
    """List all skills."""
    # Dummy results, should be retrived from SkillManager,
    # SkillManager should be accessable from 'app'
    result: list[Skill] = [
        Skill(
            name="pdf-processing",
            description="Extract text and tables from PDF files, fill PDF forms, and merge multiple PDFs. Use when working with PDF documents.",  # noqa: E501
            license="Apache-2.0",
            allowed_tools="Bash(pdftotext:*) Read",
        ),
        Skill(
            name="code-review",
            description="Review code changes for bugs, security issues, and style violations. Use when the user asks for a code review or PR review.",  # noqa: E501
            metadata={"author": "dive-team", "version": "1.0"},
        ),
        Skill(
            name="git-commit",
            description="Create well-formatted git commits with conventional commit messages. Use when the user wants to commit changes.",  # noqa: E501
            allowed_tools="Bash(git:*)",
        ),
    ]
    return DataResult(success=True, message=None, data=result)
