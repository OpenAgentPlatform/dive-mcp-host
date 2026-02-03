import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.models import DataResult
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.mcp_installer_plugin.tools.skills import (
    _get_installed_skills,
    _read_skill_content,
)

logger = logging.getLogger(__name__)

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
    _app: DiveHostAPI = Depends(get_app),
) -> DataResult[list[Skill]]:
    """List all skills."""
    installed = _get_installed_skills()
    result: list[Skill] = []
    for s in installed:
        try:
            result.append(
                Skill(
                    name=s["name"],
                    description=s.get("description") or "",
                    license=s.get("license"),
                    compatibility=s.get("compatibility"),
                    metadata=s.get("metadata"),
                    allowed_tools=s.get("allowed_tools"),
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("Skipping skill with invalid frontmatter: %s",
                         s.get("dir_name"))
            continue
    return DataResult(success=True, message=None, data=result)


@skills.get("/{skill_name}")
async def get_skill(
    skill_name: str,
    _app: DiveHostAPI = Depends(get_app),
) -> DataResult[str]:
    """Get the full content of a specific skill."""
    content = _read_skill_content(skill_name)
    if content.startswith("Error:"):
        raise HTTPException(status_code=404, detail=content)
    return DataResult(success=True, message=None, data=content)
