"""Skills API router."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.models import DataResult
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.skills import Skill, get_skill_manager

logger = logging.getLogger(__name__)

skills = APIRouter(tags=["skills"])


@skills.get("/")
async def list_skills(
    _app: DiveHostAPI = Depends(get_app),
) -> DataResult[list[Skill]]:
    """List all skills."""
    manager = get_skill_manager()
    result = manager.list_skills()
    return DataResult(success=True, message=None, data=result)


@skills.get("/{skill_name}")
async def get_skill(
    skill_name: str,
    _app: DiveHostAPI = Depends(get_app),
) -> DataResult[str]:
    """Get the full content of a specific skill."""
    manager = get_skill_manager()
    content = manager.get_skill_content(skill_name)
    if content is None:
        installed = manager.list_skills()
        if installed:
            available = ", ".join(s.name for s in installed)
            detail = (
                f"Error: Skill '{skill_name}' not found. Available skills: {available}"
            )
        else:
            detail = f"Error: Skill '{skill_name}' not found. No skills are installed."
        raise HTTPException(status_code=404, detail=detail)
    return DataResult(success=True, message=None, data=content)
