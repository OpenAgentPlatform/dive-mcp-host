"""Skills API router."""

import logging

from fastapi import APIRouter, HTTPException

from dive_mcp_host.httpd.routers.models import DataResult
from dive_mcp_host.skills.manager import SkillManager
from dive_mcp_host.skills.models import SkillMeta

logger = logging.getLogger(__name__)

skills = APIRouter(tags=["skills"])


@skills.get("/")
async def list_skills() -> DataResult[list[SkillMeta]]:
    """List all skills."""
    manager = SkillManager()
    skills = manager.list_skills()
    result: list[SkillMeta] = [s.meta for s in skills]
    return DataResult(success=True, message=None, data=result)


@skills.get("/{skill_name}")
async def get_skill(
    skill_name: str,
) -> DataResult[str]:
    """Get the full content of a specific skill."""
    manager = SkillManager()
    skill = manager.get_skill(skill_name)
    if skill is None:
        installed = manager.list_skills()
        if installed:
            available = ", ".join(s.meta.name for s in installed)
            detail = (
                f"Error: Skill '{skill_name}' not found. Available skills: {available}"
            )
        else:
            detail = f"Error: Skill '{skill_name}' not found. No skills are installed."
        raise HTTPException(status_code=404, detail=detail)
    return DataResult(success=True, message=None, data=skill.content)
