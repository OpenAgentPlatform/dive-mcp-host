import json
import re
from collections.abc import Generator
from os import getenv
from typing import Any

from sqlalchemy import make_url

POSTGRES_URI = getenv("POSTGRES_URI", "postgresql://mcp:mcp@localhost:5432/mcp")
SQLITE_URI = getenv("SQLITE_URI", "sqlite:///dummy.db")

POSTGRES_URI_ASYNC = make_url(POSTGRES_URI).set(drivername="postgresql+asyncpg")
SQLITE_URI_ASYNC = make_url(SQLITE_URI).set(drivername="sqlite+aiosqlite")


def dict_subset(superset: dict, subset: dict) -> bool:
    """Check if subset is a subset of superset.

    Args:
        superset: The dictionary that should contain all items from subset
        subset: The dictionary that should be contained within superset

    Returns:
        bool: True if subset is a subset of superset, False otherwise
    """
    assert isinstance(superset, dict)
    assert isinstance(subset, dict)
    for key, value in subset.items():
        assert key in superset, f"{key} is not in {superset}"
        if isinstance(value, dict) and isinstance(superset[key], dict):
            assert dict_subset(superset[key], value), (
                f"{key} is not a subset of {superset[key]}"
            )
        elif isinstance(value, list) and isinstance(superset[key], list):
            assert len(value) == len(superset[key]), (
                f"List lengths for {key} are different: {len(value)} != {len(superset[key])}"  # noqa: E501
            )
            for i, (subset_item, superset_item) in enumerate(
                zip(value, superset[key], strict=False)
            ):
                if isinstance(subset_item, dict) and isinstance(superset_item, dict):
                    assert dict_subset(superset_item, subset_item), (
                        f"List item {i} in {key} is not a subset"
                    )
                else:
                    assert subset_item == superset_item, (
                        f"List item {i} in {key} are not equal: {subset_item} != {superset_item}"  # noqa: E501
                    )
        else:
            assert superset[key] == value, (
                f"{key} is not equal {superset[key]} and {value}"
            )
    return True


def extract_stream(content: str) -> Generator[dict[str, Any], None, None]:
    """Extract the stream from the content.

    Args:
        content: The content to extract the stream from

    Returns: The stream from the content
    """
    assert content.startswith("data: ")
    assert content.endswith("data: [DONE]\n\n")
    data_messages = re.findall(r"data: (.*?)\n\n", content)
    for data in data_messages:
        if data != "[DONE]":
            yield json.loads(data)
