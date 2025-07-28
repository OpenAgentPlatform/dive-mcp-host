import json

import pytest

from dive_mcp_host.host.conf.llm import LLMOapConfig
from dive_mcp_host.httpd.routers.models import ModelSingleConfig


@pytest.mark.parametrize(
    "base_url",
    ["", "test://test.url", None],
)
def test_oap_model(base_url):
    """Test oap model."""
    oap_json = {
        "modelProvider": "oap",
        "model": "claude-3-7-sonnet-20250219",
        "apiKey": "test_key",
        "maxTokens": 128000,
        "defaultHeaders": {"anthropic-beta": "output-128k-2025-02-19"},
    }
    match base_url:
        case None:
            oap_json["configuration"] = None
        case "":
            oap_json["configuration"] = {"baseURL": ""}
        case _:
            oap_json["configuration"] = {"baseURL": base_url}
    parsed = ModelSingleConfig.model_validate_json(json.dumps(oap_json))

    config = parsed.to_host_llm_config()
    assert isinstance(config, LLMOapConfig)
    assert config.max_tokens == 128000
    assert config.default_headers is not None
    assert config.default_headers.get("anthropic-beta") == "output-128k-2025-02-19"
    assert config.configuration is not None
    match base_url:
        case None | "":
            assert config.configuration.base_url == "https://api.oaphub.ai/v1"
        case _:
            assert config.configuration.base_url == base_url
