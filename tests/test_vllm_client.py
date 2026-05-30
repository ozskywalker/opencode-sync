"""Unit tests for opencode_sync.vllm_client."""

from __future__ import annotations

import pytest

from opencode_sync.vllm_client import (
    ModelInfo,
    VLLMClient,
    VLLMConnectionError,
    VLLMParseError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client(response=None, error=None, base_url="http://localhost:8080/v1"):
    """Build a VLLMClient with an injected _http_get stub."""

    def _get(url: str, timeout: int) -> dict:
        if error is not None:
            raise error
        return response

    return VLLMClient(base_url=base_url, _http_get=_get)


def models_response(*ids) -> dict:
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "owned_by": "vllm"} for mid in ids],
    }


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_from_dict_basic(self):
        m = ModelInfo.from_dict({"id": "gpt-4", "object": "model", "owned_by": "openai"})
        assert m.id == "gpt-4"
        assert m.object == "model"
        assert m.owned_by == "openai"

    def test_from_dict_defaults(self):
        m = ModelInfo.from_dict({"id": "my-model"})
        assert m.object == "model"
        assert m.owned_by == ""

    def test_from_dict_extra_fields_captured(self):
        m = ModelInfo.from_dict({"id": "x", "created": 123})
        assert m.extra == {"created": 123}

    def test_from_dict_extra_excludes_known_fields(self):
        m = ModelInfo.from_dict({"id": "x", "object": "model", "owned_by": "y"})
        assert m.extra == {}


# ---------------------------------------------------------------------------
# VLLMClient.base_url normalization
# ---------------------------------------------------------------------------

class TestBaseURL:
    def test_trailing_slash_stripped(self):
        c = VLLMClient(base_url="http://host:8000/v1/", _http_get=lambda u, t: {})
        assert c.base_url == "http://host:8000/v1"

    def test_no_trailing_slash(self):
        c = VLLMClient(base_url="http://host:8000/v1", _http_get=lambda u, t: {})
        assert c.base_url == "http://host:8000/v1"

    def test_models_url_constructed_correctly(self):
        calls = []

        def _get(url, timeout):
            calls.append(url)
            return models_response()

        VLLMClient(base_url="http://myhost:9000/v1", _http_get=_get).get_models()
        assert calls == ["http://myhost:9000/v1/models"]


# ---------------------------------------------------------------------------
# VLLMClient.get_models
# ---------------------------------------------------------------------------

class TestGetModels:
    def test_returns_model_list(self):
        client = make_client(response=models_response("m1", "m2", "m3"))
        models = client.get_models()
        assert len(models) == 3
        assert models[0].id == "m1"
        assert models[2].id == "m3"

    def test_empty_data_list(self):
        client = make_client(response={"object": "list", "data": []})
        assert client.get_models() == []

    def test_skips_items_without_id(self):
        response = {
            "data": [
                {"id": "good-model"},
                {"no_id_here": True},
                {"id": "another-good"},
            ]
        }
        models = make_client(response=response).get_models()
        assert [m.id for m in models] == ["good-model", "another-good"]

    def test_skips_non_dict_items(self):
        response = {"data": [{"id": "ok"}, "not-a-dict", 42]}
        models = make_client(response=response).get_models()
        assert len(models) == 1
        assert models[0].id == "ok"

    def test_missing_data_key_raises_parse_error(self):
        client = make_client(response={"object": "list"})
        with pytest.raises(VLLMParseError, match="missing 'data' key"):
            client.get_models()

    def test_non_dict_response_raises_parse_error(self):
        client = make_client(response=["list", "of", "things"])
        with pytest.raises(VLLMParseError):
            client.get_models()

    def test_connection_error_propagates(self):
        client = make_client(error=VLLMConnectionError("refused"))
        with pytest.raises(VLLMConnectionError, match="refused"):
            client.get_models()

    def test_parse_error_propagates(self):
        client = make_client(error=VLLMParseError("bad json"))
        with pytest.raises(VLLMParseError, match="bad json"):
            client.get_models()


# ---------------------------------------------------------------------------
# VLLMClient.get_model_ids
# ---------------------------------------------------------------------------

class TestGetModelIds:
    def test_returns_list_of_strings(self):
        client = make_client(response=models_response("a/b", "c/d"))
        assert client.get_model_ids() == ["a/b", "c/d"]

    def test_empty_list(self):
        client = make_client(response={"data": []})
        assert client.get_model_ids() == []

    def test_timeout_passed_to_http_get(self):
        received = {}

        def _get(url, timeout):
            received["timeout"] = timeout
            return models_response()

        VLLMClient(base_url="http://x/v1", timeout=42, _http_get=_get).get_model_ids()
        assert received["timeout"] == 42
