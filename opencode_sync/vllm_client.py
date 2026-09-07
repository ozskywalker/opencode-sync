"""Client for querying model lists from a vLLM/llama.cpp OpenAI-compatible server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional

DEFAULT_BASE_URL = "http://localhost:8080/v1"


@dataclass
class ModelInfo:
    """One entry from GET /v1/models.

    Only ``id`` drives behavior (the config's model list is keyed on it);
    ``object``/``owned_by`` are parsed for callers that want them but nothing
    in this tool reads them.
    """

    id: str
    object: str = "model"
    owned_by: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        return cls(
            id=data["id"],
            object=data.get("object", "model"),
            owned_by=data.get("owned_by", ""),
        )


class VLLMClientError(Exception):
    pass


class VLLMConnectionError(VLLMClientError):
    pass


class VLLMParseError(VLLMClientError):
    pass


def _stdlib_http_get(url: str, timeout: int) -> dict:
    """Default HTTP GET using stdlib urllib (no external dependencies)."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise VLLMConnectionError(f"HTTP {e.code} from {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise VLLMConnectionError(f"Cannot connect to {url}: {e.reason}") from e
    except UnicodeDecodeError as e:
        # A non-UTF-8 response body is a server-side problem the CLI can't fix;
        # wrap it so the sync loop reports it like any other parse failure
        # instead of letting a raw UnicodeDecodeError escape the handler.
        raise VLLMParseError(f"Response from {url} is not valid UTF-8: {e}") from e
    except json.JSONDecodeError as e:
        raise VLLMParseError(f"Invalid JSON from {url}: {e}") from e


class VLLMClient:
    """Query model listings from a vLLM/llama.cpp OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 10,
        _http_get: Optional[Callable[[str, int], dict]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_get = _http_get if _http_get is not None else _stdlib_http_get

    def get_models(self) -> List[ModelInfo]:
        url = f"{self.base_url}/models"
        data = self._http_get(url, self.timeout)

        if not isinstance(data, dict) or "data" not in data:
            raise VLLMParseError(
                f"Unexpected response format: missing 'data' key. Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
            )

        models = []
        for item in data["data"]:
            if not isinstance(item, dict) or "id" not in item:
                continue
            models.append(ModelInfo.from_dict(item))

        return models

    def get_model_ids(self) -> List[str]:
        return [m.id for m in self.get_models()]
