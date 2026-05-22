# opencode-sync — codebase guide

## Project layout

```
opencode_sync/
├── vllm_client.py   # HTTP client: queries GET /v1/models
├── config.py        # JSONC parser, config path detection, read/write/update
└── cli.py           # Argument parsing and orchestration (entry point)

tests/
├── conftest.py      # MockVLLMServer (real threaded HTTPServer), shared fixtures
├── test_vllm_client.py
├── test_config.py
└── test_integration.py
```

## Running tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest                         # 95 tests
python -m pytest --cov=opencode_sync     # ~93% branch coverage
```

No external runtime dependencies. `pytest` and `pytest-cov` are dev-only.

## Key design decisions

### Zero runtime dependencies
Everything uses stdlib (`urllib.request`, `json`, `argparse`, `pathlib`). Don't add `requests`, `httpx`, or any third-party library for runtime use — the tool needs to work anywhere Python 3.8+ is installed, without a `pip install` step for end users.

### Injectable HTTP for unit tests
`VLLMClient.__init__` accepts an optional `_http_get` callable. Unit tests pass in a stub; integration tests use the real `_stdlib_http_get`. Don't mock `urllib` at the module level — use the injection point instead.

### URL behaviour (important invariant)
- **No `--host`/`--port`**: reads `baseURL` from the existing provider config and queries that server. The stored URL is **never changed**. Safe to run routinely.
- **With `--host`/`--port`**: queries the specified target and also overwrites `baseURL` in the config (unless `--no-url-update`).

This invariant is tested in `TestUsesExistingUrl`. Don't break it.

### JSONC parsing
`config._strip_jsonc_comments` is a hand-rolled state machine that handles `//`, `/* */`, escaped quotes inside strings, and URLs (`://`) in string values. It does **not** use regex — regex cannot correctly track string context. `_strip_trailing_commas` uses a single regex after the state machine runs. Don't replace the state machine with a regex.

### Display name preservation
`update_provider_models` rebuilds the `models` dict from the live model list, copying existing entries verbatim for IDs that survive the sync. New IDs get a generated name (`model_id.split("/")[-1]`). This means users can customise names and have them survive future syncs as long as the model ID stays the same.

### Config mutation safety
`update_provider_models` always deep-copies the input dict before modifying it. The caller's dict is never mutated. Tests verify this in `TestUpdateProviderModels::test_original_config_not_mutated`.

## Config file format

opencode uses JSONC (`opencode.jsonc`). The relevant structure this tool touches:

```jsonc
{
  "provider": {
    "<provider-id>": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "display name",
      "options": {
        "baseURL": "http://host:port/v1"   // ← may be updated
      },
      "models": {                           // ← always replaced
        "<model-id>": { "name": "display name" }
      }
    }
  },
  "model": "<model-id>",                   // ← may be updated
  "small_model": "<model-id>"              // ← may be updated
}
```

## Adding a new flag

1. Add `p.add_argument(...)` in `cli._build_parser`.
2. Thread the value through `main()` — keep the "compute what to do" and "do it" phases separate.
3. Add unit tests for the config-level logic in `test_config.py`, then an integration test in `test_integration.py`.

## Adding support for a new platform config path

Edit `config._config_candidates()`. Add the new path to the correct platform block (Windows/Darwin/else). Add a corresponding test in `TestFindConfigPath`.

## Mock server (integration tests)

`conftest.MockVLLMServer` spins up a real `http.server.HTTPServer` on a random port in a daemon thread. The `mock_server` fixture is a factory — call it once per test with the model list you want served. It tears down automatically after the test.

To simulate server errors, pass `fail_with=<status_code>` to the factory:

```python
srv = mock_server(["model-a"], fail_with=500)
```
