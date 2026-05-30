# opencode-sync

Keep [opencode](https://opencode.ai)'s model list in sync with your local [vLLM](https://github.com/vllm-project/vllm) or llama.cpp server.

When you swap models on your inference server, `opencode-sync` queries `/v1/models`, updates `opencode.jsonc`, and preserves any display names you've customised. Run it once manually, or install the shell wrapper so it happens automatically every time you start opencode.

## Requirements

- Python 3.8+
- No runtime dependencies (pure stdlib)

## Install

```bash
pip install opencode-sync
```

Or from source:

```bash
git clone https://github.com/ozskywalker/opencode-sync
cd opencode-sync
pip install -e .
```

## Usage

**Sync against the URL already in your config** (never changes the stored URL — safe to run any time):

```bash
opencode-sync
```

**Point at a specific server and update the stored URL:**

```bash
opencode-sync --host llm-server.local --port 8080
```

**Preview changes without writing anything:**

```bash
opencode-sync --dry-run
```

## Auto-sync on every opencode launch

Install a shell wrapper once:

```bash
opencode-sync install
```

This writes `~/.local/bin/opencode` — a small script that runs `opencode-sync` silently, then `exec`s the real opencode binary. opencode always starts with a fresh model list, and still launches normally if the inference server is unreachable.

Make sure `~/.local/bin` is earlier in your `$PATH` than the real opencode binary. The installer warns you if it isn't.

```
opencode-sync install [--wrapper PATH] [--opencode-bin PATH] [--force] [--dry-run]
```

## How it works

On each run, `opencode-sync`:

1. Fetches the live model list from `GET /v1/models`
2. Rebuilds `provider.models` — existing IDs keep their current display name; new IDs get a generated name from the model ID
3. If `model` or `small_model` in your config is no longer served, updates it to the first available model
4. Writes the updated config

The `model`/`small_model` fields are always written in opencode's expected `provider/model-id` format (e.g. `vllm/org/model-name`), so opencode resolves them correctly.

### URL behaviour

| Invocation | Server queried | `baseURL` in config |
|---|---|---|
| `opencode-sync` | Existing `baseURL` (fallback: `localhost:8080`) | **Unchanged** |
| `opencode-sync --host X` | `http://X:8080/v1` | Updated |
| `opencode-sync --host X --no-url-update` | `http://X:8080/v1` | **Unchanged** |

### Display names are preserved

```jsonc
// Before sync
"models": { "org/Qwen3-27B": { "name": "My Qwen label" } }

// After sync (model still served)
"models": { "org/Qwen3-27B": { "name": "My Qwen label" } }  // kept
```

## All options

```
opencode-sync [--host HOST] [--port PORT] [--provider ID] [--config PATH]
              [--dry-run] [--no-url-update] [--no-model-update] [--timeout SECONDS]
```

| Flag | Default | Description |
|---|---|---|
| `--host HOST` | from config | vLLM server hostname |
| `--port PORT` | from config | vLLM server port |
| `--provider ID` | auto-detect | Provider key to update (required only with multiple providers) |
| `--config PATH` | auto-detect | Path to `opencode.jsonc` |
| `--dry-run` | off | Show changes without writing |
| `--no-url-update` | off | Don't update `baseURL` even when `--host`/`--port` is given |
| `--no-model-update` | off | Don't update `model`/`small_model` if the active model is removed |
| `--timeout SECONDS` | 10 | HTTP request timeout |

## Config file locations

Auto-detected in this order:

| Platform | Paths checked |
|---|---|
| Linux | `$XDG_CONFIG_HOME/opencode/opencode.jsonc`, `~/.config/opencode/opencode.jsonc` |
| macOS | XDG paths above, then `~/Library/Application Support/opencode/opencode.jsonc` |
| Windows | `%APPDATA%\opencode\opencode.jsonc`, `%LOCALAPPDATA%\opencode\opencode.jsonc`, `~/.config/…` |

Override on any platform with `--config PATH`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest                      # 113 tests
python -m pytest --cov=opencode_sync  # with coverage
```
