# opencode-sync

Keep [opencode](https://opencode.ai)'s model list in sync with your local [vLLM](https://github.com/vllm-project/vllm) or llama.cpp server.

When you swap models on your inference server, `opencode-sync` queries `/v1/models` and updates `opencode.jsonc` in place. Run it once manually, or install the shell wrapper so it happens automatically every time you start opencode.

It edits **only** the model list of the providers it syncs. Your comments, formatting, and every other setting in the file are left byte-for-byte alone.

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

1. Works out which providers to sync (see [Which providers get synced](#which-providers-get-synced))
2. Fetches each one's live model list from `GET /v1/models`
3. Rebuilds that provider's `models` — surviving IDs keep their entry exactly as you wrote it; new IDs get a generated name from the model ID
4. If `model` or `small_model` points at *this* provider and its model is gone, repoints it to the first available one
5. Rewrites only those parts of the file

The `model`/`small_model` fields are written in opencode's expected `provider/model-id` format (e.g. `vllm/org/model-name`), so opencode resolves them correctly.

### What it won't touch

Everything outside the model lists it syncs. Concretely:

- **Comments and formatting.** The file is edited by splicing text, not by re-serializing it, so your `//` notes, indentation, and blank lines survive exactly.
- **Other providers.** Syncing one provider never modifies another — including `model`/`small_model` when they point somewhere else.
- **Everything else.** `$schema`, `agent`, `mcp`, `keybinds`, and any other key stay byte-identical.
- **The file itself, when nothing changed.** A no-op sync doesn't rewrite it at all.

Before writing, it re-parses the result and checks it means exactly what was intended. If that check fails it writes nothing and tells you — it never falls back to reformatting your file. Writes are atomic (temp file + rename), and a `.bak` is kept whenever content actually changes.

### Which providers get synced

| Invocation | Providers synced |
|---|---|
| `opencode-sync` | Every provider with a `baseURL`, each against its own URL |
| `opencode-sync --provider ID` | Just `ID` |
| `opencode-sync --host X` | The one matching `http://X:8080/v1` |

With no arguments it syncs everything, which is what the wrapper does. If one server is down, the others still sync; the exit code is non-zero only if every one failed. A server that's up but serving no models is skipped — your model list is never wiped because an inference server was mid-restart.

### When a server renames a model

If a server starts reporting a different ID for what is really the same model, a naive sync would drop your old entry and everything you'd tuned on it. So:

```bash
opencode-sync --provider vllm --rename aeon=qwen3.5-122B-A10B
```

This moves the entry — display name, context limits, sampling options, comments and all — to the new ID.

A sync that removes exactly one model and adds exactly one is *assumed* to be a rename, but only when the old entry has settings worth keeping. It'll tell you when it does this:

```
~ Renamed: aeon -> qwen3.5-122B-A10B (kept limit, options, reasoning, temperature, tool_call)
```

That's a guess, and a genuine model swap would inherit the wrong settings — use `--no-infer-renames` to turn it off and drop the old entry instead.

### Which model opencode defaults to

opencode resolves its default model in this order: the top-level `model` key in the config, then its own `recent[]` state file (`~/.local/state/opencode/model.json`), then the first model of the first provider. The sync keeps the provider's model list fresh, but by default it never *creates* a `model` pointer — after a server-side rename, a config that never had one still won't, and opencode falls through to its (possibly stale) recent list.

`--default-model` makes that tunable:

```bash
# After a model-set change, point config.model at the first served model
opencode-sync --default-model auto

# Never touch model/small_model at all, not even to repair a stale pointer
opencode-sync --default-model none --default-small-model none

# Pin an explicit model (provider-qualified)
opencode-sync --default-model vllm/qwen3.5-122B-A10B
```

| Mode | Behavior |
|---|---|
| `first` (default) | Legacy: never invent a pointer; only repair one that already points at the synced provider |
| `auto` | When this provider's model set changes, point `model`/`small_model` at the first served model |
| `none` | Never touch the keys |
| `provider/model-id` | Explicit pin; written whenever it isn't already there |

`--default-small-model` takes the same values and controls `small_model` independently. A pointer belonging to a different provider is never touched in any mode. `auto` only fires when the model set actually changed, so routine no-op syncs don't churn the config.

If opencode has recently-used leftovers for models that no longer exist, they can shadow the synced default (opencode skips dead entries rather than cleaning them up):

```bash
opencode-sync --prune-recent   # drop dead models from ~/.local/state/opencode/model.json
```

Entries for providers this tool doesn't sync are left alone.

### URL behaviour

| Invocation | Server queried | `baseURL` in config |
|---|---|---|
| `opencode-sync` | Existing `baseURL` (fallback: `localhost:8080`) | **Unchanged** |
| `LLAMA_ARG_HOST=X opencode-sync` | `http://X:8080/v1` | Updated |
| `LLAMA_ARG_HOST=X LLAMA_ARG_PORT=9000 opencode-sync` | `http://X:9000/v1` | Updated |
| `opencode-sync --host X` | `http://X:8080/v1` | Updated |
| `opencode-sync --host X --no-url-update` | `http://X:8080/v1` | **Unchanged** |

CLI `--host`/`--port` values override `LLAMA_ARG_HOST`/`LLAMA_ARG_PORT`.

### Your customisations are preserved

Anything you've written on a model entry survives a sync, as long as the server still reports that model ID:

```jsonc
"models": {
  "org/Qwen3-27B": {
    "name": "My Qwen label",       // kept
    "limit": { "context": 229376 }, // kept
    // Even this comment is kept.
    "options": { "temperature": 0.6 }
  }
}
```

## All options

```
opencode-sync [--host HOST] [--port PORT] [--provider ID] [--config PATH]
              [--rename OLD=NEW] [--no-infer-renames] [--dry-run]
              [--no-url-update] [--no-model-update] [--timeout SECONDS]
              [--default-model MODE] [--default-small-model MODE] [--prune-recent]
              [--version] [--no-update-check]
```

| Flag | Default | Description |
|---|---|---|
| `--host HOST` | env, config, else localhost | vLLM server hostname |
| `--port PORT` | env, config, else 8080 | vLLM server port |
| `--provider ID` | all providers | Provider key to update |
| `--config PATH` | auto-detect | Path to `opencode.jsonc` |
| `--rename OLD=NEW` | — | Move a model entry to a new ID, keeping its settings (repeatable) |
| `--no-infer-renames` | off | Don't treat a 1-in/1-out sync as a rename; drop the old entry |
| `--dry-run` | off | Print a diff of the planned changes without writing |
| `--no-url-update` | off | Don't update `baseURL` even when `--host`/`--port` is given |
| `--no-model-update` | off | Don't update `model`/`small_model` if the active model is removed |
| `--default-model MODE` | `first` | Pointer policy: `first`, `none`, `auto`, or explicit `provider/model-id` |
| `--default-small-model MODE` | `first` | Same, for `small_model` |
| `--prune-recent` | off | Drop dead models from opencode's `recent[]` state file |
| `--timeout SECONDS` | 10 | HTTP request timeout |
| `--version` | — | Print `opencode-sync <version>` and exit |
| `--no-update-check` | off | Skip the PyPI check for a newer release |

## Version banner and update checks

Every invocation prints its version as the first line of stdout:

```
opencode-sync v0.5.0
```

After the work completes (sync, install, or nothing-to-do), the tool checks
PyPI for a newer release. If one exists, a notice is printed:

```
---
Update available: opencode-sync 0.6.0 (you have 0.5.0) — https://pypi.org/project/opencode-sync/
```

Properties of the check:

- **Best-effort and silent on failure** — network errors, timeouts, and bad
  responses produce no output and never change the exit code.
- **1-second timeout**, and it runs *after* the config work, never before.
- **Dev builds don't nag** — a version like `0.5.0.dev1+g4b58801` (setuptools-scm
  output on a checkout ahead of the last tag) skips the check entirely.
- **Opt out** with `--no-update-check` (useful on air-gapped machines).

## Config file locations

opencode uses xdg-basedir on every platform, so the search order is the same everywhere:

1. `$XDG_CONFIG_HOME/opencode/opencode.jsonc` (if `XDG_CONFIG_HOME` is set)
2. `~/.config/opencode/opencode.jsonc`

Override with `--config PATH`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest                      # ~298 tests
python -m pytest --cov=opencode_sync  # with coverage
```
