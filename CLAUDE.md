# opencode-sync — codebase guide

## Project layout

```
opencode_sync/
├── vllm_client.py   # HTTP client: queries GET /v1/models
├── jsonc_edit.py    # Position-preserving JSONC scanner + text splicing (pure text->text)
├── config.py        # Config domain layer: parse, plan, surgical edit, atomic write
└── cli.py           # Argument parsing and orchestration (entry point)

tests/
├── conftest.py      # MockVLLMServer (real threaded HTTPServer), shared fixtures
├── fixtures/
│   └── advanced.jsonc   # Realistic hand-maintained config: comments, 2 providers,
│                        # per-model tuning, agent section, trailing-space canaries
├── test_vllm_client.py
├── test_jsonc_edit.py   # The scanner, in isolation
├── test_config.py
├── test_surgical.py     # Safety: what a sync must NOT disturb
└── test_integration.py
```

## Running tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest                         # 205 tests
python -m pytest --cov=opencode_sync     # ~92% branch coverage
```

No external runtime dependencies. `pytest` and `pytest-cov` are dev-only.

## Key design decisions

### Zero runtime dependencies
Everything uses stdlib (`urllib.request`, `json`, `argparse`, `pathlib`). Don't add `requests`, `httpx`, `json5`, `jsonc-parser`, or any third-party library for runtime use — the tool needs to work anywhere Python 3.8+ is installed, without a `pip install` step for end users. This is why `jsonc_edit.py` exists rather than a dependency.

### Injectable HTTP for unit tests
`VLLMClient.__init__` accepts an optional `_http_get` callable. Unit tests pass in a stub; integration tests use the real `_stdlib_http_get`. Don't mock `urllib` at the module level — use the injection point instead.

### URL behaviour (important invariant)
- **No `--host`/`--port`**: reads `baseURL` from the existing provider config and queries that server. The stored URL is **never changed**. Safe to run routinely.
- **With `--host`/`--port`**: queries the specified target and also overwrites `baseURL` in the config (unless `--no-url-update`).

This invariant is tested in `TestUsesExistingUrl`. Don't break it.

### Target selection
- `--provider ID` → that provider only.
- `--host`/`--port`/`LLAMA_ARG_*` → one server. With several providers, matched by URL; ambiguity is a hard error (`--host` names *one* server, so syncing everything against it would be wrong).
- **Neither** → every provider that has a stored `options.baseURL`, each against its own URL. This is what the `install` wrapper invokes, and a multi-provider config is the normal case.

Per-provider failures warn and continue; the exit code is 1 only if *every* provider failed. A server that is up but serving no models is skipped, never treated as "remove all models". A server you *named* being down is still a hard error — see `TestConnectionFailure`.

### Surgical editing (don't regress this)
`save_config` re-serializes a dict with `json.dumps`, which destroys every comment and reflows the file. It is therefore **greenfield-only** — for configs that don't exist yet and have nothing to preserve. Editing an existing file goes through `apply_plans_to_text`, which rewrites only the spans that actually change.

How it works:
1. `mask_comments` replaces comment characters with **spaces**, preserving length, so offsets in the masked text map 1:1 onto the original.
2. A plain JSON scanner finds the span of any value. Because comments are whitespace by then, the scanner needs no comment awareness.
3. Replacement text is spliced into the **original**, so every byte outside the span survives by construction.

Two details worth keeping:
- Surviving *and renamed* model entries are re-spliced from their **original source text**, not re-serialized — that's what preserves comments *inside* an entry.
- Edits are applied **one at a time, re-scanning between each** (`_apply_one_plan`). Edits nest: inserting an absent `options` key rebuilds the provider object, which contains the models span. Splicing both against one set of offsets makes the outer rebuild silently swallow the inner edit.

Two independent guards stop a bad edit reaching disk, and **neither ever falls back to `json.dumps`** — silently destroying a user's comments is worse than failing loudly:
- The fixpoint loop raises if the text hasn't reached the desired state within `_MAX_EDIT_ROUNDS`.
- `_verify_edit` re-parses the result and compares it to what the plain dict path would have produced, via `json.dumps` rather than `==` so the comparison is **order-sensitive** (model order is meaningful in opencode; dict equality ignores it).

If the model set is unchanged, no edit is produced and the file is not written at all — the wrapper runs on every opencode launch, so this is the common case.

### JSONC parsing
`jsonc_edit.find_comment_spans` is a hand-rolled state machine that handles `//`, `/* */`, escaped quotes inside strings, and URLs (`://`) in string values. It does **not** use regex — regex cannot correctly track string context. There is **one** state machine with two consumers, so they can't drift: `mask_comments` blanks the spans (surgical path), `config._strip_jsonc_comments` deletes them.

`mask_trailing_commas` is also string-aware, for the same reason. It replaced a regex (`,(\s*[}\]])`) that silently corrupted data: `parse_jsonc('{"a": "x,  }"}')` used to return `{'a': 'x  }'}`. If you're tempted by a regex here, that's the bug you'll reintroduce.

Newlines inside block comments are preserved when masking, so `JSONDecodeError` line numbers point at real lines in the user's file.

### Display name and tuning preservation
`apply_plan` rebuilds the `models` dict from the live model list, copying existing entries verbatim for IDs that survive. Users can customise names, context limits, and sampling options and have them survive future syncs.

This is keyed on the model **ID**, so a server-side rename would drop everything. Hence renames:
- `--rename OLD=NEW` moves an entry's whole body (and, on the text path, its comments) to a new ID.
- A 1-removed/1-added sync is *inferred* as a rename, but only when the removed entry carries keys beyond `name` — i.e. only when there is hand-tuning worth rescuing. It's a heuristic, so the CLI reports every inference it acts on and `--no-infer-renames` turns it off.

### model / small_model pointers (don't regress this)
Only ever repoint a pointer that belongs to the provider being synced. Syncing provider B must never touch a pointer aimed at provider A.

Model IDs contain slashes (`org/model-a`), so "has a slash" does **not** mean "provider-qualified". The only reliable test is whether the first segment names a provider that actually exists in `config["provider"]`. Tested in `TestCrossProviderIsolation`.

A bare (unqualified) ID is legacy output from an older version of this tool and gets normalized — but only when syncing a single provider, since a bare ID is genuinely ambiguous across several.

### Config mutation safety
`apply_plan` always deep-copies the input dict before modifying it. The caller's dict is never mutated. Tests verify this in `TestUpdateProviderModels::test_original_config_not_mutated`.

### Writes are atomic
`_atomic_write_text` writes a temp file in the target directory (same filesystem — `os.replace` is only atomic within one), fsyncs, copies the original's mode across (`mkstemp` creates 0600, so this is mandatory, not a nicety), then `os.replace`s. A rolling `.bak` is written **only when content actually changes**, so routine no-op runs don't churn it.

Read and write both use `newline=""`. `Path.read_text`/`write_text` apply universal-newline translation, which silently converts a CRLF config to LF.

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
      "models": {                           // ← rebuilt from the live model list
        "<model-id>": { "name": "display name", /* any other keys preserved */ }
      }
    }
  },
  "model": "<provider-id>/<model-id>",     // ← only if it points at the synced provider
  "small_model": "<provider-id>/<model-id>"
}
```

Everything else — `$schema`, `agent`, `mcp`, `keybinds`, and any key this tool has never heard of — is never touched, and `test_surgical.py` asserts that byte-for-byte rather than trusting it.

## Adding a new flag

1. Add `p.add_argument(...)` in `cli._build_parser`.
2. Thread the value through `main()` — keep the "compute what to do" (`plan_provider_update`) and "do it" (`apply_plan` / `apply_plans_to_text`) phases separate.
3. Add unit tests for the config-level logic in `test_config.py`, then an integration test in `test_integration.py`.

## Adding support for a new platform config path

Edit `config._config_candidates()`. opencode uses xdg-basedir everywhere, so the list is `XDG_CONFIG_HOME` (if set) then `~/.config` — there are no per-platform branches, and `APPDATA` / `~/Library/Application Support` are deliberately excluded. Add a corresponding test in `TestFindConfigPath`.

## Mock server (integration tests)

`conftest.MockVLLMServer` spins up a real `http.server.HTTPServer` on a random port in a daemon thread. The `mock_server` fixture is a factory — call it once per test with the model list you want served. It tears down automatically after the test. Each server gets its **own handler subclass**, because `models`/`fail_with` are class attributes and a shared handler would let a second server clobber the first's list — sync-all tests need two servers serving different models.

To simulate server errors, pass `fail_with=<status_code>` to the factory:

```python
srv = mock_server(["model-a"], fail_with=500)
```
