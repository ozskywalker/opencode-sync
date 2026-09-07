# opencode-sync — codebase guide

## Project layout

```
opencode_sync/
├── vllm_client.py   # HTTP client: queries GET /v1/models
├── jsonc_edit.py    # Position-preserving JSONC scanner + text splicing (pure text->text)
├── io.py            # Config path discovery, verbatim reads, atomic writes, parse_jsonc
├── planning.py      # Pure plan computation: plan_provider_update, apply_plan, ProviderPlan
├── surgical.py      # Span-level text editing that preserves comments byte-for-byte
├── state.py         # opencode model.json recent[] pruning
├── config.py        # Facade re-exporting the domain API (io/planning/surgical/state)
└── cli.py           # Argument parsing and orchestration (entry point)

tests/
├── conftest.py      # MockVLLMServer (real threaded HTTPServer), shared fixtures
├── fixtures/
│   └── advanced.jsonc   # Realistic hand-maintained config: comments, 2 providers,
│                        # per-model tuning, agent section, trailing-space canaries,
│                        # a brace-adjacent ("orphan zone") comment canary
├── test_vllm_client.py
├── test_jsonc_edit.py   # The scanner, in isolation
├── test_config.py       # Domain-level unit tests (import via the config facade)
├── test_surgical.py     # Safety: what a sync must NOT disturb
├── test_install.py
├── test_version.py
└── test_integration.py
```

Import direction (must stay acyclic): `io` and `planning` depend only on
`jsonc_edit`/stdlib; `surgical` adds `planning` + `io`; `state` adds `io`;
`config` is a facade over all four; `cli` imports the domain API from
`config`. Never make `planning` depend on `surgical` or the facade.

## Running tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest                         # 341 tests
python -m pytest --cov=opencode_sync     # ~93% branch coverage
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
`jsonc_edit.find_comment_spans` is a hand-rolled state machine that handles `//`, `/* */`, escaped quotes inside strings, and URLs (`://`) in string values. It does **not** use regex — regex cannot correctly track string context. `mask_comments` blanks the spans (surgical path); `parse_jsonc` composes `mask_comments` + `mask_trailing_commas` before handing the masked text to `json.loads`, so both JSONC-tolerant paths flow through the same state machine and can't drift. (An old `_strip_jsonc_comments` delete-the-spans variant was removed: nothing in production consumed it.)

`mask_trailing_commas` is also string-aware, for the same reason. It replaced a regex (`,(\s*[}\]])`) that silently corrupted data: `parse_jsonc('{"a": "x,  }"}')` used to return `{'a': 'x  }'}`. If you're tempted by a regex here, that's the bug you'll reintroduce.

Newlines inside block comments are preserved when masking, so `JSONDecodeError` line numbers point at real lines in the user's file.

### Display name and tuning preservation
`apply_plan` rebuilds the `models` dict from the live model list, copying existing entries verbatim for IDs that survive. Users can customise names, context limits, and sampling options and have them survive future syncs.

This is keyed on the model **ID**, so a server-side rename would drop everything. Hence renames:
- `--rename OLD=NEW` moves an entry's whole body (and, on the text path, its comments) to a new ID.
- A 1-removed/1-added sync is *inferred* as a rename, but only when the removed entry carries keys beyond `name` — i.e. only when there is hand-tuning worth rescuing. It's a heuristic, so the CLI reports every inference it acts on and `--no-infer-renames` turns it off.
- A renamed entry's `name` is **regenerated from the new ID** (when the entry has one): the old display name described the old ID, and opencode shows it in the picker, so carrying it over made every rename look like a failed sync. Everything else in the entry survives. An entry with no `name` key gains none. On the text path this splices only the name's value span, so comments around it survive byte-for-byte (`_renamed_entry_text`).

### model / small_model pointers (don't regress this)
Only ever repoint a pointer that belongs to the provider being synced. Syncing provider B must never touch a pointer aimed at provider A.

Model IDs contain slashes (`org/model-a`), so "has a slash" does **not** mean "provider-qualified". The only reliable test is whether the first segment names a provider that actually exists in `config["provider"]`. Tested in `TestCrossProviderIsolation`.

A bare (unqualified) ID is legacy output from an older version of this tool and gets normalized — but only when syncing a single provider, since a bare ID is genuinely ambiguous across several.

### Default-model pointer modes (`--default-model` / `--default-small-model`)
How opencode picks its default: `config.model` first (a missing model is a hard error, no fallback), then the `recent[]` list in `~/.local/state/opencode/model.json` (dead entries are skipped, not removed), then the first model of the first provider. opencode never discovers models from an `openai-compatible` server — the config's model list *is* the universe.

The planner modes (`_plan_model_key_updates`):
- `first` (default) — legacy: never invent a pointer, only repair one that already points at the synced provider. This is why a sync of a renamed server didn't used to change what opencode defaults to when no pointer existed.
- `none` — never touch the keys.
- `auto` — when this provider's model set actually changed, point the key at the first served model. No-op syncs don't churn the pointer (the wrapper runs on every launch).
- explicit `provider/model-id` — written whenever it differs, change or no change. Validated against the syncing provider; a pointer naming another provider is rejected *before any HTTP call* (`main()` pre-checks it against the resolved targets; the planner's ValueError is the belt, the CLI check is the braces).

Cross-provider isolation applies in every mode. `update_active_model=False` (`--no-model-update`) still disables all pointer work.

The surgical editor can now *insert* a top-level `model`/`small_model` key it has never seen (via `_append_member_edit`, which re-emits existing members from source text, so comments survive). It used to be rewrite-only because the planner never invented pointers.

`--prune-recent` drops dead models from the state file's `recent[]` (path honors `XDG_STATE_HOME`; entries for providers this tool doesn't sync are kept — it only judges what it syncs). When the first entry is pruned and a pointer update exists, the pointed-at model is seeded at the head of `recent[]`. The prune runs **independent of the config-write outcome** — including no-op syncs, where a stale `recent[]` head would otherwise keep shadowing the default forever. Under `--dry-run` it reports what *would* be pruned without touching the state file (the no-op branch honors this too — it was once a real bug that only `--dry-run` after a non-noop plan was dry).

### Explicit provider targeting is strict
Naming a provider (`--provider ID`, with or without `--host`/`--port`) is an explicit request to update *that* provider, so both failure modes are hard errors rather than surprises:
- a provider ID that isn't in the config dies (`not in the config (...)`) — the old behavior silently *created* a bogus provider block and exited 0, turning typos into config corruption;
- a named provider with no stored `options.baseURL` dies (`no options.baseURL to query`) — the old behavior silently queried `http://localhost:8080/v1` and could sync a stranger's model list into the entry. The sync-all path (no `--provider`) still warns-and-skips such providers, since nothing was named.

### Orphan-zone comments (`body_trailing`, don't regress this)
A comment between the last member and an object's closing brace belongs to no member — it describes the object. `find_object_members` captures it as `ObjectBody.body_trailing` (span from the newline after the last member to the closing brace's line start), and `render_object` re-emits it verbatim before the brace. All three rebuild paths flow through this: the models rebuild, provider append, and top-level key append. Pure-whitespace trailing blocks are dropped on rebuild (the entry loop already emits the separator newline); comment-bearing blocks survive byte-identically. Canaries live in `test_surgical.py::TestCommentsSurvive` and the shared fixture.

### EOL policy
Inserted text is translated to the file's dominant line ending (`_dominant_eol` / `_to_file_eol` in `surgical.py`): a CRLF config stays uniformly CRLF after a rebuild instead of accumulating bare-LF lines. Producers (`render_object`, `_render_value`) emit `\n`-joined text; the surgical layer owns the translation so no producer needs to know.

### Console encoding
`main()` reconfigures stdout/stderr with `errors="replace"` before printing anything: a cp1252 console that can't render a glyph degrades the glyph, not the run. Printed output sticks to cp1252-representable characters (`->`, em-dashes are fine).

### Exit codes
`0` = synced (wrote, or `--dry-run` produced a plan); `2` = nothing to do (no-op, byte-identical, nothing to apply); `1` = error (validation, every server down, write failure). `--json` emits one JSON result object on stdout (banner and progress suppressed, update-check notice moved to stderr); `--quiet` suppresses progress but never errors; `--verbose` adds detail.


### Config mutation safety
`apply_plan` always deep-copies the input dict before modifying it. The caller's dict is never mutated. Tests verify this in `TestUpdateProviderModels::test_plan_apply_does_not_mutate_input_config`.

### Writes are atomic
`_atomic_write_text` writes a temp file in the target directory (same filesystem — `os.replace` is only atomic within one), fsyncs, copies the original's mode across (`mkstemp` creates 0600, so this is mandatory, not a nicety), then `os.replace`s. A rolling `.bak` is written **only when content actually changes**, so routine no-op runs don't churn it. The `install` wrapper uses the same temp+replace discipline (`_write_wrapper`), with the executable-bit chmod guarded to POSIX and `newline="\n"` so the `/bin/sh` wrapper can never be written with CRLF endings.

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
