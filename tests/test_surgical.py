"""Safety tests: syncing must not disturb anything it wasn't asked to change.

These run against tests/fixtures/advanced.jsonc, a realistic hand-maintained config
with comments, two providers, rich per-model metadata, and a large agent section.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest

from opencode_sync.cli import main
from opencode_sync.config import (
    apply_plans_to_text,
    parse_jsonc,
    plan_provider_update,
)
from opencode_sync.jsonc_edit import (
    JsoncEditError,
    find_value_span,
    mask_comments,
)

FIXTURE = Path(__file__).parent / "fixtures" / "advanced.jsonc"
COMMENT = 'Qwen3 "thinking" recommended sampling'


@pytest.fixture()
def text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture()
def config(text) -> dict:
    return parse_jsonc(text)


def diff(a: str, b: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "before", "after"))


def span_of(text: str, path):
    return find_value_span(mask_comments(text), path)


class TestFixtureIsRealistic:
    def test_parses(self, config):
        assert set(config) == {"$schema", "provider", "agent"}
        assert set(config["provider"]) == {"node-a1b2", "node-c3d4"}

    def test_carries_tuned_metadata(self, config):
        entry = config["provider"]["node-c3d4"]["models"]["aeon"]
        assert entry["limit"]["context"] == 196608
        assert entry["options"]["top_k"] == 20

    def test_carries_comments(self, text):
        assert text.count(COMMENT) == 2


class TestNoOpIsByteIdentical:
    def test_unchanged_model_set_rewrites_nothing(self, text, config):
        plan = plan_provider_update(config, "node-a1b2", ["aeon"])
        assert plan.is_noop()
        assert apply_plans_to_text(text, config, [plan]) == text

    def test_both_providers_unchanged(self, text, config):
        plans = [
            plan_provider_update(config, "node-a1b2", ["aeon"]),
            plan_provider_update(config, "node-c3d4", ["aeon"]),
        ]
        assert apply_plans_to_text(text, config, plans) == text


class TestOnlyTheTargetSpanChanges:
    def test_everything_outside_the_models_block_is_byte_identical(self, text, config):
        plan = plan_provider_update(config, "node-a1b2", ["aeon", "new/model-x"])
        out = apply_plans_to_text(text, config, [plan])

        span = span_of(text, ["provider", "node-a1b2", "models"])
        assert out[: span.start] == text[: span.start], diff(text, out)
        assert out[len(out) - (len(text) - span.end) :] == text[span.end :], diff(text, out)

    def test_agent_section_is_byte_identical(self, text, config):
        plan = plan_provider_update(config, "node-a1b2", ["aeon", "new/model-x"])
        out = apply_plans_to_text(text, config, [plan])

        before = span_of(text, ["agent"]).of(text)
        after = span_of(out, ["agent"]).of(out)
        assert after == before

    def test_other_provider_is_byte_identical(self, text, config):
        plan = plan_provider_update(config, "node-a1b2", ["aeon", "new/model-x"])
        out = apply_plans_to_text(text, config, [plan])

        before = span_of(text, ["provider", "node-c3d4"]).of(text)
        after = span_of(out, ["provider", "node-c3d4"]).of(out)
        assert after == before

    def test_trailing_whitespace_canary_survives(self, text, config):
        # The fixture carries stray trailing spaces, as the real config does.
        plan = plan_provider_update(config, "node-a1b2", ["aeon", "new/model-x"])
        out = apply_plans_to_text(text, config, [plan])
        assert '"permission": { \n' in out
        assert "      } \n" in out


class TestCommentsSurvive:
    def test_comments_inside_a_surviving_entry(self, text, config):
        plan = plan_provider_update(config, "node-a1b2", ["aeon", "new/model-x"])
        out = apply_plans_to_text(text, config, [plan])
        assert out.count(COMMENT) == 2

    def test_comments_ride_along_a_rename(self, text, config):
        plan = plan_provider_update(config, "node-c3d4", ["qwen3.5-122B-A10B"])
        out = apply_plans_to_text(text, config, [plan])
        assert out.count(COMMENT) == 2

        entry = parse_jsonc(out)["provider"]["node-c3d4"]["models"]["qwen3.5-122B-A10B"]
        assert entry["limit"]["context"] == 196608
        assert entry["options"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

    def test_a_removed_model_takes_only_its_own_comment(self, config):
        text = (
            "{\n"
            '  "provider": {\n'
            '    "p": {\n'
            '      "models": {\n'
            "        // block-level note\n"
            '        "keep": { "name": "K" },\n'
            "        // about drop\n"
            '        "drop": { "name": "D" }\n'
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n"
        )
        cfg = parse_jsonc(text)
        plan = plan_provider_update(cfg, "p", ["keep"], infer_renames=False)
        out = apply_plans_to_text(text, cfg, [plan])
        assert "// block-level note" in out
        assert "// about drop" not in out


class TestRename:
    def test_one_to_one_rename_is_inferred(self, config):
        plan = plan_provider_update(config, "node-c3d4", ["qwen3.5-122B-A10B"])
        assert plan.renames == {"aeon": "qwen3.5-122B-A10B"}
        assert plan.added == []
        assert plan.removed == []

    def test_rename_is_a_minimal_diff(self, text, config):
        plan = plan_provider_update(config, "node-c3d4", ["qwen3.5-122B-A10B"])
        out = apply_plans_to_text(text, config, [plan])
        changed = [
            line
            for line in difflib.unified_diff(text.splitlines(), out.splitlines(), n=0)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
        assert changed == ['-        "aeon": {', '+        "qwen3.5-122B-A10B": {']

    def test_explicit_rename_overrides_inference(self, config):
        plan = plan_provider_update(
            config, "node-c3d4", ["a", "b"], renames={"aeon": "b"}
        )
        assert plan.renames == {"aeon": "b"}
        assert plan.added == ["a"]

    def test_inference_off_drops_the_entry(self, config):
        plan = plan_provider_update(
            config, "node-c3d4", ["qwen3.5-122B-A10B"], infer_renames=False
        )
        assert plan.renames == {}
        assert plan.removed == ["aeon"]

    def test_no_inference_for_a_bare_entry(self):
        # Nothing worth rescuing, so a swap stays a swap.
        cfg = {"provider": {"p": {"models": {"old": {"name": "old"}}}}}
        plan = plan_provider_update(cfg, "p", ["new"])
        assert plan.renames == {}
        assert (plan.added, plan.removed) == (["new"], ["old"])

    def test_no_inference_when_two_models_change(self, config):
        cfg = {
            "provider": {
                "p": {"models": {"a": {"name": "A", "limit": 1}, "b": {"name": "B", "limit": 2}}}
            }
        }
        plan = plan_provider_update(cfg, "p", ["c", "d"])
        assert plan.renames == {}


class TestCrossProviderIsolation:
    def test_syncing_one_provider_leaves_the_others_pointer_alone(self, config):
        # Regression: syncing node-c3d4 used to repoint model at itself.
        config["model"] = "node-a1b2/aeon"
        config["small_model"] = "node-a1b2/aeon"
        plan = plan_provider_update(config, "node-c3d4", ["qwen3.5-122B-A10B"])
        assert plan.model_key_updates == {}

    def test_own_pointer_is_repointed_when_its_model_vanishes(self, config):
        config["model"] = "node-a1b2/aeon"
        plan = plan_provider_update(config, "node-a1b2", ["other"], infer_renames=False)
        assert plan.model_key_updates == {"model": "node-a1b2/other"}

    def test_own_pointer_follows_a_rename(self, config):
        config["model"] = "node-c3d4/aeon"
        plan = plan_provider_update(config, "node-c3d4", ["qwen3.5-122B-A10B"])
        assert plan.model_key_updates == {"model": "node-c3d4/qwen3.5-122B-A10B"}

    def test_slashed_model_id_is_not_mistaken_for_a_qualifier(self):
        # "org/model-a" starts with "org", which is not a provider — so it's a bare ID.
        cfg = {"provider": {"vllm": {"models": {"org/model-a": {"name": "A"}}}}, "model": "org/model-a"}
        plan = plan_provider_update(cfg, "vllm", ["org/model-a"])
        assert plan.model_key_updates == {"model": "vllm/org/model-a"}

    def test_bare_id_is_left_alone_when_syncing_many_providers(self, config):
        # Ambiguous across providers, so guessing is exactly the wrong move.
        config["model"] = "aeon"
        plan = plan_provider_update(config, "node-c3d4", ["aeon"], normalize_bare_ids=False)
        assert plan.model_key_updates == {}


class TestStructuralInsertion:
    """Paths that add structure the file doesn't have yet, rather than replacing it."""

    def test_base_url_is_updated_in_place(self, text, config):
        plan = plan_provider_update(
            config, "node-a1b2", ["aeon"], base_url="http://elsewhere:9000/v1"
        )
        out = apply_plans_to_text(text, config, [plan])
        assert (
            parse_jsonc(out)["provider"]["node-a1b2"]["options"]["baseURL"]
            == "http://elsewhere:9000/v1"
        )
        assert out.count(COMMENT) == 2
        assert span_of(out, ["agent"]).of(out) == span_of(text, ["agent"]).of(text)

    def test_new_provider_is_appended_to_an_existing_file(self, text, config):
        plan = plan_provider_update(
            config, "fresh", ["m1"], base_url="http://fresh:8000/v1"
        )
        out = apply_plans_to_text(text, config, [plan])
        parsed = parse_jsonc(out)
        assert set(parsed["provider"]) == {"node-a1b2", "node-c3d4", "fresh"}
        assert parsed["provider"]["fresh"]["options"]["baseURL"] == "http://fresh:8000/v1"
        assert parsed["provider"]["fresh"]["models"] == {"m1": {"name": "m1"}}
        # The pre-existing providers are re-emitted from their own source text.
        assert out.count(COMMENT) == 2

    def test_provider_key_absent_entirely(self):
        text = '{\n  "$schema": "https://opencode.ai/config.json"\n}\n'
        cfg = parse_jsonc(text)
        plan = plan_provider_update(cfg, "vllm", ["m1"], base_url="http://h:8000/v1")
        out = apply_plans_to_text(text, cfg, [plan])
        parsed = parse_jsonc(out)
        assert parsed["$schema"] == "https://opencode.ai/config.json"
        assert parsed["provider"]["vllm"]["models"] == {"m1": {"name": "m1"}}

    def test_provider_without_a_models_key(self):
        text = (
            "{\n"
            '  "provider": {\n'
            '    "p": {\n'
            '      // no models yet\n'
            '      "options": { "baseURL": "http://h:8000/v1" }\n'
            "    }\n"
            "  }\n"
            "}\n"
        )
        cfg = parse_jsonc(text)
        plan = plan_provider_update(cfg, "p", ["m1"])
        out = apply_plans_to_text(text, cfg, [plan])
        assert parse_jsonc(out)["provider"]["p"]["models"] == {"m1": {"name": "m1"}}
        assert "// no models yet" in out

    def test_provider_without_an_options_key(self):
        text = '{\n  "provider": {\n    "p": {\n      "models": {}\n    }\n  }\n}\n'
        cfg = parse_jsonc(text)
        plan = plan_provider_update(cfg, "p", ["m1"], base_url="http://h:8000/v1")
        out = apply_plans_to_text(text, cfg, [plan])
        assert parse_jsonc(out)["provider"]["p"]["options"]["baseURL"] == "http://h:8000/v1"

    def test_empty_models_object(self):
        text = '{\n  "provider": {\n    "p": {\n      "models": {}\n    }\n  }\n}\n'
        cfg = parse_jsonc(text)
        plan = plan_provider_update(cfg, "p", ["m1", "m2"], infer_renames=False)
        out = apply_plans_to_text(text, cfg, [plan])
        assert list(parse_jsonc(out)["provider"]["p"]["models"]) == ["m1", "m2"]


class TestFourSpaceStyle:
    def test_indentation_style_is_matched(self):
        text = (
            "{\n"
            '    "provider": {\n'
            '        "p": {\n'
            '            "models": {\n'
            '                "keep": { "name": "K" }\n'
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        cfg = parse_jsonc(text)
        plan = plan_provider_update(cfg, "p", ["keep", "add"], infer_renames=False)
        out = apply_plans_to_text(text, cfg, [plan])
        assert '                "add": {\n                    "name": "add"\n                }' in out


class TestCrlf:
    def test_crlf_line_endings_survive(self):
        text = (
            '{\r\n  "provider": {\r\n    "p": {\r\n      "models": {\r\n'
            '        "keep": { "name": "K" }\r\n      }\r\n    }\r\n  }\r\n}\r\n'
        )
        cfg = parse_jsonc(text)
        plan = plan_provider_update(cfg, "p", ["keep"])
        assert plan.is_noop()
        assert apply_plans_to_text(text, cfg, [plan]) == text

    def test_crlf_file_is_not_converted_to_lf_on_write(self, tmp_path, mock_server):
        from opencode_sync.config import load_config_text, save_config_text

        path = tmp_path / "opencode.jsonc"
        path.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
        text = load_config_text(path)
        assert "\r\n" in text
        save_config_text(path, text)
        assert path.read_bytes() == b'{\r\n  "a": 1\r\n}\r\n'


class TestSelfCheck:
    """Two independent guards, either of which must stop a bad edit reaching disk."""

    def test_an_edit_that_never_lands_is_refused(self, text, config, monkeypatch):
        import opencode_sync.config as cfgmod

        def sabotage(text_, masked, plan):
            span = find_value_span(masked, ["provider", plan.provider_id, "models"])
            return span, '{"wrong": {"name": "wrong"}}'

        monkeypatch.setattr(cfgmod, "_models_edit", sabotage)
        plan = plan_provider_update(config, "node-a1b2", ["aeon", "new/model-x"])
        with pytest.raises(JsoncEditError, match="did not converge"):
            apply_plans_to_text(text, config, [plan])

    def test_text_meaning_something_else_is_refused(self, config):
        from opencode_sync.config import _verify_edit

        plan = plan_provider_update(config, "node-a1b2", ["aeon"])
        with pytest.raises(JsoncEditError, match="changed the config's meaning"):
            _verify_edit('{"provider": {}}', config, [plan])

    def test_model_order_is_part_of_the_check(self, config):
        from opencode_sync.config import _verify_edit

        plan = plan_provider_update(config, "node-a1b2", ["a", "b"], infer_renames=False)
        good = {
            "$schema": config["$schema"],
            "provider": {
                "node-a1b2": {
                    **config["provider"]["node-a1b2"],
                    "models": {"a": {"name": "a"}, "b": {"name": "b"}},
                },
                "node-c3d4": config["provider"]["node-c3d4"],
            },
            "agent": config["agent"],
        }
        _verify_edit(json.dumps(good), config, [plan])  # correct order passes

        swapped = json.loads(json.dumps(good))
        swapped["provider"]["node-a1b2"]["models"] = {"b": {"name": "b"}, "a": {"name": "a"}}
        with pytest.raises(JsoncEditError):
            _verify_edit(json.dumps(swapped), config, [plan])

    def test_invalid_output_is_refused(self, config):
        from opencode_sync.config import _verify_edit

        plan = plan_provider_update(config, "node-a1b2", ["aeon"])
        with pytest.raises(JsoncEditError, match="invalid JSONC"):
            _verify_edit("{not json", config, [plan])


class TestIdempotence:
    def test_second_sync_is_byte_identical(self, text, config):
        plan = plan_provider_update(config, "node-c3d4", ["qwen3.5-122B-A10B"])
        once = apply_plans_to_text(text, config, [plan])

        cfg2 = parse_jsonc(once)
        plan2 = plan_provider_update(cfg2, "node-c3d4", ["qwen3.5-122B-A10B"])
        assert plan2.is_noop()
        assert apply_plans_to_text(once, cfg2, [plan2]) == once


class TestWholeFileProperty:
    def test_every_path_span_reparses_to_its_value(self, text, config):
        """The scanner must agree with json.loads about every value in the file."""
        masked = mask_comments(text)

        def walk(obj, path=()):
            yield path, obj
            if isinstance(obj, dict):
                for key, value in obj.items():
                    yield from walk(value, path + (key,))

        checked = 0
        for path, value in walk(config):
            if not path:
                continue
            span = find_value_span(masked, path)
            assert span is not None, f"no span for {path}"
            assert parse_jsonc(span.of(text)) == value, f"mismatch at {path}"
            checked += 1
        assert checked > 50


class TestEndToEnd:
    def _write(self, tmp_path: Path) -> Path:
        path = tmp_path / "opencode.jsonc"
        path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
        return path

    def test_sync_all_hits_every_provider(self, tmp_path, mock_server, capsys):
        path = self._write(tmp_path)
        before = path.read_text()
        a = mock_server(["aeon"])
        c = mock_server(["qwen3.5-122B-A10B"])

        text = before.replace("http://node-a1b2.example.invalid:8000/v1", a.base_url)
        text = text.replace("http://node-c3d4.example.invalid:8000/v1", c.base_url)
        path.write_text(text)

        assert main(["--config", str(path)]) == 0

        out = path.read_text()
        assert out.count(COMMENT) == 2
        models = parse_jsonc(out)["provider"]
        assert set(models["node-a1b2"]["models"]) == {"aeon"}
        assert set(models["node-c3d4"]["models"]) == {"qwen3.5-122B-A10B"}
        # The rename carried the tuning across.
        assert models["node-c3d4"]["models"]["qwen3.5-122B-A10B"]["limit"]["context"] == 196608

    def test_one_server_down_does_not_block_the_other(self, tmp_path, mock_server):
        path = self._write(tmp_path)
        a = mock_server(["aeon", "extra"])
        text = path.read_text().replace("http://node-a1b2.example.invalid:8000/v1", a.base_url)
        path.write_text(text)  # node-c3d4 still points at an unroutable host

        assert main(["--config", str(path), "--timeout", "1"]) == 0
        assert set(parse_jsonc(path.read_text())["provider"]["node-a1b2"]["models"]) == {
            "aeon",
            "extra",
        }

    def test_noop_sync_does_not_touch_the_file(self, tmp_path, mock_server):
        path = self._write(tmp_path)
        a = mock_server(["aeon"])
        text = path.read_text().replace("http://node-a1b2.example.invalid:8000/v1", a.base_url)
        path.write_text(text)
        before = path.read_text()
        mtime = path.stat().st_mtime_ns

        assert main(["--config", str(path), "--provider", "node-a1b2"]) == 0
        assert path.read_text() == before
        assert path.stat().st_mtime_ns == mtime
        assert not path.with_suffix(".jsonc.bak").exists()

    def test_dry_run_prints_a_diff_and_writes_nothing(self, tmp_path, mock_server, capsys):
        path = self._write(tmp_path)
        a = mock_server(["aeon", "extra"])
        text = path.read_text().replace("http://node-a1b2.example.invalid:8000/v1", a.base_url)
        path.write_text(text)
        before = path.read_text()

        assert main(["--config", str(path), "--provider", "node-a1b2", "--dry-run"]) == 0
        assert path.read_text() == before
        out = capsys.readouterr().out
        assert '+        "extra": {' in out
        assert "[dry-run]" in out

    def test_backup_is_written_only_on_change(self, tmp_path, mock_server):
        path = self._write(tmp_path)
        a = mock_server(["aeon", "extra"])
        text = path.read_text().replace("http://node-a1b2.example.invalid:8000/v1", a.base_url)
        path.write_text(text)
        original = path.read_text()

        assert main(["--config", str(path), "--provider", "node-a1b2"]) == 0
        backup = path.with_suffix(".jsonc.bak")
        assert backup.exists()
        assert backup.read_text() == original

    def test_explicit_rename_flag(self, tmp_path, mock_server):
        path = self._write(tmp_path)
        c = mock_server(["brand-new"])
        text = path.read_text().replace("http://node-c3d4.example.invalid:8000/v1", c.base_url)
        path.write_text(text)

        rc = main(
            ["--config", str(path), "--provider", "node-c3d4", "--rename", "aeon=brand-new"]
        )
        assert rc == 0
        entry = parse_jsonc(path.read_text())["provider"]["node-c3d4"]["models"]["brand-new"]
        assert entry["limit"]["context"] == 196608
        assert path.read_text().count(COMMENT) == 2
