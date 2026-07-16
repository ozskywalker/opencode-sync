"""Tests for the position-preserving JSONC scanner."""

import json

import pytest

from opencode_sync.jsonc_edit import (
    JsoncEditError,
    RenderEntry,
    Span,
    apply_edits,
    detect_base_indent,
    detect_unit,
    find_comment_spans,
    find_object_members,
    find_value_span,
    mask_comments,
    mask_trailing_commas,
    render_object,
)


def parse(masked: str):
    return json.loads(mask_trailing_commas(mask_comments(masked)))


class TestMaskComments:
    def test_length_is_preserved(self):
        text = '{"a": 1} // tail'
        assert len(mask_comments(text)) == len(text)

    def test_line_comment_becomes_spaces(self):
        assert mask_comments('{"a": 1} // x') == '{"a": 1}     '

    def test_block_comment_becomes_spaces(self):
        assert mask_comments('{/* hi */"a": 1}') == '{' + " " * 8 + '"a": 1}'

    def test_newlines_inside_block_comments_survive(self):
        # Keeps JSONDecodeError line numbers pointing at the user's real lines.
        masked = mask_comments('{\n/* one\ntwo */\n"a": 1}')
        assert masked.count("\n") == 3
        assert masked == "{\n" + " " * 6 + "\n" + " " * 6 + '\n"a": 1}'

    def test_url_in_string_is_not_a_comment(self):
        text = '{"baseURL": "http://host:8080/v1"}'
        assert mask_comments(text) == text

    def test_escaped_quote_does_not_end_string(self):
        text = '{"a": "say \\" // not a comment"}'
        assert mask_comments(text) == text

    def test_unterminated_block_comment_consumes_to_eof(self):
        assert mask_comments('{"a": 1} /* x') == '{"a": 1}     '

    def test_comment_markers_inside_string_are_untouched(self):
        text = '{"a": "/* not a comment */"}'
        assert mask_comments(text) == text

    def test_offsets_survive_non_ascii(self):
        text = '{"a": "ééé"} // café'
        masked = mask_comments(text)
        assert len(masked) == len(text)
        assert masked.index('"a"') == text.index('"a"')


class TestFindCommentSpans:
    def test_line_comment_span_excludes_newline(self):
        text = '{"a": 1} // x\n'
        (span,) = find_comment_spans(text)
        assert span.of(text) == "// x"

    def test_finds_multiple(self):
        text = "// one\n{} /* two */"
        assert len(find_comment_spans(text)) == 2


class TestMaskTrailingCommas:
    def test_removes_trailing_comma_before_brace(self):
        assert parse('{"a": 1,}') == {"a": 1}

    def test_removes_trailing_comma_before_bracket(self):
        assert parse('{"a": [1, 2,]}') == {"a": [1, 2]}

    def test_removes_trailing_comma_across_newline(self):
        assert parse('{"a": 1,\n}') == {"a": 1}

    def test_preserves_separating_commas(self):
        assert parse('{"a": 1, "b": 2}') == {"a": 1, "b": 2}

    def test_length_is_preserved(self):
        masked = '{"a": 1,}'
        assert len(mask_trailing_commas(masked)) == len(masked)

    def test_comma_inside_a_string_value_survives(self):
        # Regression: the old regex stripper was not string-aware and turned
        # {"a": "x,  }"} into {"a": "x  }"}.
        assert parse('{"a": "x,  }"}') == {"a": "x,  }"}
        assert parse('{"note": "trailing, ]"}') == {"note": "trailing, ]"}

    def test_trailing_comma_after_a_comment(self):
        assert parse('{"a": 1, // why\n}') == {"a": 1}


class TestScanner:
    def test_finds_nested_value(self):
        text = '{"provider": {"vllm": {"models": {"a": {"name": "A"}}}}}'
        span = find_value_span(text, ["provider", "vllm", "models"])
        assert span.of(text) == '{"a": {"name": "A"}}'

    def test_missing_key_returns_none(self):
        assert find_value_span('{"a": 1}', ["nope"]) is None
        assert find_value_span('{"a": 1}', ["a", "deeper"]) is None

    def test_scalar_span(self):
        text = '{"model": "vllm/a"}'
        assert find_value_span(text, ["model"]).of(text) == '"vllm/a"'

    def test_tolerates_trailing_commas(self):
        text = '{"a": {"b": 1,},}'
        assert find_value_span(text, ["a"]).of(text) == '{"b": 1,}'

    def test_duplicate_key_takes_the_last(self):
        text = '{"a": 1, "a": 2}'
        assert find_value_span(text, ["a"]).of(text) == "2"
        assert json.loads(text)["a"] == 2

    def test_string_containing_braces(self):
        text = '{"a": "}{", "b": 2}'
        assert find_value_span(text, ["b"]).of(text) == "2"

    def test_handles_tabs_and_crlf(self):
        text = '{\r\n\t"a": {\r\n\t\t"b": 1\r\n\t}\r\n}'
        assert find_value_span(text, ["a", "b"]).of(text) == "1"

    def test_non_object_root_raises(self):
        with pytest.raises(JsoncEditError):
            find_value_span("[1, 2]", ["a"])


class TestObjectMembers:
    def test_member_spans(self):
        text = '{"a": 1, "b": 2}'
        body = find_object_members(text, find_value_span(text, []) or Span(0, len(text)))
        assert [m.key for m in body.members] == ["a", "b"]
        assert body.members[1].value_span.of(text) == "2"

    def test_comment_above_first_member_attaches_to_the_block(self):
        # It describes the block, so dropping member "a" must not delete it.
        text = '{\n  // about the block\n  "a": 1\n}'
        masked = mask_comments(text)
        body = find_object_members(masked, Span(0, len(text)))
        assert body.block_leading.of(text) == "\n  // about the block\n"
        assert body.members[0].leading.is_empty()

    def test_comment_between_members_attaches_below(self):
        text = '{\n  "a": 1,\n  // about b\n  "b": 2\n}'
        masked = mask_comments(text)
        body = find_object_members(masked, Span(0, len(text)))
        assert body.members[1].leading.of(text) == "  // about b\n"
        assert body.members[0].trailing.of(text) == ""

    def test_same_line_comment_stays_with_its_member(self):
        text = '{\n  "a": 1, // about a\n  "b": 2\n}'
        masked = mask_comments(text)
        body = find_object_members(masked, Span(0, len(text)))
        assert body.members[0].trailing.of(text) == " // about a"
        assert body.members[1].leading.is_empty()

    def test_comment_inside_a_member_value_is_part_of_its_span(self):
        text = '{\n  "a": {\n    // inner\n    "x": 1\n  }\n}'
        masked = mask_comments(text)
        body = find_object_members(masked, Span(0, len(text)))
        assert "// inner" in body.members[0].value_span.of(text)

    def test_empty_object(self):
        body = find_object_members("{}", Span(0, 2))
        assert body.members == []


class TestIndent:
    def test_detects_base_and_unit(self):
        text = '{\n  "models": {\n    "a": 1\n  }\n}'
        key = text.index('"models"')
        base = detect_base_indent(text, key)
        assert base == "  "
        assert detect_unit(text, base, text.index('"a"')) == "  "

    def test_four_space_style(self):
        text = '{\n    "models": {\n        "a": 1\n    }\n}'
        key = text.index('"models"')
        base = detect_base_indent(text, key)
        assert base == "    "
        assert detect_unit(text, base, text.index('"a"')) == "    "

    def test_unit_defaults_when_no_members(self):
        assert detect_unit('{\n  "m": {}\n}', "  ", None) == "  "


class TestRenderObject:
    def test_renders_entries(self):
        out = render_object(
            "  ", "  ", "\n", [RenderEntry("", '"a"', "1", ""), RenderEntry("", '"b"', "2", "")]
        )
        assert out == '{\n    "a": 1,\n    "b": 2\n  }'

    def test_re_emits_comments(self):
        out = render_object(
            "",
            "  ",
            "\n  // block\n",
            [RenderEntry("", '"a"', "1", " // tail"), RenderEntry("  // about b\n", '"b"', "2", "")],
        )
        assert out == '{\n  // block\n  "a": 1, // tail\n  // about b\n  "b": 2\n}'

    def test_empty(self):
        assert render_object("", "  ", "\n", []) == "{}"


class TestApplyEdits:
    def test_applies_in_reverse_so_offsets_stay_valid(self):
        text = "0123456789"
        out = apply_edits(text, [(Span(0, 2), "AA"), (Span(8, 10), "BBBB")])
        assert out == "AA234567BBBB"

    def test_rejects_overlapping_edits(self):
        with pytest.raises(JsoncEditError, match="overlapping"):
            apply_edits("0123456789", [(Span(0, 5), "x"), (Span(4, 6), "y")])

    def test_no_edits_is_identity(self):
        assert apply_edits("abc", []) == "abc"
