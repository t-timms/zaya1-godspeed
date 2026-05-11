from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import remap_to_zaya as rtz


class TestFormatZayaToolCall:
    def test_simple_tool(self):
        result = rtz.format_zaya_tool_call("file_read", {"path": "src/main.py"})
        assert result.startswith("<zyphra_tool_call>")
        assert result.endswith("</zyphra_tool_call>")
        inner = json.loads(result[len("<zyphra_tool_call>"): -len("</zyphra_tool_call>")])
        assert inner == {"name": "file_read", "arguments": {"path": "src/main.py"}}

    def test_empty_arguments(self):
        result = rtz.format_zaya_tool_call("noop", {})
        inner = json.loads(result[len("<zyphra_tool_call>"): -len("</zyphra_tool_call>")])
        assert inner["arguments"] == {}

    def test_unicode_arguments(self):
        result = rtz.format_zaya_tool_call("search", {"query": "\u00fcber"})
        inner = json.loads(result[len("<zyphra_tool_call>"): -len("</zyphra_tool_call>")])
        assert inner["arguments"]["query"] == "\u00fcber"

    def test_nested_arguments(self):
        args = {"filters": {"status": "active", "limit": 10}}
        result = rtz.format_zaya_tool_call("query", args)
        inner = json.loads(result[len("<zyphra_tool_call>"): -len("</zyphra_tool_call>")])
        assert inner["arguments"] == args


class TestFormatZayaToolResponse:
    def test_simple_response(self):
        result = rtz.format_zaya_tool_response("file contents here")
        assert result == "<zyphra_tool_response>file contents here</zyphra_tool_response>"

    def test_empty_response(self):
        result = rtz.format_zaya_tool_response("")
        assert result == "<zyphra_tool_response></zyphra_tool_response>"

    def test_multiline_response(self):
        result = rtz.format_zaya_tool_response("line1\nline2\nline3")
        assert "<zyphra_tool_response>" in result
        assert "</zyphra_tool_response>" in result
        assert "line1\nline2\nline3" in result


class TestExtractToolCalls:
    def test_single_tool_call(self):
        content = '<zyphra_tool_call>{"name": "file_read", "arguments": {"path": "x.py"}}</zyphra_tool_call>'
        calls = rtz._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "file_read"

    def test_multiple_tool_calls(self):
        content = (
            '<zyphra_tool_call>{"name": "a", "arguments": {}}</zyphra_tool_call>\n'
            '<zyphra_tool_call>{"name": "b", "arguments": {}}</zyphra_tool_call>'
        )
        calls = rtz._extract_tool_calls_from_content(content)
        assert len(calls) == 2

    def test_no_tool_calls(self):
        content = "Just some text."
        calls = rtz._extract_tool_calls_from_content(content)
        assert calls == []

    def test_malformed_json(self):
        content = '<zyphra_tool_call>not json</zyphra_tool_call>'
        calls = rtz._extract_tool_calls_from_content(content)
        assert calls == []

    def test_mixed_valid_invalid(self):
        content = (
            '<zyphra_tool_call>{"name": "ok", "arguments": {}}</zyphra_tool_call>\n'
            '<zyphra_tool_call>bad json</zyphra_tool_call>'
        )
        calls = rtz._extract_tool_calls_from_content(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "ok"


class TestParseGodspeedMessage:
    def test_system(self, sample_system_message):
        result = rtz._parse_godspeed_message(sample_system_message)
        assert result == {"role": "system", "content": "You are Godspeed, a coding agent with tools."}

    def test_user(self, sample_user_message):
        result = rtz._parse_godspeed_message(sample_user_message)
        assert result == {"role": "user", "content": "Fix the bug in src/main.py."}

    def test_assistant_no_tools(self, sample_assistant_no_tools):
        result = rtz._parse_godspeed_message(sample_assistant_no_tools)
        assert result == {"role": "assistant", "content": "Let me look at the file first."}

    def test_assistant_with_tool(self, sample_assistant_with_tool):
        result = rtz._parse_godspeed_message(sample_assistant_with_tool)
        assert result["role"] == "assistant"
        assert "<zyphra_tool_call>" in result["content"]
        assert "file_read" in result["content"]
        assert "<think>" not in result["content"]

    def test_assistant_with_thinking(self, sample_assistant_with_thinking):
        result = rtz._parse_godspeed_message(sample_assistant_with_thinking)
        assert "<think>" in result["content"]
        assert "</think>" in result["content"]
        assert "<zyphra_tool_call>" in result["content"]

    def test_assistant_multi_tool(self, sample_assistant_multi_tool):
        result = rtz._parse_godspeed_message(sample_assistant_multi_tool)
        assert result["content"].count("<zyphra_tool_call>") == 2

    def test_tool_result(self, sample_tool_result):
        result = rtz._parse_godspeed_message(sample_tool_result)
        expected = '<zyphra_tool_response>def main():\n    print("hello world")</zyphra_tool_response>'
        assert result["role"] == "tool"
        assert result["content"] == expected

    def test_tool_result_no_name(self):
        msg = {"role": "tool", "content": "some output"}
        result = rtz._parse_godspeed_message(msg)
        assert result == {"role": "tool", "content": "<zyphra_tool_response>some output</zyphra_tool_response>"}

    def test_session_end(self, sample_session_end_ok):
        result = rtz._parse_godspeed_message(sample_session_end_ok)
        assert result is None

    def test_meta(self, sample_meta):
        result = rtz._parse_godspeed_message(sample_meta)
        assert result is None

    def test_unknown_role(self):
        result = rtz._parse_godspeed_message({"role": "unknown", "content": "..."})
        assert result is None


class TestQualityGates:
    def test_dangerous_commands_detected(self, sample_dangerous):
        assert rtz._check_dangerous_commands([sample_dangerous])

    def test_dangerous_commands_clean(self, sample_system_message, sample_user_message):
        assert not rtz._check_dangerous_commands([sample_system_message, sample_user_message])

    def test_errors_detected(self, sample_tool_result_error):
        assert rtz._check_errors([sample_tool_result_error])

    def test_errors_clean(self, sample_tool_result):
        assert not rtz._check_errors([sample_tool_result])

    def test_count_tool_calls(self):
        messages = [
            {"role": "assistant", "content": '<zyphra_tool_call>{"name":"a","arguments":{}}</zyphra_tool_call>'},
            {"role": "assistant", "content": '<zyphra_tool_call>{"name":"b","arguments":{}}</zyphra_tool_call>'},
        ]
        assert rtz._count_tool_calls(messages) == 2

    def test_count_tool_calls_none(self):
        messages = [{"role": "assistant", "content": "Hello world"}]
        assert rtz._count_tool_calls(messages) == 0

    def test_tool_names_used(self):
        messages = [
            {"role": "assistant", "content": (
                '<zyphra_tool_call>{"name":"file_read","arguments":{}}</zyphra_tool_call>\n'
                '<zyphra_tool_call>{"name":"grep_search","arguments":{}}</zyphra_tool_call>'
            )},
        ]
        names = rtz._tool_names_used(messages)
        assert names == {"file_read", "grep_search"}

    def test_tool_names_used_malformed_skipped(self):
        messages = [
            {"role": "assistant", "content": (
                '<zyphra_tool_call>{"name":"ok","arguments":{}}</zyphra_tool_call>\n'
                '<zyphra_tool_call>bad</zyphra_tool_call>'
            )},
        ]
        names = rtz._tool_names_used(messages)
        assert names == {"ok"}


class TestJaccardSimilarity:
    def test_identical(self):
        assert rtz._jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert rtz._jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial(self):
        assert rtz._jaccard_similarity({"a", "b"}, {"b", "c"}) == 1 / 3

    def test_both_empty(self):
        assert rtz._jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert rtz._jaccard_similarity(set(), {"a"}) == 0.0
        assert rtz._jaccard_similarity({"a"}, set()) == 0.0


class TestEstimateTokens:
    def test_empty(self):
        assert rtz._estimate_tokens([]) == 0

    def test_simple_content(self):
        messages = [{"role": "user", "content": "Hello world! This is a test."}]
        tokens = rtz._estimate_tokens(messages)
        assert tokens == len("Hello world! This is a test.") // 4

    def test_no_content_key(self):
        messages = [{"role": "system"}]
        assert rtz._estimate_tokens(messages) == 0

    def test_non_string_content(self):
        messages = [{"role": "assistant", "content": 12345}]
        assert rtz._estimate_tokens(messages) == 0


class TestRemapSession:
    def test_full_ok_conversation(self, tmp_conversation_file):
        result = rtz.remap_session(
            tmp_conversation_file,
            expected_tools={},
            min_jaccard=0.0,
        )
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) >= 3
        roles = [m["role"] for m in result["messages"]]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

    def test_session_fail_exit_code(self, tmp_conversation_file):
        with open(tmp_conversation_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "system", "content": "agent"}) + "\n")
            f.write(json.dumps({"role": "user", "content": "help"}) + "\n")
            f.write(json.dumps({"role": "session_end", "exit_code": 1}) + "\n")
        result = rtz.remap_session(tmp_conversation_file, expected_tools={})
        assert result is None

    def test_dangerous_command_skipped(self, tmp_conversation_file):
        with open(tmp_conversation_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "system", "content": "agent"}) + "\n")
            f.write(json.dumps({"role": "user", "content": "help"}) + "\n")
            f.write(json.dumps({"dangerous_command": True}) + "\n")
        result = rtz.remap_session(tmp_conversation_file, expected_tools={})
        assert result is None

    def test_tool_error_skipped(self, tmp_conversation_file):
        with open(tmp_conversation_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "system", "content": "agent"}) + "\n")
            f.write(json.dumps({"role": "user", "content": "help"}) + "\n")
            f.write(json.dumps({"role": "tool", "name": "x", "content": "err", "is_error": True}) + "\n")
        result = rtz.remap_session(tmp_conversation_file, expected_tools={})
        assert result is None

    def test_no_tool_calls_skipped(self, tmp_conversation_file):
        with open(tmp_conversation_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "system", "content": "agent"}) + "\n")
            f.write(json.dumps({"role": "user", "content": "hello"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "hi there"}) + "\n")
            f.write(json.dumps({"role": "session_end", "exit_code": 0}) + "\n")
        result = rtz.remap_session(tmp_conversation_file, expected_tools={})
        assert result is None

    def test_token_budget_exceeded(self, tmp_conversation_file):
        long_content = "x" * 20000
        with open(tmp_conversation_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "system", "content": "agent"}) + "\n")
            f.write(json.dumps({"role": "user", "content": long_content}) + "\n")
            f.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": '<zyphra_tool_call>{"name":"x","arguments":{}}</zyphra_tool_call>',
                    }
                )
                + "\n"
            )
            f.write(json.dumps({"role": "tool", "content": "ok"}) + "\n")
            f.write(json.dumps({"role": "session_end", "exit_code": 0}) + "\n")
        result = rtz.remap_session(tmp_conversation_file, expected_tools={}, max_tokens=4096)
        assert result is None

    def test_too_few_messages(self, tmp_conversation_file):
        with open(tmp_conversation_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "content": "hi"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "hello"}) + "\n")
        result = rtz.remap_session(tmp_conversation_file, expected_tools={})
        assert result is None

    def test_empty_file(self, tmp_conversation_file):
        with open(tmp_conversation_file, "w", encoding="utf-8") as f:
            f.write("")
        result = rtz.remap_session(tmp_conversation_file, expected_tools={})
        assert result is None

    def test_jaccard_gate_fail(self, tmp_path):
        filepath = tmp_path / "task-01.session.conversation.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "system", "content": "agent"}) + "\n")
            f.write(json.dumps({"role": "user", "content": "fix bug"}) + "\n")
            f.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": '<zyphra_tool_call>{"name":"file_read","arguments":{}}</zyphra_tool_call>',
                    }
                )
                + "\n"
            )
            f.write(json.dumps({"role": "tool", "content": "ok"}) + "\n")
            f.write(json.dumps({"role": "session_end", "exit_code": 0}) + "\n")
        expected_tools = {"task-01": {"tool_x", "tool_y"}}
        result = rtz.remap_session(filepath, expected_tools=expected_tools, min_jaccard=0.7)
        assert result is None

    def test_jaccard_gate_pass(self, tmp_path):
        filepath = tmp_path / "task-01.session.conversation.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "system", "content": "agent"}) + "\n")
            f.write(json.dumps({"role": "user", "content": "fix bug"}) + "\n")
            f.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": '<zyphra_tool_call>{"name":"file_read","arguments":{}}</zyphra_tool_call>',
                    }
                )
                + "\n"
            )
            f.write(json.dumps({"role": "tool", "content": "ok"}) + "\n")
            f.write(json.dumps({"role": "session_end", "exit_code": 0}) + "\n")
        expected_tools = {"task-01": {"file_read", "grep_search"}}
        result = rtz.remap_session(filepath, expected_tools=expected_tools, min_jaccard=0.5)
        assert result is not None


class TestLoadExpectedTools:
    def test_none_path(self):
        assert rtz.load_expected_tools(None) == {}

    def test_nonexistent_path(self):
        assert rtz.load_expected_tools("/nonexistent/path.jsonl") == {}

    def test_valid_file(self, tmp_tasks_file, base_tasks):
        expected = rtz.load_expected_tools(str(tmp_tasks_file))
        assert len(expected) == len(base_tasks)
        for task in base_tasks:
            assert task["task_id"] in expected


class TestRemapDirectory:
    def test_nonexistent_dir(self):
        result = rtz.remap_directory(
            input_dir="/nonexistent/path",
            output_path="/tmp/out.jsonl",
        )
        assert result == 0

    def test_no_conversation_files(self, tmp_path):
        out = tmp_path / "out.jsonl"
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = rtz.remap_directory(
            input_dir=str(empty_dir),
            output_path=str(out),
        )
        assert result == 0

    def test_dry_run(self, tmp_conversation_file, tmp_output_file):
        result = rtz.remap_directory(
            input_dir=str(tmp_conversation_file.parent),
            output_path=str(tmp_output_file),
            dry_run=True,
        )
        assert result >= 0
        content = tmp_output_file.read_text(encoding="utf-8").strip()
        assert content == ""
