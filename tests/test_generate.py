from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import generate as gen


class TestHasToolUse:
    def test_zaya_tool_call_format(self):
        messages = [
            {"role": "assistant", "content": '<zyphra_tool_call>{"name":"x","arguments":{}}</zyphra_tool_call>'},
        ]
        assert gen._has_tool_use(messages)

    def test_name_keyword(self):
        messages = [
            {"role": "assistant", "content": '{"name": "file_read"}'},
        ]
        assert gen._has_tool_use(messages)

    def test_no_tool_use(self):
        messages = [
            {"role": "assistant", "content": "Just text, no tools."},
        ]
        assert not gen._has_tool_use(messages)

    def test_only_user_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert not gen._has_tool_use(messages)

    def test_empty_messages(self):
        assert not gen._has_tool_use([])


class TestParseConversation:
    def test_valid_multiturn_trajectory(self, tmp_path):
        tool_call_str = '<zyphra_tool_call>{"name":"file_read","arguments":{}}</zyphra_tool_call>'
        filepath = _write_conversation(tmp_path, [
            {"role": "system", "content": "Agent with tools."},
            {"role": "user", "content": "Fix bug."},
            {"role": "assistant", "content": tool_call_str},
            {"role": "tool", "content": "file content"},
            {"role": "assistant", "content": "Bug fixed."},
        ])
        result = gen.parse_conversation(filepath)
        assert result is not None
        assert len(result) == 5

    def test_too_few_messages(self, tmp_path):
        filepath = _write_conversation(tmp_path, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        result = gen.parse_conversation(filepath)
        assert result is None

    def test_no_tool_use(self, tmp_path):
        filepath = _write_conversation(tmp_path, [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        result = gen.parse_conversation(filepath)
        assert result is None

    def test_single_turn_only(self, tmp_path):
        tool_call_str = '<zyphra_tool_call>{"name":"file_read","arguments":{}}</zyphra_tool_call>'
        filepath = _write_conversation(tmp_path, [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": tool_call_str},
            {"role": "tool", "content": "content"},
        ])
        result = gen.parse_conversation(filepath)
        assert result is None

    def test_malformed_lines_skipped(self, tmp_path):
        filepath = _write_conversation_raw(tmp_path, [
            '{"role": "system", "content": "agent"}\n',
            "not valid json\n",
            '{"role": "user", "content": "hi"}\n',
            '{"role": "assistant", "content": "<zyphra_tool_call>{\\\"name\\\":\\\"x\\\"}</zyphra_tool_call>"}\n',
            '{"role": "tool", "content": "result"}\n',
            '{"role": "assistant", "content": "done"}\n',
        ])
        result = gen.parse_conversation(filepath)
        assert result is not None
        assert len(result) == 5

    def test_blank_lines_skipped(self, tmp_path):
        filepath = _write_conversation_raw(tmp_path, [
            '\n',
            '{"role": "system", "content": "agent"}\n',
            '{"role": "user", "content": "do something"}\n',
            '{"role": "assistant", "content": "<zyphra_tool_call>x</zyphra_tool_call>"}\n',
            '{"role": "tool", "content": "result"}\n',
            '\n',
            '{"role": "assistant", "content": "done"}\n',
        ])
        result = gen.parse_conversation(filepath)
        assert result is not None
        assert len(result) == 5


class TestFilterQuality:
    def test_good_messages_pass(self):
        messages = [
            {"role": "assistant", "content": "Here is the fix for your code."},
        ]
        assert gen.filter_quality(messages)

    def test_short_content_fails(self):
        messages = [
            {"role": "assistant", "content": "OK"},
        ]
        assert not gen.filter_quality(messages)

    def test_i_cannot_rejection(self):
        messages = [
            {"role": "assistant", "content": "I cannot use the tool because..."},
        ]
        assert not gen.filter_quality(messages)

    def test_im_unable_rejection(self):
        messages = [
            {"role": "assistant", "content": "I'm unable to call that tool."},
        ]
        assert not gen.filter_quality(messages)

    def test_non_assistant_ignored(self):
        messages = [
            {"role": "user", "content": "I cannot do this"},
        ]
        assert gen.filter_quality(messages)

    def test_empty_messages(self):
        assert gen.filter_quality([])


class TestTruncateLongContent:
    def test_no_truncation_needed(self):
        messages = [
            {"role": "assistant", "content": "Short response."},
        ]
        result = gen.truncate_long_content(messages, max_chars=8000)
        assert result[0]["content"] == "Short response."

    def test_truncation_applied(self):
        long_content = "x" * 10000
        messages = [
            {"role": "tool", "content": long_content},
        ]
        result = gen.truncate_long_content(messages, max_chars=8000)
        assert len(result[0]["content"]) < len(long_content)
        assert "[truncated" in result[0]["content"]
        assert result[0]["content"][:4] == "xxxx"

    def test_exact_boundary(self):
        content = "x" * 8000
        messages = [
            {"role": "tool", "content": content},
        ]
        result = gen.truncate_long_content(messages, max_chars=8000)
        assert len(result[0]["content"]) == 8000

    def test_preserves_role(self):
        messages = [
            {"role": "system", "content": "x" * 9000},
            {"role": "user", "content": "y" * 100},
        ]
        result = gen.truncate_long_content(messages, max_chars=8000)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert len(result[0]["content"]) < 9000
        assert len(result[1]["content"]) == 100


class TestFindSessions:
    def test_nonexistent_dir(self):
        result = gen.find_sessions("/nonexistent/path")
        assert result == []

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training = Path(tmpdir) / "training"
            training.mkdir()
            result = gen.find_sessions(str(tmpdir))
            assert isinstance(result, list)
            assert result == []

    def test_finds_conversation_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            training = base / "training"
            training.mkdir()
            (training / "session1.conversation.jsonl").touch()
            (training / "session2.conversation.jsonl").touch()
            (training / "notes.txt").touch()
            result = gen.find_sessions(str(base))
            assert len(result) == 2


def _write_conversation(tmp_path, messages):
    filepath = tmp_path / "session.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return filepath


def _write_conversation_raw(tmp_path, lines):
    filepath = tmp_path / "session.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return filepath
