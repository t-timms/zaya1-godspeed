from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_system_message():
    return {
        "role": "system",
        "content": "You are Godspeed, a coding agent with tools.",
    }


@pytest.fixture
def sample_user_message():
    return {
        "role": "user",
        "content": "Fix the bug in src/main.py.",
    }


@pytest.fixture
def sample_assistant_no_tools():
    return {
        "role": "assistant",
        "content": "Let me look at the file first.",
    }


@pytest.fixture
def sample_assistant_with_tool():
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "file_read",
                    "arguments": '{"path": "src/main.py"}',
                },
            },
        ],
    }


@pytest.fixture
def sample_assistant_with_thinking():
    return {
        "role": "assistant",
        "content": "",
        "thinking": "I need to read the file first.",
        "tool_calls": [
            {
                "function": {
                    "name": "grep_search",
                    "arguments": '{"pattern": "bug"}',
                },
            },
        ],
    }


@pytest.fixture
def sample_assistant_multi_tool():
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "file_read",
                    "arguments": '{"path": "src/main.py"}',
                },
            },
            {
                "function": {
                    "name": "grep_search",
                    "arguments": '{"pattern": "def main"}',
                },
            },
        ],
    }


@pytest.fixture
def sample_tool_result():
    return {
        "role": "tool",
        "name": "file_read",
        "content": 'def main():\n    print("hello world")',
    }


@pytest.fixture
def sample_tool_result_error():
    return {
        "role": "tool",
        "name": "file_read",
        "content": "File not found",
        "is_error": True,
    }


@pytest.fixture
def sample_session_end_ok():
    return {"role": "session_end", "exit_code": 0}


@pytest.fixture
def sample_session_end_fail():
    return {"role": "session_end", "exit_code": 1}


@pytest.fixture
def sample_meta():
    return {"role": "meta", "key": "value"}


@pytest.fixture
def sample_dangerous():
    return {"role": "meta", "dangerous_command": True}


@pytest.fixture
def full_ok_conversation():
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": "",
            "thinking": "Let me read the file.",
            "tool_calls": [
                {"function": {"name": "file_read", "arguments": '{"path": "src/main.py"}'}},
            ],
        },
        {"role": "tool", "name": "file_read", "content": "print('hello')"},
        {
            "role": "assistant",
            "content": "The file looks fine. Let me check for issues.",
            "thinking": "No obvious bug. Let me search.",
            "tool_calls": [
                {"function": {"name": "grep_search", "arguments": '{"pattern": "TODO"}'}},
            ],
        },
        {"role": "tool", "name": "grep_search", "content": "No matches."},
        {"role": "assistant", "content": "No issues found."},
        {"role": "session_end", "exit_code": 0},
    ]


@pytest.fixture
def base_tasks():
    return [
        {
            "task_id": "task-01",
            "prompt": "Fix the bug in auth.py where users cannot log in.",
            "expected_tools": ["file_read", "grep_search", "file_edit"],
            "expected_tool_sequence": ["file_read", "grep_search", "file_edit"],
            "difficulty": "easy",
            "success_criteria": "Users can log in.",
        },
        {
            "task_id": "task-02",
            "prompt": "Add pagination to the GET /users endpoint in routes.py.",
            "expected_tools": ["file_read", "grep_search", "file_edit", "test_runner"],
            "expected_tool_sequence": ["file_read", "grep_search", "file_edit", "test_runner"],
            "difficulty": "medium",
            "success_criteria": "Pagination works.",
        },
        {
            "task_id": "task-03",
            "prompt": "Implement a circuit breaker for database connections in models.py.",
            "expected_tools": ["file_read", "file_write", "test_runner"],
            "expected_tool_sequence": ["file_read", "file_write", "test_runner"],
            "difficulty": "hard",
            "success_criteria": "Circuit breaker works.",
        },
        {
            "task_id": "task-04",
            "prompt": "Add input validation to all API endpoints in middleware.ts.",
            "expected_tools": ["grep_search", "file_read", "file_edit"],
            "expected_tool_sequence": ["grep_search", "file_read", "file_edit"],
            "difficulty": "medium",
            "success_criteria": "Input validated.",
        },
        {
            "task_id": "task-05",
            "prompt": "Write unit tests for the auth handler in handler/user.go.",
            "expected_tools": ["glob_search", "file_read", "file_write"],
            "expected_tool_sequence": ["glob_search", "file_read", "file_write"],
            "difficulty": "easy",
            "success_criteria": "Tests written.",
        },
    ]


@pytest.fixture
def tmp_conversation_file(full_ok_conversation):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".conversation.jsonl", delete=False, encoding="utf-8") as f:
        for msg in full_ok_conversation:
            f.write(json.dumps(msg) + "\n")
    yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def tmp_conversation_dir(tmp_conversation_file):
    return tmp_conversation_file.parent


@pytest.fixture
def tmp_tasks_file(base_tasks):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for task in base_tasks:
            f.write(json.dumps(task) + "\n")
    yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def tmp_output_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        pass
    p = Path(f.name)
    yield p
    p.unlink(missing_ok=True)
