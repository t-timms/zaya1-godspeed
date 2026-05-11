"""Expand the 20-task Godspeed benchmark suite into 200+ variant tasks.

Prevents the Shopify failure mode: synthetic data that only covers benchmark
tasks produces a model that scores well offline but fails on real usage.

Mutation categories (from context doc):
  1. Language:  Python -> JS/TS, Go, Rust; add type annotations; swap frameworks
  2. Structure: rename files, reorganize directories, add/remove modules
  3. Composition:  combine two easy tasks into one medium task
  4. Constraints: "without modifying imports", "under N lines", "no new deps"
  5. Error inject: introduce a known bug, require agent to find and fix it
  6. OOD: tasks NOT in the benchmark suite — real repo issues, edge cases

Output: mutated_tasks.jsonl — each line is a task variant with metadata.
Fields: task_id, parent_task_id, mutation_type, prompt, expected_tools,
        expected_tool_sequence, difficulty, success_criteria, is_ood

Usage:
    python scripts/mutate_tasks.py \
        --tasks benchmarks/tasks.jsonl \
        --output data/mutated_tasks.jsonl \
        --count 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from copy import deepcopy
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)

_LANG_MUTATIONS = {
    "python": {
        "from .config import": "import { Config } from './config';",
        "def ": "function ",
        "return": "return",
        "import": "import",
        "class ": "class ",
        "__init__": "constructor",
        "self": "this",
        "None": "null",
        "True": "true",
        "False": "false",
        "async def ": "async function ",
        "    ": "  ",
        "pip install": "npm install",
        "pytest": "jest",
        "ruff": "biome",
        "pyproject.toml": "package.json",
        ".py": ".ts",
        "Python": "TypeScript",
        "python": "typescript",
    },
    "typescript": {
        "app.ts": "main.go",
        "express": "net/http",
        "npm run": "go run",
        "npx tsc": "go build",
        "package.json": "go.mod",
        "middleware.ts": "middleware.go",
        ".ts": ".go",
        "TypeScript": "Go",
        "typescript": "go",
    },
    "go": {
        "main.go": "main.rs",
        "net/http": "actix-web",
        "go build": "cargo build",
        "go.mod": "Cargo.toml",
        ".go": ".rs",
        "Go": "Rust",
        "go": "rust",
        "fmt.Errorf": "anyhow!",
        "json.NewEncoder": "serde_json::to_string",
    },
    "rust": {
        "main.rs": "lib.py",
        "actix-web": "fastapi",
        "cargo build": "uv run pytest",
        "Cargo.toml": "pyproject.toml",
        ".rs": ".py",
        "Rust": "Python",
        "rust": "python",
        "anyhow!": "ValueError",
        "serde_json": "pydantic",
    },
}

_FILE_RENAMES = {
    "auth.py": ["auth_manager.py", "authentication.py", "auth/service.py", "security.py"],
    "models.py": ["models/user.py", "schemas.py", "domain_models.py", "entities.py"],
    "routes.py": ["api/routes.py", "endpoints.py", "views.py", "controllers.py"],
    "utils.py": ["helpers.py", "utils/string_utils.py", "common.py", "lib/utils.py"],
    "config.py": ["settings.py", "config/app_config.py", "environment.py"],
    "main.py": ["app.py", "server.py", "api.py", "application.py"],
    "middleware.ts": ["middleware/auth.ts", "interceptors.ts", "plugins/logger.ts"],
    "index.ts": ["server.ts", "app.ts", "main_server.ts"],
    "handler/user.go": ["handler/user_handler.go", "handler/profile.go", "api/user.go"],
}

_CONSTRAINTS = [
    "without modifying any existing imports",
    "in fewer than 15 lines total",
    "without adding any new dependencies",
    "while keeping all existing tests passing",
    "without using any third-party libraries",
    "using only standard library functions",
    "with type annotations on every new function",
    "without changing the public API surface",
    "in a single file — do not create new files",
    "following the existing error handling pattern exactly",
]

_ERROR_INJECTIONS = [
    ("Missing null check in get_user(): user can be None but not handled.", "medium"),
    ("Off-by-one error in pagination: page_size=0 causes division by zero.", "medium"),
    ("Race condition: cache.get() and cache.set() not atomic under concurrent access.", "hard"),
    ("Memory leak: event listener registered but never removed in componentDidMount.", "hard"),
    ("SQL injection: user_id passed directly to query string, not parameterized.", "high"),
    ("Wrong comparison: 'is' used instead of '==' for string equality check.", "easy"),
    ("Import error: circular import between models.py and routes.py.", "medium"),
    ("Time zone bug: datetime.utcnow() used but should be datetime.now(UTC).", "medium"),
    ("Type error: function returns Optional[str] but caller assumes str.", "medium"),
    ("Regex catastrophic backtracking on long input strings.", "hard"),
]

_OOD_TASKS = [
    {
        "task_id": "ood-db-migration-01",
        "prompt": "Write a database migration that adds a 'last_login' timestamp column to the users table. "
                  "Use alembic conventions. The migration must be reversible.",
        "expected_tools": ["glob_search", "file_read", "file_write", "shell"],
        "expected_tool_sequence": ["glob_search", "file_read", "file_write", "shell"],
        "difficulty": "medium",
        "success_criteria": "Migration file created, runs forward and backward without errors",
    },
    {
        "task_id": "ood-api-pagination-01",
        "prompt": "Add cursor-based pagination to the GET /users endpoint. "
                  "Return a 'next_cursor' field in the response. Handle empty results.",
        "expected_tools": ["grep_search", "file_read", "file_edit", "test_runner"],
        "expected_tool_sequence": ["grep_search", "file_read", "file_edit", "test_runner"],
        "difficulty": "hard",
        "success_criteria": "Pagination implemented with cursor field, tests pass",
    },
    {
        "task_id": "ood-log-sanitize-01",
        "prompt": "Audit all log statements for potential PII leakage. "
                  "Replace any logging of email addresses, IP addresses, or user IDs with hashed versions.",
        "expected_tools": ["grep_search", "file_read", "file_edit", "verify"],
        "expected_tool_sequence": ["grep_search", "file_read", "file_edit", "verify"],
        "difficulty": "medium",
        "success_criteria": "PII removed from log statements, hash function used for identifiers",
    },
    {
        "task_id": "ood-circuit-breaker-01",
        "prompt": "Implement a circuit breaker for the database connection pool. "
                  "After 3 consecutive failures, open the circuit for 30 seconds. "
                  "Log state transitions.",
        "expected_tools": ["file_read", "grep_search", "file_write", "test_runner"],
        "expected_tool_sequence": ["file_read", "grep_search", "file_write", "test_runner"],
        "difficulty": "hard",
        "success_criteria": "Circuit breaker implemented, state transitions logged, tests pass",
    },
    {
        "task_id": "ood-env-validation-01",
        "prompt": "Add startup validation for all required environment variables. "
                  "If DATABASE_URL, JWT_SECRET, or REDIS_URL are missing, fail fast with a clear error message.",
        "expected_tools": ["file_read", "file_edit", "shell", "test_runner"],
        "expected_tool_sequence": ["file_read", "file_edit", "shell", "test_runner"],
        "difficulty": "easy",
        "success_criteria": "Missing env vars cause clear error on startup, all present allows normal start",
    },
    {
        "task_id": "ood-rate-limit-config-01",
        "prompt": "Make the rate limiter configurable via environment variables. "
                  "Add RATE_LIMIT_REQUESTS and RATE_LIMIT_WINDOW_SECONDS env vars with sensible defaults.",
        "expected_tools": ["grep_search", "file_read", "file_edit", "verify", "test_runner"],
        "expected_tool_sequence": ["grep_search", "file_read", "file_edit", "verify", "test_runner"],
        "difficulty": "medium",
        "success_criteria": "Rate limit configurable via env vars, defaults applied, tests pass",
    },
    {
        "task_id": "ood-health-check-01",
        "prompt": "Add a GET /health endpoint that returns 200 if the database is reachable, "
                  "503 if not. Include uptime and version info in the response.",
        "expected_tools": ["file_read", "grep_search", "file_edit", "test_runner"],
        "expected_tool_sequence": ["file_read", "grep_search", "file_edit", "test_runner"],
        "difficulty": "easy",
        "success_criteria": "Health endpoint returns 200 with DB reachable, 503 when not",
    },
    {
        "task_id": "ood-metrics-endpoint-01",
        "prompt": "Add a GET /metrics endpoint that exposes Prometheus-compatible metrics "
                  "for request count, request duration histogram, and error count by endpoint.",
        "expected_tools": ["file_read", "grep_search", "file_write", "file_edit"],
        "expected_tool_sequence": ["file_read", "grep_search", "file_write", "file_edit"],
        "difficulty": "hard",
        "success_criteria": "Metrics endpoint returns valid Prometheus format with all required counters",
    },
    {
        "task_id": "ood-request-id-01",
        "prompt": "Add X-Request-ID header propagation. Generate a UUID if the header is missing. "
                  "Include the request ID in all log messages for the request lifetime.",
        "expected_tools": ["grep_search", "file_read", "file_edit", "file_write"],
        "expected_tool_sequence": ["grep_search", "file_read", "file_edit", "file_write"],
        "difficulty": "medium",
        "success_criteria": "Request ID generated, propagated in headers and logs",
    },
    {
        "task_id": "ood-retry-backoff-01",
        "prompt": "Add exponential backoff with jitter to the HTTP client retry logic. "
                  "Replace the fixed 1-second retry delay with: 1s, 2s, 4s, 8s (max 30s). "
                  "Add +/- 10% random jitter.",
        "expected_tools": ["grep_search", "file_read", "file_edit", "test_runner", "verify"],
        "expected_tool_sequence": ["grep_search", "file_read", "file_edit", "test_runner", "verify"],
        "difficulty": "hard",
        "success_criteria": "Exponential backoff with jitter implemented, tests verify delay intervals",
    },
]

_ADDITIONAL_FRAMEWORKS = {
    "python": ["Django", "Flask", "Litestar", "Sanic"],
    "typescript": ["NestJS", "Koa", "Hono", "Fastify"],
    "go": ["Gin", "Echo", "Fiber", "Chi"],
    "rust": ["Axum", "Rocket", "Warp", "Tide"],
}


def _language_swap(task: dict, lang_from: str, lang_to: str) -> dict:
    """Swap language/framework references in a task."""
    mutations = _LANG_MUTATIONS.get(lang_to, {})
    new_task = deepcopy(task)
    new_task["task_id"] = f"{task['task_id']}-lang-{lang_to}"
    new_task["prompt"] = task["prompt"]
    for old, new in mutations.items():
        new_task["prompt"] = new_task["prompt"].replace(old, new)
    return new_task


def _rename_files(task: dict) -> dict:
    """Rename files referenced in a task prompt."""
    new_task = deepcopy(task)
    new_task["task_id"] = f"{task['task_id']}-rename"
    for old_name, new_names in _FILE_RENAMES.items():
        if old_name in new_task["prompt"]:
            new_task["prompt"] = new_task["prompt"].replace(old_name, random.choice(new_names))
            break
    return new_task


def _add_constraint(task: dict) -> dict:
    """Add a constraint to the task prompt."""
    new_task = deepcopy(task)
    constraint = random.choice(_CONSTRAINTS)
    new_task["task_id"] = f"{task['task_id']}-constraint"
    new_task["prompt"] = f"{task['prompt']}\n\nConstraint: {constraint}."
    if task["difficulty"] == "easy":
        new_task["difficulty"] = "medium"
    return new_task


def _inject_error(task: dict) -> dict:
    """Add a known bug/error description to the task."""
    error, severity = random.choice(_ERROR_INJECTIONS)
    new_task = deepcopy(task)
    new_task["task_id"] = f"{task['task_id']}-error-inject"
    new_task["prompt"] = f"{task['prompt']}\n\nAdditionally, there's a known bug: {error}"
    if severity == "hard" and task["difficulty"] != "hard":
        new_task["difficulty"] = "hard"
    return new_task


def _compose_tasks(task1: dict, task2: dict) -> dict:
    """Combine two tasks into one compound task."""
    new_task = deepcopy(task1)
    new_task["task_id"] = f"{task1['task_id']}-compose-{task2['task_id']}"
    new_task["prompt"] = f"{task1['prompt']}\n\nAlso: {task2['prompt']}"
    new_task["expected_tools"] = list(set(task1.get("expected_tools", []) + task2.get("expected_tools", [])))
    new_task["expected_tool_sequence"] = (
        task1.get("expected_tool_sequence", []) + task2.get("expected_tool_sequence", [])
    )
    new_task["difficulty"] = "hard" if task1["difficulty"] == "hard" or task2["difficulty"] == "hard" else "medium"
    new_task["success_criteria"] = f"{task1.get('success_criteria', '')} AND {task2.get('success_criteria', '')}"
    return new_task


def _framework_swap(task: dict) -> dict:
    """Swap the framework mentioned in a task."""
    new_task = deepcopy(task)
    new_task["task_id"] = f"{task['task_id']}-fw-swap"
    for lang, frameworks in _ADDITIONAL_FRAMEWORKS.items():
        for fw in frameworks:
            if fw.lower() in task["prompt"].lower():
                new_fw = random.choice([f for f in frameworks if f != fw])
                new_task["prompt"] = task["prompt"].replace(fw, new_fw)
                return new_task
    return new_task


def mutate_tasks(
    base_tasks: list[dict],
    target_count: int,
    ood_ratio: float = 0.30,
) -> list[dict]:
    """Generate mutated task variants from the base benchmark suite.

    Args:
        base_tasks: Original 20 tasks from benchmarks/tasks.jsonl.
        target_count: Desired number of total variant tasks.
        ood_ratio: Fraction of tasks that should be OOD (not in benchmark).

    Returns:
        List of mutated task dicts.
    """
    mutations: list[dict] = []
    mutators = [
        _language_swap,
        _rename_files,
        _add_constraint,
        _inject_error,
        _framework_swap,
    ]
    languages = ["python", "typescript", "go", "rust"]

    random.seed(42)

    ood_target = int(target_count * ood_ratio)
    variant_target = target_count - ood_target

    if not base_tasks:
        variant_target = 0

    while len(mutations) < variant_target:
        task = random.choice(base_tasks)

        mutator = random.choice(mutators)
        if random.random() < 0.15:
            t2 = random.choice(base_tasks)
            if t2["task_id"] != task["task_id"]:
                mutated = _compose_tasks(task, t2)
                mutator = _compose_tasks
            else:
                if mutator == _language_swap:
                    mutated = mutator(task, "python", random.choice(languages))
                else:
                    mutated = mutator(task)
        elif mutator == _language_swap:
            mutated = mutator(task, "python", random.choice(languages))
        else:
            mutated = mutator(task)

        if mutated["task_id"] not in {m["task_id"] for m in mutations}:
            mutated["parent_task_id"] = task["task_id"]
            mutated["mutation_type"] = mutator.__name__.lstrip("_")
            mutated["is_ood"] = False
            mutations.append(mutated)

    # Add OOD tasks
    for ood_task in _OOD_TASKS[:ood_target]:
        ood_task["parent_task_id"] = None
        ood_task["mutation_type"] = "ood"
        ood_task["is_ood"] = True
        mutations.append(ood_task)

    random.shuffle(mutations)
    return mutations[:target_count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand Godspeed benchmark tasks into 200+ variants"
    )
    parser.add_argument("--tasks", default="benchmarks/tasks.jsonl")
    parser.add_argument("--output", default="data/mutated_tasks.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--ood-ratio", type=float, default=0.30, help="Fraction of OOD tasks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        tasks_path = (
            Path("C:/Users/ttimm/Documents/Project Portfolio/godspeed-coding-agent")
            / args.tasks
        )
    if not tasks_path.exists():
        logger.error("Tasks file not found: %s", args.tasks)
        raise SystemExit(1)

    base_tasks: list[dict] = []
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                base_tasks.append(json.loads(line))

    logger.info("Loaded %d base tasks from %s", len(base_tasks), tasks_path)

    mutated = mutate_tasks(base_tasks, args.count, args.ood_ratio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task in mutated:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    ood_count = sum(1 for t in mutated if t.get("is_ood"))
    diff_counts = {
        "easy": sum(1 for t in mutated if t["difficulty"] == "easy"),
        "medium": sum(1 for t in mutated if t["difficulty"] == "medium"),
        "hard": sum(1 for t in mutated if t["difficulty"] == "hard"),
    }

    logger.info(
        "Generated %d tasks (%d OOD, %s) -> %s",
        len(mutated),
        ood_count,
        ", ".join(f"{k}: {v}" for k, v in diff_counts.items()),
        output_path,
    )


if __name__ == "__main__":
    main()
