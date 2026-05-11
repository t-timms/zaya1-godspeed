from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import mutate_tasks as mt


class TestLanguageSwap:
    def test_python_to_typescript(self, base_tasks):
        task = base_tasks[0]
        result = mt._language_swap(task, "python", "typescript")
        assert result["task_id"] == "task-01-lang-typescript"
        assert result["difficulty"] == task["difficulty"]
        assert result is not task

    def test_go_to_rust(self, base_tasks):
        task = base_tasks[4]
        result = mt._language_swap(task, "go", "rust")
        assert result["task_id"] == "task-05-lang-rust"
        assert "main.rs" in result["prompt"] or "go" in result["prompt"].lower()

    def test_does_not_modify_original(self, base_tasks):
        task = base_tasks[0]
        original_id = task["task_id"]
        mt._language_swap(task, "python", "go")
        assert task["task_id"] == original_id


class TestRenameFiles:
    def test_renames_files(self, base_tasks):
        task = base_tasks[0]
        result = mt._rename_files(task)
        assert result["task_id"] == "task-01-rename"
        assert result["prompt"] != task["prompt"] or "auth.py" not in result["prompt"]

    def test_no_rename_if_no_match(self):
        task = {"task_id": "t", "prompt": "do something", "difficulty": "easy"}
        result = mt._rename_files(task)
        assert result["prompt"] == "do something"


class TestAddConstraint:
    def test_adds_constraint(self, base_tasks):
        task = base_tasks[0]
        result = mt._add_constraint(task)
        assert result["task_id"] == "task-01-constraint"
        assert "Constraint:" in result["prompt"]

    def test_upgrades_difficulty(self, base_tasks):
        task = base_tasks[0]
        assert task["difficulty"] == "easy"
        result = mt._add_constraint(task)
        assert result["difficulty"] == "medium"


class TestInjectError:
    def test_adds_error(self, base_tasks):
        task = base_tasks[0]
        result = mt._inject_error(task)
        assert result["task_id"] == "task-01-error-inject"
        assert "known bug" in result["prompt"]


class TestComposeTasks:
    def test_combines_two_tasks(self, base_tasks):
        task1 = base_tasks[0]
        task2 = base_tasks[1]
        result = mt._compose_tasks(task1, task2)
        assert result["task_id"] == "task-01-compose-task-02"
        assert task1["prompt"] in result["prompt"]
        assert task2["prompt"] in result["prompt"]
        assert len(result["expected_tools"]) >= len(task1["expected_tools"])
        assert result["difficulty"] == "medium"

    def test_hard_when_one_hard(self, base_tasks):
        result = mt._compose_tasks(base_tasks[0], base_tasks[2])
        assert result["difficulty"] == "hard"


class TestFrameworkSwap:
    def test_swaps_framework(self):
        task = {"task_id": "t", "prompt": "Refactor the Django app.", "difficulty": "easy"}
        result = mt._framework_swap(task)
        assert result["task_id"] == "t-fw-swap"

    def test_no_swap_if_no_framework(self):
        task = {"task_id": "t", "prompt": "No framework here.", "difficulty": "easy"}
        result = mt._framework_swap(task)
        assert result["prompt"] == "No framework here."


class TestMutateTasks:
    def test_produces_target_count(self, base_tasks):
        result = mt.mutate_tasks(base_tasks, target_count=30, ood_ratio=0.0)
        assert len(result) == 30

    def test_deduplicates_task_ids(self, base_tasks):
        result = mt.mutate_tasks(base_tasks, target_count=50, ood_ratio=0.0)
        ids = [t["task_id"] for t in result]
        assert len(ids) == len(set(ids))

    def test_ood_tasks_included(self, base_tasks):
        result = mt.mutate_tasks(base_tasks, target_count=50, ood_ratio=0.30)
        ood_count = sum(1 for t in result if t.get("is_ood"))
        expected_min = int(50 * 0.30)
        assert ood_count >= expected_min - 10

    def test_all_have_metadata(self, base_tasks):
        result = mt.mutate_tasks(base_tasks, target_count=20, ood_ratio=0.0)
        for task in result:
            assert "task_id" in task
            assert "prompt" in task
            assert "difficulty" in task
            assert "mutation_type" in task

    def test_non_ood_have_parent(self, base_tasks):
        result = mt.mutate_tasks(base_tasks, target_count=20, ood_ratio=0.0)
        for task in result:
            assert "parent_task_id" in task
            assert task["parent_task_id"] is not None

    def test_seed_reproducibility(self, base_tasks):
        result1 = mt.mutate_tasks(base_tasks, target_count=30, ood_ratio=0.0)
        result2 = mt.mutate_tasks(base_tasks, target_count=30, ood_ratio=0.0)
        ids1 = sorted(t["task_id"] for t in result1)
        ids2 = sorted(t["task_id"] for t in result2)
        assert ids1 == ids2

    def test_empty_base_tasks(self):
        result = mt.mutate_tasks([], target_count=10, ood_ratio=0.30)
        assert len(result) == len(mt._OOD_TASKS[:3])

    def test_difficulty_distribution(self, base_tasks):
        result = mt.mutate_tasks(base_tasks, target_count=50, ood_ratio=0.30)
        diffs = {t["difficulty"] for t in result}
        assert len(diffs) >= 2
