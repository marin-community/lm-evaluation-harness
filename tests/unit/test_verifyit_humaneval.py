"""HumanEval function calls preserve typed values and pass@1 semantics."""

import uuid

import pytest
from verifyit.grade import InvalidTask

from lm_eval.verifyit_humaneval import pass_at_k, score


REFERENCE = """
def check(candidate):
    assert candidate((1, 2), {"value": 3}) == (3, 3)
    assert candidate((4, 5), {"value": 6}) == (9, 6)
check(solution)
"""


@pytest.mark.parametrize(
    "body,expected",
    [
        ("return (sum(values), mapping['value'])", 1.0),
        ("return None", 0.0),
        ("print(values); return (sum(values), mapping['value'])", 1.0),
    ],
)
def test_typed_function_results(body, expected):
    code = f"def solution(values, mapping):\n    {body}\n"
    assert pass_at_k([REFERENCE], [[code]], [1]) == {"pass@1": expected}


def test_unsupported_pass_at_k_configuration_aborts():
    with pytest.raises(InvalidTask, match="pass@1"):
        pass_at_k([REFERENCE], [["def solution(): pass"]], [2])


def test_wrong_return_type_scores_zero():
    reference = (
        "def check(candidate):\n    assert candidate()[0] == 1\ncheck(solution)\n"
    )
    assert pass_at_k([reference], [["def solution(): return None"]], [1]) == {
        "pass@1": 0.0
    }


def test_default_memory_budget_still_rejects_oversized_candidate():
    size = 600 * 1024 * 1024
    reference = (
        f"def check(candidate):\n    assert candidate() == {size}\ncheck(solution)\n"
    )
    prediction = f"def solution():\n    value = bytearray({size})\n    for i in range(0, len(value), 4096): value[i] = 1\n    return len(value)\n"
    assert not score(
        reference, prediction, "verifyit-memory-default-" + uuid.uuid4().hex
    )


def test_missing_runtime_executable_remains_an_infrastructure_error(
    tmp_path, monkeypatch
):
    from lm_eval import verifyit_humaneval

    monkeypatch.setattr(verifyit_humaneval, "DOCKER", str(tmp_path / "missing-runtime"))
    with pytest.raises(FileNotFoundError):
        score(REFERENCE, "def solution(*args): return None", "missing-runtime")
