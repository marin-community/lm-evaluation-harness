"""HumanEval function calls preserve typed values and pass@1 semantics."""

import json
import os
import sys
import uuid

import pytest
from verifyit.grade import InvalidTask, Status
from verifyit.preparation.errors import PreparationError

from lm_eval import verifyit_humaneval
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
    with pytest.raises(InvalidTask):
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
    monkeypatch.setattr(verifyit_humaneval, "DOCKER", str(tmp_path / "missing-runtime"))
    with pytest.raises(PreparationError) as failure:
        score(REFERENCE, "def solution(*args): return None", "missing-runtime")
    assert failure.value.verdict.status is Status.INFRA_ERROR
    assert failure.value.verdict.reward == 0


def test_blank_message_trusted_setup_failure_is_invalid():
    reference = "raise AssertionError()\ndef check(candidate): pass\ncheck(solution)"
    with pytest.raises(InvalidTask):
        pass_at_k([reference], [["def solution(): return 1"]], [1])


def test_trusted_check_cannot_swallow_failed_candidate_transport():
    reference = "def check(candidate):\n    try: candidate()\n    except Exception: pass\ncheck(solution)"
    prediction = "def solution(): raise ValueError()"
    assert pass_at_k([reference], [[prediction]], [1]) == {"pass@1": 0.0}


def test_humaneval_host_launch_failure_preserves_infrastructure_status(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verifyit_humaneval, "DOCKER", "/unavailable-docker-executable")
    (tmp_path / "input.json").write_text(
        json.dumps(
            {
                "reference": "def check(candidate):\n    assert candidate() == 1\ncheck(f)",
                "prediction": "def f(): return 1",
                "entry": "f",
                "name": "host-launch-test",
                "worker_options": {},
            }
        )
    )
    namespace = {}
    exec(verifyit_humaneval.TRUSTED_TEST, namespace)  # noqa: S102 - Execute the trusted supervisor fixture.
    with pytest.raises(PreparationError) as failure:
        namespace["test_candidate"]()
    assert failure.value.verdict.status is Status.INFRA_ERROR
    assert failure.value.verdict.reward == 0
    metadata = json.loads((tmp_path / "execution.json").read_text())
    assert metadata["preparation_failure"]["status"] == Status.INFRA_ERROR.value
    assert metadata["candidate_calls"] == 0


@pytest.mark.parametrize(
    "scenario,expected_stage",
    [
        ("launch_failure", "candidate_launch"),
        ("never_started", "candidate_launch"),
        ("malformed_state", "candidate_launch"),
        ("candidate_exit125", None),
        ("cleanup_daemon_failure", "candidate_cleanup"),
        ("cleanup_container_remains", "candidate_cleanup"),
        ("already_removed", None),
    ],
)
def test_humaneval_lifecycle_preserves_failure_status_and_reaps_cli(
    scenario,
    expected_stage,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    docker = tmp_path / "docker"
    docker.write_text(
        "#!"
        + sys.executable
        + "\n"
        + f"scenario = {scenario!r}\n"
        + r"""
import json
import os
import sys
from pathlib import Path

if sys.argv[1] == "run":
    Path("owned-pid").write_text(str(os.getpid()))
    sys.stdin.readline()
    if scenario in ("launch_failure", "never_started", "malformed_state", "candidate_exit125"):
        sys.exit(125)
    print('{"ready": true}', flush=True)
    sys.stdin.readline()
    print("1", flush=True)
    sys.stdin.read()
elif sys.argv[1:3] == ["image", "inspect"]:
    sys.exit(0)
elif sys.argv[1:3] == ["container", "inspect"]:
    if scenario == "launch_failure":
        sys.exit(1)
    if scenario == "malformed_state":
        print('{"StartedAt": "not-a-timestamp"}')
        sys.exit(0)
    print(json.dumps({"StartedAt": "0001-01-01T00:00:00Z" if scenario == "never_started" else "2026-10-02T00:00:00Z"}))
elif sys.argv[1] == "rm":
    sys.exit(125 if scenario in ("cleanup_daemon_failure", "cleanup_container_remains", "already_removed") else 0)
elif sys.argv[1:3] == ["container", "ls"]:
    if scenario == "cleanup_daemon_failure":
        sys.exit(125)
    if scenario == "cleanup_container_remains":
        print("lifecycle-owned")
else:
    sys.exit(2)
"""
    )
    docker.chmod(0o700)
    monkeypatch.setattr(verifyit_humaneval, "DOCKER", str(docker))
    (tmp_path / "input.json").write_text(
        json.dumps(
            {
                "reference": "def check(candidate):\n    assert candidate() == 1\ncheck(f)",
                "prediction": "def f(): return 1",
                "entry": "f",
                "name": "lifecycle-owned",
                "worker_options": {},
            }
        )
    )
    namespace = {}
    exec(verifyit_humaneval.TRUSTED_TEST, namespace)  # noqa: S102 - Execute the trusted supervisor fixture.
    if expected_stage:
        with pytest.raises(PreparationError) as failure:
            namespace["test_candidate"]()
        assert failure.value.verdict.status is Status.INFRA_ERROR
        assert failure.value.verdict.reward == 0
    elif scenario == "candidate_exit125":
        with pytest.raises(ValueError):
            namespace["test_candidate"]()
    else:
        namespace["test_candidate"]()
    metadata = json.loads((tmp_path / "execution.json").read_text())
    if expected_stage:
        assert metadata["preparation_failure"]["stage"] == expected_stage
        assert metadata["preparation_failure"]["status"] == Status.INFRA_ERROR.value
    else:
        assert "preparation_failure" not in metadata
    assert metadata["candidate_calls"] == (
        0
        if scenario
        in ("launch_failure", "never_started", "malformed_state", "candidate_exit125")
        else 1
    )
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "owned-pid").read_text()), 0)

    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    reference = "def check(candidate):\n    assert candidate() == 1\ncheck(f)"
    if expected_stage:
        with pytest.raises(PreparationError) as failure:
            verifyit_humaneval.grade_function(
                reference, "def f(): return 1", "lifecycle-owned"
            )
        assert failure.value.verdict.status is Status.INFRA_ERROR
        assert failure.value.verdict.reward == 0
    else:
        verdict = verifyit_humaneval.grade_function(
            reference, "def f(): return 1", "lifecycle-owned"
        )
        assert verdict.status is Status.SCORED
        assert verdict.reward == (0 if scenario == "candidate_exit125" else 1)


def test_image_inspection_and_candidate_share_one_deadline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docker = tmp_path / "docker"
    docker.write_text(
        "#!"
        + sys.executable
        + "\n"
        + f"candidate_pid_path = {str(tmp_path / 'candidate-cli-pid')!r}\n"
        + r"""
import os
import sys
import time
from pathlib import Path

if sys.argv[1:3] == ["image", "inspect"]:
    time.sleep(2)
elif sys.argv[1] == "run":
    Path(candidate_pid_path).write_text(str(os.getpid()))
    sys.stdin.readline()
    time.sleep(2)
    print('{"ready": true}', flush=True)
    sys.stdin.readline()
    print("1", flush=True)
    sys.stdin.read()
elif sys.argv[1] != "rm":
    sys.exit(2)
"""
    )
    docker.chmod(0o700)
    monkeypatch.setattr(verifyit_humaneval, "DOCKER", str(docker))
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    reference = "def check(candidate):\n    assert candidate() == 1\ncheck(f)"
    verdict = verifyit_humaneval.grade_function(
        reference, "def f(): return 1", "shared-deadline", timeout=3
    )
    assert verdict.status is Status.SCORED
    assert verdict.reward == 0
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "candidate-cli-pid").read_text()), 0)
