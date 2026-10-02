"""Run HumanEval assertions in a trusted supervisor with an isolated function worker."""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from verifyit.grade import InvalidTask, Status, run
from verifyit.spec import ScriptSpec, render_spec

from lm_eval.verifyit_function_worker import MAX_BYTES, decode, encode, read, write


SOURCE_HASHES = {
    "7f017b66a2b6fae0261c760f48247634fe74cadb46d3f7734feb5f547069a362",
    "5f77adbf75b4342a6113fea30168ea57b4e5c3877ffd3b4942cb338b3edb2c95",
}

DOCKER = shutil.which("docker")

IMAGE = "python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e"


def enable(config):
    if config.get("output_type") != "generate_until" or config.get("repeats", 1) != 1:
        raise InvalidTask("HumanEval requires one generated response")
    if config.get("doc_to_target") != "{{test}}\ncheck({{entry_point}})":
        raise InvalidTask("HumanEval requires the source test reference")
    definitions = config.get("metric_list")
    if not isinstance(definitions, list) or len(definitions) != 1:
        raise InvalidTask("HumanEval requires its single pass@1 metric")
    definition = definitions[0]
    if (
        set(definition) != {"metric", "aggregation", "higher_is_better", "k"}
        or definition.get("k") != [1]
        or definition.get("aggregation") != "mean"
        or definition.get("higher_is_better") is not True
    ):
        raise InvalidTask("HumanEval metric configuration changed")
    callback = definition["metric"]
    path = (
        Path(callback.__code__.co_filename)
        if callable(callback) and hasattr(callback, "__code__")
        else None
    )
    if (
        path is None
        or callback.__name__ != "pass_at_k"
        or hashlib.sha256(path.read_bytes()).hexdigest() not in SOURCE_HASHES
    ):
        raise InvalidTask("HumanEval source metric changed")
    definition["metric"] = pass_at_k


def entry_point(reference):
    tree = ast.parse(reference)
    last = tree.body[-1] if tree.body else None
    if not (
        isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Call)
        and isinstance(last.value.func, ast.Name)
        and last.value.func.id in ("check", "_verifyit_check")
        and len(last.value.args) == 1
        and isinstance(last.value.args[0], ast.Name)
        and not last.value.keywords
    ):
        raise InvalidTask("HumanEval reference must end with check(entry_point)")
    return last.value.args[0].id


@contextmanager
def candidate_function(
    prediction,
    entry,
    name,
    *,
    image=IMAGE,
    result_observation="identity",
    max_bytes=MAX_BYTES,
    memory_bytes=256 * 1024 * 1024,
    resource_limit_bytes=None,
):
    if DOCKER is None:
        raise RuntimeError("Docker executable is unavailable")
    if memory_bytes is not None and (
        type(memory_bytes) is not int or memory_bytes <= 0
    ):
        raise InvalidTask("Candidate memory limit must be a positive byte count")
    if resource_limit_bytes is not None and (
        type(resource_limit_bytes) is not int or resource_limit_bytes <= 0
    ):
        raise InvalidTask("Candidate resource limit must be a positive byte count")
    worker = Path(__file__).with_name("verifyit_function_worker.py").resolve()
    process = subprocess.Popen(  # noqa: S603 - fixed worker command; candidate source travels over stdin.
        [
            DOCKER,
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--pids-limit",
            "64",
            *(["--memory", str(memory_bytes)] if memory_bytes is not None else []),
            "--cpus",
            "1",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",  # noqa: S108 - private container tmpfs.
            "--mount",
            f"type=bind,source={worker},target=/worker.py,readonly",
            image,
            "python",
            "-I",
            "/worker.py",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        write(
            process.stdin,
            {
                "code": prediction,
                "entry_point": entry,
                "result_observation": result_observation,
                "max_bytes": max_bytes,
                **(
                    {"resource_limit_bytes": resource_limit_bytes}
                    if resource_limit_bytes is not None
                    else {}
                ),
            },
        )
        if read(process.stdout) != {"ready": True}:
            raise ValueError("Candidate worker failed to initialize")

        def candidate(*args, **kwargs):
            write(process.stdin, encode((args, kwargs)), max_bytes)
            return decode(read(process.stdout, max_bytes))

        yield candidate
    finally:
        subprocess.run(  # noqa: S603 - fixed Docker inspection/cleanup command.
            [DOCKER, "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        process.wait(timeout=10)


def score(
    reference,
    prediction,
    name,
    *,
    image=IMAGE,
    result_observation="identity",
    max_bytes=MAX_BYTES,
    memory_bytes=256 * 1024 * 1024,
    resource_limit_bytes=None,
):
    entry = entry_point(reference)
    calls = 0
    try:
        with candidate_function(
            prediction,
            entry,
            name,
            image=image,
            result_observation=result_observation,
            max_bytes=max_bytes,
            memory_bytes=memory_bytes,
            resource_limit_bytes=resource_limit_bytes,
        ) as remote:

            def candidate(*args, **kwargs):
                nonlocal calls
                calls += 1
                return remote(*args, **kwargs)

            try:
                namespace = {entry: candidate}
                # Benchmark-owned source assertions, never candidate code.
                exec(reference, namespace)  # noqa: S102
                if calls == 0:
                    raise InvalidTask("HumanEval reference executed no candidate calls")
                return True
            except (AssertionError, ValueError, BrokenPipeError, EOFError):
                return False
            except (InvalidTask, TimeoutError):
                raise
            except Exception as error:
                if calls:
                    return False
                raise InvalidTask(
                    "Trusted reference failed before calling candidate"
                ) from error
    except (ValueError, BrokenPipeError, EOFError):
        return False


def pass_at_k(references, predictions, k=None):
    if (
        k not in (1, [1])
        or len(references) != 1
        or len(predictions) != 1
        or len(predictions[0]) != 1
    ):
        raise InvalidTask("HumanEval integration requires one sample and pass@1")
    reference, prediction = references[0], predictions[0][0]
    if not isinstance(reference, str) or not isinstance(prediction, str):
        raise InvalidTask("HumanEval requires string reference and candidate")
    entry_point(reference)
    subprocess.run(  # noqa: S603 - fixed Docker inspection/cleanup command.
        [DOCKER, "image", "inspect", IMAGE],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    name = "verifyit-humaneval-" + uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix="verifyit-humaneval-") as directory:
            tests = Path(directory)
            payload = tests / "input.json"
            payload.write_text(
                json.dumps(
                    {"reference": reference, "prediction": prediction, "name": name}
                )
            )
            (tests / "run.sh").write_text('exec "$@"\n')
            spec = ScriptSpec(
                "run.sh",
                args=(sys.executable, "-m", "lm_eval.verifyit_humaneval", str(payload)),
                timeout=30,
                verdict_file="result.json",
            )
            config = tests / "verifier.toml"
            config.write_text(render_spec(spec))
            verdict = run(config, tests)
    finally:
        subprocess.run(  # noqa: S603 - fixed Docker inspection/cleanup command.
            [DOCKER, "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    if verdict.status != Status.SCORED:
        raise RuntimeError(
            f"HumanEval verifier failed: {verdict.status}: {verdict.detail}"
        )
    return {"pass@1": verdict.reward}


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    try:
        passed = score(payload["reference"], payload["prediction"], payload["name"])
        verdict = {
            "status": "scored",
            "reward": float(passed),
            "detail": {"image": IMAGE},
        }
    except InvalidTask as error:
        verdict = {
            "status": "invalid_task",
            "reward": 0,
            "detail": {"error": str(error)},
        }
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "result.json").write_text(
        json.dumps(verdict)
    )


if __name__ == "__main__":
    main()
