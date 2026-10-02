"""Prepare recognized source response formats for opt-in native verification."""

import ast
import inspect
from pathlib import Path


def prepare_responses(task, doc, responses, filter_name):
    """Decode the reserved NQ/Trivia invalid-extraction value without grading it."""
    from verifyit.spec import EmptyOutputPolicy

    name = task.config.task
    profiles = {
        "nq_open": ("google-research-datasets/nq_open", None, "{{answer}}"),
        "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "{{answer.aliases}}"),
    }
    if name not in profiles:
        return responses, EmptyOutputPolicy.GRADE

    from verifyit.adapters.harness_validation import (
        compiled_source_functions,
        pinned_source_bytes,
        source_functions_match,
    )
    from verifyit.grade import InvalidTask

    if (
        task.config.dataset_path,
        task.config.dataset_name,
        task.config.doc_to_target,
    ) != profiles[name]:
        raise InvalidTask("short-answer task target contract changed")
    references = doc.get("answer")
    if name == "triviaqa":
        references = references.get("aliases") if isinstance(references, dict) else None
    if (
        not isinstance(references, list)
        or not references
        or any(
            not isinstance(reference, str) or not reference.strip()
            for reference in references
        )
    ):
        raise InvalidTask(
            "short-answer task requires nonempty textual reference aliases"
        )

    functions = {
        "strict_answer": "marked_short_answer_filter",
        "extract_answer": "short_answer_filter",
    }
    expected = functions.get(filter_name)
    definitions = task.config.filter_list
    selected = [item for item in definitions or [] if item.get("name") == filter_name]
    if expected is None or len(selected) != 1:
        raise InvalidTask("short-answer task filter contract changed")
    pipeline = selected[0].get("filter", [])
    if (
        len(pipeline) != 2
        or pipeline[0].get("function") != "custom"
        or pipeline[1] != {"function": "take_first"}
    ):
        raise InvalidTask(
            "short-answer task requires source extraction then take_first"
        )
    function = pipeline[0].get("filter_fn")
    if (
        not inspect.isfunction(function)
        or function.__module__ != "eval.lm_eval_tasks.short_answer_extraction"
        or function.__name__ != expected
    ):
        raise InvalidTask("short-answer task requires its pinned extraction function")
    path = Path(function.__code__.co_filename)
    source = pinned_source_bytes(
        path, {"ea7e332b93d9ce71501649685f423d68a653a2ad402893be08f3432a23c2a262"}
    )
    if source is None:
        raise InvalidTask("short-answer extraction source changed")
    names = {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    codes = [
        code
        for code in compiled_source_functions(source, str(path))
        if code.co_name in names
    ]
    namespace = function.__globals__
    if (
        not source_functions_match(namespace, codes, {})
        or namespace.get("INVALID_SHORT_ANSWER") != "[invalid]"
    ):
        raise InvalidTask("short-answer extraction function or reserved value changed")
    return (
        ["" if response == "[invalid]" else response for response in responses],
        EmptyOutputPolicy.ZERO,
    )


def execution_metrics(task, doc, responses, filter_name):
    """Run recognized execution contracts through core modes."""
    if task.config.task != "humaneval":
        return None
    from verifyit.adapters.harness_validation import (
        compiled_source_functions,
        pinned_source_bytes,
        source_functions_match,
    )
    from verifyit.grade import InvalidTask

    from lm_eval.verifyit_humaneval import SOURCE_HASHES, enable, pass_at_k

    config = dict(vars(task.config))
    config["metric_list"] = [dict(item) for item in config["metric_list"]]
    callback = config["metric_list"][0]["metric"]
    enable(config)
    if (
        task.config.dataset_path != "openai/openai_humaneval"
        or task.config.doc_to_text != "{{prompt}}"
        or task.config.process_results is not None
        or filter_name != "create_test"
        or len(responses) != 1
    ):
        raise InvalidTask("HumanEval task or response contract changed")
    definitions = task.config.filter_list
    if not isinstance(definitions, list) or len(definitions) != 1:
        raise InvalidTask("HumanEval requires its source prediction builder")
    pipeline = definitions[0].get("filter", [])
    if (
        definitions[0].get("name") != "create_test"
        or len(pipeline) != 1
        or pipeline[0].get("function") != "custom"
    ):
        raise InvalidTask("HumanEval requires its source prediction builder")
    builder = pipeline[0].get("filter_fn")
    path = Path(callback.__code__.co_filename)
    source = pinned_source_bytes(path, SOURCE_HASHES)
    namespace = callback.__globals__
    if (
        source is None
        or namespace.get("pass_at_k") is not callback
        or namespace.get("build_predictions") is not builder
        or not source_functions_match(
            namespace,
            compiled_source_functions(source, str(path)),
            {"pass_at_k": (None,)},
        )
    ):
        raise InvalidTask("HumanEval source callable graph changed")
    truncate = namespace.get("truncate_at_stop")
    if not inspect.isfunction(truncate):
        raise InvalidTask("HumanEval stop preparation changed")
    stop_source = pinned_source_bytes(
        Path(truncate.__code__.co_filename),
        {"dfc35a741f1e7a6489f7088dfe119de106045fce4d7bed9a33d04eebf1a75eca"},
    )
    if stop_source is None:
        raise InvalidTask("HumanEval stop preparation source changed")
    expected_stops = {}
    exec(compile(stop_source, truncate.__code__.co_filename, "exec"), expected_stops)  # noqa: S102 - hash-pinned data preparation only.
    expected_truncate = expected_stops["truncate_at_stop"]
    if (
        truncate.__code__ != expected_truncate.__code__
        or truncate.__defaults__ != expected_truncate.__defaults__
        or namespace.get("HUMANEVAL_STOP_SEQUENCES")
        != expected_stops["HUMANEVAL_STOP_SEQUENCES"]
    ):
        raise InvalidTask("HumanEval stop preparation contract changed")
    metrics = pass_at_k([task.doc_to_target(doc)], responses, [1])
    task.aggregation().setdefault("pass@1", task.aggregation()["pass_at_k"])
    task.higher_is_better().setdefault("pass@1", task.higher_is_better()["pass_at_k"])
    return metrics
