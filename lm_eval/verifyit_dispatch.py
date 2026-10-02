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
