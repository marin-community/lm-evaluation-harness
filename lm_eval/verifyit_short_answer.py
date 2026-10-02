"""Client-owned raw short-answer capture and source-policy preparation."""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class ShortAnswerInputs:
    responses: tuple[tuple[str, ...], ...]
    references: tuple[str, ...]


def short_answer_filter_keys(task):
    """Validate the supported configured cohort and preserve its filter ordering."""
    from harbor_config.errors import error_category
    from verifyit.grade import Status, finalize_preparation_failure
    from verifyit.preparation.errors import InvalidPreparation, PreparationFailure

    definitions = task.config.filter_list
    if (
        type(definitions) is not list
        or len(definitions) != 2
        or any(
            type(item) is not dict or type(item.get("name")) is not str
            for item in definitions
        )
        or {item["name"] for item in definitions} != {"strict_answer", "extract_answer"}
        or any(
            type(item.get("filter")) is not list
            or len(item["filter"]) != 2
            or type(item["filter"][0]) is not dict
            or set(item["filter"][0]) != {"function", "filter_fn"}
            or item["filter"][0]["function"] != "custom"
            or type(item["filter"][1]) is not dict
            or item["filter"][1] != {"function": "take_first"}
            for item in definitions
        )
    ):
        failure = PreparationFailure(
            Status.INVALID_TASK,
            error_category("InvalidTask"),
            "InvalidTask",
            "short-answer task requires strict_answer and extract_answer filters",
            "filter_contract",
        )
        raise InvalidPreparation(
            failure, finalize_preparation_failure(**asdict(failure))
        )
    return tuple(item["name"] for item in definitions)


def capture_short_answer(doc, requests, task_name):
    """Snapshot raw completions and reference aliases without answer selection."""
    from harbor_config.errors import error_category
    from verifyit.grade import InvalidTask, Status, finalize_preparation_failure
    from verifyit.preparation.errors import PreparationError, PreparationFailure

    if type(doc) is not dict:
        raise InvalidTask("short-answer task record must be a plain mapping")
    references = doc.get("answer")
    if task_name == "triviaqa":
        references = references.get("aliases") if type(references) is dict else None
    if (
        type(references) is not list
        or not references
        or any(type(value) is not str or not value.strip() for value in references)
    ):
        raise InvalidTask(
            "short-answer task requires nonempty textual reference aliases"
        )
    if len(requests) != 1:
        raise InvalidTask("short-answer task requires one generation request")
    if any(type(request.resps) is not list for request in requests):
        failure = PreparationFailure(
            Status.INFRA_ERROR,
            error_category("TypeError"),
            "TypeError",
            "short-answer raw completions must be response lists",
            "structure",
        )
        raise PreparationError(failure, finalize_preparation_failure(**asdict(failure)))
    responses = tuple(tuple(request.resps) for request in requests)
    if any(type(value) is not str for batch in responses for value in batch):
        failure = PreparationFailure(
            Status.INFRA_ERROR,
            error_category("TypeError"),
            "TypeError",
            "short-answer provider completion must be text",
            "structure",
        )
        raise PreparationError(failure, finalize_preparation_failure(**asdict(failure)))
    return ShortAnswerInputs(responses, tuple(references))


def prepare_short_answer(inputs, config, filter_name):
    """Apply the named source extraction and first-completion policy."""
    from eval.lm_eval_tasks import short_answer_extraction as source
    from verifyit.adapters.harness_validation import pinned_source_bytes
    from verifyit.grade import InvalidTask

    from lm_eval.verifyit_dispatch import prepare_responses

    doc = {"answer": list(inputs.references)}
    if config.task == "triviaqa":
        doc = {"answer": {"aliases": list(inputs.references)}}
    prepare_responses(SimpleNamespace(config=config), doc, [], filter_name)
    # Source extraction calls this helper with its default stop tuple. Validate
    # the actual callable and defaults, rather than only its imported name.
    path = Path(source.truncate_at_stop.__code__.co_filename)
    stop_source = pinned_source_bytes(
        path, {"dfc35a741f1e7a6489f7088dfe119de106045fce4d7bed9a33d04eebf1a75eca"}
    )
    if stop_source is None:
        raise InvalidTask("short-answer stop source changed")
    namespace = {}
    exec(  # noqa: S102 - immutable hash-pinned source policy.
        compile(stop_source, str(path), "exec"), namespace
    )
    expected = namespace["truncate_at_stop"]
    actual = source.truncate_at_stop
    if (
        actual.__code__ != expected.__code__
        or actual.__defaults__ != expected.__defaults__
        or config.generation_kwargs.get("until")
        != namespace["SHORT_ANSWER_STOP_SEQUENCES"]
    ):
        raise InvalidTask("short-answer stop contract changed")
    batch = inputs.responses[0]
    if batch:
        if filter_name == "strict_answer":
            candidate = source.extract_marked_short_answer(batch[0])
            answer_format = (
                "contract" if candidate != source.INVALID_SHORT_ANSWER else "invalid"
            )
        else:
            extraction = source.extract_short_answer(batch[0])
            candidate, answer_format = extraction.answer, extraction.format.value
    else:
        candidate, answer_format = "", "missing"
    prepared, empty_output = prepare_responses(
        SimpleNamespace(config=config), doc, [candidate], filter_name
    )
    candidate = prepared[0]
    provenance = {
        "policy": "evalchemy_short_answer_v1",
        "filter": filter_name,
        "format": answer_format,
        "selection": "take_first",
        "reserved_value": "[invalid] -> empty",
        "stop_policy": "evalchemy_end_of_turn_v1",
        "stop_source_sha256": hashlib.sha256(stop_source).hexdigest(),
        "extractor_source_sha256": "ea7e332b93d9ce71501649685f423d68a653a2ad402893be08f3432a23c2a262",
        "raw_sha256": hashlib.sha256(
            json.dumps(asdict(inputs), ensure_ascii=False).encode()
        ).hexdigest(),
    }
    return candidate, empty_output, provenance


def short_answer_verdict(inputs, config, filter_name, options):
    """Compose client preparation with core normalization, Exact and MAX."""
    from verifyit.adapters.harness_native import exact_match

    candidate, empty_output, provenance = prepare_short_answer(
        inputs, config, filter_name
    )
    verdict = exact_match(
        candidate, inputs.references, empty_output=empty_output, **options
    )
    return candidate, replace(
        verdict, detail={**verdict.detail, "source_preparation": provenance}
    )


def short_answer_metrics(task, doc, requests, filter_name, timeout):
    """Account trusted capture and serialization in the record/filter deadline.

    Extraction, normalization and grading share one hard worker deadline. Trusted
    parent capture and serialization are synchronous; reject late successes.
    """
    from harbor_config.errors import error_category
    from verifyit.adapters.harness_native import native_config_route
    from verifyit.execution.worker import call_bounded
    from verifyit.grade import InvalidTask, Status, finalize_preparation_failure
    from verifyit.preparation.errors import (
        InvalidPreparation,
        PreparationError,
        PreparationFailure,
    )

    from lm_eval.verifyit_dispatch import prepare_responses

    started = time.monotonic()
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("verifyit_timeout must be finite and positive")
    try:
        short_answer_filter_keys(task)
        inputs = capture_short_answer(doc, requests, task.config.task)
        prepare_responses(task, doc, [], filter_name)
        method = task.process_results
        metric = task._metric_fn_list.get("exact_match")
        options = task._metric_fn_kwargs.get("exact_match", {})
        metric_config = [
            {"metric": name, **task._metric_fn_kwargs.get(name, {})}
            for name in task._metric_fn_list
        ]
        if (
            getattr(method, "__qualname__", None) != "ConfigurableTask.process_results"
            or getattr(method, "__module__", None) != "lm_eval.api.task"
            or getattr(metric, "__module__", None) != "lm_eval.api.metrics"
            or getattr(metric, "__name__", None) != "exact_match_fn"
            or task.config.process_results is not None
            or native_config_route(
                {"output_type": task.OUTPUT_TYPE, "metric_list": metric_config}
            )
            != "exact_match"
        ):
            raise InvalidTask("short-answer native metric contract changed")
        config = SimpleNamespace(
            **{
                name: getattr(task.config, name)
                for name in (
                    "task",
                    "dataset_path",
                    "dataset_name",
                    "doc_to_target",
                    "filter_list",
                    "generation_kwargs",
                )
            }
        )
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("short-answer record exceeded its deadline")
        candidate, verdict = call_bounded(
            short_answer_verdict,
            inputs,
            config,
            filter_name,
            options,
            timeout=remaining,
        )
        if time.monotonic() - started >= timeout:
            raise TimeoutError("short-answer record exceeded its deadline")
    except InvalidTask as error:
        if isinstance(error, PreparationError):
            raise
        failure = PreparationFailure(
            Status.INVALID_TASK,
            error_category(type(error).__name__),
            type(error).__name__,
            str(error),
            "short_answer",
        )
        raise InvalidPreparation(
            failure, finalize_preparation_failure(**asdict(failure))
        ) from error
    except (TimeoutError, RuntimeError, OSError) as error:
        if isinstance(error, PreparationError):
            raise
        failure = PreparationFailure(
            Status.INFRA_ERROR,
            error_category(type(error).__name__),
            type(error).__name__,
            str(error),
            "short_answer",
        )
        raise PreparationError(
            failure, finalize_preparation_failure(**asdict(failure))
        ) from error
    requests[0].filtered_resps[filter_name] = candidate
    return {"exact_match": verdict.reward}, {
        **verdict.detail["source_preparation"],
        "normalization": verdict.detail["preparation"],
    }
