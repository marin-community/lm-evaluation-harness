"""Short-answer cutover retains raw evidence and source filter behavior."""

from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict
from verifyit.grade import Status
from verifyit.preparation.errors import InvalidPreparation, PreparationError

from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.verifyit_short_answer import capture_short_answer, short_answer_metrics


source = pytest.importorskip("eval.lm_eval_tasks.short_answer_extraction")


class LocalTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = DatasetDict(
            validation=Dataset.from_list(
                [{"question": "capital?", "answer": ["Paris", "PARIS"]}]
            )
        )


class LocalLM(LM):
    def __init__(self, responses):
        super().__init__()
        self.responses = responses

    def generate_until(self, requests):
        return self.responses

    def loglikelihood(self, requests):
        raise AssertionError("unexpected likelihood request")

    def loglikelihood_rolling(self, requests):
        raise AssertionError("unexpected rolling request")


@pytest.fixture
def task():
    result = LocalTask(
        config={
            "task": "nq_open",
            "dataset_path": "google-research-datasets/nq_open",
            "output_type": "generate_until",
            "validation_split": "validation",
            "doc_to_text": "{{question}}",
            "doc_to_target": "{{answer}}",
            "generation_kwargs": {
                "until": list(source.truncate_at_stop.__defaults__[0]),
                "do_sample": False,
            },
            "filter_list": [
                {
                    "name": name,
                    "filter": [
                        {"function": "custom", "filter_fn": function},
                        {"function": "take_first"},
                    ],
                }
                for name, function in [
                    ("strict_answer", source.marked_short_answer_filter),
                    ("extract_answer", source.short_answer_filter),
                ]
            ],
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "ignore_case": True,
                    "ignore_punctuation": True,
                }
            ],
        }
    )
    result.set_fewshot_seed(0)
    return result


def test_raw_capture_retains_all_completions_and_aliases():
    doc = {"answer": ["Paris", "PARIS"]}
    request = SimpleNamespace(resps=["Answer: wrong", "Answer: Paris"])
    captured = capture_short_answer(doc, [request], "nq_open")
    request.resps.reverse()
    doc["answer"].clear()
    assert captured.responses == (("Answer: wrong", "Answer: Paris"),)
    assert captured.references == ("Paris", "PARIS")


@pytest.mark.parametrize(
    "response,strict,extracted",
    [
        ("Answer: wrong\nFinal answer: Paris", 1.0, 1.0),
        ("The answer is Paris", 0.0, 1.0),
        ("Answer: Paris\nQuestion: other\nAnswer: wrong", 1.0, 1.0),
        ("[invalid]", 0.0, 0.0),
        ("", 0.0, 0.0),
    ],
)
def test_evaluator_preserves_both_source_filter_results(
    task, response, strict, extracted
):
    result = evaluate(
        LocalLM([response]), {"nq_open": task}, bootstrap_iters=0, verifyit_enabled=True
    )
    assert result["results"]["nq_open"]["exact_match,strict_answer"] == strict
    assert result["results"]["nq_open"]["exact_match,extract_answer"] == extracted
    for sample in result["samples"]["nq_open"]:
        assert sample["verifyit_preparation"]["raw_sha256"]
        assert sample["resps"] == [[response]]


def test_first_completion_is_used_even_when_later_completion_is_correct(task):
    task.config.repeats = 2
    result = evaluate(
        LocalLM(["Answer: wrong", "Answer: Paris"]),
        {"nq_open": task},
        bootstrap_iters=0,
        verifyit_enabled=True,
    )
    assert result["results"]["nq_open"]["exact_match,extract_answer"] == 0.0


@pytest.mark.parametrize("raw", [[], [""]])
def test_invalid_aliases_fail_before_missing_completion(task, raw):
    request = SimpleNamespace(resps=raw, filtered_resps={})
    with pytest.raises(InvalidPreparation) as caught:
        short_answer_metrics(task, {"answer": []}, [request], "extract_answer", 120)
    assert caught.value.failure.status == Status.INVALID_TASK
    assert caught.value.verdict.reward == 0.0


def test_missing_completion_scores_zero_and_has_missing_provenance(task):
    request = SimpleNamespace(resps=[], filtered_resps={})
    metrics, provenance = short_answer_metrics(
        task, {"answer": ["Paris"]}, [request], "extract_answer", 120
    )
    assert metrics == {"exact_match": 0.0}
    assert provenance["format"] == "missing"


def test_worker_deadline_is_infrastructure_failure_at_minimum_score(task):
    request = SimpleNamespace(resps=["Answer: Paris"], filtered_resps={})
    with pytest.raises(PreparationError) as caught:
        short_answer_metrics(
            task, {"answer": ["Paris"]}, [request], "extract_answer", 0.01
        )
    assert caught.value.failure.status == Status.INFRA_ERROR
    assert caught.value.verdict.reward == 0.0


def test_nontext_provider_response_is_infrastructure_failure_after_alias_validation(
    task,
):
    request = SimpleNamespace(resps=[None], filtered_resps={})
    with pytest.raises(InvalidPreparation) as invalid:
        short_answer_metrics(task, {"answer": []}, [request], "extract_answer", 120)
    assert invalid.value.verdict.status == Status.INVALID_TASK
    with pytest.raises(PreparationError) as infrastructure:
        short_answer_metrics(
            task, {"answer": ["Paris"]}, [request], "extract_answer", 120
        )
    assert infrastructure.value.failure.stage == "structure"
    assert infrastructure.value.verdict.status == Status.INFRA_ERROR
    assert infrastructure.value.verdict.reward == 0.0


@pytest.mark.parametrize(
    "filters", [None, [{}], [{"name": "strict_answer"}, {"name": "strict_answer"}]]
)
def test_evaluator_invalid_filter_cohort_fails_with_typed_minimum_verdict(
    task, filters
):
    task.config.filter_list = filters
    with pytest.raises(InvalidPreparation) as caught:
        evaluate(
            LocalLM(["Answer: Paris"]),
            {"nq_open": task},
            bootstrap_iters=0,
            verifyit_enabled=True,
        )
    assert caught.value.failure.stage == "filter_contract"
    assert caught.value.verdict.status == Status.INVALID_TASK
    assert caught.value.verdict.reward == 0.0


def test_opaque_provider_list_is_rejected_before_custom_iteration(task):
    class OpaqueList(list):
        def __iter__(self):
            raise AssertionError("provider iteration must not run in parent")

    request = SimpleNamespace(resps=OpaqueList(["Answer: Paris"]), filtered_resps={})
    with pytest.raises(PreparationError) as caught:
        short_answer_metrics(
            task, {"answer": ["Paris"]}, [request], "extract_answer", 120
        )
    assert caught.value.verdict.status == Status.INFRA_ERROR
    assert caught.value.verdict.reward == 0.0


@pytest.mark.parametrize(
    "pipeline",
    [
        None,
        [None, {"function": "take_first"}],
        [
            {
                "function": "custom",
                "filter_fn": source.short_answer_filter,
                "extra": True,
            },
            {"function": "take_first"},
        ],
    ],
)
def test_evaluator_malformed_filter_pipeline_has_typed_minimum_verdict(task, pipeline):
    task.config.filter_list[1]["filter"] = pipeline
    with pytest.raises(InvalidPreparation) as caught:
        evaluate(
            LocalLM(["Answer: Paris"]),
            {"nq_open": task},
            bootstrap_iters=0,
            verifyit_enabled=True,
        )
    assert caught.value.verdict.status == Status.INVALID_TASK
    assert caught.value.verdict.reward == 0.0
