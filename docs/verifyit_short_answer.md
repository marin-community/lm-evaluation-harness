# Native NQ-Open and TriviaQA grading

The Marin Evalchemy overrides for `nq_open` and `triviaqa` support the
`evaluate` and `simple_evaluate` Python entry points with
`verifyit_enabled=True`. Install Evalchemy with its `verifyit` extra and this
companion harness in the same environment. The default disabled path uses the source filters and
metrics without importing verifyit.

For an already loaded Evalchemy task dictionary and model:

```python
from lm_eval.evaluator import evaluate

result = evaluate(
    lm=lm,
    task_dict=tasks,
    verifyit_enabled=True,
    verifyit_timeout=120.0,
    log_samples=True,
)
```

The supported task contract requires both `strict_answer` and `extract_answer`
with their pinned Evalchemy extractors followed by `take_first`. Configured filter
ordering is retained; missing, duplicate, or unsupported filters fail explicitly.
Raw completion strings and reference aliases are captured before extraction.
The named `evalchemy_short_answer_v1` policy retains source stop truncation,
last-answer selection, presentation cleanup, and the 256-character final-line
fallback. The first completion is selected. Reserved invalid-extraction output, empty extracted
answers, and missing completions score zero. The existing response-preparation
helper handles the reserved output after extraction, before core normalization. Reference aliases must be nonempty strings and
are validated before missing-candidate handling.

Core text preparation applies the configured regex, case, and punctuation
normalization. NQ retains its article-removal regex; TriviaQA has no article
removal. Existing Exact and MAX primitives compare all aliases. Each sample's
`verifyit_preparation` records input and source hashes, extraction format,
selection policy, and normalization options. It contains no answer text.

`verifyit_timeout` is a finite positive per-record/filter budget, defaulting to
120 seconds. One worker bounds extraction, normalization, and grading together.
Trusted parent capture and serialization consume the same budget synchronously;
they cannot be interrupted by the worker deadline. Late results fail closed.
Malformed tasks raise `InvalidPreparation` with `invalid_task`; nontext provider
responses and deadline failures raise `PreparationError` with `infra_error`.
Both carry a minimum-score verdict. They are distinct from a scored wrong answer.
