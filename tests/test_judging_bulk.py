from __future__ import annotations

from persona_vectors.judging import LLMJudgeScorer, _parse_batch_output_text
from persona_vectors.types import JudgeRequest, TraitArtifacts, TraitDefinition


class FakeClient:
    model_name = "fake-judge"

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        if "REFUSAL" in prompt:
            return "REFUSAL"
        return "73"

    def submit_chat_batch(self, *, requests, metadata=None):  # noqa: ANN001
        self._last_requests = requests
        return "batch_123", "/tmp/fake_requests.jsonl"

    def wait_for_batch_output(self, *, batch_id: str, poll_seconds: int = 10, timeout_seconds: int = 3600) -> str:
        return "\n".join(
            [
                '{"custom_id":"r1","response":{"body":{"choices":[{"message":{"content":"42"}}]}}}',
                '{"custom_id":"r2","response":{"body":{"choices":[{"message":{"content":"REFUSAL"}}]}}}',
            ]
        )

    def fetch_batch_output(self, *, batch_id: str) -> str:
        return self.wait_for_batch_output(batch_id=batch_id)


def _artifacts() -> TraitArtifacts:
    return TraitArtifacts(
        trait=TraitDefinition(name="evil", description="bad"),
        instruction_pairs=[],
        extraction_questions=[],
        evaluation_questions=[],
        evaluation_prompt="Q: {question}\nA: {answer}",
    )


def test_score_many_online_preserves_input_order() -> None:
    scorer = LLMJudgeScorer(FakeClient())
    requests = [
        JudgeRequest(request_id="r1", question="q1", answer="a1", prompt="prompt one"),
        JudgeRequest(request_id="r2", question="q2", answer="a2", prompt="prompt two"),
    ]
    results, submission = scorer.score_many(
        requests=requests,
        artifacts=_artifacts(),
        mode="online",
        parallelism=4,
    )
    assert submission is None
    assert results is not None
    assert [result.request_id for result in results] == ["r1", "r2"]
    assert all(result.score == 73.0 for result in results)


def test_parse_batch_output_handles_refusal() -> None:
    payload = "\n".join(
        [
            '{"custom_id":"r1","response":{"body":{"choices":[{"message":{"content":"100"}}]}}}',
            '{"custom_id":"r2","response":{"body":{"choices":[{"message":{"content":"REFUSAL"}}]}}}',
        ]
    )
    parsed = _parse_batch_output_text(output_text=payload, include_logprobs=False)
    assert parsed["r1"].score == 100.0
    assert parsed["r2"].score is None
