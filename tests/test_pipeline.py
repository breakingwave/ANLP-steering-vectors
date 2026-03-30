from __future__ import annotations

from contextlib import nullcontext

from persona_vectors.pipeline import ExtractionPipeline
from persona_vectors.types import (
    ExtractorConfig,
    GeneratedSample,
    InstructionPair,
    JudgeRequest,
    JudgeResult,
    TraitArtifacts,
    TraitDefinition,
)


class FakeBackend:
    model_name = "fake-target"
    num_layers = 2

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> GeneratedSample:
        response_text = "positive response" if "show" in system_prompt else "negative response"
        return GeneratedSample(prompt_token_ids=[1, 2, 3], response_token_ids=[4, 5], response_text=response_text)

    def response_layer_means(self, *, prompt_token_ids: list[int], response_token_ids: list[int]) -> list[list[float]]:
        if response_token_ids != [4, 5]:
            raise AssertionError("unexpected response ids")
        return [[1.0, 1.0], [3.0, 0.0]] if len(prompt_token_ids) == 3 else [[0.0, 0.0], [1.0, 1.0]]

    def steering_scope(self, *, layer_index: int, vector: list[float], alpha: float):
        return nullcontext()


class BackendWithSignedActivations(FakeBackend):
    def response_layer_means(self, *, prompt_token_ids: list[int], response_token_ids: list[int]) -> list[list[float]]:
        if prompt_token_ids != [1, 2, 3]:
            raise AssertionError("unexpected prompt ids")
        return [[2.0, 0.0], [4.0, 0.0]] if response_token_ids == [4, 5] else [[0.0, 1.0], [1.0, 2.0]]

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> GeneratedSample:
        if "show" in system_prompt:
            return GeneratedSample(prompt_token_ids=[1, 2, 3], response_token_ids=[4, 5], response_text="positive response")
        return GeneratedSample(prompt_token_ids=[1, 2, 3], response_token_ids=[6, 7], response_text="negative response")


class FakeScorer:
    def score(self, *, question: str, answer: str, artifacts: TraitArtifacts) -> float | None:
        return 90.0 if "positive" in answer else 10.0

    def score_many(
        self,
        *,
        requests: list[JudgeRequest],
        artifacts: TraitArtifacts,
        mode: str = "online",
        parallelism: int = 1,
        batch_behavior: str = "blocking_poll",
        batch_id: str | None = None,
        batch_output_path: str | None = None,
        artifact_dir: str | None = None,
        poll_seconds: int = 10,
        timeout_seconds: int = 3600,
        batch_label: str = "judge",
    ) -> tuple[list[JudgeResult] | None, object]:
        results = []
        for request in requests:
            score = 90.0 if "positive" in request.answer else 10.0
            results.append(JudgeResult(request_id=request.request_id, score=score))
        return results, None


def make_artifacts() -> TraitArtifacts:
    return TraitArtifacts(
        trait=TraitDefinition(name="evil", description="bad"),
        instruction_pairs=[InstructionPair(positive="show the trait", negative="avoid the trait")],
        extraction_questions=["q1", "q2"],
        evaluation_questions=["eval q1"],
        evaluation_prompt="Question: {question}\nAnswer: {answer}",
    )


def test_pipeline_extracts_difference_of_means_and_selects_largest_norm() -> None:
    pipeline = ExtractionPipeline(BackendWithSignedActivations(), scorer=FakeScorer())
    bundle, samples = pipeline.run(
        artifacts=make_artifacts(),
        config=ExtractorConfig(rollouts_per_question=1, random_seed=7),
    )
    assert len(samples) == 4
    assert bundle.selected_layer == 2
    assert bundle.selection_strategy == "max_norm"
    layer_1 = bundle.layers[0]
    layer_2 = bundle.layers[1]
    assert layer_1.vector == [2.0, -1.0]
    assert layer_2.vector == [3.0, -2.0]
    assert layer_2.norm > layer_1.norm
    assert bundle.metadata["kept_positive_samples"] == 2
    assert bundle.metadata["kept_negative_samples"] == 2


def test_pipeline_without_scorer_keeps_all_samples() -> None:
    pipeline = ExtractionPipeline(BackendWithSignedActivations())
    bundle, samples = pipeline.run(
        artifacts=make_artifacts(),
        config=ExtractorConfig(rollouts_per_question=1),
    )
    assert len(samples) == 4
    assert bundle.sample_count == 4
