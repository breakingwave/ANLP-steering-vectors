from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from persona_vectors.types import GeneratedSample, JudgeBatchSubmission, JudgeRequest, JudgeResult, TraitArtifacts, TraitDefinition


class TextCompletionClient(Protocol):
    model_name: str

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        ...


class TraitScorer(Protocol):
    def score(self, *, question: str, answer: str, artifacts: TraitArtifacts) -> float | None:
        ...

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
    ) -> tuple[list[JudgeResult] | None, JudgeBatchSubmission | None]:
        ...


class TargetModelBackend(Protocol):
    model_name: str
    num_layers: int

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
        ...

    def response_layer_means(self, *, prompt_token_ids: list[int], response_token_ids: list[int]) -> list[list[float]]:
        ...

    def steering_scope(self, *, layer_index: int, vector: list[float], alpha: float) -> AbstractContextManager[None]:
        ...


class TraitArtifactGenerator(Protocol):
    def generate(self, trait: TraitDefinition) -> TraitArtifacts:
        ...
