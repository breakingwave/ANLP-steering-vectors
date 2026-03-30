from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TraitDefinition:
    name: str
    description: str


@dataclass(slots=True)
class InstructionPair:
    positive: str
    negative: str

    def to_dict(self) -> dict[str, str]:
        return {"pos": self.positive, "neg": self.negative}


@dataclass(slots=True)
class TraitArtifacts:
    trait: TraitDefinition
    instruction_pairs: list[InstructionPair]
    extraction_questions: list[str]
    evaluation_questions: list[str]
    evaluation_prompt: str
    source_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trait": asdict(self.trait),
            "instruction": [pair.to_dict() for pair in self.instruction_pairs],
            "extraction_questions": list(self.extraction_questions),
            "evaluation_questions": list(self.evaluation_questions),
            "eval_prompt": self.evaluation_prompt,
            "source_model": self.source_model,
        }


@dataclass(slots=True)
class ExtractorConfig:
    rollouts_per_question: int = 10
    max_new_tokens: int = 192
    temperature: float = 0.8
    top_p: float = 0.95
    positive_threshold: float = 50.0
    negative_threshold: float = 50.0
    layer_selection: str = "max_norm"
    steering_alpha: float = 1.0
    steering_max_new_tokens: int = 192
    steering_questions_limit: int | None = None
    random_seed: int | None = None
    judge_mode: str = "online"
    judge_parallelism: int = 1
    judge_batch_behavior: str = "blocking_poll"
    judge_batch_id: str | None = None
    judge_batch_output: str | None = None
    judge_batch_poll_seconds: int = 10
    judge_batch_timeout_seconds: int = 3600
    judge_artifact_dir: str | None = None


@dataclass(slots=True)
class GeneratedSample:
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_text: str


@dataclass(slots=True)
class ActivationSample:
    prompt_kind: str
    instruction_index: int
    question_index: int
    question: str
    response_text: str
    score: float | None
    layer_means: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JudgeRequest:
    request_id: str
    question: str
    answer: str
    prompt: str


@dataclass(slots=True)
class JudgeResult:
    request_id: str
    score: float | None
    raw_text: str | None = None
    token_logprobs: list[dict[str, Any]] | None = None
    error: str | None = None


@dataclass(slots=True)
class JudgeBatchSubmission:
    batch_id: str
    metadata_path: str
    request_path: str


@dataclass(slots=True)
class PersonaVectorLayer:
    layer_index: int
    vector: list[float]
    norm: float
    positive_count: int
    negative_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PersonaVectorBundle:
    trait: TraitDefinition
    model_name: str
    token_pooling: str
    selected_layer: int
    selection_strategy: str
    layers: list[PersonaVectorLayer]
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trait": asdict(self.trait),
            "model_name": self.model_name,
            "token_pooling": self.token_pooling,
            "selected_layer": self.selected_layer,
            "selection_strategy": self.selection_strategy,
            "sample_count": self.sample_count,
            "layers": [layer.to_dict() for layer in self.layers],
            "metadata": self.metadata,
        }
