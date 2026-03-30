from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from persona_vectors.interfaces import TargetModelBackend, TraitScorer
from persona_vectors.judging import BatchSubmissionPending, _build_judge_prompt
from persona_vectors.selection import MaxNormLayerSelector, SteeringLayerSelector
from persona_vectors.types import (
    ActivationSample,
    ExtractorConfig,
    GeneratedSample,
    JudgeRequest,
    PersonaVectorBundle,
    PersonaVectorLayer,
    TraitArtifacts,
)


@dataclass(slots=True)
class _CandidateSample:
    prompt_kind: str
    instruction_index: int
    question_index: int
    question: str
    generated: GeneratedSample

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_kind": self.prompt_kind,
            "instruction_index": self.instruction_index,
            "question_index": self.question_index,
            "question": self.question,
            "prompt_token_ids": self.generated.prompt_token_ids,
            "response_token_ids": self.generated.response_token_ids,
            "response_text": self.generated.response_text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> _CandidateSample:
        return cls(
            prompt_kind=str(payload["prompt_kind"]),
            instruction_index=int(payload["instruction_index"]),
            question_index=int(payload["question_index"]),
            question=str(payload["question"]),
            generated=GeneratedSample(
                prompt_token_ids=[int(token) for token in payload["prompt_token_ids"]],  # type: ignore[index]
                response_token_ids=[int(token) for token in payload["response_token_ids"]],  # type: ignore[index]
                response_text=str(payload["response_text"]),
            ),
        )


class ExtractionPipeline:
    def __init__(self, backend: TargetModelBackend, *, scorer: TraitScorer | None = None) -> None:
        self._backend = backend
        self._scorer = scorer

    def run(self, *, artifacts: TraitArtifacts, config: ExtractorConfig) -> tuple[PersonaVectorBundle, list[ActivationSample]]:
        samples = self._collect_samples(artifacts=artifacts, config=config)
        bundle = self._build_bundle(artifacts=artifacts, samples=samples, config=config)
        return bundle, samples

    def _collect_samples(self, *, artifacts: TraitArtifacts, config: ExtractorConfig) -> list[ActivationSample]:
        candidates_path = _candidate_path(config=config)
        if self._scorer is not None and config.judge_mode == "batch" and config.judge_batch_output and candidates_path and candidates_path.exists():
            candidates = _load_candidates(candidates_path)
        else:
            candidates = self._generate_candidates(artifacts=artifacts, config=config)
            if self._scorer is not None and config.judge_mode == "batch" and candidates_path is not None:
                _save_candidates(path=candidates_path, candidates=candidates)

        score_by_request_id: dict[str, float | None] = {}
        if self._scorer is not None:
            judge_requests = [
                JudgeRequest(
                    request_id=_request_id(candidate),
                    question=candidate.question,
                    answer=candidate.generated.response_text,
                    prompt=_build_judge_prompt(candidate.question, candidate.generated.response_text, artifacts),
                )
                for candidate in candidates
            ]
            results, submission = self._scorer.score_many(
                requests=judge_requests,
                artifacts=artifacts,
                mode=config.judge_mode,
                parallelism=config.judge_parallelism,
                batch_behavior=config.judge_batch_behavior,
                batch_id=config.judge_batch_id,
                batch_output_path=config.judge_batch_output,
                artifact_dir=config.judge_artifact_dir,
                poll_seconds=config.judge_batch_poll_seconds,
                timeout_seconds=config.judge_batch_timeout_seconds,
                batch_label="judge_batch_extract",
            )
            if submission is not None and results is None:
                raise BatchSubmissionPending(submission)
            if results is None:
                raise ValueError("Judge scorer did not return results")
            for result in results:
                score_by_request_id[result.request_id] = result.score

        samples: list[ActivationSample] = []
        for candidate in candidates:
            request_id = _request_id(candidate)
            score = score_by_request_id.get(request_id) if self._scorer is not None else None
            if self._scorer is not None and not _passes_filter(prompt_kind=candidate.prompt_kind, score=score, config=config):
                continue
            layer_means = self._backend.response_layer_means(
                prompt_token_ids=candidate.generated.prompt_token_ids,
                response_token_ids=candidate.generated.response_token_ids,
            )
            samples.append(
                ActivationSample(
                    prompt_kind=candidate.prompt_kind,
                    instruction_index=candidate.instruction_index,
                    question_index=candidate.question_index,
                    question=candidate.question,
                    response_text=candidate.generated.response_text,
                    score=score,
                    layer_means=layer_means,
                )
            )

        if not samples:
            raise ValueError("No samples survived filtering; lower thresholds or disable the judge filter")
        return samples

    def _generate_candidates(self, *, artifacts: TraitArtifacts, config: ExtractorConfig) -> list[_CandidateSample]:
        candidates: list[_CandidateSample] = []
        for instruction_index, instruction_pair in enumerate(artifacts.instruction_pairs):
            for prompt_kind, system_prompt in (("positive", instruction_pair.positive), ("negative", instruction_pair.negative)):
                for question_index, question in enumerate(artifacts.extraction_questions):
                    for rollout_index in range(config.rollouts_per_question):
                        seed = None
                        if config.random_seed is not None:
                            seed = config.random_seed + instruction_index * 100_000 + question_index * 1_000 + rollout_index
                        generated = self._backend.generate(
                            system_prompt=system_prompt,
                            user_prompt=question,
                            max_new_tokens=config.max_new_tokens,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            seed=seed,
                        )
                        candidates.append(
                            _CandidateSample(
                                prompt_kind=prompt_kind,
                                instruction_index=instruction_index,
                                question_index=question_index,
                                question=question,
                                generated=generated,
                            )
                        )
        return candidates

    def _build_bundle(
        self,
        *,
        artifacts: TraitArtifacts,
        samples: list[ActivationSample],
        config: ExtractorConfig,
    ) -> PersonaVectorBundle:
        grouped: dict[str, dict[int, list[np.ndarray]]] = {
            "positive": defaultdict(list),
            "negative": defaultdict(list),
        }
        for sample in samples:
            for layer_index, layer_mean in enumerate(sample.layer_means, start=1):
                grouped[sample.prompt_kind][layer_index].append(np.asarray(layer_mean, dtype=float))

        layers: list[PersonaVectorLayer] = []
        for layer_index in range(1, self._backend.num_layers + 1):
            positive = grouped["positive"][layer_index]
            negative = grouped["negative"][layer_index]
            if not positive or not negative:
                raise ValueError(f"Layer {layer_index} is missing positive or negative samples after filtering")
            vector = np.mean(positive, axis=0) - np.mean(negative, axis=0)
            layers.append(
                PersonaVectorLayer(
                    layer_index=layer_index,
                    vector=vector.tolist(),
                    norm=float(np.linalg.norm(vector)),
                    positive_count=len(positive),
                    negative_count=len(negative),
                )
            )

        selector = MaxNormLayerSelector()
        if config.layer_selection == "steering":
            if self._scorer is None:
                raise ValueError("Steering-based layer selection requires a scorer")
            selector = SteeringLayerSelector(self._scorer)
        selection = selector.select(layers, backend=self._backend, artifacts=artifacts, config=config)

        kept_scores = [sample.score for sample in samples if sample.score is not None]
        metadata = {
            "source_artifact_model": artifacts.source_model,
            "kept_positive_samples": sum(sample.prompt_kind == "positive" for sample in samples),
            "kept_negative_samples": sum(sample.prompt_kind == "negative" for sample in samples),
            "mean_judge_score": float(np.mean(kept_scores)) if kept_scores else None,
            **selection.metrics,
        }
        return PersonaVectorBundle(
            trait=artifacts.trait,
            model_name=self._backend.model_name,
            token_pooling="response_mean",
            selected_layer=selection.selected_layer,
            selection_strategy=selection.strategy,
            layers=layers,
            sample_count=len(samples),
            metadata=metadata,
        )


def _passes_filter(*, prompt_kind: str, score: float | None, config: ExtractorConfig) -> bool:
    if score is None:
        return False
    if prompt_kind == "positive":
        return score > config.positive_threshold
    return score < config.negative_threshold


def _request_id(candidate: _CandidateSample) -> str:
    digest = hashlib.sha1(candidate.generated.response_text.encode("utf-8")).hexdigest()[:12]
    return f"inst{candidate.instruction_index}_q{candidate.question_index}_{candidate.prompt_kind}_{digest}"


def _candidate_path(*, config: ExtractorConfig) -> Path | None:
    if config.judge_artifact_dir is None:
        return None
    return Path(config.judge_artifact_dir) / "judge_batch_extract_candidates.jsonl"


def _save_candidates(*, path: Path, candidates: list[_CandidateSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.to_dict()) + "\n")


def _load_candidates(path: Path) -> list[_CandidateSample]:
    return [_CandidateSample.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
