from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from persona_vectors.interfaces import TargetModelBackend, TraitScorer
from persona_vectors.types import ExtractorConfig, PersonaVectorLayer, TraitArtifacts


@dataclass(slots=True)
class LayerSelectionResult:
    selected_layer: int
    strategy: str
    metrics: dict[str, float]


class MaxNormLayerSelector:
    def select(self, layers: list[PersonaVectorLayer], *_args: object, **_kwargs: object) -> LayerSelectionResult:
        selected = max(layers, key=lambda item: item.norm)
        return LayerSelectionResult(
            selected_layer=selected.layer_index,
            strategy="max_norm",
            metrics={f"layer_{layer.layer_index}_norm": layer.norm for layer in layers},
        )


class SteeringLayerSelector:
    MIN_NORM_PERCENTILE = 25

    def __init__(self, scorer: TraitScorer) -> None:
        self._scorer = scorer

    def select(
        self,
        layers: list[PersonaVectorLayer],
        *,
        backend: TargetModelBackend,
        artifacts: TraitArtifacts,
        config: ExtractorConfig,
    ) -> LayerSelectionResult:
        questions = artifacts.evaluation_questions or artifacts.extraction_questions
        if config.steering_questions_limit is not None:
            questions = questions[: config.steering_questions_limit]

        all_norms = np.array([layer.norm for layer in layers])
        norm_threshold = float(np.percentile(all_norms, self.MIN_NORM_PERCENTILE))
        candidate_layers = [layer for layer in layers if layer.norm >= norm_threshold]
        skipped = len(layers) - len(candidate_layers)
        if skipped:
            print(
                f"SteeringLayerSelector: skipping {skipped} layer(s) with norm < "
                f"{norm_threshold:.4f} (p{self.MIN_NORM_PERCENTILE} threshold)"
            )
        if not candidate_layers:
            candidate_layers = list(layers)

        skipped_indices = {layer.layer_index for layer in layers} - {layer.layer_index for layer in candidate_layers}
        scores: dict[str, float] = {f"layer_{idx}_steering_score": float("-inf") for idx in skipped_indices}

        for layer in candidate_layers:
            layer_vector = np.asarray(layer.vector, dtype=float)
            norm = np.linalg.norm(layer_vector)
            if norm == 0:
                scores[f"layer_{layer.layer_index}_steering_score"] = float("-inf")
                continue

            # Use the raw (un-normalized) difference-in-means vector so that layers with a
            # stronger contrastive signal (higher norm) naturally apply a larger perturbation
            # and reveal themselves as better candidates. Normalization was removed because it
            # collapsed all layers to unit length, causing degenerate early layers (e.g. layer 1,
            # norm ~0.05) to produce massive relative perturbations that break generation and
            # falsely score as high trait expression.
            raw_vector = layer_vector.tolist()
            question_scores: list[float] = []
            for question_index, question in enumerate(questions):
                seed = None if config.random_seed is None else config.random_seed + question_index
                with backend.steering_scope(layer_index=layer.layer_index, vector=raw_vector, alpha=config.steering_alpha):
                    sample = backend.generate(
                        system_prompt="",
                        user_prompt=question,
                        max_new_tokens=config.steering_max_new_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        seed=seed,
                    )
                score = self._scorer.score(question=question, answer=sample.response_text, artifacts=artifacts)
                if score is not None:
                    question_scores.append(score)

            scores[f"layer_{layer.layer_index}_steering_score"] = float(np.mean(question_scores)) if question_scores else float("-inf")

        selected_key, selected_value = max(scores.items(), key=lambda item: item[1])
        selected_layer = int(selected_key.split("_")[1])
        return LayerSelectionResult(
            selected_layer=selected_layer,
            strategy="steering_eval",
            metrics={**scores, "best_steering_score": selected_value},
        )
