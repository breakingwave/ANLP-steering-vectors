from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_vectors.artifact_generation import LLMTraitArtifactGenerator, load_trait_artifacts, save_trait_artifacts
from persona_vectors.hf import HuggingFaceCausalLMBackend
from persona_vectors.judging import BatchSubmissionPending, LLMJudgeScorer, LogitWeightedJudgeScorer
from persona_vectors.openai_client import OpenAICompletionClient
from persona_vectors.pipeline import ExtractionPipeline
from persona_vectors.serialization import save_run_bundle
from persona_vectors.traits import PAPER_TRAITS
from persona_vectors.types import ExtractorConfig, TraitDefinition


def main() -> None:
    parser = argparse.ArgumentParser(prog="persona-vectors", description="Extract persona steering vectors from decoder-only LLMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_generate_artifacts_parser(subparsers)
    _build_extract_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


def _build_generate_artifacts_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("generate-artifacts", help="Generate instruction pairs, questions, and a judge rubric")
    parser.add_argument("--trait-preset", choices=list(PAPER_TRAITS.keys()), help="Use a paper-defined trait preset")
    parser.add_argument("--trait-name", help="Custom trait name (not needed with --trait-preset)")
    parser.add_argument("--trait-description", help="Custom trait description (not needed with --trait-preset)")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--generator-base-url")
    parser.add_argument("--generator-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=_run_generate_artifacts)


def _build_extract_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("extract", help="Generate contrastive responses and extract per-layer persona vectors")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--positive-threshold", type=float, default=50.0)
    parser.add_argument("--negative-threshold", type=float, default=50.0)
    parser.add_argument("--layer-selection", choices=("max_norm", "steering"), default="max_norm")
    parser.add_argument("--steering-alpha", type=float, default=1.0)
    parser.add_argument("--steering-max-new-tokens", type=int, default=192)
    parser.add_argument("--steering-questions-limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-scoring", choices=("regex", "logit_weighted"), default="regex",
                        help="Scoring method: regex (default) or logit_weighted (paper-aligned)")
    parser.add_argument("--judge-mode", choices=("online", "batch"), default="online")
    parser.add_argument("--judge-parallelism", type=int, default=1)
    parser.add_argument("--judge-batch-behavior", choices=("submit_exit", "blocking_poll"), default="blocking_poll")
    parser.add_argument("--judge-batch-id")
    parser.add_argument("--judge-batch-output")
    parser.add_argument("--judge-batch-poll-seconds", type=int, default=10)
    parser.add_argument("--judge-batch-timeout-seconds", type=int, default=3600)
    parser.set_defaults(func=_run_extract)


def _run_generate_artifacts(args: argparse.Namespace) -> None:
    if args.trait_preset:
        trait = PAPER_TRAITS[args.trait_preset]
    elif args.trait_name and args.trait_description:
        trait = TraitDefinition(name=args.trait_name, description=args.trait_description)
    else:
        raise SystemExit("Either --trait-preset or both --trait-name and --trait-description are required")
    client = OpenAICompletionClient(
        model_name=args.generator_model,
        base_url=args.generator_base_url,
        api_key_env=args.generator_api_key_env,
    )
    generator = LLMTraitArtifactGenerator(client)
    artifacts = generator.generate(trait)
    save_trait_artifacts(artifacts, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "source_model": artifacts.source_model}, indent=2))


def _run_extract(args: argparse.Namespace) -> None:
    artifacts = load_trait_artifacts(args.artifacts)
    backend = HuggingFaceCausalLMBackend(
        args.target_model,
        load_in_4bit=args.load_in_4bit,
    )
    scorer = None
    if args.judge_model:
        judge_client = OpenAICompletionClient(
            model_name=args.judge_model,
            base_url=args.judge_base_url,
            api_key_env=args.judge_api_key_env,
        )
        if args.judge_scoring == "logit_weighted":
            scorer = LogitWeightedJudgeScorer(judge_client)
        else:
            scorer = LLMJudgeScorer(judge_client)
    config = ExtractorConfig(
        rollouts_per_question=args.rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        layer_selection=args.layer_selection,
        steering_alpha=args.steering_alpha,
        steering_max_new_tokens=args.steering_max_new_tokens,
        steering_questions_limit=args.steering_questions_limit,
        random_seed=args.seed,
        judge_mode=args.judge_mode,
        judge_parallelism=args.judge_parallelism,
        judge_batch_behavior=args.judge_batch_behavior,
        judge_batch_id=args.judge_batch_id,
        judge_batch_output=args.judge_batch_output,
        judge_batch_poll_seconds=args.judge_batch_poll_seconds,
        judge_batch_timeout_seconds=args.judge_batch_timeout_seconds,
        judge_artifact_dir=args.output_dir,
    )
    pipeline = ExtractionPipeline(backend, scorer=scorer)
    try:
        bundle, samples = pipeline.run(artifacts=artifacts, config=config)
    except BatchSubmissionPending as exc:
        print(json.dumps({
            "status": "batch_submitted",
            "batch_id": exc.submission.batch_id,
            "metadata_path": exc.submission.metadata_path,
            "request_path": exc.submission.request_path,
        }, indent=2))
        return
    save_run_bundle(output_dir=args.output_dir, artifacts=artifacts, bundle=bundle, samples=samples)
    print(json.dumps({
        "output_dir": str(Path(args.output_dir).resolve()),
        "selected_layer": bundle.selected_layer,
        "selection_strategy": bundle.selection_strategy,
        "sample_count": bundle.sample_count,
    }, indent=2))


if __name__ == "__main__":
    main()
