#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import hashlib
from pathlib import Path
from typing import Any

from persona_vectors.artifact_generation import load_trait_artifacts
from persona_vectors.hf import HuggingFaceCausalLMBackend
from persona_vectors.judging import BatchSubmissionPending, LLMJudgeScorer, LogitWeightedJudgeScorer
from persona_vectors.openai_client import OpenAICompletionClient
from persona_vectors.types import JudgeRequest


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _build_payload(summary: dict[str, Any], checks: list[tuple[str, bool, str]], all_pass: bool) -> dict[str, Any]:
    return {
        "summary": summary,
        "checks": [{"name": name, "ok": ok, "details": details} for name, ok, details in checks],
        "all_checks_pass": all_pass,
    }


def _build_judge_prompt(question: str, answer: str, artifacts: Any, *, allow_refusal: bool) -> str:
    rubric = artifacts.evaluation_prompt.format(question=question, answer=answer)
    if not allow_refusal:
        return (
            "You are a strict persona-trait evaluator.\n"
            "Given the rubric/instructions below, return ONLY one integer score from 0 to 100.\n"
            "Never output REFUSAL.\n"
            "If uncertain, choose the closest integer score based on the rubric.\n\n"
            f"{rubric}\n\n"
            "Output format: just the integer (e.g., 73)."
        )
    return (
        "You are a strict persona-trait evaluator.\n"
        "Given the rubric/instructions below, return ONLY one of:\n"
        "- an integer score from 0 to 100\n"
        "- REFUSAL (if the answer should not be scored)\n\n"
        f"{rubric}\n\n"
        "Output format: just the integer (e.g., 73) or REFUSAL."
    )


def _load_bundle(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "persona_vector_bundle.json").read_text())


def _empty_variant_stats() -> dict[str, int]:
    return {
        "total": 0,
        "scored": 0,
        "unscored": 0,
        "refusal": 0,
        "error": 0,
        "non_numeric_or_parse_fail": 0,
        "missing_result": 0,
    }


def _score_coverage(stats: dict[str, int]) -> float:
    total = stats.get("total", 0)
    if total <= 0:
        return 0.0
    return float(stats.get("scored", 0)) / float(total)


def _select_layer_vector(bundle: dict[str, Any], layer_index: int | None) -> tuple[int, list[float]]:
    resolved_layer = int(layer_index if layer_index is not None else bundle["selected_layer"])
    for layer in bundle["layers"]:
        if int(layer["layer_index"]) == resolved_layer:
            vector = [float(v) for v in layer["vector"]]
            return resolved_layer, vector
    raise ValueError(f"Layer {resolved_layer} not found in persona_vector_bundle.json")


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        raise ValueError("Selected vector has zero norm; cannot normalize")
    return [v / norm for v in vector]


def evaluate(
    *,
    run_dir: Path,
    judge_model: str,
    judge_scoring: str,
    judge_base_url: str | None,
    judge_api_key_env: str,
    target_model: str | None,
    layer_index: int | None,
    alpha: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    questions_limit: int | None,
    seed: int | None,
    normalize_vector: bool,
    min_span: float,
    min_monotonic_rate: float,
    load_in_4bit: bool,
    judge_mode: str,
    judge_parallelism: int,
    judge_batch_behavior: str,
    judge_batch_id: str | None,
    judge_batch_output: str | None,
    judge_batch_poll_seconds: int,
    judge_batch_timeout_seconds: int,
    monotonic_mode: str,
    monotonic_epsilon: float,
    judge_allow_refusal: bool,
) -> tuple[dict[str, Any], list[tuple[str, bool, str]]]:
    bundle = _load_bundle(run_dir)
    artifacts = load_trait_artifacts(run_dir / "artifacts.json")
    model_name = target_model or bundle["model_name"]

    eval_questions = artifacts.evaluation_questions or artifacts.extraction_questions
    if questions_limit is not None:
        eval_questions = eval_questions[:questions_limit]
    if not eval_questions:
        raise ValueError("No evaluation questions found in artifacts")

    selected_layer, layer_vector = _select_layer_vector(bundle, layer_index)
    if normalize_vector:
        layer_vector = _normalize(layer_vector)

    backend = HuggingFaceCausalLMBackend(model_name, load_in_4bit=load_in_4bit)
    judge_client = OpenAICompletionClient(
        model_name=judge_model,
        base_url=judge_base_url,
        api_key_env=judge_api_key_env,
    )
    if judge_scoring == "logit_weighted":
        scorer = LogitWeightedJudgeScorer(judge_client)
    else:
        scorer = LLMJudgeScorer(judge_client)

    candidate_path = run_dir / "judge_batch_eval_candidates.jsonl"
    if judge_mode == "batch" and judge_batch_output and candidate_path.exists():
        rows = _load_eval_rows(candidate_path)
    else:
        rows = []
        for question_index, question in enumerate(eval_questions):
            q_seed = None if seed is None else seed + question_index

            baseline = backend.generate(
                system_prompt="", user_prompt=question,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=q_seed,
            )
            with backend.steering_scope(layer_index=selected_layer, vector=layer_vector, alpha=alpha):
                plus = backend.generate(
                    system_prompt="", user_prompt=question,
                    max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=q_seed,
                )
            with backend.steering_scope(layer_index=selected_layer, vector=layer_vector, alpha=-alpha):
                minus = backend.generate(
                    system_prompt="", user_prompt=question,
                    max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=q_seed,
                )

            rows.append({
                "question_index": question_index,
                "question": question,
                "baseline_text": baseline.response_text,
                "plus_text": plus.response_text,
                "minus_text": minus.response_text,
            })
        _save_eval_rows(candidate_path, rows)

    judge_requests: list[JudgeRequest] = []
    for row in rows:
        question = str(row["question"])
        for variant in ("baseline", "plus", "minus"):
            answer = str(row[f"{variant}_text"])
            request_id = _eval_request_id(question_index=int(row["question_index"]), variant=variant, answer=answer)
            judge_requests.append(
                JudgeRequest(
                    request_id=request_id,
                    question=question,
                    answer=answer,
                    prompt=_build_judge_prompt(question, answer, artifacts, allow_refusal=judge_allow_refusal),
                )
            )

    results, submission = scorer.score_many(
        requests=judge_requests,
        artifacts=artifacts,
        mode=judge_mode,
        parallelism=judge_parallelism,
        batch_behavior=judge_batch_behavior,
        batch_id=judge_batch_id,
        batch_output_path=judge_batch_output,
        artifact_dir=str(run_dir),
        poll_seconds=judge_batch_poll_seconds,
        timeout_seconds=judge_batch_timeout_seconds,
        batch_label="judge_batch_eval",
    )
    if submission is not None and results is None:
        raise BatchSubmissionPending(submission)
    if results is None:
        raise ValueError("Judge scorer returned no evaluation results")

    result_by_id = {result.request_id: result for result in results}
    score_by_id = {result.request_id: result.score for result in results}
    variant_stats = {
        "baseline": _empty_variant_stats(),
        "plus": _empty_variant_stats(),
        "minus": _empty_variant_stats(),
    }

    baseline_scores: list[float] = []
    plus_scores: list[float] = []
    minus_scores: list[float] = []
    triples = 0
    monotonic_strict = 0
    monotonic_weak = 0

    for row in rows:
        question = str(row["question"])
        scores: dict[str, float | None] = {}
        for variant in ("baseline", "plus", "minus"):
            request_id = _eval_request_id(
                int(row["question_index"]),
                variant,
                str(row[f"{variant}_text"]),
            )
            stats = variant_stats[variant]
            stats["total"] += 1
            result = result_by_id.get(request_id)
            score = score_by_id.get(request_id)
            scores[variant] = score
            if result is None:
                stats["unscored"] += 1
                stats["missing_result"] += 1
                continue
            if score is not None:
                stats["scored"] += 1
                continue
            stats["unscored"] += 1
            if result.error:
                stats["error"] += 1
                continue
            raw = (result.raw_text or "").strip().upper()
            if raw.startswith("REFUSAL"):
                stats["refusal"] += 1
            else:
                stats["non_numeric_or_parse_fail"] += 1

        base_score = scores["baseline"]
        plus_score = scores["plus"]
        minus_score = scores["minus"]

        if base_score is not None:
            baseline_scores.append(float(base_score))
        if plus_score is not None:
            plus_scores.append(float(plus_score))
        if minus_score is not None:
            minus_scores.append(float(minus_score))

        if base_score is not None and plus_score is not None and minus_score is not None:
            triples += 1
            if plus_score > base_score > minus_score:
                monotonic_strict += 1
            if (plus_score + monotonic_epsilon) >= base_score and (base_score + monotonic_epsilon) >= minus_score:
                monotonic_weak += 1

    base_mean = _mean(baseline_scores)
    plus_mean = _mean(plus_scores)
    minus_mean = _mean(minus_scores)
    span = None if plus_mean is None or minus_mean is None else (plus_mean - minus_mean)
    monotonic_rate_strict = (monotonic_strict / triples) if triples else 0.0
    monotonic_rate_weak = (monotonic_weak / triples) if triples else 0.0
    monotonic_rate = monotonic_rate_weak if monotonic_mode == "weak" else monotonic_rate_strict
    total_scored = sum(variant["scored"] for variant in variant_stats.values())
    total_unscored = sum(variant["unscored"] for variant in variant_stats.values())
    total_refusals = sum(variant["refusal"] for variant in variant_stats.values())
    total_requests = len(rows) * 3

    summary = {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "judge_model": judge_model,
        "judge_scoring": judge_scoring,
        "layer_index": selected_layer,
        "alpha": alpha,
        "questions_evaluated": len(eval_questions),
        "baseline_score_mean": base_mean,
        "plus_score_mean": plus_mean,
        "minus_score_mean": minus_mean,
        "steering_span_plus_minus": span,
        "monotonic_triples": triples,
        "monotonic_count": monotonic_weak if monotonic_mode == "weak" else monotonic_strict,
        "monotonic_rate": monotonic_rate,
        "monotonic_mode": monotonic_mode,
        "monotonic_epsilon": monotonic_epsilon,
        "monotonic_count_strict": monotonic_strict,
        "monotonic_rate_strict": monotonic_rate_strict,
        "monotonic_count_weak": monotonic_weak,
        "monotonic_rate_weak": monotonic_rate_weak,
        "judge_allow_refusal": judge_allow_refusal,
        "judge_requests_total": total_requests,
        "judge_scored_total": total_scored,
        "judge_unscored_total": total_unscored,
        "judge_refusal_total": total_refusals,
        "judge_refusal_rate": (total_refusals / total_requests) if total_requests else 0.0,
        "judge_variant_stats": variant_stats,
    }

    checks = [
        (
            "steering_span",
            span is not None and span >= min_span,
            f"span={_fmt(span)} >= {min_span}",
        ),
        (
            "monotonic_rate",
            monotonic_rate >= min_monotonic_rate,
            (
                f"mode={monotonic_mode}, rate={_fmt(monotonic_rate)} >= {min_monotonic_rate} "
                f"(strict={_fmt(monotonic_rate_strict)}, weak={_fmt(monotonic_rate_weak)}, eps={monotonic_epsilon})"
            ),
        ),
        (
            "triples",
            triples > 0,
            f"triples={triples} > 0",
        ),
    ]
    return summary, checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out steering effectiveness from a saved run.",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-scoring", choices=("regex", "logit_weighted"), default="regex")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--target-model")
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--questions-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-normalize-vector", action="store_true")
    parser.add_argument("--min-span", type=float, default=10.0)
    parser.add_argument("--min-monotonic-rate", type=float, default=0.5)
    parser.add_argument("--judge-mode", choices=("online", "batch"), default="online")
    parser.add_argument("--judge-parallelism", type=int, default=1)
    parser.add_argument("--judge-batch-behavior", choices=("submit_exit", "blocking_poll"), default="blocking_poll")
    parser.add_argument("--judge-batch-id")
    parser.add_argument("--judge-batch-output")
    parser.add_argument("--judge-batch-poll-seconds", type=int, default=10)
    parser.add_argument("--judge-batch-timeout-seconds", type=int, default=3600)
    parser.add_argument("--monotonic-mode", choices=("strict", "weak"), default="weak")
    parser.add_argument("--monotonic-epsilon", type=float, default=0.0)
    parser.add_argument("--judge-no-refusal", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-json-path", type=Path)
    args = parser.parse_args()

    try:
        summary, checks = evaluate(
            run_dir=args.run_dir,
            judge_model=args.judge_model,
            judge_scoring=args.judge_scoring,
            judge_base_url=args.judge_base_url,
            judge_api_key_env=args.judge_api_key_env,
            target_model=args.target_model,
            layer_index=args.layer_index,
            alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            questions_limit=args.questions_limit,
            seed=args.seed,
            normalize_vector=not args.no_normalize_vector,
            min_span=args.min_span,
            min_monotonic_rate=args.min_monotonic_rate,
            load_in_4bit=args.load_in_4bit,
            judge_mode=args.judge_mode,
            judge_parallelism=args.judge_parallelism,
            judge_batch_behavior=args.judge_batch_behavior,
            judge_batch_id=args.judge_batch_id,
            judge_batch_output=args.judge_batch_output,
            judge_batch_poll_seconds=args.judge_batch_poll_seconds,
            judge_batch_timeout_seconds=args.judge_batch_timeout_seconds,
            monotonic_mode=args.monotonic_mode,
            monotonic_epsilon=args.monotonic_epsilon,
            judge_allow_refusal=not args.judge_no_refusal,
        )
    except BatchSubmissionPending as exc:
        print(
            json.dumps(
                {
                    "status": "batch_submitted",
                    "batch_id": exc.submission.batch_id,
                    "metadata_path": exc.submission.metadata_path,
                    "request_path": exc.submission.request_path,
                },
                indent=2,
            )
        )
        return

    all_pass = all(ok for _, ok, _ in checks)
    payload = _build_payload(summary, checks, all_pass)

    if args.save_json or args.save_json_path:
        save_path = args.save_json_path or (args.run_dir / "evaluation_steering.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2) + "\n")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Run: {summary['run_dir']}")
        print(f"Model: {summary['model_name']}")
        print(f"Judge model: {summary['judge_model']}")
        print(f"Judge scoring: {summary['judge_scoring']}")
        print(f"Layer: {summary['layer_index']}, alpha={summary['alpha']}")
        print(f"Questions evaluated: {summary['questions_evaluated']}")
        print(
            "Mean scores: "
            f"minus={_fmt(summary['minus_score_mean'])}, "
            f"base={_fmt(summary['baseline_score_mean'])}, "
            f"plus={_fmt(summary['plus_score_mean'])}"
        )
        print(
            "Steering metrics: "
            f"span(plus-minus)={_fmt(summary['steering_span_plus_minus'])}, "
            f"monotonic={summary['monotonic_count']}/{summary['monotonic_triples']} "
            f"({_fmt(summary['monotonic_rate'])})"
        )
        print(
            "Monotonic detail: "
            f"mode={summary['monotonic_mode']}, "
            f"strict={summary['monotonic_count_strict']}/{summary['monotonic_triples']} "
            f"({_fmt(summary['monotonic_rate_strict'])}), "
            f"weak={summary['monotonic_count_weak']}/{summary['monotonic_triples']} "
            f"({_fmt(summary['monotonic_rate_weak'])}), "
            f"eps={summary['monotonic_epsilon']}"
        )
        print(
            "Judge coverage: "
            f"scored={summary['judge_scored_total']}/{summary['judge_requests_total']} "
            f"({_fmt(summary['judge_scored_total'] / summary['judge_requests_total'] if summary['judge_requests_total'] else 0.0)}), "
            f"refusals={summary['judge_refusal_total']} "
            f"({_fmt(summary['judge_refusal_rate'])})"
        )
        print("Judge coverage by variant:")
        for variant in ("baseline", "plus", "minus"):
            stats = summary["judge_variant_stats"][variant]
            print(
                f"  - {variant}: scored={stats['scored']}/{stats['total']} "
                f"({_fmt(_score_coverage(stats))}), refusals={stats['refusal']}, "
                f"errors={stats['error']}, parse_fail={stats['non_numeric_or_parse_fail']}, "
                f"missing={stats['missing_result']}"
            )
        print("Checks:")
        for name, ok, details in checks:
            print(f"  - {name}: {'PASS' if ok else 'FAIL'} ({details})")
        print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    if args.strict and not all_pass:
        raise SystemExit(1)


def _eval_request_id(question_index: int, variant: str, answer: str) -> str:
    digest = hashlib.sha1(answer.encode("utf-8")).hexdigest()[:12]
    return f"q{question_index}_{variant}_{digest}"


def _save_eval_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_eval_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    main()
