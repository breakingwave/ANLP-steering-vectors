#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import analyze_run
import evaluate_steering
from persona_vectors.judging import BatchSubmissionPending


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _build_payload(
    *,
    run_dir: Path,
    static_summary: dict[str, Any],
    static_checks: list[tuple[str, bool, str]],
    static_ok: bool,
    steering_summary: dict[str, Any],
    steering_checks: list[tuple[str, bool, str]],
    steering_ok: bool,
    overall_ok: bool,
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "static": {
            "summary": static_summary,
            "checks": [{"name": n, "ok": ok, "details": d} for n, ok, d in static_checks],
            "all_checks_pass": static_ok,
        },
        "steering": {
            "summary": steering_summary,
            "checks": [{"name": n, "ok": ok, "details": d} for n, ok, d in steering_checks],
            "all_checks_pass": steering_ok,
        },
        "all_checks_pass": overall_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command evaluation for extracted persona vectors.",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--min-score-gap", type=float, default=20.0)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--min-selected-norm", type=float, default=0.5)
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
    parser.add_argument("--monotonic-mode", choices=("strict", "weak"), default="weak")
    parser.add_argument("--monotonic-epsilon", type=float, default=0.0)
    parser.add_argument("--judge-no-refusal", action="store_true")
    parser.add_argument("--judge-mode", choices=("online", "batch"), default="online")
    parser.add_argument("--judge-parallelism", type=int, default=1)
    parser.add_argument("--judge-batch-behavior", choices=("submit_exit", "blocking_poll"), default="blocking_poll")
    parser.add_argument("--judge-batch-id")
    parser.add_argument("--judge-batch-output")
    parser.add_argument("--judge-batch-poll-seconds", type=int, default=10)
    parser.add_argument("--judge-batch-timeout-seconds", type=int, default=3600)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-json-path", type=Path)
    args = parser.parse_args()

    static_summary, static_checks = analyze_run.analyze(
        run_dir=args.run_dir,
        min_score_gap=args.min_score_gap,
        min_samples=args.min_samples,
        min_selected_norm=args.min_selected_norm,
    )

    try:
        steering_summary, steering_checks = evaluate_steering.evaluate(
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

    static_ok = all(ok for _, ok, _ in static_checks)
    steering_ok = all(ok for _, ok, _ in steering_checks)
    overall_ok = static_ok and steering_ok
    payload = _build_payload(
        run_dir=args.run_dir,
        static_summary=static_summary,
        static_checks=static_checks,
        static_ok=static_ok,
        steering_summary=steering_summary,
        steering_checks=steering_checks,
        steering_ok=steering_ok,
        overall_ok=overall_ok,
    )

    if args.save_json or args.save_json_path:
        save_path = args.save_json_path or (args.run_dir / "evaluation_vectors.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2) + "\n")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=== Static Extraction Quality ===")
        print(f"Model: {static_summary['model_name']}")
        print(f"Trait: {static_summary['trait_name']}")
        print(
            "Scores: "
            f"pos={_fmt(static_summary['positive_score_mean'])}, "
            f"neg={_fmt(static_summary['negative_score_mean'])}, "
            f"gap={_fmt(static_summary['score_gap_pos_minus_neg'])}"
        )
        print(
            f"Samples: total={static_summary['sample_count']} "
            f"(pos={static_summary['positive_samples']}, neg={static_summary['negative_samples']})"
        )
        print(
            f"Selected layer: {static_summary['selected_layer']} "
            f"(norm={_fmt(static_summary['selected_layer_norm'])})"
        )
        for name, ok, details in static_checks:
            print(f"  - {name}: {'PASS' if ok else 'FAIL'} ({details})")
        print(f"Static overall: {'PASS' if static_ok else 'FAIL'}")
        print()

        print("=== Held-out Steering Effect ===")
        print(f"Judge: {steering_summary['judge_model']}")
        print(
            "Means: "
            f"minus={_fmt(steering_summary['minus_score_mean'])}, "
            f"base={_fmt(steering_summary['baseline_score_mean'])}, "
            f"plus={_fmt(steering_summary['plus_score_mean'])}"
        )
        print(
            f"Span(+/-): {_fmt(steering_summary['steering_span_plus_minus'])}, "
            f"Monotonic: {steering_summary['monotonic_count']}/"
            f"{steering_summary['monotonic_triples']} "
            f"({_fmt(steering_summary['monotonic_rate'])})"
        )
        for name, ok, details in steering_checks:
            print(f"  - {name}: {'PASS' if ok else 'FAIL'} ({details})")
        print(f"Steering overall: {'PASS' if steering_ok else 'FAIL'}")
        print()
        print(f"Overall: {'PASS' if overall_ok else 'FAIL'}")

    if args.strict and not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
