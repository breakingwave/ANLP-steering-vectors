#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_TRAITS = ("evil", "sycophancy", "hallucination")


@dataclass(frozen=True)
class RunProfile:
    rollouts: int
    max_new_tokens: int
    temperature: float
    top_p: float
    positive_threshold: float
    negative_threshold: float
    layer_selection: str
    judge_scoring: str
    steering_alpha: float
    steering_max_new_tokens: int
    steering_questions_limit: int | None
    extract_seed: int
    load_in_4bit: bool
    eval_alpha: float
    eval_no_normalize: bool
    eval_max_new_tokens: int
    eval_temperature: float
    eval_top_p: float
    eval_judge_scoring: str
    eval_questions_limit: int | None
    eval_seed: int
    eval_min_span: float
    eval_min_monotonic_rate: float
    eval_monotonic_mode: str
    eval_monotonic_epsilon: float
    eval_judge_allow_refusal: bool


PROFILES: dict[str, RunProfile] = {
    # paper_closest: matches the paper's methodology as closely as possible.
    # - steering_alpha=2.0 with raw (un-normalized) vectors — the effective value found to
    #   produce meaningful steering effects on Llama-3.1-8B-Instruct (phase transition at ~2.0).
    # - eval_no_normalize=True — matches paper's h_ℓ ← h_ℓ + α·v_ℓ formula (raw vector).
    "paper_closest": RunProfile(
        rollouts=10,
        max_new_tokens=192,
        temperature=0.8,
        top_p=0.95,
        positive_threshold=50.0,
        negative_threshold=50.0,
        layer_selection="steering",
        judge_scoring="logit_weighted",
        steering_alpha=2.0,
        steering_max_new_tokens=192,
        steering_questions_limit=None,
        extract_seed=42,
        load_in_4bit=False,
        eval_alpha=2.0,
        eval_no_normalize=True,
        eval_max_new_tokens=96,
        eval_temperature=0.7,
        eval_top_p=0.9,
        eval_judge_scoring="logit_weighted",
        eval_questions_limit=20,
        eval_seed=123,
        eval_min_span=10.0,
        eval_min_monotonic_rate=0.5,
        eval_monotonic_mode="weak",
        eval_monotonic_epsilon=0.0,
        eval_judge_allow_refusal=True,
    ),
    "optimized": RunProfile(
        rollouts=8,
        max_new_tokens=160,
        temperature=0.8,
        top_p=0.95,
        positive_threshold=50.0,
        negative_threshold=50.0,
        layer_selection="steering",
        judge_scoring="logit_weighted",
        steering_alpha=2.0,
        steering_max_new_tokens=160,
        steering_questions_limit=8,
        extract_seed=42,
        load_in_4bit=True,
        eval_alpha=2.0,
        eval_no_normalize=True,
        eval_max_new_tokens=96,
        eval_temperature=0.7,
        eval_top_p=0.9,
        eval_judge_scoring="regex",
        eval_questions_limit=8,
        eval_seed=123,
        eval_min_span=8.0,
        eval_min_monotonic_rate=0.4,
        eval_monotonic_mode="weak",
        eval_monotonic_epsilon=0.0,
        eval_judge_allow_refusal=False,
    ),
    "smoke_batch_3traits": RunProfile(
        rollouts=2,
        max_new_tokens=96,
        temperature=0.7,
        top_p=0.9,
        positive_threshold=50.0,
        negative_threshold=50.0,
        layer_selection="steering",
        judge_scoring="logit_weighted",
        steering_alpha=2.0,
        steering_max_new_tokens=96,
        steering_questions_limit=4,
        extract_seed=123,
        load_in_4bit=True,
        eval_alpha=2.0,
        eval_no_normalize=True,
        eval_max_new_tokens=72,
        eval_temperature=0.7,
        eval_top_p=0.9,
        eval_judge_scoring="regex",
        eval_questions_limit=4,
        eval_seed=123,
        eval_min_span=5.0,
        eval_min_monotonic_rate=0.4,
        eval_monotonic_mode="weak",
        eval_monotonic_epsilon=0.0,
        eval_judge_allow_refusal=False,
    ),
}


def _run(command: list[str]) -> None:
    rendered = " ".join(command)
    print(f"$ {rendered}")
    subprocess.run(command, cwd=ROOT, check=True)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return "unknown"
    return result.stdout.strip()


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _write_manifest(
    *,
    run_dir: Path,
    trait: str,
    run_id: str,
    profile_name: str,
    profile: RunProfile,
    target_model: str,
    generator_model: str,
    judge_model: str,
    stage: str,
    s3_uri: str | None,
) -> None:
    payload = {
        "run_id": run_id,
        "created_at_utc": _now_utc(),
        "trait": trait,
        "stage": stage,
        "target_model": target_model,
        "generator_model": generator_model,
        "judge_model": judge_model,
        "profile_name": profile_name,
        "profile": asdict(profile),
        "git_commit": _git_commit(),
        "run_dir": str(run_dir.resolve()),
        "s3_uri": s3_uri,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _split_traits(raw: str) -> tuple[str, ...]:
    traits = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not traits:
        raise ValueError("At least one trait is required")
    return traits


def _upload_if_enabled(
    *,
    run_dir: Path,
    enabled: bool,
    bucket: str | None,
    prefix: str,
    region: str | None,
    strict: bool,
) -> str | None:
    if not enabled:
        return None
    if not bucket:
        raise ValueError("S3 upload enabled but --s3-bucket is missing")
    from persona_vectors.s3_persistence import upload_directory

    try:
        uploaded = upload_directory(
            local_dir=run_dir,
            bucket=bucket,
            prefix=prefix.strip("/"),
            region=region,
        )
        print(f"Uploaded {len(uploaded)} files to s3://{bucket}/{prefix.strip('/')}")
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        print(f"WARNING: S3 upload failed for {run_dir}: {exc}")
    return f"s3://{bucket}/{prefix.strip('/')}"


def _run_trait(
    *,
    trait: str,
    run_id: str,
    run_root: Path,
    artifacts_root: Path,
    target_model: str,
    generator_model: str,
    judge_model: str,
    profile_name: str,
    profile: RunProfile,
    s3_upload: bool,
    s3_bucket: str | None,
    s3_prefix: str,
    s3_region: str | None,
    s3_strict: bool,
    extract_judge_mode: str,
    extract_judge_parallelism: int,
    extract_judge_batch_behavior: str,
    extract_judge_batch_poll_seconds: int,
    extract_judge_batch_timeout_seconds: int,
    eval_judge_mode: str,
    eval_judge_parallelism: int,
    eval_judge_batch_behavior: str,
    eval_judge_batch_poll_seconds: int,
    eval_judge_batch_timeout_seconds: int,
) -> None:
    run_dir = run_root / f"{trait}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    artifacts_path = artifacts_root / f"{trait}_{run_id}.json"

    print(f"\n=== [{trait}] generate-artifacts ===")
    _run(
        [
            "persona-vectors",
            "generate-artifacts",
            "--trait-preset",
            trait,
            "--generator-model",
            generator_model,
            "--output",
            str(artifacts_path),
        ]
    )

    print(f"\n=== [{trait}] extract ({profile_name}) ===")
    extract_judge_scoring = profile.judge_scoring
    if extract_judge_mode == "batch" and extract_judge_scoring == "logit_weighted":
        # Batch chat-completions can fail for some model/parameter combinations when
        # requesting top logprobs. Use regex scoring for batch robustness.
        print(
            "NOTE: switching extract judge scoring from "
            "logit_weighted -> regex for batch mode compatibility."
        )
        extract_judge_scoring = "regex"
    extract_cmd = [
        "persona-vectors",
        "extract",
        "--artifacts",
        str(artifacts_path),
        "--target-model",
        target_model,
        "--judge-model",
        judge_model,
        "--judge-scoring",
        extract_judge_scoring,
        "--judge-mode",
        extract_judge_mode,
        "--judge-parallelism",
        str(extract_judge_parallelism),
        "--judge-batch-behavior",
        extract_judge_batch_behavior,
        "--judge-batch-poll-seconds",
        str(extract_judge_batch_poll_seconds),
        "--judge-batch-timeout-seconds",
        str(extract_judge_batch_timeout_seconds),
        "--output-dir",
        str(run_dir),
        "--rollouts",
        str(profile.rollouts),
        "--max-new-tokens",
        str(profile.max_new_tokens),
        "--temperature",
        str(profile.temperature),
        "--top-p",
        str(profile.top_p),
        "--positive-threshold",
        str(profile.positive_threshold),
        "--negative-threshold",
        str(profile.negative_threshold),
        "--layer-selection",
        profile.layer_selection,
        "--steering-alpha",
        str(profile.steering_alpha),
        "--steering-max-new-tokens",
        str(profile.steering_max_new_tokens),
        "--seed",
        str(profile.extract_seed),
    ]
    if profile.load_in_4bit:
        extract_cmd.append("--load-in-4bit")
    if profile.steering_questions_limit is not None:
        extract_cmd.extend(["--steering-questions-limit", str(profile.steering_questions_limit)])
    _run(extract_cmd)

    s3_trait_prefix = f"{s3_prefix.strip('/')}/{run_id}/{trait}"
    s3_uri = _upload_if_enabled(
        run_dir=run_dir,
        enabled=s3_upload,
        bucket=s3_bucket,
        prefix=s3_trait_prefix,
        region=s3_region,
        strict=s3_strict,
    )
    _write_manifest(
        run_dir=run_dir,
        trait=trait,
        run_id=run_id,
        profile_name=profile_name,
        profile=profile,
        target_model=target_model,
        generator_model=generator_model,
        judge_model=judge_model,
        stage="post_extract",
        s3_uri=s3_uri,
    )

    print(f"\n=== [{trait}] evaluate_vectors ({profile_name}) ===")
    evaluate_cmd = [
        "python",
        "evaluate_vectors.py",
        "--run-dir",
        str(run_dir),
        "--judge-model",
        judge_model,
        "--judge-scoring",
        profile.eval_judge_scoring,
        "--judge-mode",
        eval_judge_mode,
        "--judge-parallelism",
        str(eval_judge_parallelism),
        "--judge-batch-behavior",
        eval_judge_batch_behavior,
        "--judge-batch-poll-seconds",
        str(eval_judge_batch_poll_seconds),
        "--judge-batch-timeout-seconds",
        str(eval_judge_batch_timeout_seconds),
        "--alpha",
        str(profile.eval_alpha),
        "--max-new-tokens",
        str(profile.eval_max_new_tokens),
        "--temperature",
        str(profile.eval_temperature),
        "--top-p",
        str(profile.eval_top_p),
        "--seed",
        str(profile.eval_seed),
        "--min-span",
        str(profile.eval_min_span),
        "--min-monotonic-rate",
        str(profile.eval_min_monotonic_rate),
        "--monotonic-mode",
        profile.eval_monotonic_mode,
        "--monotonic-epsilon",
        str(profile.eval_monotonic_epsilon),
        "--json",
        "--save-json",
    ]
    if not profile.eval_judge_allow_refusal:
        evaluate_cmd.append("--judge-no-refusal")
    if profile.load_in_4bit:
        evaluate_cmd.append("--load-in-4bit")
    if profile.eval_no_normalize:
        evaluate_cmd.append("--no-normalize-vector")
    if profile.eval_questions_limit is not None:
        evaluate_cmd.extend(["--questions-limit", str(profile.eval_questions_limit)])
    _run(evaluate_cmd)

    s3_uri = _upload_if_enabled(
        run_dir=run_dir,
        enabled=s3_upload,
        bucket=s3_bucket,
        prefix=s3_trait_prefix,
        region=s3_region,
        strict=s3_strict,
    )
    _write_manifest(
        run_dir=run_dir,
        trait=trait,
        run_id=run_id,
        profile_name=profile_name,
        profile=profile,
        target_model=target_model,
        generator_model=generator_model,
        judge_model=judge_model,
        stage="post_eval",
        s3_uri=s3_uri,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full 3-trait persona-vector pipeline "
            "(generate -> extract -> evaluate -> optional S3 upload)."
        )
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="optimized",
        help=(
            "Run profile: paper_closest (higher fidelity), optimized (faster/lower cost), "
            "or smoke_batch_3traits (smallest full 3-trait batch smoke test)."
        ),
    )
    parser.add_argument(
        "--traits",
        default="evil,sycophancy,hallucination",
        help="Comma-separated trait presets.",
    )
    parser.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--generator-model", default="gpt-5.4-nano")
    parser.add_argument("--judge-model", default="gpt-5.4-nano")
    parser.add_argument("--run-id", help="Run id tag appended to output directories. Defaults to UTC timestamp.")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "runs" / "artifacts")
    parser.add_argument("--s3-upload", action="store_true")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix", default="persona-vectors/runs")
    parser.add_argument("--s3-region")
    parser.add_argument("--s3-strict", action="store_true", help="Fail immediately if any upload fails.")
    parser.add_argument("--extract-judge-mode", choices=("online", "batch"), default="online")
    parser.add_argument("--extract-judge-parallelism", type=int, default=1)
    parser.add_argument(
        "--extract-judge-batch-behavior",
        choices=("submit_exit", "blocking_poll"),
        default="blocking_poll",
    )
    parser.add_argument("--extract-judge-batch-poll-seconds", type=int, default=10)
    parser.add_argument("--extract-judge-batch-timeout-seconds", type=int, default=3600)
    parser.add_argument("--eval-judge-mode", choices=("online", "batch"), default="online")
    parser.add_argument("--eval-judge-parallelism", type=int, default=1)
    parser.add_argument(
        "--eval-judge-batch-behavior",
        choices=("submit_exit", "blocking_poll"),
        default="blocking_poll",
    )
    parser.add_argument("--eval-judge-batch-poll-seconds", type=int, default=10)
    parser.add_argument("--eval-judge-batch-timeout-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    traits = _split_traits(args.traits)
    profile = PROFILES[args.profile]

    print("=== Persona vectors full run ===")
    print(f"profile: {args.profile}")
    print(f"run_id: {run_id}")
    print(f"traits: {', '.join(traits)}")
    if args.s3_upload:
        print(f"s3: s3://{args.s3_bucket}/{args.s3_prefix.strip('/')}/{run_id}/<trait>")

    for trait in traits:
        _run_trait(
            trait=trait,
            run_id=run_id,
            run_root=args.run_root,
            artifacts_root=args.artifacts_root,
            target_model=args.target_model,
            generator_model=args.generator_model,
            judge_model=args.judge_model,
            profile_name=args.profile,
            profile=profile,
            s3_upload=args.s3_upload,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_region=args.s3_region,
            s3_strict=args.s3_strict,
            extract_judge_mode=args.extract_judge_mode,
            extract_judge_parallelism=args.extract_judge_parallelism,
            extract_judge_batch_behavior=args.extract_judge_batch_behavior,
            extract_judge_batch_poll_seconds=args.extract_judge_batch_poll_seconds,
            extract_judge_batch_timeout_seconds=args.extract_judge_batch_timeout_seconds,
            eval_judge_mode=args.eval_judge_mode,
            eval_judge_parallelism=args.eval_judge_parallelism,
            eval_judge_batch_behavior=args.eval_judge_batch_behavior,
            eval_judge_batch_poll_seconds=args.eval_judge_batch_poll_seconds,
            eval_judge_batch_timeout_seconds=args.eval_judge_batch_timeout_seconds,
        )

    print("\n=== All requested traits completed successfully ===")


if __name__ == "__main__":
    main()
