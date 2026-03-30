from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from persona_vectors.interfaces import TextCompletionClient
from persona_vectors.types import JudgeBatchSubmission, JudgeRequest, JudgeResult, TraitArtifacts


def _build_judge_prompt(question: str, answer: str, artifacts: TraitArtifacts) -> str:
    rubric = artifacts.evaluation_prompt.format(question=question, answer=answer)
    return (
        "You are a strict persona-trait evaluator.\n"
        "Given the rubric/instructions below, return ONLY one of:\n"
        "- an integer score from 0 to 100\n"
        "- REFUSAL (if the answer should not be scored)\n\n"
        f"{rubric}\n\n"
        "Output format: just the integer (e.g., 73) or REFUSAL."
    )


def _parse_score_from_text(raw: str) -> float | None:
    if raw.upper().startswith("REFUSAL"):
        return None
    match = re.search(r"\b(100|[1-9]?\d)\b", raw)
    if not match:
        return None
    return float(match.group(1))


def _ensure_artifact_dir(path: str | None) -> Path | None:
    if path is None:
        return None
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _run_online(score_fn: Any, requests: list[JudgeRequest], parallelism: int) -> list[JudgeResult]:
    if parallelism <= 1:
        return [score_fn(request) for request in requests]

    results: dict[str, JudgeResult] = {}
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(score_fn, request): request.request_id for request in requests}
        for future in as_completed(futures):
            rid = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = JudgeResult(request_id=rid, score=None, error=str(exc))
            results[rid] = result

    return [results[request.request_id] for request in requests]


class BatchSubmissionPending(RuntimeError):
    def __init__(self, submission: JudgeBatchSubmission) -> None:
        self.submission = submission
        super().__init__(
            f"Judge batch submitted: id={submission.batch_id}, metadata={submission.metadata_path}"
        )


class LLMJudgeScorer:
    def __init__(self, client: TextCompletionClient) -> None:
        self._client = client

    def score(self, *, question: str, answer: str, artifacts: TraitArtifacts) -> float | None:
        prompt = _build_judge_prompt(question, answer, artifacts)
        raw = self._client.complete(prompt, temperature=0.0).strip()
        return _parse_score_from_text(raw)

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
        if mode == "online":
            return _run_online(self._score_single, requests, parallelism), None
        if not hasattr(self._client, "submit_chat_batch"):
            raise ValueError("Batch mode requires OpenAICompletionClient")
        return self._score_many_batch(
            requests=requests,
            batch_behavior=batch_behavior,
            batch_id=batch_id,
            batch_output_path=batch_output_path,
            artifact_dir=artifact_dir,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            batch_label=batch_label,
            include_logprobs=False,
        )

    def _score_single(self, request: JudgeRequest) -> JudgeResult:
        raw = self._client.complete(request.prompt, temperature=0.0).strip()
        return JudgeResult(request_id=request.request_id, score=_parse_score_from_text(raw), raw_text=raw)

    def _score_many_batch(
        self,
        *,
        requests: list[JudgeRequest],
        batch_behavior: str,
        batch_id: str | None,
        batch_output_path: str | None,
        artifact_dir: str | None,
        poll_seconds: int,
        timeout_seconds: int,
        batch_label: str,
        include_logprobs: bool,
    ) -> tuple[list[JudgeResult] | None, JudgeBatchSubmission | None]:
        directory = _ensure_artifact_dir(artifact_dir)
        request_path = None if directory is None else directory / f"{batch_label}_request.jsonl"
        results_path = None if directory is None else directory / f"{batch_label}_results.jsonl"
        meta_path = None if directory is None else directory / f"{batch_label}_meta.json"

        if batch_output_path:
            output_text = Path(batch_output_path).read_text()
            results = _parse_batch_output_text(output_text=output_text, include_logprobs=include_logprobs)
            if results_path is not None:
                results_path.write_text(output_text if output_text.endswith("\n") else output_text + "\n")
            return _ordered_results(requests=requests, parsed=results), None

        if batch_id:
            output_text = self._client.fetch_batch_output(batch_id=batch_id)  # type: ignore[attr-defined]
            results = _parse_batch_output_text(output_text=output_text, include_logprobs=include_logprobs)
            if results_path is not None:
                results_path.write_text(output_text if output_text.endswith("\n") else output_text + "\n")
            return _ordered_results(requests=requests, parsed=results), None

        payloads = [
            _build_chat_batch_request(
                request=request,
                model_name=self._client.model_name,
                include_logprobs=include_logprobs,
            )
            for request in requests
        ]
        submitted_batch_id, tmp_request_path = self._client.submit_chat_batch(  # type: ignore[attr-defined]
            requests=payloads,
            metadata={"kind": batch_label, "created_at": str(int(time.time()))},
        )

        if request_path is not None:
            request_path.write_text("\n".join(json.dumps(payload) for payload in payloads) + "\n")
        if meta_path is not None:
            meta_path.write_text(
                json.dumps(
                    {
                        "batch_id": submitted_batch_id,
                        "batch_label": batch_label,
                        "tmp_request_path": tmp_request_path,
                        "request_path": str(request_path) if request_path is not None else None,
                        "submitted_at": int(time.time()),
                    },
                    indent=2,
                )
                + "\n"
            )

        submission = JudgeBatchSubmission(
            batch_id=submitted_batch_id,
            metadata_path=str(meta_path) if meta_path is not None else "",
            request_path=str(request_path) if request_path is not None else tmp_request_path,
        )

        if batch_behavior == "submit_exit":
            return None, submission

        output_text = self._client.wait_for_batch_output(  # type: ignore[attr-defined]
            batch_id=submitted_batch_id,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
        if results_path is not None:
            results_path.write_text(output_text if output_text.endswith("\n") else output_text + "\n")
        results = _parse_batch_output_text(output_text=output_text, include_logprobs=include_logprobs)
        return _ordered_results(requests=requests, parsed=results), submission


class LogitWeightedJudgeScorer:
    def __init__(self, client: object) -> None:
        self._client = client

    def score(self, *, question: str, answer: str, artifacts: TraitArtifacts) -> float | None:
        prompt = _build_judge_prompt(question, answer, artifacts)
        text, token_logprobs = self._client.complete_with_logprobs(  # type: ignore[attr-defined]
            prompt, temperature=0.0, max_completion_tokens=8, top_logprobs=5,
        )
        return _score_from_logprobs(text=text, token_logprobs=token_logprobs)

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
        if mode == "online":
            return _run_online(self._score_single, requests, parallelism), None

        delegate = LLMJudgeScorer(self._client)  # type: ignore[arg-type]
        return delegate._score_many_batch(
            requests=requests,
            batch_behavior=batch_behavior,
            batch_id=batch_id,
            batch_output_path=batch_output_path,
            artifact_dir=artifact_dir,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            batch_label=batch_label,
            include_logprobs=True,
        )

    def _score_single(self, request: JudgeRequest) -> JudgeResult:
        text, token_logprobs = self._client.complete_with_logprobs(  # type: ignore[attr-defined]
            request.prompt,
            temperature=0.0,
            max_completion_tokens=8,
            top_logprobs=5,
        )
        score = _score_from_logprobs(text=text, token_logprobs=token_logprobs)
        return JudgeResult(
            request_id=request.request_id,
            score=score,
            raw_text=text,
            token_logprobs=token_logprobs,
        )


def _score_from_logprobs(*, text: str, token_logprobs: list[dict[str, Any]]) -> float | None:
    if text.strip().upper().startswith("REFUSAL"):
        return None
    if not token_logprobs:
        return _parse_score_from_text(text)

    weighted_sum = 0.0
    weight_total = 0.0
    for token_str, logprob in token_logprobs[0].items():
        token_stripped = token_str.strip()
        if token_stripped.isdigit():
            value = int(token_stripped)
            if 0 <= value <= 100:
                prob = math.exp(logprob)
                weighted_sum += value * prob
                weight_total += prob

    if weight_total == 0:
        return None
    return weighted_sum / weight_total


def _build_chat_batch_request(
    *,
    request: JudgeRequest,
    model_name: str,
    include_logprobs: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 16 if include_logprobs else 32,
    }
    if include_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    return {
        "custom_id": request.request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _parse_batch_output_text(*, output_text: str, include_logprobs: bool) -> dict[str, JudgeResult]:
    parsed: dict[str, JudgeResult] = {}

    for line in output_text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        request_id = payload.get("custom_id")
        if not request_id:
            continue

        error = payload.get("error")
        if error:
            parsed[request_id] = JudgeResult(request_id=request_id, score=None, error=str(error))
            continue

        body = payload.get("response", {}).get("body", {})
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict)).strip()
        else:
            text = str(content).strip()

        token_logprobs: list[dict[str, Any]] | None = None
        if include_logprobs:
            token_logprobs = []
            logprobs_content = ((choice.get("logprobs") or {}).get("content")) or []
            for token_info in logprobs_content:
                entry: dict[str, Any] = {}
                for top in token_info.get("top_logprobs") or []:
                    token = top.get("token")
                    logprob = top.get("logprob")
                    if token is not None and logprob is not None:
                        entry[str(token)] = float(logprob)
                token_logprobs.append(entry)
            score = _score_from_logprobs(text=text, token_logprobs=token_logprobs)
        else:
            score = _parse_score_from_text(text)

        parsed[request_id] = JudgeResult(
            request_id=request_id,
            score=score,
            raw_text=text,
            token_logprobs=token_logprobs,
        )

    return parsed


def _ordered_results(*, requests: list[JudgeRequest], parsed: dict[str, JudgeResult]) -> list[JudgeResult]:
    return [
        parsed.get(request.request_id)
        or JudgeResult(request_id=request.request_id, score=None, error="missing_result_for_request")
        for request in requests
    ]
