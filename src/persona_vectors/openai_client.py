from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

from openai import BadRequestError, OpenAI


class OpenAICompletionClient:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
    ) -> None:
        key = api_key or os.getenv(api_key_env)
        if not key:
            raise ValueError(f"Missing API key. Set {api_key_env} or pass api_key explicitly.")
        self.model_name = model_name
        self._client = OpenAI(api_key=key, base_url=base_url)

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = self._client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )
        return response.output_text

    def complete_with_logprobs(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_completion_tokens: int = 8,
        top_logprobs: int = 5,
    ) -> tuple[str, list[dict[str, Any]]]:
        safe_top_logprobs = max(0, min(top_logprobs, 5))
        safe_max_completion_tokens = max(1, max_completion_tokens)

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=safe_max_completion_tokens,
                logprobs=True,
                top_logprobs=safe_top_logprobs,
            )
        except BadRequestError as exc:
            message = str(exc).lower()
            if "max_tokens or model output limit was reached" not in message:
                raise
            retry_max_completion_tokens = max(16, safe_max_completion_tokens * 2)
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=retry_max_completion_tokens,
                logprobs=True,
                top_logprobs=safe_top_logprobs,
            )

        choice = response.choices[0]
        text = choice.message.content or ""
        token_logprobs: list[dict[str, Any]] = []
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                entry: dict[str, Any] = {}
                for top in token_info.top_logprobs:
                    entry[top.token] = top.logprob
                token_logprobs.append(entry)

        return text, token_logprobs

    def submit_chat_batch(
        self,
        *,
        requests: list[dict[str, Any]],
        metadata: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as handle:
            for request in requests:
                handle.write(json.dumps(request) + "\n")
            input_path = handle.name

        with open(input_path, "rb") as file_handle:
            file_ref = self._client.files.create(file=file_handle, purpose="batch")

        batch = self._client.batches.create(
            input_file_id=file_ref.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata or {},
        )
        return batch.id, input_path

    def wait_for_batch_output(
        self,
        *,
        batch_id: str,
        poll_seconds: int = 10,
        timeout_seconds: int = 3600,
    ) -> str:
        deadline = time.time() + timeout_seconds
        while True:
            batch = self._client.batches.retrieve(batch_id)
            status = batch.status
            if status == "completed":
                if not batch.output_file_id:
                    raise RuntimeError(f"Batch {batch_id} completed without output_file_id")
                output = self._client.files.content(batch.output_file_id)
                return output.text
            if status in {"failed", "expired", "cancelled"}:
                detail = self._format_batch_failure(batch=batch)
                raise RuntimeError(f"Batch {batch_id} ended with status={status}. {detail}")
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for batch {batch_id}")
            time.sleep(max(1, poll_seconds))

    def fetch_batch_output(self, *, batch_id: str) -> str:
        batch = self._client.batches.retrieve(batch_id)
        if batch.status != "completed":
            detail = self._format_batch_failure(batch=batch)
            raise RuntimeError(
                f"Batch {batch_id} is not completed (status={batch.status}). {detail}"
            )
        if not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} completed without output_file_id")
        output = self._client.files.content(batch.output_file_id)
        return output.text

    def _format_batch_failure(self, *, batch: Any) -> str:
        parts: list[str] = []
        errors = getattr(batch, "errors", None)
        if errors:
            try:
                data = getattr(errors, "data", None) or []
                if data:
                    first = data[0]
                    code = first.get("code")
                    message = first.get("message")
                    param = first.get("param")
                    if code:
                        parts.append(f"code={code}")
                    if param:
                        parts.append(f"param={param}")
                    if message:
                        parts.append(f"message={message}")
            except Exception:  # noqa: BLE001
                parts.append(f"errors={errors}")

        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            parts.append(f"error_file_id={error_file_id}")
            try:
                error_file = self._client.files.content(error_file_id)
                lines = [line for line in error_file.text.splitlines() if line.strip()]
                if lines:
                    first_payload = json.loads(lines[0])
                    err = first_payload.get("error") or {}
                    msg = err.get("message") or str(err)
                    if msg:
                        parts.append(f"first_error={msg}")
            except Exception:  # noqa: BLE001
                parts.append("failed_to_read_error_file")

        if not parts:
            return "No additional error details returned by OpenAI batch API."
        return " ".join(parts)
