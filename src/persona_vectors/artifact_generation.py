from __future__ import annotations

import json
from pathlib import Path

from persona_vectors.interfaces import TextCompletionClient
from persona_vectors.prompts import ARTIFACT_GENERATION_PROMPT
from persona_vectors.types import InstructionPair, TraitArtifacts, TraitDefinition


class LLMTraitArtifactGenerator:
    def __init__(self, client: TextCompletionClient) -> None:
        self._client = client

    def generate(self, trait: TraitDefinition) -> TraitArtifacts:
        prompt = ARTIFACT_GENERATION_PROMPT.replace("__TRAIT_NAME__", trait.name).replace(
            "__TRAIT_DESCRIPTION__", trait.description
        )
        raw = self._client.complete(prompt, temperature=0.0)
        payload = _parse_json_payload(raw)
        pairs = payload.get("instruction")
        questions = payload.get("questions")
        eval_prompt = payload.get("eval prompt") or payload.get("eval_prompt")
        if not isinstance(pairs, list) or len(pairs) < 1:
            raise ValueError("Artifact generator did not return any instruction pairs")
        if not isinstance(questions, list) or len(questions) < 2:
            raise ValueError("Artifact generator did not return enough questions")
        if not isinstance(eval_prompt, str) or "{question}" not in eval_prompt or "{answer}" not in eval_prompt:
            raise ValueError("Evaluation prompt must include {question} and {answer} placeholders")

        instruction_pairs = [
            InstructionPair(positive=item["pos"].strip(), negative=item["neg"].strip())
            for item in pairs
        ]
        midpoint = len(questions) // 2
        extraction_questions = [str(item).strip() for item in questions[:midpoint]]
        evaluation_questions = [str(item).strip() for item in questions[midpoint:]]
        return TraitArtifacts(
            trait=trait,
            instruction_pairs=instruction_pairs,
            extraction_questions=extraction_questions,
            evaluation_questions=evaluation_questions,
            evaluation_prompt=eval_prompt.strip(),
            source_model=self._client.model_name,
        )


def load_trait_artifacts(path: str | Path) -> TraitArtifacts:
    payload = json.loads(Path(path).read_text())
    trait_payload = payload.get("trait") or {}
    extraction_questions = payload.get("extraction_questions")
    evaluation_questions = payload.get("evaluation_questions")
    if extraction_questions is None or evaluation_questions is None:
        questions = payload.get("questions") or []
        midpoint = len(questions) // 2
        extraction_questions = questions[:midpoint]
        evaluation_questions = questions[midpoint:]
    return TraitArtifacts(
        trait=TraitDefinition(
            name=trait_payload.get("name") or payload.get("trait_name") or "trait",
            description=trait_payload.get("description") or payload.get("trait_description") or "",
        ),
        instruction_pairs=[
            InstructionPair(positive=item["pos"].strip(), negative=item["neg"].strip())
            for item in payload["instruction"]
        ],
        extraction_questions=[str(item).strip() for item in extraction_questions],
        evaluation_questions=[str(item).strip() for item in evaluation_questions],
        evaluation_prompt=(payload.get("eval_prompt") or payload.get("eval prompt") or "").strip(),
        source_model=payload.get("source_model"),
    )


def save_trait_artifacts(artifacts: TraitArtifacts, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifacts.to_dict(), indent=2) + "\n")


def _parse_json_payload(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("{"):
                text = candidate
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return json.loads(_escape_newlines_inside_strings(candidate))


def _escape_newlines_inside_strings(text: str) -> str:
    escaped: list[str] = []
    in_string = False
    backslash_count = 0
    for char in text:
        if char == '"' and backslash_count % 2 == 0:
            in_string = not in_string
        if char == "\n" and in_string:
            escaped.append("\\n")
            backslash_count = 0
            continue
        escaped.append(char)
        if char == "\\":
            backslash_count += 1
        else:
            backslash_count = 0
    return "".join(escaped)
