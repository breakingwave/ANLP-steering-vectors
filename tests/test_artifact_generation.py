from persona_vectors.artifact_generation import LLMTraitArtifactGenerator, _parse_json_payload
from persona_vectors.types import TraitDefinition


class FakeClient:
    model_name = "fake-generator"

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        assert "instruction pairs" in prompt
        return """```json
        {
          "instruction": [
            {"pos": "show the trait", "neg": "avoid the trait"}
          ],
          "questions": ["q1", "q2", "q3", "q4"],
          "eval prompt": "Question: {question}\nAnswer: {answer}"
        }
        ```"""


def test_parse_json_payload_handles_fenced_json() -> None:
    payload = _parse_json_payload("```json\n{\"value\": 1}\n```")
    assert payload == {"value": 1}


def test_generate_trait_artifacts_splits_questions_evenly() -> None:
    generator = LLMTraitArtifactGenerator(FakeClient())
    artifacts = generator.generate(TraitDefinition(name="evil", description="bad"))
    assert artifacts.source_model == "fake-generator"
    assert len(artifacts.instruction_pairs) == 1
    assert artifacts.extraction_questions == ["q1", "q2"]
    assert artifacts.evaluation_questions == ["q3", "q4"]
    assert "{question}" in artifacts.evaluation_prompt
    assert "{answer}" in artifacts.evaluation_prompt
