from __future__ import annotations

import json
from pathlib import Path

from persona_vectors.types import ActivationSample, PersonaVectorBundle, TraitArtifacts


def save_run_bundle(
    *,
    output_dir: str | Path,
    artifacts: TraitArtifacts,
    bundle: PersonaVectorBundle,
    samples: list[ActivationSample],
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "artifacts.json").write_text(json.dumps(artifacts.to_dict(), indent=2) + "\n")
    (target / "persona_vector_bundle.json").write_text(json.dumps(bundle.to_dict(), indent=2) + "\n")
    with (target / "samples.jsonl").open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict()) + "\n")
