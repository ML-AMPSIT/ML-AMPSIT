"""Deterministic output layout for analysis artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def safe_stem(value: object) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    ).strip("_") or "analysis"


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path
    run: Path
    figures: Path
    tables: Path
    models: Path
    reports: Path

    @classmethod
    def for_run(
        cls, config, *, name, model, sample_count, importance, transform
    ) -> "ArtifactLayout":
        root = Path(config["output_pathname"])
        run_name = "__".join((
            safe_stem(name),
            f"N{int(sample_count)}",
            safe_stem(model),
            safe_stem(importance),
            safe_stem(transform),
        ))
        run = root / "analysis_outputs" / run_name
        layout = cls(
            root=root,
            run=run,
            figures=run / "figures",
            tables=run / "tables",
            models=run / "models",
            reports=run / "reports",
        )
        layout.ensure()
        return layout

    def ensure(self) -> "ArtifactLayout":
        for path in (self.root, self.run, self.figures, self.tables, self.models, self.reports):
            path.mkdir(parents=True, exist_ok=True)
        return self


def comparison_directory(config) -> Path:
    path = Path(config["output_pathname"]) / "analysis_outputs" / "loop_comparisons"
    path.mkdir(parents=True, exist_ok=True)
    return path
