from __future__ import annotations

import os
import re
import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = Path(
    os.environ.get(
        "TASK_WORKSPACE",
        str(TASK_ROOT / "workspace"),
    )
).resolve()

REPORT_CANDIDATES = (
    WORKSPACE / "report.md",
    WORKSPACE / "scientific_report.md",
    WORKSPACE / "results.md",
)


def _load_report() -> str:
    for path in REPORT_CANDIDATES:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return text

    raise AssertionError(
        "No scientific report was found. Expected one of: "
        + ", ".join(path.name for path in REPORT_CANDIDATES)
    )


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def test_scientific_report_exists_and_has_substantive_content():
    report = _load_report()

    assert len(report.strip()) >= 1000, (
        "Scientific report is too short to document the investigation "
        "and validation."
    )


def test_report_documents_physical_model_and_parameters():
    report = _normalized(_load_report())

    physical_terms = (
        "thermal",
        "heat",
        "conduct",
        "interface",
    )

    assert all(term in report for term in physical_terms), (
        "Report must explain the relevant thermal/physical model."
    )

    parameter_terms = (
        "parameter",
        "calibrat",
        "fit",
    )

    assert any(term in report for term in parameter_terms), (
        "Report must document the parameter calibration procedure."
    )


def test_report_documents_visible_validation():
    report = _normalized(_load_report())

    experiment_names = (
        "heating",
        "cooling",
        "moderate",
    )

    for experiment in experiment_names:
        assert experiment in report, (
            f"Report does not document the {experiment} experiment."
        )

    metric_terms = (
        "rmse",
        "mean absolute",
        "mae",
        "max error",
    )

    assert any(term in report for term in metric_terms), (
        "Report must include quantitative validation metrics."
    )


def test_report_documents_numerical_and_reproducibility_details():
    report = _normalized(_load_report())

    numerical_terms = (
        "numerical",
        "solver",
        "integration",
        "stability",
    )

    assert any(term in report for term in numerical_terms), (
        "Report must discuss relevant numerical considerations."
    )

    reproducibility_terms = (
        "reproduc",
        "run",
        "command",
        "environment",
    )

    assert any(term in report for term in reproducibility_terms), (
        "Report must provide enough information to reproduce the results."
    )


def test_report_includes_units_for_physical_quantities():
    report = _normalized(_load_report())

    unit_terms = (
        "°c",
        "deg c",
        "celsius",
        "w/m",
        "w m",
        "k",
    )

    assert any(term in report for term in unit_terms), (
        "Report should include units for relevant physical quantities."
    )