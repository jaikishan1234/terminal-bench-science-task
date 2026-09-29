from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


TASK_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = TASK_ROOT / "workspace"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.calibration import ThermalCalibrator
from src.data import load_datasets


INITIAL_GUESSES = (
    (2.0, 2.0),
    (5.0, 10.0),
    (10.0, 5.0),
    (20.0, 15.0),
    (40.0, 30.0),
)

MIN_CONDUCTIVITY = 0.0
MAX_CONDUCTIVITY = 100.0

MAX_RELATIVE_SPREAD = 0.10


def test_calibration_is_finite_and_positive():
    datasets = load_datasets(WORKSPACE / "data")
    calibrator = ThermalCalibrator()

    results = [
        calibrator.fit(datasets, initial_guess=guess)
        for guess in INITIAL_GUESSES
    ]

    for result in results:
        assert result.success, (
            f"Calibration failed from one of the initial guesses: "
            f"{result.message}"
        )

        assert np.isfinite(result.conductivity_a)
        assert np.isfinite(result.conductivity_b)
        assert np.isfinite(result.cost)

        assert (
            MIN_CONDUCTIVITY
            < result.conductivity_a
            < MAX_CONDUCTIVITY
        ), (
            "Calibrated conductivity_a must be finite and physically "
            "positive."
        )

        assert (
            MIN_CONDUCTIVITY
            < result.conductivity_b
            < MAX_CONDUCTIVITY
        ), (
            "Calibrated conductivity_b must be finite and physically "
            "positive."
        )

        assert result.cost >= 0.0
        assert np.isfinite(result.cost)


def test_calibration_is_stable_across_initial_guesses():
    datasets = load_datasets(WORKSPACE / "data")
    calibrator = ThermalCalibrator()

    results = [
        calibrator.fit(datasets, initial_guess=guess)
        for guess in INITIAL_GUESSES
    ]

    conductivity_a = np.array(
        [result.conductivity_a for result in results],
        dtype=float,
    )

    conductivity_b = np.array(
        [result.conductivity_b for result in results],
        dtype=float,
    )

    def relative_spread(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        return float((np.max(values) - np.min(values)) / mean)

    spread_a = relative_spread(conductivity_a)
    spread_b = relative_spread(conductivity_b)

    assert spread_a <= MAX_RELATIVE_SPREAD, (
        f"conductivity_a is too sensitive to the initial guess: "
        f"relative spread={spread_a:.4f}"
    )

    assert spread_b <= MAX_RELATIVE_SPREAD, (
        f"conductivity_b is too sensitive to the initial guess: "
        f"relative spread={spread_b:.4f}"
    )