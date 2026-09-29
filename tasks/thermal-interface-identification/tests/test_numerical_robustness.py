from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


TASK_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = Path(
    os.environ.get(
        "TASK_WORKSPACE",
        str(TASK_ROOT / "workspace"),
    )
).resolve()

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.calibration import VISIBLE_EXPERIMENTS, fit_visible_data
from src.data import load_datasets
from src.thermal_model import ThermalModel


def test_visible_simulations_are_finite():
    datasets = load_datasets(WORKSPACE / "data")
    result = fit_visible_data(datasets)

    assert result.success

    model = ThermalModel()

    for experiment_name, experiment in VISIBLE_EXPERIMENTS.items():
        dataset = datasets[experiment_name]

        temperature_field = model.simulate(
            conductivity_a=result.conductivity_a,
            conductivity_b=result.conductivity_b,
            experiment=experiment,
            sample_times=dataset.time,
        )

        assert temperature_field.ndim == 2
        assert temperature_field.shape[0] == len(dataset.time)

        assert np.all(np.isfinite(temperature_field)), (
            f"{experiment_name} produced non-finite temperatures."
        )

        predicted = model.extract_sensors(temperature_field)

        assert predicted.shape == dataset.temperatures.shape
        assert np.all(np.isfinite(predicted)), (
            f"{experiment_name} produced non-finite sensor predictions."
        )


def test_simulated_temperatures_remain_within_reasonable_bounds():
    datasets = load_datasets(WORKSPACE / "data")
    result = fit_visible_data(datasets)

    assert result.success

    model = ThermalModel()

    for experiment_name, experiment in VISIBLE_EXPERIMENTS.items():
        dataset = datasets[experiment_name]

        temperature_field = model.simulate(
            conductivity_a=result.conductivity_a,
            conductivity_b=result.conductivity_b,
            experiment=experiment,
            sample_times=dataset.time,
        )

        minimum_temperature = float(np.min(temperature_field))
        maximum_temperature = float(np.max(temperature_field))

        boundary_minimum = min(
            experiment.initial_temperature,
            experiment.left_temperature,
            experiment.right_temperature,
        )

        boundary_maximum = max(
            experiment.initial_temperature,
            experiment.left_temperature,
            experiment.right_temperature,
        )

        tolerance = 5.0

        assert minimum_temperature >= boundary_minimum - tolerance, (
            f"{experiment_name} produced an unexpectedly low temperature: "
            f"{minimum_temperature:.4f} °C"
        )

        assert maximum_temperature <= boundary_maximum + tolerance, (
            f"{experiment_name} produced an unexpectedly high temperature: "
            f"{maximum_temperature:.4f} °C"
        )


def test_calibration_rejects_invalid_initial_conductivities():
    datasets = load_datasets(WORKSPACE / "data")

    from src.calibration import ThermalCalibrator

    calibrator = ThermalCalibrator()

    invalid_guesses = (
        (0.0, 5.0),
        (-1.0, 5.0),
        (5.0, 0.0),
        (5.0, -1.0),
    )

    for guess in invalid_guesses:
        try:
            calibrator.fit(datasets, initial_guess=guess)
        except ValueError:
            continue

        raise AssertionError(
            f"Calibration accepted invalid initial guess: {guess}"
        )