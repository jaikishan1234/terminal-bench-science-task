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
from src.validation import calculate_metrics


MAX_OVERALL_RMSE = 0.5
MAX_EXPERIMENT_RMSE = 0.5
MAX_HEATING_INTERFACE_JUMP_RMSE = 0.5


def _calibrate():
    datasets = load_datasets(WORKSPACE / "data")
    result = fit_visible_data(datasets)

    assert result.success, (
        "Calibration did not converge successfully."
    )

    return datasets, result


def _predict_experiment(
    model: ThermalModel,
    result,
    experiment_name: str,
    dataset,
) -> np.ndarray:
    temperature_field = model.simulate(
        conductivity_a=result.conductivity_a,
        conductivity_b=result.conductivity_b,
        experiment=VISIBLE_EXPERIMENTS[experiment_name],
        sample_times=dataset.time,
    )

    return model.extract_sensors(temperature_field)


def _interface_jump_rmse(
    measured: np.ndarray,
    predicted: np.ndarray,
) -> float:
    measured_jump = (
        measured[:, 1]
        - measured[:, 2]
    )

    predicted_jump = (
        predicted[:, 1]
        - predicted[:, 2]
    )

    error = predicted_jump - measured_jump

    return float(np.sqrt(np.mean(error**2)))


def test_all_visible_experiments_fit_within_tolerance():
    datasets, result = _calibrate()

    model = ThermalModel()

    all_measured = []
    all_predicted = []

    for experiment_name in VISIBLE_EXPERIMENTS:
        dataset = datasets[experiment_name]

        predicted = _predict_experiment(
            model,
            result,
            experiment_name,
            dataset,
        )

        metrics = calculate_metrics(
            predicted,
            dataset.temperatures,
        )

        assert metrics.rmse <= MAX_EXPERIMENT_RMSE, (
            f"{experiment_name} RMSE "
            f"{metrics.rmse:.4f} °C exceeds "
            f"{MAX_EXPERIMENT_RMSE:.4f} °C."
        )

        all_measured.append(dataset.temperatures)
        all_predicted.append(predicted)

        if experiment_name == "heating":
            jump_rmse = _interface_jump_rmse(
                dataset.temperatures,
                predicted,
            )

            assert jump_rmse <= MAX_HEATING_INTERFACE_JUMP_RMSE, (
                f"Heating interface-jump RMSE "
                f"{jump_rmse:.4f} °C exceeds "
                f"{MAX_HEATING_INTERFACE_JUMP_RMSE:.4f} °C."
            )

    all_measured = np.concatenate(
        all_measured,
        axis=0,
    )

    all_predicted = np.concatenate(
        all_predicted,
        axis=0,
    )

    overall = calculate_metrics(
        all_predicted,
        all_measured,
    )

    assert overall.rmse <= MAX_OVERALL_RMSE, (
        f"Overall visible RMSE "
        f"{overall.rmse:.4f} °C exceeds "
        f"{MAX_OVERALL_RMSE:.4f} °C."
    )