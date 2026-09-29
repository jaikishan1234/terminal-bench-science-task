from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TASK_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = Path(
    os.environ.get(
        "TASK_WORKSPACE",
        str(TASK_ROOT / "workspace"),
    )
).resolve()

# Evaluator-only data. This file must never be copied into workspace/.
HIDDEN_DATA = (
    TASK_ROOT.parent.parent
    / "task-design"
    / "hidden"
    / "reference_hidden_validation.csv"
)

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.calibration import fit_visible_data
from src.data import load_datasets
from src.thermal_model import Experiment, ThermalModel
from src.validation import calculate_metrics


MAX_HIDDEN_RMSE = 0.5
MAX_HIDDEN_INTERFACE_JUMP_RMSE = 0.5


def _load_hidden_data():
    assert HIDDEN_DATA.exists(), (
        f"Evaluator hidden data is missing: {HIDDEN_DATA}"
    )

    dataframe = pd.read_csv(HIDDEN_DATA)

    required_columns = (
        "time",
        "sensor_a_bulk",
        "sensor_a_interface",
        "sensor_b_interface",
        "sensor_b_bulk",
    )

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    assert not missing, (
        "Hidden evaluator data is missing columns: "
        + ", ".join(missing)
    )

    values = dataframe.loc[:, required_columns].to_numpy(dtype=float)

    assert np.all(np.isfinite(values)), (
        "Hidden evaluator data contains non-finite values."
    )

    return values[:, 0], values[:, 1:]


def _interface_jump_rmse(predicted, measured):
    predicted_jump = predicted[:, 1] - predicted[:, 2]
    measured_jump = measured[:, 1] - measured[:, 2]

    error = predicted_jump - measured_jump

    return float(np.sqrt(np.mean(error**2)))


def test_model_generalizes_to_unseen_boundary_conditions():
    datasets = load_datasets(WORKSPACE / "data")

    result = fit_visible_data(datasets)

    assert result.success, (
        f"Visible-data calibration failed: {result.message}"
    )

    assert np.isfinite(result.conductivity_a)
    assert np.isfinite(result.conductivity_b)

    hidden_time, hidden_measured = _load_hidden_data()

    hidden_experiment = Experiment(
        name="hidden_generalization",
        initial_temperature=20.0,
        left_temperature=65.0,
        right_temperature=10.0,
    )

    model = ThermalModel()

    temperature_field = model.simulate(
        conductivity_a=result.conductivity_a,
        conductivity_b=result.conductivity_b,
        experiment=hidden_experiment,
        sample_times=hidden_time,
    )

    predicted = model.extract_sensors(temperature_field)

    assert predicted.shape == hidden_measured.shape
    assert np.all(np.isfinite(predicted)), (
        "Hidden-condition simulation produced non-finite predictions."
    )

    metrics = calculate_metrics(predicted, hidden_measured)

    assert metrics.rmse <= MAX_HIDDEN_RMSE, (
        f"Hidden-condition RMSE {metrics.rmse:.4f} °C exceeds "
        f"{MAX_HIDDEN_RMSE:.4f} °C."
    )

    jump_rmse = _interface_jump_rmse(
        predicted,
        hidden_measured,
    )

    assert jump_rmse <= MAX_HIDDEN_INTERFACE_JUMP_RMSE, (
        f"Hidden interface-jump RMSE {jump_rmse:.4f} °C exceeds "
        f"{MAX_HIDDEN_INTERFACE_JUMP_RMSE:.4f} °C."
    )