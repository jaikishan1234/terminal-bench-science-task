from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "time",
    "sensor_a_bulk",
    "sensor_a_interface",
    "sensor_b_interface",
    "sensor_b_bulk",
)

SENSOR_COLUMNS = (
    "sensor_a_bulk",
    "sensor_a_interface",
    "sensor_b_interface",
    "sensor_b_bulk",
)


@dataclass(frozen=True)
class ExperimentalDataset:
    """Temperature measurements from one thermal experiment."""

    name: str
    time: np.ndarray
    temperatures: np.ndarray

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return SENSOR_COLUMNS

    @property
    def number_of_samples(self) -> int:
        return len(self.time)


def load_dataset(
    path: str | Path,
    name: str | None = None,
) -> ExperimentalDataset:
    """Load one experimental dataset from CSV."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    dataframe = dataframe.loc[:, REQUIRED_COLUMNS].copy()

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    values = dataframe.to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Dataset contains non-finite values: {path}")

    time = values[:, 0]
    temperatures = values[:, 1:]

    if np.any(np.diff(time) < 0):
        raise ValueError(f"Time values must be sorted: {path}")

    if len(time) < 2:
        raise ValueError("Dataset must contain at least two samples.")

    return ExperimentalDataset(
        name=name or path.stem,
        time=time,
        temperatures=temperatures,
    )


def load_datasets(
    data_directory: str | Path,
) -> dict[str, ExperimentalDataset]:
    """Load all visible calibration experiments."""

    data_directory = Path(data_directory)

    experiments = (
        "heating",
        "cooling",
        "moderate_heating",
    )

    datasets: dict[str, ExperimentalDataset] = {}

    for experiment in experiments:
        path = data_directory / f"{experiment}.csv"

        datasets[experiment] = load_dataset(
            path,
            name=experiment,
        )

    return datasets