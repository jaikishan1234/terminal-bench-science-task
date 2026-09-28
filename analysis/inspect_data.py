from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_datasets


DATA_DIRECTORY = PROJECT_ROOT / "data"


def summarize_experiment(name: str, dataset) -> None:
    """Print basic statistics for one experimental dataset."""

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"{'=' * 60}")

    print(f"Samples: {dataset.number_of_samples}")
    print(
        f"Time range: "
        f"{dataset.time[0]:.2f} -> {dataset.time[-1]:.2f} s"
    )

    print("\nSensor statistics:")

    for index, sensor_name in enumerate(dataset.sensor_names):
        values = dataset.temperatures[:, index]

        print(
            f"  {sensor_name:20s} "
            f"min={np.min(values):8.3f} °C  "
            f"max={np.max(values):8.3f} °C  "
            f"final={values[-1]:8.3f} °C"
        )

    interface_jump = (
        dataset.temperatures[:, 1]
        - dataset.temperatures[:, 2]
    )

    print("\nInterface temperature difference:")
    print(f"  Maximum: {np.max(interface_jump):.3f} °C")
    print(f"  Minimum: {np.min(interface_jump):.3f} °C")
    print(f"  Final:   {interface_jump[-1]:.3f} °C")

    print("\nData quality:")

    finite = np.isfinite(dataset.temperatures)

    print(f"  All values finite: {bool(np.all(finite))}")
    print(f"  Temperature mean: {np.mean(dataset.temperatures):.3f} °C")
    print(f"  Temperature std:  {np.std(dataset.temperatures):.3f} °C")


def main() -> None:
    """Inspect all visible experimental datasets."""

    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {DATA_DIRECTORY}"
        )

    datasets = load_datasets(DATA_DIRECTORY)

    print("Thermal experiment inspection")
    print(f"Data directory: {DATA_DIRECTORY}")

    for name, dataset in datasets.items():
        summarize_experiment(name, dataset)


if __name__ == "__main__":
    main()