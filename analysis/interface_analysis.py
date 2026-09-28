from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import VISIBLE_EXPERIMENTS, fit_visible_data
from src.data import load_datasets
from src.thermal_model import ThermalModel


DATA_DIRECTORY = PROJECT_ROOT / "data"


def interface_difference(values: np.ndarray) -> np.ndarray:
    """Return the temperature difference across the material interface."""

    return values[:, 1] - values[:, 2]


def summarize_difference(
    measured: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Print numerical comparison of measured and predicted interface jumps."""

    measured_jump = interface_difference(measured)
    predicted_jump = interface_difference(predicted)

    difference_error = predicted_jump - measured_jump

    print(
        f"Measured maximum jump:  "
        f"{np.max(measured_jump):10.4f} °C"
    )

    print(
        f"Predicted maximum jump: "
        f"{np.max(predicted_jump):10.4f} °C"
    )

    print(
        f"Measured final jump:     "
        f"{measured_jump[-1]:10.4f} °C"
    )

    print(
        f"Predicted final jump:    "
        f"{predicted_jump[-1]:10.4f} °C"
    )

    print(
        f"Maximum absolute jump error:"
        f"{np.max(np.abs(difference_error)):10.4f} °C"
    )

    print(
        f"Mean absolute jump error:   "
        f"{np.mean(np.abs(difference_error)):10.4f} °C"
    )


def main() -> None:
    """Analyze interface behavior across all visible experiments."""

    datasets = load_datasets(DATA_DIRECTORY)

    calibration = fit_visible_data(datasets)

    model = ThermalModel()

    print("Thermal interface analysis")
    print("=" * 70)

    print("\nCalibrated starter model")
    print("-" * 70)
    print(f"k_A: {calibration.conductivity_a:.6f}")
    print(f"k_B: {calibration.conductivity_b:.6f}")
    print(f"Cost: {calibration.cost:.6f}")

    for name, dataset in datasets.items():
        experiment = VISIBLE_EXPERIMENTS[name]

        temperature_field = model.simulate(
            conductivity_a=calibration.conductivity_a,
            conductivity_b=calibration.conductivity_b,
            experiment=experiment,
            sample_times=dataset.time,
        )

        predicted = model.extract_sensors(temperature_field)

        print("\n" + "=" * 70)
        print(f"Experiment: {name}")
        print("=" * 70)

        summarize_difference(
            measured=dataset.temperatures,
            predicted=predicted,
        )

        measured_jump = interface_difference(
            dataset.temperatures
        )

        predicted_jump = interface_difference(
            predicted
        )

        print("\nInterface jump at selected times:")

        sample_indices = np.linspace(
            0,
            len(dataset.time) - 1,
            5,
            dtype=int,
        )

        for index in sample_indices:
            print(
                f"  t={dataset.time[index]:7.1f} s | "
                f"measured={measured_jump[index]:9.4f} °C | "
                f"model={predicted_jump[index]:9.4f} °C"
            )


if __name__ == "__main__":
    main()