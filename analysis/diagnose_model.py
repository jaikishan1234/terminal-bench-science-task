from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import fit_visible_data, VISIBLE_EXPERIMENTS
from src.data import load_datasets
from src.validation import calculate_metrics
from src.thermal_model import ThermalModel


DATA_DIRECTORY = PROJECT_ROOT / "data"


def print_experiment_diagnostics(
    name: str,
    measured: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Print validation results for one experiment."""

    metrics = calculate_metrics(
        predicted,
        measured,
    )

    measured_jump = measured[:, 1] - measured[:, 2]
    predicted_jump = predicted[:, 1] - predicted[:, 2]

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"{'=' * 60}")

    print(f"Overall RMSE:     {metrics.rmse:.4f} °C")
    print(f"Maximum error:    {metrics.max_error:.4f} °C")
    print(f"Mean absolute error: {metrics.mean_absolute_error:.4f} °C")

    print("\nInterface temperature difference:")

    print(
        f"  Measured maximum:  "
        f"{np.max(measured_jump):.4f} °C"
    )

    print(
        f"  Model maximum:     "
        f"{np.max(predicted_jump):.4f} °C"
    )

    print(
        f"  Measured final:    "
        f"{measured_jump[-1]:.4f} °C"
    )

    print(
        f"  Model final:       "
        f"{predicted_jump[-1]:.4f} °C"
    )


def main() -> None:
    """Fit the starter model and diagnose its limitations."""

    datasets = load_datasets(DATA_DIRECTORY)

    print("Thermal model diagnostic")
    print(f"Data directory: {DATA_DIRECTORY}")

    result = fit_visible_data(datasets)

    print("\nStarter calibration result")
    print("--------------------------")
    print(f"k_A:      {result.conductivity_a:.6f}")
    print(f"k_B:      {result.conductivity_b:.6f}")
    print(f"Cost:     {result.cost:.6f}")
    print(f"Success:  {result.success}")
    print(f"Message:  {result.message}")

    model = ThermalModel()

    all_errors: list[np.ndarray] = []

    for name, dataset in datasets.items():
        experiment = VISIBLE_EXPERIMENTS[name]

        field = model.simulate(
            conductivity_a=result.conductivity_a,
            conductivity_b=result.conductivity_b,
            experiment=experiment,
            sample_times=dataset.time,
        )

        predicted = model.extract_sensors(field)

        print_experiment_diagnostics(
            name=name,
            measured=dataset.temperatures,
            predicted=predicted,
        )

        all_errors.append(
            predicted - dataset.temperatures
        )

    combined_error = np.concatenate(
        [error.ravel() for error in all_errors]
    )

    overall_rmse = float(
        np.sqrt(np.mean(combined_error**2))
    )

    overall_max_error = float(
        np.max(np.abs(combined_error))
    )

    print(f"\n{'=' * 60}")
    print("Combined validation")
    print(f"{'=' * 60}")
    print(f"Overall RMSE:  {overall_rmse:.4f} °C")
    print(f"Overall max:   {overall_max_error:.4f} °C")


if __name__ == "__main__":
    main()