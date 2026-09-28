from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import ThermalCalibrator
from src.data import load_datasets


DATA_DIRECTORY = PROJECT_ROOT / "data"

INITIAL_GUESSES = (
    (2.0, 2.0),
    (5.0, 10.0),
    (10.0, 5.0),
    (20.0, 15.0),
    (40.0, 30.0),
)


def main() -> None:
    """Evaluate starter calibration from multiple initial guesses."""

    datasets = load_datasets(DATA_DIRECTORY)

    calibrator = ThermalCalibrator()

    print("Starter calibration sensitivity analysis")
    print("=" * 70)

    print("\nInitial guess -> fitted parameters\n")

    results = []

    for initial_guess in INITIAL_GUESSES:
        result = calibrator.fit(
            datasets,
            initial_guess=initial_guess,
        )

        results.append(result)

        print(
            f"Initial: "
            f"k_A={initial_guess[0]:6.2f}, "
            f"k_B={initial_guess[1]:6.2f}"
        )

        print(
            f"Fitted:  "
            f"k_A={result.conductivity_a:10.6f}, "
            f"k_B={result.conductivity_b:10.6f}"
        )

        print(f"Cost:    {result.cost:12.6f}")
        print(f"Success: {result.success}")
        print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    successful_results = [
        result
        for result in results
        if result.success
    ]

    print(
        f"Successful calibrations: "
        f"{len(successful_results)}/{len(results)}"
    )

    if successful_results:
        conductivity_a_values = [
            result.conductivity_a
            for result in successful_results
        ]

        conductivity_b_values = [
            result.conductivity_b
            for result in successful_results
        ]

        costs = [
            result.cost
            for result in successful_results
        ]

        print(
            f"k_A range: "
            f"{min(conductivity_a_values):.6f} "
            f"-> "
            f"{max(conductivity_a_values):.6f}"
        )

        print(
            f"k_B range: "
            f"{min(conductivity_b_values):.6f} "
            f"-> "
            f"{max(conductivity_b_values):.6f}"
        )

        print(
            f"Cost range: "
            f"{min(costs):.6f} "
            f"-> "
            f"{max(costs):.6f}"
        )


if __name__ == "__main__":
    main()