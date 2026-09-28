import numpy as np
import matplotlib.pyplot as plt

from prototype import (
    TRUE_PARAMETERS,
    calibration_experiments,
    simulate,
)


def main():
    print("=" * 70)
    print("THERMAL INTERFACE TEMPERATURE DIAGNOSTIC")
    print("=" * 70)

    sample_times = np.linspace(
        0.0,
        2000.0,
        101,
    )

    experiment = calibration_experiments[0]

    print(f"\nExperiment: {experiment['name']}")

    temperatures = simulate(
        TRUE_PARAMETERS[0],
        TRUE_PARAMETERS[1],
        TRUE_PARAMETERS[2],
        experiment,
        sample_times,
    )

    # The current prototype exposes:
    #
    # sensor 1 -> Material A
    # sensor 2 -> near interface
    # sensor 3 -> Material B
    #
    # For this diagnostic we directly inspect the two cells
    # adjacent to the physical interface.

    interface_a_index = 4
    interface_b_index = 5

    temperature_a = temperatures[:, 0] if False else None

    # Re-run the physical state calculation through a small
    # helper that exposes the interface temperatures.
    #
    # The current simulate() function intentionally returns
    # only the three sensor measurements, so this diagnostic
    # cannot directly access both interface cells yet.

    print("\nCurrent prototype limitation:")
    print(
        "simulate() returns only the three configured sensors."
    )

    print(
        "We need to expose the two cells immediately adjacent "
        "to the interface."
    )

    print("\nNext model change:")
    print("  A-side interface temperature")
    print("  B-side interface temperature")
    print("  Interface temperature difference")

    print("\nThis diagnostic is intentionally incomplete.")
    print(
        "It confirms that the model API needs to expose "
        "the interface state before we redesign the data."
    )


if __name__ == "__main__":
    main()