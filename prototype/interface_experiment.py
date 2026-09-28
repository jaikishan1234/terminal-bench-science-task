import csv
from pathlib import Path

import numpy as np

from interface_model import (
    N_A,
    simulate_full,
)


# ============================================================
# Experiment configuration
# ============================================================

EXPERIMENTS = [
    {
        "name": "heating",
        "initial_temperature": 20.0,
        "left_temperature": 100.0,
        "right_temperature": 20.0,
    },
    {
        "name": "cooling",
        "initial_temperature": 100.0,
        "left_temperature": 20.0,
        "right_temperature": 20.0,
    },
    {
        "name": "moderate_heating",
        "initial_temperature": 20.0,
        "left_temperature": 80.0,
        "right_temperature": 20.0,
    },
]


# ============================================================
# Reference parameters
# ============================================================

TRUE_K_A = 12.0
TRUE_K_B = 4.5
TRUE_R_INTERFACE = 0.003


# ============================================================
# Sampling
# ============================================================

SAMPLE_TIMES = np.linspace(
    0.0,
    2000.0,
    101,
)


# ============================================================
# Sensor locations
# ============================================================

# Full temperature field:
#
# A0 A1 A2 A3 A4 | B5 B6 B7 B8 B9
#                 ^
#              interface
#
# Sensors:
#
# Sensor 1 -> inside Material A
# Sensor 2 -> A-side interface cell
# Sensor 3 -> B-side interface cell
# Sensor 4 -> inside Material B

SENSOR_INDICES = [
    1,
    N_A - 1,
    N_A,
    N_A + 2,
]


SENSOR_NAMES = [
    "sensor_a_bulk",
    "sensor_a_interface",
    "sensor_b_interface",
    "sensor_b_bulk",
]


# ============================================================
# Data generation
# ============================================================

def generate_experiment(experiment, rng):
    """
    Generate noisy measurements for one experiment.
    """

    temperatures = simulate_full(
        k_a=TRUE_K_A,
        k_b=TRUE_K_B,
        r_interface=TRUE_R_INTERFACE,
        initial_temperature=experiment[
            "initial_temperature"
        ],
        left_temperature=experiment[
            "left_temperature"
        ],
        right_temperature=experiment[
            "right_temperature"
        ],
        sample_times=SAMPLE_TIMES,
    )

    clean_measurements = temperatures[
        SENSOR_INDICES,
        :
    ]

    noise = rng.normal(
        loc=0.0,
        scale=0.05,
        size=clean_measurements.shape,
    )

    noisy_measurements = (
        clean_measurements + noise
    )

    return clean_measurements, noisy_measurements


# ============================================================
# CSV writer
# ============================================================

def write_csv(
    output_path,
    experiment,
    measurements,
):
    """
    Write one experiment to CSV.
    """

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "time",
                *SENSOR_NAMES,
            ]
        )

        for index, time in enumerate(
            SAMPLE_TIMES
        ):
            writer.writerow(
                [
                    f"{time:.6f}",
                    *[
                        f"{measurements[sensor, index]:.8f}"
                        for sensor in range(
                            len(SENSOR_INDICES)
                        )
                    ],
                ]
            )


# ============================================================
# Diagnostics
# ============================================================

def print_interface_statistics(
    experiment,
    clean_measurements,
):
    """
    Print the temperature difference across the interface.
    """

    a_interface = clean_measurements[1]
    b_interface = clean_measurements[2]

    temperature_jump = (
        a_interface - b_interface
    )

    print(
        f"\n{experiment['name']}"
    )
    print("-" * 70)

    print(
        f"Maximum A-side interface temperature: "
        f"{np.max(a_interface):.6f} °C"
    )

    print(
        f"Maximum B-side interface temperature: "
        f"{np.max(b_interface):.6f} °C"
    )

    print(
        f"Maximum interface temperature jump: "
        f"{np.max(temperature_jump):.6f} °C"
    )

    print(
        f"Final interface temperature jump: "
        f"{temperature_jump[-1]:.6f} °C"
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("INTERFACE-AWARE EXPERIMENT DESIGN")
    print("=" * 70)

    print("\nSensor layout:")
    print("  Sensor 1 -> Material A bulk")
    print("  Sensor 2 -> Material A interface side")
    print("  Sensor 3 -> Material B interface side")
    print("  Sensor 4 -> Material B bulk")

    print("\nReference parameters:")
    print(f"  k_A = {TRUE_K_A}")
    print(f"  k_B = {TRUE_K_B}")
    print(
        f"  R_interface = "
        f"{TRUE_R_INTERFACE}"
    )

    output_directory = Path(
        "prototype/interface_data"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(42)

    for experiment in EXPERIMENTS:
        clean, noisy = generate_experiment(
            experiment,
            rng,
        )

        print_interface_statistics(
            experiment,
            clean,
        )

        output_path = (
            output_directory
            / f"{experiment['name']}.csv"
        )

        write_csv(
            output_path,
            experiment,
            noisy,
        )

        print(
            f"Saved: {output_path}"
        )

    print("\n" + "=" * 70)
    print("EXPERIMENT GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()