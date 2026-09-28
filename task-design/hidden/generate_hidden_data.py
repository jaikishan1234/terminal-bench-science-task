from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermal_model import Experiment, ThermalModel


TRUE_CONDUCTIVITY_A = 12.0
TRUE_CONDUCTIVITY_B = 4.5

HIDDEN_EXPERIMENT = Experiment(
    name="hidden_validation",
    initial_temperature=20.0,
    left_temperature=65.0,
    right_temperature=10.0,
)

SAMPLE_TIMES = np.linspace(0.0, 2000.0, 101)

NOISE_SEED = 42
NOISE_STD = 0.05


def generate_hidden_dataset() -> pd.DataFrame:
    """Generate deterministic hidden validation measurements."""

    model = ThermalModel()

    temperature_field = model.simulate(
        conductivity_a=TRUE_CONDUCTIVITY_A,
        conductivity_b=TRUE_CONDUCTIVITY_B,
        experiment=HIDDEN_EXPERIMENT,
        sample_times=SAMPLE_TIMES,
    )

    sensors = model.extract_sensors(temperature_field)

    rng = np.random.default_rng(NOISE_SEED)

    noisy_sensors = sensors + rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=sensors.shape,
    )

    return pd.DataFrame(
        {
            "time": SAMPLE_TIMES,
            "sensor_a_bulk": noisy_sensors[:, 0],
            "sensor_a_interface": noisy_sensors[:, 1],
            "sensor_b_interface": noisy_sensors[:, 2],
            "sensor_b_bulk": noisy_sensors[:, 3],
        }
    )


def main() -> None:
    """Generate the hidden validation dataset."""

    output_path = (
        PROJECT_ROOT
        / "task-design"
        / "hidden"
        / "hidden_validation.csv"
    )

    dataframe = generate_hidden_dataset()

    dataframe.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    print(f"Generated: {output_path}")
    print(f"Samples: {len(dataframe)}")
    print(f"Noise seed: {NOISE_SEED}")
    print(f"Noise standard deviation: {NOISE_STD}")


if __name__ == "__main__":
    main()