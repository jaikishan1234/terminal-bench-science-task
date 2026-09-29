from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.data import ExperimentalDataset
from src.thermal_model import Experiment, ThermalModel


@dataclass(frozen=True)
class CalibrationResult:
    """Result of fitting the starter thermal model."""

    conductivity_a: float
    conductivity_b: float
    cost: float
    success: bool
    message: str


VISIBLE_EXPERIMENTS = {
    "heating": Experiment(
        name="heating",
        initial_temperature=20.0,
        left_temperature=100.0,
        right_temperature=20.0,
    ),
    "cooling": Experiment(
        name="cooling",
        initial_temperature=100.0,
        left_temperature=20.0,
        right_temperature=20.0,
    ),
    "moderate_heating": Experiment(
        name="moderate_heating",
        initial_temperature=20.0,
        left_temperature=80.0,
        right_temperature=20.0,
    ),
}


class ThermalCalibrator:
    """
    Calibrate the starter thermal model against experimental data.

    The starter model estimates only the thermal conductivities of
    materials A and B.
    """

    def __init__(self, model: ThermalModel | None = None):
        self.model = model or ThermalModel()

    def _predict(
        self,
        parameters: np.ndarray,
        dataset: ExperimentalDataset,
    ) -> np.ndarray:
        """Generate sensor predictions for one experiment."""

        conductivity_a = float(parameters[0])
        conductivity_b = float(parameters[1])

        if dataset.name not in VISIBLE_EXPERIMENTS:
            raise ValueError(
                f"Unknown visible experiment: {dataset.name}"
            )

        experiment = VISIBLE_EXPERIMENTS[dataset.name]

        temperature_field = self.model.simulate(
            conductivity_a=conductivity_a,
            conductivity_b=conductivity_b,
            experiment=experiment,
            sample_times=dataset.time,
        )

        return self.model.extract_sensors(temperature_field)

    def _residuals(
        self,
        log_parameters: np.ndarray,
        datasets: dict[str, ExperimentalDataset],
    ) -> np.ndarray:
        """
        Calculate normalized residuals across all experiments.

        Parameters are optimized in log-space so that thermal
        conductivities remain positive.
        """

        parameters = np.exp(log_parameters)

        residuals: list[np.ndarray] = []

        for name in VISIBLE_EXPERIMENTS:
            if name not in datasets:
                raise ValueError(
                    f"Missing required experiment: {name}"
                )

            dataset = datasets[name]

            predicted = self._predict(
                parameters,
                dataset,
            )

            residuals.append(
                (predicted - dataset.temperatures).ravel()
            )

        return np.concatenate(residuals)

    def fit(
        self,
        datasets: dict[str, ExperimentalDataset],
        initial_guess: tuple[float, float] = (10.0, 5.0),
    ) -> CalibrationResult:
        """
        Fit k_A and k_B to the supplied visible experiments.
        """

        if not datasets:
            raise ValueError("At least one dataset is required.")

        initial_guess_array = np.asarray(
            initial_guess,
            dtype=float,
        )

        if initial_guess_array.shape != (2,):
            raise ValueError(
                "initial_guess must contain exactly two values."
            )

        if np.any(initial_guess_array <= 0):
            raise ValueError(
                "Initial conductivity values must be positive."
            )

        result = least_squares(
            self._residuals,
            x0=np.log(initial_guess_array),
            args=(datasets,),
            bounds=(
                np.log([0.5, 0.5]),
                np.log([50.0, 50.0]),
            ),
            method="trf",
        )

        fitted_parameters = np.exp(result.x)

        return CalibrationResult(
            conductivity_a=float(fitted_parameters[0]),
            conductivity_b=float(fitted_parameters[1]),
            cost=float(result.cost),
            success=bool(result.success),
            message=str(result.message),
        )


def fit_visible_data(
    datasets: dict[str, ExperimentalDataset],
) -> CalibrationResult:
    """Convenience function for calibrating the visible experiments."""

    calibrator = ThermalCalibrator()

    return calibrator.fit(datasets)