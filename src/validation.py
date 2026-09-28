from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationMetrics:
    """Error metrics for one model prediction."""

    rmse: float
    max_error: float
    mean_absolute_error: float


def calculate_metrics(
    predicted: np.ndarray,
    measured: np.ndarray,
) -> ValidationMetrics:
    """
    Calculate numerical error metrics between predictions and measurements.

    Both arrays must have the same shape.
    """

    predicted = np.asarray(predicted, dtype=float)
    measured = np.asarray(measured, dtype=float)

    if predicted.shape != measured.shape:
        raise ValueError(
            "Predicted and measured arrays must have the same shape."
        )

    if predicted.size == 0:
        raise ValueError(
            "Predicted and measured arrays cannot be empty."
        )

    if not np.all(np.isfinite(predicted)):
        raise ValueError(
            "Predicted values contain non-finite values."
        )

    if not np.all(np.isfinite(measured)):
        raise ValueError(
            "Measured values contain non-finite values."
        )

    error = predicted - measured

    rmse = float(np.sqrt(np.mean(error**2)))
    max_error = float(np.max(np.abs(error)))
    mean_absolute_error = float(np.mean(np.abs(error)))

    return ValidationMetrics(
        rmse=rmse,
        max_error=max_error,
        mean_absolute_error=mean_absolute_error,
    )


def calculate_sensor_metrics(
    predicted: np.ndarray,
    measured: np.ndarray,
    sensor_names: tuple[str, ...],
) -> dict[str, ValidationMetrics]:
    """
    Calculate metrics independently for every sensor.
    """

    predicted = np.asarray(predicted, dtype=float)
    measured = np.asarray(measured, dtype=float)

    if predicted.ndim != 2:
        raise ValueError(
            "Predicted data must have shape (samples, sensors)."
        )

    if measured.shape != predicted.shape:
        raise ValueError(
            "Predicted and measured arrays must have the same shape."
        )

    if len(sensor_names) != predicted.shape[1]:
        raise ValueError(
            "Number of sensor names must match the number of sensors."
        )

    metrics: dict[str, ValidationMetrics] = {}

    for index, sensor_name in enumerate(sensor_names):
        metrics[sensor_name] = calculate_metrics(
            predicted[:, index],
            measured[:, index],
        )

    return metrics


def compare_predictions(
    predicted: np.ndarray,
    measured: np.ndarray,
) -> dict[str, float]:
    """
    Return a compact summary of prediction quality.
    """

    metrics = calculate_metrics(
        predicted,
        measured,
    )

    return {
        "rmse": metrics.rmse,
        "max_error": metrics.max_error,
        "mean_absolute_error": metrics.mean_absolute_error,
    }