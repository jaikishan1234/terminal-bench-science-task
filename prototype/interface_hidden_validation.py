import numpy as np

from interface_solution import (
    fit_parameters,
    load_dataset,
    simulate,
)


TRUE_PARAMETERS = np.array(
    [
        12.0,
        4.5,
        0.003,
    ],
    dtype=float,
)


VISIBLE_DATASETS = {
    "heating": load_dataset("heating"),
    "cooling": load_dataset("cooling"),
    "moderate_heating": load_dataset("moderate_heating"),
}


HIDDEN_EXPERIMENT = {
    "name": "hidden_validation",
    "initial_temperature": 20.0,
    "left_temperature": 65.0,
    "right_temperature": 10.0,
}


SECOND_HIDDEN_EXPERIMENT = {
    "name": "hidden_reversed_gradient",
    "initial_temperature": 60.0,
    "left_temperature": 30.0,
    "right_temperature": 90.0,
}


SENSOR_NAMES = [
    "Material A interior",
    "Material A interface",
    "Material B interface",
    "Material B interior",
]


def calculate_metrics(predicted, reference):
    error = predicted - reference

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    max_error = np.max(
        np.abs(error)
    )

    return rmse, max_error


def validate_experiment(
    fitted_parameters,
    experiment,
):
    sample_times = np.linspace(
        0.0,
        2000.0,
        101,
    )

    reference = simulate(
        TRUE_PARAMETERS[0],
        TRUE_PARAMETERS[1],
        TRUE_PARAMETERS[2],
        experiment,
        sample_times,
    )

    prediction = simulate(
        fitted_parameters[0],
        fitted_parameters[1],
        fitted_parameters[2],
        experiment,
        sample_times,
    )

    error = prediction - reference

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    max_error = np.max(
        np.abs(error)
    )

    return (
        reference,
        prediction,
        rmse,
        max_error,
    )


def print_sensor_metrics(
    prediction,
    reference,
):
    print("\nPer-sensor error:")

    sensor_count = min(
        len(SENSOR_NAMES),
        prediction.shape[1],
    )

    for sensor_index in range(
        sensor_count
    ):
        sensor_error = (
            prediction[:, sensor_index]
            - reference[:, sensor_index]
        )

        sensor_rmse = np.sqrt(
            np.mean(sensor_error ** 2)
        )

        sensor_max = np.max(
            np.abs(sensor_error)
        )

        print(
            f"  {SENSOR_NAMES[sensor_index]}: "
            f"RMSE={sensor_rmse:.8f} °C, "
            f"Max={sensor_max:.8f} °C"
        )


def print_experiment_info(experiment):
    print(
        f"\nInitial temperature = "
        f"{experiment['initial_temperature']} °C"
    )

    print(
        f"Left boundary = "
        f"{experiment['left_temperature']} °C"
    )

    print(
        f"Right boundary = "
        f"{experiment['right_temperature']} °C"
    )


def main():
    print("=" * 60)
    print("HIDDEN EXPERIMENT VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Fit using visible calibration data only
    # --------------------------------------------------------

    fitted_parameters, result = fit_parameters(
        VISIBLE_DATASETS,
    )

    print("\nParameters recovered from visible data:")

    print(
        f"  k_A          = "
        f"{fitted_parameters[0]:.8f}"
    )

    print(
        f"  k_B          = "
        f"{fitted_parameters[1]:.8f}"
    )

    print(
        f"  R_interface  = "
        f"{fitted_parameters[2]:.8f}"
    )

    print(
        f"\nOptimizer success = "
        f"{result.success}"
    )

    print(
        f"Optimizer cost    = "
        f"{result.cost:.8f}"
    )

    # --------------------------------------------------------
    # Hidden experiment #1
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("HIDDEN EXPERIMENT #1")
    print("=" * 60)

    print_experiment_info(
        HIDDEN_EXPERIMENT
    )

    (
        reference,
        prediction,
        rmse,
        max_error,
    ) = validate_experiment(
        fitted_parameters,
        HIDDEN_EXPERIMENT,
    )

    print("\nPrediction error:")

    print(
        f"  RMSE      = "
        f"{rmse:.8f} °C"
    )

    print(
        f"  Max error = "
        f"{max_error:.8f} °C"
    )

    print_sensor_metrics(
        prediction,
        reference,
    )

    # --------------------------------------------------------
    # Hidden experiment #2
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("HIDDEN EXPERIMENT #2")
    print("=" * 60)

    print_experiment_info(
        SECOND_HIDDEN_EXPERIMENT
    )

    (
        reference,
        prediction,
        rmse,
        max_error,
    ) = validate_experiment(
        fitted_parameters,
        SECOND_HIDDEN_EXPERIMENT,
    )

    print("\nPrediction error:")

    print(
        f"  RMSE      = "
        f"{rmse:.8f} °C"
    )

    print(
        f"  Max error = "
        f"{max_error:.8f} °C"
    )

    print_sensor_metrics(
        prediction,
        reference,
    )

    print("\n")
    print("=" * 60)
    print("ALL HIDDEN VALIDATIONS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()