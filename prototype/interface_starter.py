import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ============================================================
# Physical configuration
# ============================================================

L_A = 0.01
L_B = 0.02

RHO_A = 7800.0
RHO_B = 2700.0

CP_A = 500.0
CP_B = 900.0

N_A = 5
N_B = 5

DX_A = L_A / N_A
DX_B = L_B / N_B

TOTAL_CELLS = N_A + N_B


# ============================================================
# Experiments
# ============================================================

EXPERIMENTS = {
    "heating": {
        "initial_temperature": 20.0,
        "left_temperature": 100.0,
        "right_temperature": 20.0,
    },
    "cooling": {
        "initial_temperature": 100.0,
        "left_temperature": 20.0,
        "right_temperature": 20.0,
    },
    "moderate_heating": {
        "initial_temperature": 20.0,
        "left_temperature": 80.0,
        "right_temperature": 20.0,
    },
}


# ============================================================
# Sensor layout
# ============================================================

SENSOR_NAMES = [
    "sensor_a_bulk",
    "sensor_a_interface",
    "sensor_b_interface",
    "sensor_b_bulk",
]

SENSOR_INDICES = [
    1,
    N_A - 1,
    N_A,
    N_A + 2,
]


# ============================================================
# Data loading
# ============================================================

DATA_DIRECTORY = Path(
    "prototype/interface_data"
)


def load_dataset(experiment_name):
    """
    Load one generated experimental dataset.
    """

    path = (
        DATA_DIRECTORY
        / f"{experiment_name}.csv"
    )

    times = []
    measurements = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            times.append(
                float(row["time"])
            )

            measurements.append(
                [
                    float(
                        row[sensor_name]
                    )
                    for sensor_name in SENSOR_NAMES
                ]
            )

    return (
        np.asarray(times),
        np.asarray(measurements).T,
    )


# ============================================================
# STARTER MODEL
# ============================================================

def simulate_perfect_contact(
    k_a,
    k_b,
    experiment,
    sample_times,
):
    """
    Starter model.

    IMPORTANT:
    This model assumes perfect thermal contact.

    There is no interface-resistance parameter.
    """

    initial_temperature = experiment[
        "initial_temperature"
    ]

    left_temperature = experiment[
        "left_temperature"
    ]

    right_temperature = experiment[
        "right_temperature"
    ]

    initial_state = np.full(
        TOTAL_CELLS,
        initial_temperature,
        dtype=float,
    )

    conductance_a = k_a / DX_A
    conductance_b = k_b / DX_B

    def rhs(t, temperatures):
        dTdt = np.zeros(
            TOTAL_CELLS
        )

        # ----------------------------------------------------
        # Material A
        # ----------------------------------------------------

        heat_left = conductance_a * (
            left_temperature
            - temperatures[0]
        )

        heat_right = conductance_a * (
            temperatures[0]
            - temperatures[1]
        )

        dTdt[0] = (
            heat_left - heat_right
        ) / (
            RHO_A * CP_A * DX_A
        )

        for i in range(1, N_A - 1):
            heat_in = conductance_a * (
                temperatures[i - 1]
                - temperatures[i]
            )

            heat_out = conductance_a * (
                temperatures[i]
                - temperatures[i + 1]
            )

            dTdt[i] = (
                heat_in - heat_out
            ) / (
                RHO_A * CP_A * DX_A
            )

        # ----------------------------------------------------
        # Perfect A/B contact
        # ----------------------------------------------------

        interface_a = N_A - 1
        interface_b = N_A

        # There is no additional resistance here.
        #
        # The starter model treats the interface as an
        # ordinary conductive connection.

        heat_in_a = conductance_a * (
            temperatures[interface_a - 1]
            - temperatures[interface_a]
        )

        heat_interface = conductance_a * (
            temperatures[interface_a]
            - temperatures[interface_b]
        )

        dTdt[interface_a] = (
            heat_in_a - heat_interface
        ) / (
            RHO_A * CP_A * DX_A
        )

        heat_interface = conductance_b * (
            temperatures[interface_a]
            - temperatures[interface_b]
        )

        heat_out_b = conductance_b * (
            temperatures[interface_b]
            - temperatures[interface_b + 1]
        )

        dTdt[interface_b] = (
            heat_interface - heat_out_b
        ) / (
            RHO_B * CP_B * DX_B
        )

        # ----------------------------------------------------
        # Material B
        # ----------------------------------------------------

        for i in range(
            N_A + 1,
            TOTAL_CELLS - 1,
        ):
            heat_in = conductance_b * (
                temperatures[i - 1]
                - temperatures[i]
            )

            heat_out = conductance_b * (
                temperatures[i]
                - temperatures[i + 1]
            )

            dTdt[i] = (
                heat_in - heat_out
            ) / (
                RHO_B * CP_B * DX_B
            )

        # ----------------------------------------------------
        # Right boundary
        # ----------------------------------------------------

        last = TOTAL_CELLS - 1

        heat_in = conductance_b * (
            temperatures[last - 1]
            - temperatures[last]
        )

        heat_right = conductance_b * (
            temperatures[last]
            - right_temperature
        )

        dTdt[last] = (
            heat_in - heat_right
        ) / (
            RHO_B * CP_B * DX_B
        )

        return dTdt

    solution = solve_ivp(
        rhs,
        (
            sample_times[0],
            sample_times[-1],
        ),
        initial_state,
        t_eval=sample_times,
        method="BDF",
        rtol=1e-7,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(
            solution.message
        )

    return solution.y


# ============================================================
# Extract starter-model sensors
# ============================================================

def extract_sensors(
    temperatures,
):
    return temperatures[
        SENSOR_INDICES,
        :,
    ]


# ============================================================
# Parameter fitting
# ============================================================

def residuals(
    log_parameters,
    datasets,
):
    """
    Fit k_A and k_B.

    Log parameters keep both conductivities positive.
    """

    k_a = np.exp(
        log_parameters[0]
    )

    k_b = np.exp(
        log_parameters[1]
    )

    all_residuals = []

    for (
        experiment_name,
        experiment,
    ) in EXPERIMENTS.items():

        times, observed = (
            datasets[experiment_name]
        )

        predicted_full = (
            simulate_perfect_contact(
                k_a=k_a,
                k_b=k_b,
                experiment=experiment,
                sample_times=times,
            )
        )

        predicted = extract_sensors(
            predicted_full
        )

        error = (
            predicted - observed
        )

        all_residuals.extend(
            error.ravel()
        )

    return np.asarray(
        all_residuals
    )


def fit_model(datasets):
    """
    Fit the perfect-contact starter model.
    """

    initial_guess = np.log(
        [
            10.0,
            5.0,
        ]
    )

    result = least_squares(
        residuals,
        initial_guess,
        args=(datasets,),
        bounds=(
            np.log(
                [
                    0.5,
                    0.5,
                ]
            ),
            np.log(
                [
                    50.0,
                    50.0,
                ]
            ),
        ),
        max_nfev=100,
    )

    k_a = np.exp(
        result.x[0]
    )

    k_b = np.exp(
        result.x[1]
    )

    return (
        k_a,
        k_b,
        result,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    predicted,
    observed,
):
    error = (
        predicted - observed
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    max_error = np.max(
        np.abs(error)
    )

    return (
        rmse,
        max_error,
    )


# ============================================================
# Main diagnostics
# ============================================================

def main():
    print("=" * 70)
    print(
        "INTERFACE-AWARE STARTER MODEL"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    datasets = {}

    for experiment_name in EXPERIMENTS:
        datasets[
            experiment_name
        ] = load_dataset(
            experiment_name
        )

    print("\nLoaded experiments:")

    for experiment_name in datasets:
        times, measurements = (
            datasets[
                experiment_name
            ]
        )

        print(
            f"  {experiment_name}: "
            f"{len(times)} time points, "
            f"{measurements.shape[0]} sensors"
        )

    # --------------------------------------------------------
    # Fit model
    # --------------------------------------------------------

    print(
        "\nFitting perfect-contact starter model..."
    )

    k_a, k_b, result = fit_model(
        datasets
    )

    print("\nRecovered parameters:")
    print(
        f"  k_A = {k_a:.6f}"
    )
    print(
        f"  k_B = {k_b:.6f}"
    )

    print(
        f"\nOptimizer success: "
        f"{result.success}"
    )

    print(
        f"Optimizer cost: "
        f"{result.cost:.8f}"
    )

    # --------------------------------------------------------
    # Per-experiment diagnostics
    # --------------------------------------------------------

    all_errors = []

    print(
        "\nPer-experiment diagnostics:"
    )
    print("-" * 70)

    for (
        experiment_name,
        experiment,
    ) in EXPERIMENTS.items():

        times, observed = (
            datasets[
                experiment_name
            ]
        )

        predicted_full = (
            simulate_perfect_contact(
                k_a=k_a,
                k_b=k_b,
                experiment=experiment,
                sample_times=times,
            )
        )

        predicted = extract_sensors(
            predicted_full
        )

        error = (
            predicted - observed
        )

        all_errors.extend(
            error.ravel()
        )

        print(
            f"\n{experiment_name}"
        )

        for sensor_index in range(
            len(SENSOR_NAMES)
        ):
            rmse, max_error = (
                calculate_metrics(
                    predicted[
                        sensor_index
                    ],
                    observed[
                        sensor_index
                    ],
                )
            )

            print(
                f"  {SENSOR_NAMES[sensor_index]}:"
            )

            print(
                f"    RMSE      = "
                f"{rmse:.6f} °C"
            )

            print(
                f"    Max error = "
                f"{max_error:.6f} °C"
            )

        # ----------------------------------------------------
        # Interface jump
        # ----------------------------------------------------

        measured_jump = (
            observed[1]
            - observed[2]
        )

        predicted_jump = (
            predicted[1]
            - predicted[2]
        )

        measured_jump_max = np.max(
            np.abs(
                measured_jump
            )
        )

        predicted_jump_max = np.max(
            np.abs(
                predicted_jump
            )
        )

        print(
            "\n  Interface temperature jump:"
        )

        print(
            f"    Measured maximum = "
            f"{measured_jump_max:.6f} °C"
        )

        print(
            f"    Model maximum    = "
            f"{predicted_jump_max:.6f} °C"
        )

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    all_errors = np.asarray(
        all_errors
    )

    overall_rmse = np.sqrt(
        np.mean(
            all_errors ** 2
        )
    )

    overall_max_error = np.max(
        np.abs(
            all_errors
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "OVERALL STARTER-MODEL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOverall RMSE = "
        f"{overall_rmse:.6f} °C"
    )

    print(
        f"Overall max error = "
        f"{overall_max_error:.6f} °C"
    )

    print(
        "\nStarter-model assumption:"
    )

    print(
        "  Perfect thermal contact"
    )

    print(
        "\nExperimental system:"
    )

    print(
        "  Imperfect thermal contact"
    )


if __name__ == "__main__":
    main()