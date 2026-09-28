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
# Sensor configuration
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
# Data
# ============================================================

DATA_DIRECTORY = Path(
    "prototype/interface_data"
)


def load_dataset(experiment_name):
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
                        row[sensor]
                    )
                    for sensor in SENSOR_NAMES
                ]
            )

    return (
        np.asarray(times),
        np.asarray(measurements).T,
    )


# ============================================================
# Interface-aware physical model
# ============================================================

def simulate(
    k_a,
    k_b,
    r_interface,
    experiment,
    sample_times,
):
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
    conductance_interface = (
        1.0 / r_interface
    )

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
        # A-side interface cell
        # ----------------------------------------------------

        interface_a = N_A - 1
        interface_b = N_A

        heat_in_a = conductance_a * (
            temperatures[interface_a - 1]
            - temperatures[interface_a]
        )

        heat_interface = (
            conductance_interface
            * (
                temperatures[interface_a]
                - temperatures[interface_b]
            )
        )

        dTdt[interface_a] = (
            heat_in_a - heat_interface
        ) / (
            RHO_A * CP_A * DX_A
        )

        # ----------------------------------------------------
        # B-side interface cell
        # ----------------------------------------------------

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
# Sensor extraction
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
    k_a = np.exp(
        log_parameters[0]
    )

    k_b = np.exp(
        log_parameters[1]
    )

    r_interface = np.exp(
        log_parameters[2]
    )

    all_residuals = []

    for (
        experiment_name,
        experiment,
    ) in EXPERIMENTS.items():

        times, observed = (
            datasets[
                experiment_name
            ]
        )

        predicted_full = simulate(
            k_a=k_a,
            k_b=k_b,
            r_interface=r_interface,
            experiment=experiment,
            sample_times=times,
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


def fit_parameters(datasets):
    initial_guess = np.log(
        [
            10.0,
            5.0,
            0.003,
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
                    0.0001,
                ]
            ),
            np.log(
                [
                    50.0,
                    50.0,
                    0.02,
                ]
            ),
        ),
        max_nfev=200,
    )

    parameters = np.exp(
        result.x
    )

    return (
        parameters,
        result,
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("INTERFACE-AWARE PARAMETER RECOVERY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load measurements
    # --------------------------------------------------------

    datasets = {}

    for experiment_name in EXPERIMENTS:
        datasets[
            experiment_name
        ] = load_dataset(
            experiment_name
        )

    print("\nLoaded experimental data.")

    # --------------------------------------------------------
    # Fit physical parameters
    # --------------------------------------------------------

    print(
        "\nFitting:"
    )

    print(
        "  k_A"
    )

    print(
        "  k_B"
    )

    print(
        "  R_interface"
    )

    parameters, result = (
        fit_parameters(
            datasets
        )
    )

    k_a = parameters[0]
    k_b = parameters[1]
    r_interface = parameters[2]

    print("\nRecovered parameters:")
    print(
        f"  k_A = "
        f"{k_a:.8f}"
    )

    print(
        f"  k_B = "
        f"{k_b:.8f}"
    )

    print(
        f"  R_interface = "
        f"{r_interface:.8f}"
    )

    print(
        f"\nOptimizer success: "
        f"{result.success}"
    )

    print(
        f"Optimizer cost: "
        f"{result.cost:.10f}"
    )

    # --------------------------------------------------------
    # Compare against known reference values
    # --------------------------------------------------------

    true_parameters = np.array(
        [
            12.0,
            4.5,
            0.003,
        ]
    )

    recovered_error = (
        parameters
        - true_parameters
    )

    print(
        "\nParameter recovery error:"
    )

    print(
        f"  k_A error = "
        f"{recovered_error[0]:.8f}"
    )

    print(
        f"  k_B error = "
        f"{recovered_error[1]:.8f}"
    )

    print(
        f"  R error   = "
        f"{recovered_error[2]:.8f}"
    )

    # --------------------------------------------------------
    # Validate every visible experiment
    # --------------------------------------------------------

    print(
        "\nVisible experiment validation:"
    )

    print(
        "-" * 70
    )

    all_errors = []

    for (
        experiment_name,
        experiment,
    ) in EXPERIMENTS.items():

        times, observed = (
            datasets[
                experiment_name
            ]
        )

        predicted_full = simulate(
            k_a=k_a,
            k_b=k_b,
            r_interface=r_interface,
            experiment=experiment,
            sample_times=times,
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

        rmse = np.sqrt(
            np.mean(
                error ** 2
            )
        )

        max_error = np.max(
            np.abs(error)
        )

        print(
            f"\n{experiment_name}"
        )

        print(
            f"  RMSE      = "
            f"{rmse:.8f} °C"
        )

        print(
            f"  Max error = "
            f"{max_error:.8f} °C"
        )

        measured_jump = (
            observed[1]
            - observed[2]
        )

        predicted_jump = (
            predicted[1]
            - predicted[2]
        )

        print(
            f"  Measured max interface jump = "
            f"{np.max(np.abs(measured_jump)):.8f} °C"
        )

        print(
            f"  Predicted max interface jump = "
            f"{np.max(np.abs(predicted_jump)):.8f} °C"
        )

    # --------------------------------------------------------
    # Overall visible-data error
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
        "OVERALL VISIBLE-DATA RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRMSE = "
        f"{overall_rmse:.8f} °C"
    )

    print(
        f"Max error = "
        f"{overall_max_error:.8f} °C"
    )


if __name__ == "__main__":
    main()