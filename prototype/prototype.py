import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ============================================================
# 1. PHYSICAL SYSTEM
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


# ============================================================
# 2. THERMAL MODEL
# ============================================================

def simulate(
    k_a,
    k_b,
    r_interface,
    experiment,
    sample_times,
):
    initial_temperature = experiment["initial_temperature"]
    left_temperature = experiment["left_temperature"]
    right_temperature = experiment["right_temperature"]

    n = N_A + N_B

    initial_state = np.full(n, initial_temperature)

    conductance_a = k_a / DX_A
    conductance_b = k_b / DX_B
    conductance_interface = 1.0 / r_interface

    def rhs(t, temperatures):
        dTdt = np.zeros(n)

        # Left boundary
        heat_left = conductance_a * (
            left_temperature - temperatures[0]
        )

        dTdt[0] += heat_left / (RHO_A * CP_A * DX_A)

        # Material A
        for i in range(1, N_A):
            heat_in = conductance_a * (
                temperatures[i - 1] - temperatures[i]
            )

            heat_out = conductance_a * (
                temperatures[i] - temperatures[i + 1]
            )

            net_heat = heat_in - heat_out

            dTdt[i] += net_heat / (
                RHO_A * CP_A * DX_A
            )

        # Interface
        interface_a = N_A - 1
        interface_b = N_A

        heat_interface = conductance_interface * (
            temperatures[interface_a]
            - temperatures[interface_b]
        )

        dTdt[interface_a] -= (
            heat_interface
            / (RHO_A * CP_A * DX_A)
        )

        dTdt[interface_b] += (
            heat_interface
            / (RHO_B * CP_B * DX_B)
        )

        # Material B
        for i in range(N_A + 1, n - 1):
            heat_in = conductance_b * (
                temperatures[i - 1] - temperatures[i]
            )

            heat_out = conductance_b * (
                temperatures[i] - temperatures[i + 1]
            )

            net_heat = heat_in - heat_out

            dTdt[i] += net_heat / (
                RHO_B * CP_B * DX_B
            )

        # Right boundary
        last = n - 1

        heat_right = conductance_b * (
            temperatures[last] - right_temperature
        )

        dTdt[last] -= (
            heat_right
            / (RHO_B * CP_B * DX_B)
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
        raise RuntimeError(solution.message)

    temperatures = solution.y

    sensor_1 = temperatures[1]
    sensor_2 = temperatures[N_A - 1]
    sensor_3 = temperatures[N_A + 2]

    return np.column_stack(
        [
            sensor_1,
            sensor_2,
            sensor_3,
        ]
    )


# ============================================================
# 3. VISIBLE CALIBRATION EXPERIMENTS
# ============================================================

calibration_experiments = [
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
# 4. HIDDEN VALIDATION EXPERIMENT
# ============================================================

hidden_experiment = {
    "name": "hidden_boundary_condition",
    "initial_temperature": 20.0,
    "left_temperature": 65.0,
    "right_temperature": 10.0,
}


# ============================================================
# 5. GROUND-TRUTH PARAMETERS
# ============================================================

TRUE_PARAMETERS = np.array(
    [
        12.0,
        4.5,
        0.003,
    ]
)


# ============================================================
# 6. GENERATE VISIBLE EXPERIMENTS
# ============================================================

def generate_calibration_data():
    rng = np.random.default_rng(42)

    datasets = []

    sample_times = np.linspace(
        0.0,
        2000.0,
        101,
    )

    for experiment in calibration_experiments:

        clean = simulate(
            TRUE_PARAMETERS[0],
            TRUE_PARAMETERS[1],
            TRUE_PARAMETERS[2],
            experiment,
            sample_times,
        )

        noise = rng.normal(
            loc=0.0,
            scale=0.05,
            size=clean.shape,
        )

        measured = clean + noise

        datasets.append(
            {
                "experiment": experiment,
                "times": sample_times,
                "measurements": measured,
            }
        )

    return datasets


# ============================================================
# 7. PARAMETER FITTING
# ============================================================

def residuals(parameters, datasets):
    k_a, k_b, r_interface = parameters

    all_residuals = []

    for dataset in datasets:

        predicted = simulate(
            k_a,
            k_b,
            r_interface,
            dataset["experiment"],
            dataset["times"],
        )

        error = (
            predicted
            - dataset["measurements"]
        )

        all_residuals.extend(
            error.ravel()
        )

    return np.asarray(all_residuals)


def fit_parameters(
    datasets,
    initial_guess,
):
    result = least_squares(
        residuals,
        x0=initial_guess,
        args=(datasets,),
        bounds=(
            [
                0.5,
                0.5,
                0.0001,
            ],
            [
                50.0,
                50.0,
                0.02,
            ],
        ),
        verbose=0,
    )

    return result


# ============================================================
# 8. HIDDEN EXPERIMENT VALIDATION
# ============================================================

def validate_hidden_experiment(
    fitted_parameters,
):
    sample_times = np.linspace(
        0.0,
        2000.0,
        101,
    )

    # The verifier would have access to this clean
    # reference data. The agent would NOT.
    reference = simulate(
        TRUE_PARAMETERS[0],
        TRUE_PARAMETERS[1],
        TRUE_PARAMETERS[2],
        hidden_experiment,
        sample_times,
    )

    prediction = simulate(
        fitted_parameters[0],
        fitted_parameters[1],
        fitted_parameters[2],
        hidden_experiment,
        sample_times,
    )

    error = prediction - reference

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    max_error = np.max(
        np.abs(error)
    )

    return rmse, max_error


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "THERMAL PARAMETER IDENTIFICATION PROTOTYPE"
    )
    print("=" * 60)

    print("\nTrue parameters:")
    print(
        f"  k_A              = "
        f"{TRUE_PARAMETERS[0]}"
    )
    print(
        f"  k_B              = "
        f"{TRUE_PARAMETERS[1]}"
    )
    print(
        f"  R_interface      = "
        f"{TRUE_PARAMETERS[2]}"
    )

    datasets = generate_calibration_data()

    print("\nVisible calibration experiments:")

    for dataset in datasets:
        print(
            f"  {dataset['experiment']['name']}: "
            f"{len(dataset['times'])} time points"
        )

    initial_guesses = [
        [5.0, 2.0, 0.010],
        [20.0, 10.0, 0.001],
        [8.0, 8.0, 0.005],
    ]

    recovered_parameters = []

    print("\nParameter recovery:")
    print("-" * 60)

    for index, guess in enumerate(
        initial_guesses,
        start=1,
    ):

        result = fit_parameters(
            datasets,
            np.array(
                guess,
                dtype=float,
            ),
        )

        recovered = result.x

        recovered_parameters.append(
            recovered
        )

        print(f"\nRun {index}")

        print("  Initial guess:")
        print(
            f"    k_A = {guess[0]}"
        )
        print(
            f"    k_B = {guess[1]}"
        )
        print(
            f"    R   = {guess[2]}"
        )

        print("\n  Recovered:")

        print(
            f"    k_A = "
            f"{recovered[0]:.6f}"
        )

        print(
            f"    k_B = "
            f"{recovered[1]:.6f}"
        )

        print(
            f"    R   = "
            f"{recovered[2]:.6f}"
        )

        print(
            f"\n  Cost = "
            f"{result.cost:.8f}"
        )

        print(
            f"  Success = "
            f"{result.success}"
        )

    # Use the first recovered parameter set
    # for hidden validation.
    fitted = recovered_parameters[0]

    rmse, max_error = validate_hidden_experiment(
        fitted
    )

    print("\n")
    print("=" * 60)
    print("HIDDEN EXPERIMENT VALIDATION")
    print("=" * 60)

    print(
        f"\nRMSE      = {rmse:.8f} °C"
    )

    print(
        f"Max error = {max_error:.8f} °C"
    )

    print("\nValidation completed.")


if __name__ == "__main__":
    main()