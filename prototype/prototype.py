import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ============================================================
# 1. PHYSICAL SYSTEM
# ============================================================

# Two-layer slab
L_A = 0.01          # Material A thickness: 10 mm
L_B = 0.02          # Material B thickness: 20 mm

RHO_A = 7800.0      # kg/m^3
RHO_B = 2700.0      # kg/m^3

CP_A = 500.0        # J/(kg K)
CP_B = 900.0        # J/(kg K)

# Number of temperature cells in each material
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
    """
    Simulate temperature evolution through the two-layer slab.

    Parameters
    ----------
    k_a : float
        Thermal conductivity of material A.

    k_b : float
        Thermal conductivity of material B.

    r_interface : float
        Thermal resistance at the A/B interface.

    experiment : dict
        Boundary and initial conditions.

    sample_times : np.ndarray
        Times at which temperatures are requested.

    Returns
    -------
    np.ndarray
        Temperature values at the requested sensor locations.
    """

    initial_temperature = experiment["initial_temperature"]
    left_temperature = experiment["left_temperature"]
    right_temperature = experiment["right_temperature"]

    n = N_A + N_B

    # Initial temperature of every cell
    initial_state = np.full(n, initial_temperature)

    # --------------------------------------------------------
    # Helper: calculate thermal conductance between cells
    # --------------------------------------------------------

    # Conductance inside material A
    conductance_a = k_a / DX_A

    # Conductance inside material B
    conductance_b = k_b / DX_B

    # Interface conductance
    conductance_interface = 1.0 / r_interface

    def rhs(t, temperatures):
        dTdt = np.zeros(n)

        # ----------------------------------------------------
        # Left boundary
        # ----------------------------------------------------

        heat_left = conductance_a * (
            left_temperature - temperatures[0]
        )

        dTdt[0] += heat_left / (RHO_A * CP_A * DX_A)

        # ----------------------------------------------------
        # Material A internal cells
        # ----------------------------------------------------

        for i in range(1, N_A):
            heat_in = conductance_a * (
                temperatures[i - 1] - temperatures[i]
            )

            heat_out = conductance_a * (
                temperatures[i] - temperatures[i + 1]
            )

            net_heat = heat_in - heat_out

            dTdt[i] += net_heat / (RHO_A * CP_A * DX_A)

        # ----------------------------------------------------
        # A/B interface
        # ----------------------------------------------------

        interface_a = N_A - 1
        interface_b = N_A

        heat_interface = conductance_interface * (
            temperatures[interface_a]
            - temperatures[interface_b]
        )

        dTdt[interface_a] -= (
            heat_interface / (RHO_A * CP_A * DX_A)
        )

        dTdt[interface_b] += (
            heat_interface / (RHO_B * CP_B * DX_B)
        )

        # ----------------------------------------------------
        # Material B internal cells
        # ----------------------------------------------------

        for i in range(N_A + 1, n - 1):
            heat_in = conductance_b * (
                temperatures[i - 1] - temperatures[i]
            )

            heat_out = conductance_b * (
                temperatures[i] - temperatures[i + 1]
            )

            net_heat = heat_in - heat_out

            dTdt[i] += net_heat / (RHO_B * CP_B * DX_B)

        # ----------------------------------------------------
        # Right boundary
        # ----------------------------------------------------

        last = n - 1

        heat_right = conductance_b * (
            temperatures[last] - right_temperature
        )

        dTdt[last] -= (
            heat_right / (RHO_B * CP_B * DX_B)
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

    # Three sensors:
    #
    # sensor_1 -> inside material A
    # sensor_2 -> near A/B interface
    # sensor_3 -> inside material B

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
# 3. EXPERIMENT DEFINITIONS
# ============================================================

experiments = [
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
# 4. GROUND-TRUTH PARAMETERS
# ============================================================

# These are SECRET values from the perspective of the
# eventual benchmark agent.
#
# For our prototype, we know them because we are generating
# the synthetic experimental data.

TRUE_PARAMETERS = np.array(
    [
        12.0,      # k_A
        4.5,       # k_B
        0.003,     # interface resistance
    ]
)


# ============================================================
# 5. GENERATE SYNTHETIC EXPERIMENTAL DATA
# ============================================================

def generate_experiments():
    rng = np.random.default_rng(42)

    generated = []

    sample_times = np.linspace(0.0, 2000.0, 101)

    for experiment in experiments:
        clean = simulate(
            TRUE_PARAMETERS[0],
            TRUE_PARAMETERS[1],
            TRUE_PARAMETERS[2],
            experiment,
            sample_times,
        )

        # Small measurement noise
        noise = rng.normal(
            loc=0.0,
            scale=0.05,
            size=clean.shape,
        )

        measured = clean + noise

        generated.append(
            {
                "experiment": experiment,
                "times": sample_times,
                "measurements": measured,
            }
        )

    return generated


# ============================================================
# 6. PARAMETER FITTING
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

        error = predicted - dataset["measurements"]

        all_residuals.extend(
            error.ravel()
        )

    return np.asarray(all_residuals)


def fit_parameters(datasets, initial_guess):
    result = least_squares(
        residuals,
        x0=initial_guess,
        args=(datasets,),
        bounds=(
            [
                0.5,       # minimum k_A
                0.5,       # minimum k_B
                0.0001,    # minimum interface resistance
            ],
            [
                50.0,      # maximum k_A
                50.0,      # maximum k_B
                0.02,      # maximum interface resistance
            ],
        ),
        verbose=0,
    )

    return result


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 60)
    print("THERMAL PARAMETER IDENTIFICATION PROTOTYPE")
    print("=" * 60)

    print("\nTrue parameters:")
    print(f"  k_A              = {TRUE_PARAMETERS[0]}")
    print(f"  k_B              = {TRUE_PARAMETERS[1]}")
    print(f"  R_interface      = {TRUE_PARAMETERS[2]}")

    datasets = generate_experiments()

    print("\nGenerated experiments:")
    for dataset in datasets:
        print(
            f"  {dataset['experiment']['name']}: "
            f"{len(dataset['times'])} time points"
        )

    # Several different starting guesses.
    initial_guesses = [
        [5.0, 2.0, 0.010],
        [20.0, 10.0, 0.001],
        [8.0, 8.0, 0.005],
    ]

    print("\nParameter recovery:")
    print("-" * 60)

    for index, guess in enumerate(initial_guesses, start=1):

        result = fit_parameters(
            datasets,
            np.array(guess, dtype=float),
        )

        recovered = result.x

        print(f"\nRun {index}")
        print(f"  Initial guess:")
        print(f"    k_A = {guess[0]}")
        print(f"    k_B = {guess[1]}")
        print(f"    R   = {guess[2]}")

        print("\n  Recovered:")
        print(f"    k_A = {recovered[0]:.6f}")
        print(f"    k_B = {recovered[1]:.6f}")
        print(f"    R   = {recovered[2]:.6f}")

        print(f"\n  Cost = {result.cost:.8f}")
        print(f"  Success = {result.success}")


if __name__ == "__main__":
    main()