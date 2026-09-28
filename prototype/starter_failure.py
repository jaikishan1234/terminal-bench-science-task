import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ============================================================
# Physical configuration
# ============================================================

L_A = 0.01          # Material A thickness (m)
L_B = 0.02          # Material B thickness (m)

RHO_A = 7800.0      # Material A density (kg/m^3)
RHO_B = 2700.0      # Material B density (kg/m^3)

CP_A = 500.0        # Material A specific heat (J/kg/K)
CP_B = 900.0        # Material B specific heat (J/kg/K)

N_A = 5
N_B = 5

DX_A = L_A / N_A
DX_B = L_B / N_B

TOTAL_CELLS = N_A + N_B


# ============================================================
# Known physical parameters used to generate experiments
# ============================================================

TRUE_K_A = 12.0
TRUE_K_B = 4.5

# The real experiments contain imperfect thermal contact.
#
# IMPORTANT:
# The starter model below does NOT know about this parameter.
#
TRUE_R_INTERFACE = 0.003


# ============================================================
# Experiment definitions
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


TIME_POINTS = np.linspace(0.0, 2000.0, 101)

SENSOR_INDICES = [
    2,  # inside material A
    4,  # close to the interface
    7,  # inside material B
]


# ============================================================
# Conductance helpers
# ============================================================

def layer_conductance(k, dx):
    """
    Thermal conductance between neighboring cells.
    """
    return k / dx


def interface_conductance(k_a, k_b, r_interface):
    """
    Effective conductance between the last cell of A
    and the first cell of B.

    This is used only by the REAL physical model.
    """
    resistance_a = (dx_for_interface("A") / 2.0) / k_a
    resistance_b = (dx_for_interface("B") / 2.0) / k_b

    total_resistance = (
        resistance_a
        + r_interface
        + resistance_b
    )

    return 1.0 / total_resistance


def dx_for_interface(material):
    if material == "A":
        return DX_A
    return DX_B


# ============================================================
# REAL physical model
# ============================================================

def simulate_real_model(
    k_a,
    k_b,
    r_interface,
    initial_temperature,
    left_temperature,
    right_temperature,
):
    """
    Generates the experimental/reference data.

    This model includes interface resistance.
    """

    def rhs(t, temperatures):
        dTdt = np.zeros(TOTAL_CELLS)

        # ----------------------------------------------------
        # Material A
        # ----------------------------------------------------

        for i in range(N_A):
            temperature = temperatures[i]

            if i == 0:
                left_flux = (
                    layer_conductance(k_a, DX_A)
                    * (left_temperature - temperature)
                )
            else:
                left_flux = (
                    layer_conductance(k_a, DX_A)
                    * (temperatures[i - 1] - temperature)
                )

            if i == N_A - 1:
                # Real imperfect interface.
                right_flux = (
                    interface_conductance(
                        k_a,
                        k_b,
                        r_interface,
                    )
                    * (temperatures[N_A] - temperature)
                )
            else:
                right_flux = (
                    layer_conductance(k_a, DX_A)
                    * (temperatures[i + 1] - temperature)
                )

            heat_capacity = RHO_A * CP_A * DX_A

            dTdt[i] = (left_flux + right_flux) / heat_capacity

        # ----------------------------------------------------
        # Material B
        # ----------------------------------------------------

        for local_i in range(N_B):
            i = N_A + local_i

            temperature = temperatures[i]

            if local_i == 0:
                # Real imperfect interface.
                left_flux = (
                    interface_conductance(
                        k_a,
                        k_b,
                        r_interface,
                    )
                    * (temperatures[N_A - 1] - temperature)
                )
            else:
                left_flux = (
                    layer_conductance(k_b, DX_B)
                    * (temperatures[i - 1] - temperature)
                )

            if local_i == N_B - 1:
                right_flux = (
                    layer_conductance(k_b, DX_B)
                    * (right_temperature - temperature)
                )
            else:
                right_flux = (
                    layer_conductance(k_b, DX_B)
                    * (temperatures[i + 1] - temperature)
                )

            heat_capacity = RHO_B * CP_B * DX_B

            dTdt[i] = (left_flux + right_flux) / heat_capacity

        return dTdt

    initial_state = np.full(
        TOTAL_CELLS,
        initial_temperature,
        dtype=float,
    )

    solution = solve_ivp(
        rhs,
        (TIME_POINTS[0], TIME_POINTS[-1]),
        initial_state,
        t_eval=TIME_POINTS,
        method="BDF",
        rtol=1e-7,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y


# ============================================================
# STARTER MODEL
# ============================================================

def simulate_starter_model(
    k_a,
    k_b,
    initial_temperature,
    left_temperature,
    right_temperature,
):
    """
    Starter model supplied to the agent.

    IMPORTANT LIMITATION:

    It assumes perfect thermal contact between material A
    and material B.

    There is NO interface resistance parameter here.
    """

    def rhs(t, temperatures):
        dTdt = np.zeros(TOTAL_CELLS)

        # ----------------------------------------------------
        # Material A
        # ----------------------------------------------------

        for i in range(N_A):
            temperature = temperatures[i]

            if i == 0:
                left_flux = (
                    layer_conductance(k_a, DX_A)
                    * (left_temperature - temperature)
                )
            else:
                left_flux = (
                    layer_conductance(k_a, DX_A)
                    * (temperatures[i - 1] - temperature)
                )

            if i == N_A - 1:
                # ------------------------------------------------
                # STARTER MODEL ASSUMPTION:
                #
                # Perfect contact between A and B.
                # ------------------------------------------------
                right_flux = (
                    layer_conductance(k_a, DX_A)
                    * (temperatures[N_A] - temperature)
                )
            else:
                right_flux = (
                    layer_conductance(k_a, DX_A)
                    * (temperatures[i + 1] - temperature)
                )

            heat_capacity = RHO_A * CP_A * DX_A

            dTdt[i] = (left_flux + right_flux) / heat_capacity

        # ----------------------------------------------------
        # Material B
        # ----------------------------------------------------

        for local_i in range(N_B):
            i = N_A + local_i

            temperature = temperatures[i]

            if local_i == 0:
                # ------------------------------------------------
                # Perfect contact.
                #
                # No additional interface resistance.
                # ------------------------------------------------
                left_flux = (
                    layer_conductance(k_b, DX_B)
                    * (temperatures[N_A - 1] - temperature)
                )
            else:
                left_flux = (
                    layer_conductance(k_b, DX_B)
                    * (temperatures[i - 1] - temperature)
                )

            if local_i == N_B - 1:
                right_flux = (
                    layer_conductance(k_b, DX_B)
                    * (right_temperature - temperature)
                )
            else:
                right_flux = (
                    layer_conductance(k_b, DX_B)
                    * (temperatures[i + 1] - temperature)
                )

            heat_capacity = RHO_B * CP_B * DX_B

            dTdt[i] = (left_flux + right_flux) / heat_capacity

        return dTdt

    initial_state = np.full(
        TOTAL_CELLS,
        initial_temperature,
        dtype=float,
    )

    solution = solve_ivp(
        rhs,
        (TIME_POINTS[0], TIME_POINTS[-1]),
        initial_state,
        t_eval=TIME_POINTS,
        method="BDF",
        rtol=1e-7,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y


# ============================================================
# Synthetic experimental data
# ============================================================

def generate_experimental_data():
    """
    Generate measurements from the real physical model.

    The starter model does not get access to the true
    interface resistance.
    """

    rng = np.random.default_rng(42)

    datasets = {}

    for experiment in EXPERIMENTS:
        temperatures = simulate_real_model(
            k_a=TRUE_K_A,
            k_b=TRUE_K_B,
            r_interface=TRUE_R_INTERFACE,
            initial_temperature=experiment["initial_temperature"],
            left_temperature=experiment["left_temperature"],
            right_temperature=experiment["right_temperature"],
        )

        measurements = temperatures[SENSOR_INDICES, :].copy()

        noise = rng.normal(
            loc=0.0,
            scale=0.05,
            size=measurements.shape,
        )

        measurements += noise

        datasets[experiment["name"]] = measurements

    return datasets


# ============================================================
# Starter-model fitting
# ============================================================

def fit_starter_model(datasets):
    """
    Fit only k_A and k_B because the starter model has no
    interface-resistance parameter.
    """

    def residuals(log_parameters):
        k_a = np.exp(log_parameters[0])
        k_b = np.exp(log_parameters[1])

        all_residuals = []

        for experiment in EXPERIMENTS:
            predicted = simulate_starter_model(
                k_a=k_a,
                k_b=k_b,
                initial_temperature=experiment["initial_temperature"],
                left_temperature=experiment["left_temperature"],
                right_temperature=experiment["right_temperature"],
            )

            predicted_sensors = predicted[SENSOR_INDICES, :]

            observed = datasets[experiment["name"]]

            all_residuals.extend(
                (predicted_sensors - observed).ravel()
            )

        return np.asarray(all_residuals)

    initial_guess = np.array([
        np.log(10.0),
        np.log(5.0),
    ])

    result = least_squares(
        residuals,
        initial_guess,
        bounds=(
            np.log([0.5, 0.5]),
            np.log([50.0, 50.0]),
        ),
        max_nfev=100,
    )

    k_a = np.exp(result.x[0])
    k_b = np.exp(result.x[1])

    return k_a, k_b, result


# ============================================================
# Diagnostics
# ============================================================

def calculate_metrics(predicted, observed):
    error = predicted - observed

    rmse = np.sqrt(np.mean(error ** 2))
    max_error = np.max(np.abs(error))

    return rmse, max_error


def main():
    print("=" * 70)
    print("THERMAL INTERFACE STARTER-MODEL DIAGNOSTIC")
    print("=" * 70)

    print("\nGenerating experimental measurements...")

    datasets = generate_experimental_data()

    print("Experiments:")
    for experiment in EXPERIMENTS:
        print(f"  - {experiment['name']}")

    print("\nFitting starter model...")
    print("Starter model assumes PERFECT thermal contact.")

    k_a, k_b, result = fit_starter_model(datasets)

    print("\nRecovered starter-model parameters:")
    print(f"  k_A = {k_a:.6f}")
    print(f"  k_B = {k_b:.6f}")

    print(f"\nOptimizer success: {result.success}")
    print(f"Optimizer cost:    {result.cost:.8f}")

    print("\nPer-experiment validation:")
    print("-" * 70)

    total_errors = []

    for experiment in EXPERIMENTS:
        predicted = simulate_starter_model(
            k_a=k_a,
            k_b=k_b,
            initial_temperature=experiment["initial_temperature"],
            left_temperature=experiment["left_temperature"],
            right_temperature=experiment["right_temperature"],
        )

        predicted_sensors = predicted[SENSOR_INDICES, :]
        observed = datasets[experiment["name"]]

        rmse, max_error = calculate_metrics(
            predicted_sensors,
            observed,
        )

        total_errors.extend(
            (predicted_sensors - observed).ravel()
        )

        print(f"\n{experiment['name']}")
        print(f"  RMSE:      {rmse:.6f} °C")
        print(f"  Max error: {max_error:.6f} °C")

    total_errors = np.asarray(total_errors)

    overall_rmse = np.sqrt(
        np.mean(total_errors ** 2)
    )

    overall_max_error = np.max(
        np.abs(total_errors)
    )

    print("\n" + "=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)

    print(f"RMSE:      {overall_rmse:.6f} °C")
    print(f"Max error: {overall_max_error:.6f} °C")

    print("\nKnown physical parameters:")
    print(f"  True k_A          = {TRUE_K_A}")
    print(f"  True k_B          = {TRUE_K_B}")
    print(f"  True R_interface  = {TRUE_R_INTERFACE}")

    print("\nStarter-model limitation:")
    print("  The model assumes perfect thermal contact.")
    print("  The experimental system contains interface resistance.")

    print("\nThe large residuals indicate that changing only")
    print("k_A and k_B is insufficient to explain the data.")


if __name__ == "__main__":
    main()