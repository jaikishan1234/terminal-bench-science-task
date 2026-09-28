import numpy as np
from scipy.integrate import solve_ivp


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
# Interface-aware thermal model
# ============================================================

def simulate_full(
    k_a,
    k_b,
    r_interface,
    initial_temperature,
    left_temperature,
    right_temperature,
    sample_times,
):
    """
    Simulate the complete temperature field.

    Cell layout:

        A0 A1 A2 A3 A4 | B5 B6 B7 B8 B9
                        ^
                     interface
    """

    initial_state = np.full(
        TOTAL_CELLS,
        initial_temperature,
        dtype=float,
    )

    conductance_a = k_a / DX_A
    conductance_b = k_b / DX_B
    conductance_interface = 1.0 / r_interface

    def rhs(t, temperatures):
        dTdt = np.zeros(TOTAL_CELLS)

        # ----------------------------------------------------
        # Left boundary: A0
        # ----------------------------------------------------

        heat_left = conductance_a * (
            left_temperature - temperatures[0]
        )

        heat_right = conductance_a * (
            temperatures[0] - temperatures[1]
        )

        dTdt[0] = (
            heat_left - heat_right
        ) / (RHO_A * CP_A * DX_A)

        # ----------------------------------------------------
        # Material A internal cells: A1 -> A3
        # ----------------------------------------------------

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
            ) / (RHO_A * CP_A * DX_A)

        # ----------------------------------------------------
        # A-side interface cell: A4
        # ----------------------------------------------------

        interface_a = N_A - 1
        interface_b = N_A

        heat_in_a = conductance_a * (
            temperatures[interface_a - 1]
            - temperatures[interface_a]
        )

        heat_across_interface = (
            conductance_interface
            * (
                temperatures[interface_a]
                - temperatures[interface_b]
            )
        )

        dTdt[interface_a] = (
            heat_in_a - heat_across_interface
        ) / (RHO_A * CP_A * DX_A)

        # ----------------------------------------------------
        # B-side interface cell: B5
        # ----------------------------------------------------

        heat_across_interface = (
            conductance_interface
            * (
                temperatures[interface_a]
                - temperatures[interface_b]
            )
        )

        heat_out_b = conductance_b * (
            temperatures[interface_b]
            - temperatures[interface_b + 1]
        )

        dTdt[interface_b] = (
            heat_across_interface - heat_out_b
        ) / (RHO_B * CP_B * DX_B)

        # ----------------------------------------------------
        # Material B internal cells: B6 -> B8
        # ----------------------------------------------------

        for i in range(N_A + 1, TOTAL_CELLS - 1):
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
            ) / (RHO_B * CP_B * DX_B)

        # ----------------------------------------------------
        # Right boundary: B9
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
        ) / (RHO_B * CP_B * DX_B)

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

    return solution.y


# ============================================================
# Interface diagnostics
# ============================================================

def get_interface_temperatures(temperatures):
    """
    Return the temperatures immediately on each side
    of the interface.
    """

    interface_a = temperatures[N_A - 1]
    interface_b = temperatures[N_A]

    return interface_a, interface_b


def get_interface_temperature_jump(temperatures):
    """
    Return:

        T_A_interface - T_B_interface
    """

    interface_a, interface_b = (
        get_interface_temperatures(temperatures)
    )

    return interface_a - interface_b


# ============================================================
# Diagnostic experiment
# ============================================================

def main():
    print("=" * 70)
    print("INTERFACE TEMPERATURE DIAGNOSTIC")
    print("=" * 70)

    sample_times = np.linspace(
        0.0,
        2000.0,
        101,
    )

    experiment = {
        "initial_temperature": 20.0,
        "left_temperature": 100.0,
        "right_temperature": 20.0,
    }

    k_a = 12.0
    k_b = 4.5
    r_interface = 0.003

    temperatures = simulate_full(
        k_a=k_a,
        k_b=k_b,
        r_interface=r_interface,
        initial_temperature=experiment[
            "initial_temperature"
        ],
        left_temperature=experiment[
            "left_temperature"
        ],
        right_temperature=experiment[
            "right_temperature"
        ],
        sample_times=sample_times,
    )

    interface_a, interface_b = (
        get_interface_temperatures(temperatures)
    )

    temperature_jump = (
        get_interface_temperature_jump(temperatures)
    )

    print("\nInterface diagnostics:")
    print("-" * 70)

    print(
        f"Maximum A-side interface temperature: "
        f"{np.max(interface_a):.6f} °C"
    )

    print(
        f"Maximum B-side interface temperature: "
        f"{np.max(interface_b):.6f} °C"
    )

    print(
        f"Maximum temperature jump: "
        f"{np.max(np.abs(temperature_jump)):.6f} °C"
    )

    print(
        f"Temperature jump at final time: "
        f"{temperature_jump[-1]:.6f} °C"
    )

    print("\nInterface cells:")
    print(
        f"  Material A interface cell index: {N_A - 1}"
    )
    print(
        f"  Material B interface cell index: {N_A}"
    )


if __name__ == "__main__":
    main()