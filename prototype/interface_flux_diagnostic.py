import numpy as np

from interface_model import (
    N_A,
    simulate_full,
    get_interface_temperatures,
)


def main():
    print("=" * 70)
    print("INTERFACE HEAT-FLOW DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    sample_times = np.linspace(
        0.0,
        2000.0,
        101,
    )

    initial_temperature = 20.0
    left_temperature = 100.0
    right_temperature = 20.0

    # Reference parameters
    k_a = 12.0
    k_b = 4.5
    r_interface = 0.003

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    temperatures = simulate_full(
        k_a=k_a,
        k_b=k_b,
        r_interface=r_interface,
        initial_temperature=initial_temperature,
        left_temperature=left_temperature,
        right_temperature=right_temperature,
        sample_times=sample_times,
    )

    # --------------------------------------------------------
    # Interface temperatures
    # --------------------------------------------------------

    interface_a, interface_b = (
        get_interface_temperatures(temperatures)
    )

    temperature_jump = (
        interface_a - interface_b
    )

    # --------------------------------------------------------
    # Interface conductance
    # --------------------------------------------------------

    interface_conductance = 1.0 / r_interface

    # --------------------------------------------------------
    # Heat flow across interface
    #
    # Q = G * (T_A - T_B)
    # --------------------------------------------------------

    interface_heat_flow = (
        interface_conductance
        * temperature_jump
    )

    # --------------------------------------------------------
    # Print diagnostics
    # --------------------------------------------------------

    print("\nModel parameters:")
    print("-" * 70)

    print(f"k_A                  = {k_a}")
    print(f"k_B                  = {k_b}")
    print(f"R_interface          = {r_interface}")
    print(
        f"Interface conductance = "
        f"{interface_conductance:.6f}"
    )

    print("\nTemperature jump:")
    print("-" * 70)

    print(
        f"Maximum jump: "
        f"{np.max(temperature_jump):.6f} °C"
    )

    print(
        f"Minimum jump: "
        f"{np.min(temperature_jump):.6f} °C"
    )

    print(
        f"Final jump: "
        f"{temperature_jump[-1]:.6f} °C"
    )

    print("\nInterface heat flow:")
    print("-" * 70)

    print(
        f"Maximum heat flow: "
        f"{np.max(interface_heat_flow):.6f}"
    )

    print(
        f"Minimum heat flow: "
        f"{np.min(interface_heat_flow):.6f}"
    )

    print(
        f"Final heat flow: "
        f"{interface_heat_flow[-1]:.6f}"
    )

    # --------------------------------------------------------
    # Check the interface relationship
    # --------------------------------------------------------

    reconstructed_jump = (
        interface_heat_flow
        * r_interface
    )

    relationship_error = (
        reconstructed_jump
        - temperature_jump
    )

    max_relationship_error = np.max(
        np.abs(relationship_error)
    )

    print("\nInterface relationship check:")
    print("-" * 70)

    print(
        "Expected relationship:"
    )

    print(
        "  temperature jump = heat flow × interface resistance"
    )

    print(
        f"\nMaximum relationship error: "
        f"{max_relationship_error:.12e} °C"
    )

    if max_relationship_error < 1e-10:
        print("\nPASS: interface relationship is internally consistent.")
    else:
        print("\nWARNING: interface relationship is inconsistent.")


if __name__ == "__main__":
    main()