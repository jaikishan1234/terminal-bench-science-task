import numpy as np
import matplotlib.pyplot as plt

from starter_failure import (
    EXPERIMENTS,
    SENSOR_INDICES,
    TIME_POINTS,
    generate_experimental_data,
    fit_starter_model,
    simulate_starter_model,
)


def main():
    print("=" * 70)
    print("THERMAL STARTER-MODEL FAILURE DIAGNOSTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Generate the same synthetic experimental measurements
    # --------------------------------------------------------

    print("\nGenerating experimental measurements...")

    datasets = generate_experimental_data()

    # --------------------------------------------------------
    # Fit the limited starter model
    # --------------------------------------------------------

    print("Fitting starter model...")

    k_a, k_b, result = fit_starter_model(datasets)

    print("\nStarter-model parameters:")
    print(f"  k_A = {k_a:.6f}")
    print(f"  k_B = {k_b:.6f}")
    print(f"  Cost = {result.cost:.6f}")

    # --------------------------------------------------------
    # Create one figure for each experiment
    # --------------------------------------------------------

    for experiment in EXPERIMENTS:
        name = experiment["name"]

        observed = datasets[name]

        predicted_full = simulate_starter_model(
            k_a=k_a,
            k_b=k_b,
            initial_temperature=experiment["initial_temperature"],
            left_temperature=experiment["left_temperature"],
            right_temperature=experiment["right_temperature"],
        )

        predicted = predicted_full[SENSOR_INDICES, :]

        # ----------------------------------------------------
        # Print residual statistics
        # ----------------------------------------------------

        print(f"\n{name}")
        print("-" * 70)

        for sensor_number in range(len(SENSOR_INDICES)):
            residual = (
                predicted[sensor_number]
                - observed[sensor_number]
            )

            rmse = np.sqrt(
                np.mean(residual ** 2)
            )

            max_error = np.max(
                np.abs(residual)
            )

            print(
                f"  Sensor {sensor_number + 1}: "
                f"RMSE = {rmse:.6f} °C, "
                f"Max error = {max_error:.6f} °C"
            )

        # ----------------------------------------------------
        # Plot measured vs predicted temperature
        # ----------------------------------------------------

        plt.figure(figsize=(11, 7))

        sensor_names = [
            "Sensor 1 - Material A",
            "Sensor 2 - Near Interface",
            "Sensor 3 - Material B",
        ]

        for sensor_number in range(3):
            plt.plot(
                TIME_POINTS,
                observed[sensor_number],
                label=f"{sensor_names[sensor_number]} measured",
            )

            plt.plot(
                TIME_POINTS,
                predicted[sensor_number],
                linestyle="--",
                label=f"{sensor_names[sensor_number]} starter model",
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title(
            f"Starter Model vs Experimental Data: {name}"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"prototype/{name}_temperature_comparison.png"
        )

        plt.savefig(
            filename,
            dpi=150,
        )

        plt.close()

        print(f"\nSaved: {filename}")

        # ----------------------------------------------------
        # Plot residuals separately
        # ----------------------------------------------------

        plt.figure(figsize=(11, 7))

        for sensor_number in range(3):
            residual = (
                predicted[sensor_number]
                - observed[sensor_number]
            )

            plt.plot(
                TIME_POINTS,
                residual,
                label=sensor_names[sensor_number],
            )

        plt.axhline(
            0.0,
            linestyle="--",
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Prediction Error (°C)")
        plt.title(
            f"Starter Model Residuals: {name}"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"prototype/{name}_residuals.png"
        )

        plt.savefig(
            filename,
            dpi=150,
        )

        plt.close()

        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()