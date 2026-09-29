from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ThermalConfig:
    """Physical and numerical configuration for the layered wall."""

    length_a: float = 0.01
    length_b: float = 0.02

    rho_a: float = 7800.0
    rho_b: float = 2700.0

    cp_a: float = 500.0
    cp_b: float = 900.0

    cells_a: int = 5
    cells_b: int = 5


@dataclass(frozen=True)
class Experiment:
    """Boundary and initial conditions for one experiment."""

    name: str
    initial_temperature: float
    left_temperature: float
    right_temperature: float


class ThermalModel:
    """
    One-dimensional transient thermal model for two layered materials.

    The current model assumes perfect thermal contact between the
    two materials. The interface therefore has no independent thermal
    resistance parameter.
    """

    def __init__(self, config: ThermalConfig | None = None):
        self.config = config or ThermalConfig()

        self.dx_a = self.config.length_a / self.config.cells_a
        self.dx_b = self.config.length_b / self.config.cells_b

        self.total_cells = self.config.cells_a + self.config.cells_b

        self.interface_a = self.config.cells_a - 1
        self.interface_b = self.config.cells_a

    def _material_properties(
        self,
        conductivity_a: float,
        conductivity_b: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return conductivity, density and heat-capacity arrays."""

        if conductivity_a <= 0 or conductivity_b <= 0:
            raise ValueError("Thermal conductivities must be positive.")

        conductivity = np.concatenate(
            [
                np.full(self.config.cells_a, conductivity_a),
                np.full(self.config.cells_b, conductivity_b),
            ]
        )

        density = np.concatenate(
            [
                np.full(self.config.cells_a, self.config.rho_a),
                np.full(self.config.cells_b, self.config.rho_b),
            ]
        )

        heat_capacity = np.concatenate(
            [
                np.full(self.config.cells_a, self.config.cp_a),
                np.full(self.config.cells_b, self.config.cp_b),
            ]
        )

        return conductivity, density, heat_capacity

    def _rhs(
        self,
        _time: float,
        temperature: np.ndarray,
        conductivity_a: float,
        conductivity_b: float,
        experiment: Experiment,
    ) -> np.ndarray:
        """Compute dT/dt for the finite-volume discretization."""

        conductivity, density, heat_capacity = self._material_properties(
            conductivity_a,
            conductivity_b,
        )

        heat_flux = np.zeros(self.total_cells + 1)

        # Left boundary.
        heat_flux[0] = (
            2.0
            * conductivity[0]
            * (experiment.left_temperature - temperature[0])
            / self.dx_a
        )

        # Internal cell faces.
        for face in range(1, self.total_cells):
            left_cell = face - 1
            right_cell = face

            dx_left = self.dx_a if left_cell < self.config.cells_a else self.dx_b
            dx_right = self.dx_a if right_cell < self.config.cells_a else self.dx_b

            resistance = (
                dx_left / (2.0 * conductivity[left_cell])
                + dx_right / (2.0 * conductivity[right_cell])
            )

            heat_flux[face] = (
                temperature[left_cell] - temperature[right_cell]
            ) / resistance

        # Right boundary.
        heat_flux[-1] = (
            2.0
            * conductivity[-1]
            * (temperature[-1] - experiment.right_temperature)
            / self.dx_b
        )

        d_temperature = np.zeros(self.total_cells)

        for cell in range(self.total_cells):
            volume = self.dx_a if cell < self.config.cells_a else self.dx_b

            energy_rate = heat_flux[cell] - heat_flux[cell + 1]

            d_temperature[cell] = energy_rate / (
                density[cell] * heat_capacity[cell] * volume
            )

        return d_temperature

    def simulate(
        self,
        conductivity_a: float,
        conductivity_b: float,
        experiment: Experiment,
        sample_times: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate the temperature history at requested sample times.

        Returns:
            Array with shape (n_times, n_cells).
        """

        sample_times = np.asarray(sample_times, dtype=float)

        if sample_times.ndim != 1:
            raise ValueError("sample_times must be one-dimensional.")

        if len(sample_times) == 0:
            raise ValueError("sample_times cannot be empty.")

        if np.any(np.diff(sample_times) < 0):
            raise ValueError("sample_times must be sorted.")

        initial_temperature = np.full(
            self.total_cells,
            experiment.initial_temperature,
            dtype=float,
        )

        solution = solve_ivp(
            self._rhs,
            (
                float(sample_times[0]),
                float(sample_times[-1]),
            ),
            initial_temperature,
            t_eval=sample_times,
            args=(
                conductivity_a,
                conductivity_b,
                experiment,
            ),
            method="BDF",
            rtol=1e-7,
            atol=1e-9,
        )

        if not solution.success:
            raise RuntimeError(
                f"Thermal simulation failed: {solution.message}"
            )

        return solution.y.T

    def sensor_indices(self) -> dict[str, int]:
        """Return the indices of the four measurement locations."""

        return {
            "A_interior": 1,
            "A_interface": self.interface_a,
            "B_interface": self.interface_b,
            "B_interior": self.interface_b + 2,
        }

    def extract_sensors(self, temperature_field: np.ndarray) -> np.ndarray:
        """Extract the four experimental sensor locations."""

        indices = list(self.sensor_indices().values())

        return temperature_field[:, indices]