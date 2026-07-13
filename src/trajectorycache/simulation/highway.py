"""
Highway simulation engine.

Models a one-dimensional highway segment with vehicles moving at
configurable speeds. Generates vehicle state snapshots used by
TrajectoryCache to compute spatial urgency.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Vehicle:
    """Single vehicle on the highway."""

    vehicle_id: int
    x: float  # Position along highway (metres)
    speed: float  # Speed (m/s)
    direction: int  # +1 = forward, -1 = backward
    lane: int = 0

    def step(self, dt: float) -> None:
        """Advance vehicle by one time step."""
        self.x += self.speed * self.direction * dt

    def to_dict(self) -> dict:
        return {
            "id": self.vehicle_id,
            "x": self.x,
            "speed": self.speed,
            "direction": self.direction,
            "lane": self.lane,
        }


class HighwaySimulation:
    """
    Discrete-time highway simulation.

    Parameters
    ----------
    road_length : float
        Total highway length in metres.
    n_vehicles : int
        Number of vehicles to simulate.
    dt : float
        Time step in seconds.
    mean_speed : float
        Mean vehicle speed (m/s). Default ~ 90 km/h.
    speed_std : float
        Standard deviation of vehicle speeds (m/s).
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        road_length: float = 10_000.0,
        n_vehicles: int = 50,
        dt: float = 1.0,
        mean_speed: float = 25.0,
        speed_std: float = 5.0,
        platoon_size: int = 10,
        platoon_gap: float = 30.0,
        seed: int | None = None,
        unidirectional: bool = False,
    ) -> None:
        self.road_length = road_length
        self.n_vehicles = n_vehicles
        self.dt = dt
        self.mean_speed = mean_speed
        self.speed_std = speed_std
        self.platoon_size = max(1, platoon_size)
        self.platoon_gap = platoon_gap
        self.unidirectional = unidirectional
        self.t: float = 0.0

        rng = np.random.default_rng(seed)
        self._rng = rng

        self.vehicles: list[Vehicle] = self._spawn_vehicles(rng)

    # ------------------------------------------------------------------

    def _spawn_vehicles(self, rng: np.random.Generator) -> list[Vehicle]:
        """
        Spawn vehicles in platoons.

        Vehicles are grouped into platoons of `platoon_size`. All members of a
        platoon share the same speed and direction, positioned within
        `platoon_gap` metres of the platoon leader. This approximates the
        emergent clustering produced by SUMO's Krauss car-following model.
        """
        vehicles = []
        vid = 0
        n_platoons = max(1, self.n_vehicles // self.platoon_size)

        for _ in range(n_platoons):
            # Platoon leader properties
            leader_x = float(rng.uniform(0, self.road_length))
            leader_speed = float(
                np.clip(rng.normal(self.mean_speed, self.speed_std), 5.0, 50.0)
            )
            direction = 1 if self.unidirectional else int(rng.choice([-1, 1]))

            for i in range(self.platoon_size):
                if vid >= self.n_vehicles:
                    break
                # Small offset within the platoon (tight cluster)
                offset = float(rng.uniform(-self.platoon_gap, self.platoon_gap))
                x = (leader_x + offset) % self.road_length
                # Tiny speed variation within platoon (Krauss-like following)
                speed = float(
                    np.clip(
                        leader_speed * rng.normal(1.0, 0.03),
                        5.0,
                        50.0,
                    )
                )
                vehicles.append(
                    Vehicle(
                        vehicle_id=vid,
                        x=x,
                        speed=speed,
                        direction=direction,
                    )
                )
                vid += 1

        return vehicles

    def step(self) -> list[dict]:
        """Advance simulation by one dt and return current vehicle states."""
        self.t += self.dt
        for veh in self.vehicles:
            veh.step(self.dt)
            # Wrap around road boundaries
            if veh.x < 0:
                veh.x = self.road_length + veh.x
            elif veh.x > self.road_length:
                veh.x = veh.x - self.road_length
        return [v.to_dict() for v in self.vehicles]

    def run(self, n_steps: int) -> Iterator[tuple[float, list[dict]]]:
        """Yield (timestamp, vehicle_states) for each step."""
        for _ in range(n_steps):
            states = self.step()
            yield self.t, states

    def snapshot(self) -> list[dict]:
        """Return current vehicle states without advancing time."""
        return [v.to_dict() for v in self.vehicles]

    @property
    def current_time(self) -> float:
        return self.t
