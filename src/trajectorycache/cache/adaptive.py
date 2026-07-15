"""
AdaptiveSpatialUrgencyCache: density-adaptive urgency weighting.

The fixed-weight SpatialUrgencyCache (SU) uses a single tuned urgency weight W for all
conditions. The density sweep (paper Section 6) shows this is suboptimal: the
spatial urgency signal helps at low-to-moderate vehicle density but saturates
and becomes harmful at high density, so the best W is itself a function of how
many vehicles are currently near the cache.

AdaptiveSpatialUrgencyCache removes the manual W hyperparameter by setting the
urgency weight at each eviction from the locally sensed vehicle density. It uses
the full urgency weight (w_max) when the coverage zone is sparse and ramps the
weight linearly to zero as density approaches the empirically/analytically
predicted crossover, after which the policy reduces to sliding-window LFU.

    W(rho) = w_max * clamp( (rho_high - rho) / (rho_high - rho_low), 0, 1 )

where rho is the number of vehicles currently within the relevance radius of the
cache, and [rho_low, rho_high] brackets the crossover density. This makes the
controller's only inputs quantities the RSU can observe directly, and ties its
calibration to the predictive crossover model of Section 6.
"""

from __future__ import annotations

from .trajectory import SpatialUrgencyCache


class AdaptiveSpatialUrgencyCache(SpatialUrgencyCache):
    """SpatialUrgencyCache (SU) with density-adaptive urgency weight."""

    name: str = "AdaptiveSU"

    def __init__(
        self,
        capacity: int,
        w_max: float = 0.2,
        rho_low: float = 16.0,
        rho_high: float = 64.0,
        cache_center: float = 5000.0,
        pop_window: float = 300.0,
        t_pred: float = 30.0,
        alpha_d: float = 0.1,
        r_rel: float = 800.0,
    ) -> None:
        # Initialise the parent at w_max; self.W is overwritten per eviction.
        super().__init__(
            capacity,
            urgency_weight=w_max,
            pop_window=pop_window,
            t_pred=t_pred,
            alpha_d=alpha_d,
            r_rel=r_rel,
        )
        self.w_max = w_max
        self.rho_low = rho_low
        self.rho_high = rho_high
        self.cache_center = cache_center

    def _local_density(self, vehicles: list[dict]) -> int:
        """Count vehicles currently within r_rel of the cache."""
        c = 0
        for v in vehicles:
            if abs(v["x"] - self.cache_center) <= self.r_rel:
                c += 1
        return c

    def _adaptive_w(self, rho: float) -> float:
        """Map local density to an urgency weight (high when sparse)."""
        span = self.rho_high - self.rho_low
        if span <= 0:
            return self.w_max if rho <= self.rho_low else 0.0
        frac = (self.rho_high - rho) / span
        frac = max(0.0, min(1.0, frac))
        return self.w_max * frac

    def _score_all(self, catalog, vehicles, t):
        # Set the urgency weight from current local density, then score as usual.
        self.W = self._adaptive_w(self._local_density(vehicles))
        return super()._score_all(catalog, vehicles, t)


# Backward-compatible alias (formerly AdaptiveTrajectoryCache).
AdaptiveTrajectoryCache = AdaptiveSpatialUrgencyCache
