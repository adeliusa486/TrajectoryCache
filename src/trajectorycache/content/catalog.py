"""
Content catalog and Zipf-distributed request generator.

Content items are geo-tagged (have a fixed location on the highway),
and requests follow a Zipf popularity distribution - a standard
assumption in caching literature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContentItem:
    """A single piece of geo-tagged content."""

    item_id: int
    location: float  # Position on highway (metres)
    size_mb: float = 1.0
    category: str = "generic"


class ContentCatalog:
    """
    Fixed catalog of geo-tagged content items.

    Parameters
    ----------
    n_items : int
        Total number of items in the catalog.
    road_length : float
        Highway length used to distribute item locations.
    zipf_alpha : float
        Zipf skew parameter (higher -> more concentrated popularity).
    seed : int, optional
        RNG seed.
    """

    def __init__(
        self,
        n_items: int = 200,
        road_length: float = 10_000.0,
        active_zone_length: float = 1600.0,
        zipf_alpha: float = 1.2,
        seed: Optional[int] = None,
    ) -> None:
        self.n_items = n_items
        self.road_length = road_length
        self.active_zone_length = active_zone_length
        self.zipf_alpha = zipf_alpha
        self._rng = np.random.default_rng(seed)

        self._items: Dict[int, ContentItem] = self._build_catalog()
        self._popularity_weights = self._build_zipf_weights()

    # ------------------------------------------------------------------

    def _build_catalog(self) -> Dict[int, ContentItem]:
        start_loc = (self.road_length - self.active_zone_length) / 2.0
        end_loc = start_loc + self.active_zone_length
        locations = self._rng.uniform(start_loc, end_loc, size=self.n_items)
        return {
            i: ContentItem(item_id=i, location=float(locations[i]))
            for i in range(self.n_items)
        }

    def _build_zipf_weights(self) -> np.ndarray:
        ranks = np.arange(1, self.n_items + 1, dtype=float)
        weights = 1.0 / (ranks**self.zipf_alpha)
        return weights / weights.sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_vehicle_requests(
        self,
        vehicles: List[dict],
        r_request: float = 800.0,
    ) -> List["ContentItem"]:
        """
        Generate requests driven by vehicle positions.

        Each vehicle in `vehicles` looks ahead by up to `r_request` metres
        and requests the nearest content item in that direction, weighted
        by Zipf popularity.  This couples requests to spatial positions,
        which is the fundamental assumption of the paper's vehicular model.

        Parameters
        ----------
        vehicles : list of dicts
            Each dict has keys ``x`` (float, metres), ``speed`` (float),
            ``direction`` (int, +1 or -1).
        r_request : float
            Look-ahead distance in metres.
        """
        requests: List[ContentItem] = []
        for veh in vehicles:
            x_v = veh["x"]
            direction = veh.get("direction", 1)
            speed = veh.get("speed", 0.0)
            if speed <= 0:
                continue

            # Find items physically ahead of vehicle within r_request
            candidates = []
            for iid, item in self._items.items():
                dist = (item.location - x_v) * direction
                if 0 < dist <= r_request:
                    candidates.append(iid)

            if not candidates:
                continue

            # Weight by Zipf popularity, renormalize over candidates
            weights = self._popularity_weights[candidates]
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            chosen_idx = int(self._rng.choice(len(candidates), p=weights))
            requests.append(self._items[candidates[chosen_idx]])

        return requests

    def generate_requests(self, n_requests: int) -> List["ContentItem"]:
        """Generate n_requests items drawn from Zipf popularity (legacy API)."""
        indices = self._rng.choice(self.n_items, size=n_requests, p=self._popularity_weights)
        return [self._items[int(i)] for i in indices]

    def item(self, item_id: int) -> ContentItem:
        """Look up a content item by ID."""
        return self._items[item_id]

    def location_map(self) -> Dict[int, float]:
        """Return {item_id: location} for all catalog items."""
        return {iid: item.location for iid, item in self._items.items()}

    def __len__(self) -> int:
        return self.n_items

    def __iter__(self) -> Iterator[ContentItem]:
        return iter(self._items.values())
