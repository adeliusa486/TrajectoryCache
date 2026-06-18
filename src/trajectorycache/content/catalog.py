"""
Content catalog and Zipf-distributed request generator.

Content items are geo-tagged (have a fixed location on the highway),
and requests follow a Zipf popularity distribution — a standard
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
    location: float    # Position on highway (metres)
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
        Zipf skew parameter (higher → more concentrated popularity).
    seed : int, optional
        RNG seed.
    """

    def __init__(
        self,
        n_items: int = 200,
        road_length: float = 10_000.0,
        zipf_alpha: float = 1.2,
        seed: Optional[int] = None,
    ) -> None:
        self.n_items = n_items
        self.road_length = road_length
        self.zipf_alpha = zipf_alpha
        self._rng = np.random.default_rng(seed)

        self._items: Dict[int, ContentItem] = self._build_catalog()
        self._popularity_weights = self._build_zipf_weights()

    # ------------------------------------------------------------------

    def _build_catalog(self) -> Dict[int, ContentItem]:
        locations = self._rng.uniform(0, self.road_length, size=self.n_items)
        return {
            i: ContentItem(item_id=i, location=float(locations[i]))
            for i in range(self.n_items)
        }

    def _build_zipf_weights(self) -> np.ndarray:
        ranks = np.arange(1, self.n_items + 1, dtype=float)
        weights = 1.0 / (ranks ** self.zipf_alpha)
        return weights / weights.sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_request(self) -> ContentItem:
        """Draw one item according to Zipf popularity."""
        idx = int(self._rng.choice(self.n_items, p=self._popularity_weights))
        return self._items[idx]

    def generate_requests(self, n_requests: int) -> List[ContentItem]:
        """Generate a batch of content requests."""
        indices = self._rng.choice(
            self.n_items, size=n_requests, p=self._popularity_weights
        )
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
