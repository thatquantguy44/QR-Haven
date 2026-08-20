"""Locate-request allocation engine for the borrow demand surface.

Greedy binary knapsack allocator: for each CUSIP, approved quantity is the
smaller of the requested quantity and available inventory.  Requests are
prioritised by revenue density (fee_bps × requested_qty) so that high-value
locates consume capacity first when inventory is constrained.

Reference: spec § Phase 3 — Downstream Applications / Locate Allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LocateRequest:
    """A single inbound locate request to be allocated."""

    locate_id: str
    cusip: str
    client_id: str
    requested_qty_shares: float
    fee_bps: float               # recommended fee from calibrator
    timestamp: datetime


@dataclass
class AllocationResult:
    """Outcome of allocating one LocateRequest."""

    locate_id: str
    cusip: str
    client_id: str
    requested_qty_shares: float
    approved_qty_shares: float
    fee_bps: float
    status: str                  # "approved" | "partial" | "rejected"
    revenue_usd: float = 0.0     # fee_bps/10000 * approved_qty * price (if price known)


@dataclass
class InventorySnapshot:
    """Available-to-lend inventory for a single CUSIP at a point in time."""

    cusip: str
    available_shares: float
    price_usd: float = 0.0       # used for revenue_usd computation; 0 if unknown


class LocateAllocator:
    """Greedy revenue-maximising allocator over a batch of locate requests.

    Usage
    -----
    allocator = LocateAllocator(inventory)
    results = allocator.allocate_batch(requests)

    Parameters
    ----------
    inventory : dict[str, InventorySnapshot]
        Mapping from CUSIP to available inventory at the time of allocation.
    min_fill_ratio : float
        Minimum fraction of requested_qty to approve; requests that can only be
        partially filled below this threshold are rejected outright.  Default 0.0
        (approve any partial fill).
    """

    def __init__(
        self,
        inventory: dict[str, InventorySnapshot],
        min_fill_ratio: float = 0.0,
    ) -> None:
        if not (0.0 <= min_fill_ratio <= 1.0):
            raise ValueError("min_fill_ratio must be in [0, 1]")
        # mutable copy — depleted as allocations are made
        self._remaining: dict[str, float] = {
            cusip: max(snap.available_shares, 0.0)
            for cusip, snap in inventory.items()
        }
        self._prices: dict[str, float] = {
            cusip: snap.price_usd for cusip, snap in inventory.items()
        }
        self.min_fill_ratio = min_fill_ratio

    def _revenue_density(self, req: LocateRequest) -> float:
        """fee_bps × requested_qty — used to sort requests by value."""
        return req.fee_bps * req.requested_qty_shares

    def allocate(self, request: LocateRequest) -> AllocationResult:
        """Allocate a single locate request against current remaining inventory."""
        avail = self._remaining.get(request.cusip, 0.0)
        requested = max(request.requested_qty_shares, 0.0)

        if avail <= 0.0 or requested <= 0.0:
            return AllocationResult(
                locate_id=request.locate_id,
                cusip=request.cusip,
                client_id=request.client_id,
                requested_qty_shares=requested,
                approved_qty_shares=0.0,
                fee_bps=request.fee_bps,
                status="rejected",
            )

        approved = min(requested, avail)
        fill_ratio = approved / requested if requested > 0 else 0.0

        if fill_ratio < self.min_fill_ratio:
            return AllocationResult(
                locate_id=request.locate_id,
                cusip=request.cusip,
                client_id=request.client_id,
                requested_qty_shares=requested,
                approved_qty_shares=0.0,
                fee_bps=request.fee_bps,
                status="rejected",
            )

        self._remaining[request.cusip] = avail - approved

        price = self._prices.get(request.cusip, 0.0)
        revenue = (request.fee_bps / 10_000) * approved * price if price > 0 else 0.0

        status = "approved" if approved >= requested else "partial"
        return AllocationResult(
            locate_id=request.locate_id,
            cusip=request.cusip,
            client_id=request.client_id,
            requested_qty_shares=requested,
            approved_qty_shares=approved,
            fee_bps=request.fee_bps,
            status=status,
            revenue_usd=revenue,
        )

    def allocate_batch(
        self, requests: list[LocateRequest]
    ) -> list[AllocationResult]:
        """Allocate a batch of requests sorted by descending revenue density.

        Requests for the same CUSIP compete for inventory in order of their
        fee_bps × qty score, so high-value locates are served first.

        Returns
        -------
        List of AllocationResult objects in the same order as *requests*
        (results are re-indexed to input order after sorting).
        """
        # Sort by descending revenue density, preserving original index for output order
        indexed = sorted(
            enumerate(requests),
            key=lambda x: self._revenue_density(x[1]),
            reverse=True,
        )

        results: list[AllocationResult | None] = [None] * len(requests)
        for orig_idx, req in indexed:
            results[orig_idx] = self.allocate(req)

        return results  # type: ignore[return-value]

    def remaining_inventory(self) -> dict[str, float]:
        """Return a copy of remaining available shares per CUSIP."""
        return dict(self._remaining)
