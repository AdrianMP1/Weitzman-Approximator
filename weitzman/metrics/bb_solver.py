"""
Branch-and-bound exact solver for Weitzman diversity

Extends ground-truth computation from n \\approx 12 (brute-force enumeration) to
n \\approx 36 by exploiting the closest-pair branching structure of the recursion:

    W(S) = d(g, h) + max( W(S∖{g}),  W(S∖{h}) )

where (g, h) is the closest pair in S.  At each node the solver branches on
which member of the closest pair is removed first, reducing the search space
from O(n!) to O(2^n) sub-problems.  Sub-problems are indexed by the bitmask
of remaining elements and memoised in an LRU cache (bounded at MAX_MEMO
entries, kept for subsets of size ≤ STORE_IF_K_LEQ).

Note: the upper_bound() method is implemented but not yet wired into the DFS
for pruning.  The current speedup comes entirely from memoisation and
best-first branch ordering.
"""

from __future__ import annotations

import math
import logging
import numpy as np

from numpy.typing import NDArray
from collections import OrderedDict

logger = logging.getLogger(__name__)

MAX_MEMO      = 500_000
STORE_IF_K_LEQ = 20

class WeitzmanBBSolver:
    """
    Exact Weitzman solver via closest-pair branching with LRU memoisation.

    Parameters
    ----------
    D : (n, n) symmetric pairwise distance matrix (numpy array or nested list).
    """

    def __init__(self, D: NDArray[np.float64] | list[list[float]]) -> None:
        if not isinstance(D, np.ndarray):
            D = np.array(D, dtype=float)
        self.D = D
        self.n = D.shape[0]
        self._neighbors: list[list[int]] = self._build_neighbors()

        self.incumbent: float = 0.0 # global lower bound (best full solution found)

        self._memo:   OrderedDict[int, float] = OrderedDict()
        self._choice: dict[int, str]          = {} # "L" or "R" (or "B" for ties)
        self._nodes_visited: int = 0

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _build_neighbors(self) -> list[list[int]]:
        """For each vertex i, pre-sort all j ≠ i by ascending distance."""
        return [
            sorted(
                (j for j in range(self.n) if j != i),
                key=lambda j: (self.D[i, j], j),
            )
            for i in range(self.n)
        ]

    # ------------------------------------------------------------------
    # Set primitives (bitmask representation)
    # ------------------------------------------------------------------

    def _closest_pair(self, mask: int) -> tuple[int, int, float]:
        """
        Return (g, h, d) for the closest pair in the subset encoded by mask.
        Tie-breaking: smallest distance, then smallest g, then smallest h.
        """

        best_d = math.inf
        best_g = -1
        best_h = -1

        # For each i in S, get its nearest neighbor
        for i in range(self.n):
            if not (mask >> i) & 1:
                continue

            # Iterate from nearest outward
            for j in self._neighbors[i]:
                if (mask >> j) & 1:
                    d   = float(self.D[i, j])
                    g, h = (i, j) if i < j else (j, i)
                    if d < best_d or (
                        d == best_d and (g < best_g or (g == best_g and h < best_h))
                    ):
                        best_d, best_g, best_h = d, g, h
                    break                      # neighbors are sorted; first hit is NN
        return best_g, best_h, best_d

    def _diameter(self, mask: int) -> float:
        """Maximum pairwise distance within the subset encoded by mask."""
        best = 0.0

        for i in range(self.n):
            if not (mask >> i) & 1:
                continue

            # Scan neighbors in reverse (largest distance first)
            for j in reversed(self._neighbors[i]):
                if (mask >> j) & 1:
                    d = float(self.D[i, j])
                    if d > best:
                        best = d
                    break # First valid j is farthest in S for this i
        return best

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def upper_bound(self, mask: int) -> float:
        """
        Admissible upper bound: diameter(S) × (|S| - 1).

        Each of the |S|-1 removal steps contributes at most the diameter,
        so this is always ≥ W(S).  Currently not wired into the DFS;
        adding 'if accum + upper_bound(mask) <= self.incumbent: return -inf'
        at the top of _dfs would enable genuine pruning.
        """
        k = mask.bit_count()
        if k <= 1:
            return 0.0
        return self._diameter(mask) * (k - 1)

    def lower_bound(self, mask: int) -> float:
        """
        Greedy lower bound: repeatedly remove the g-member of the closest pair.

        Produces a valid (possibly sub-optimal) removal sequence value, used
        to initialise the incumbent and to order branches during DFS.
        """
        val = 0.0
        cur = mask
        while cur.bit_count() > 1:
            # Get closest pair
            g, _, gain = self._closest_pair(cur)
            val += gain

            # Remove g from mask
            cur &= ~(1 << g)
        return val

    # ------------------------------------------------------------------
    # DFS
    # ------------------------------------------------------------------

    def _dfs(self, mask: int, accum: float) -> float:
        # Get how many elements remain
        k = mask.bit_count()

        # Base case: single element contributes 0
        if k <= 1:
            self.incumbent = max(self.incumbent, accum)
            return 0.0

        # Memo hit (LRU refresh)
        if mask in self._memo:
            val = self._memo.pop(mask)
            self._memo[mask] = val # Refresh LRU
            if val != float("-inf"):
                self.incumbent = max(self.incumbent, accum + val)
            return val

        self._nodes_visited += 1

        # Get the pair: link and representative
        g, h, gain = self._closest_pair(mask)

        # Make children
        mask_l = mask & ~(1 << g)
        mask_r = mask & ~(1 << h)

        child_accum = accum + gain

        # Branch ordering: explore the child with better greedy lower bound first
        # to tighten the incumbent as quickly as possible.
        lb_l = self.lower_bound(mask_l) if mask_l.bit_count() > 1 else 0.0
        lb_r = self.lower_bound(mask_r) if mask_r.bit_count() > 1 else 0.0

        # Recursion
        if child_accum + lb_l >= child_accum + lb_r:
            v_l = self._dfs(mask_l, child_accum)
            v_r = self._dfs(mask_r, child_accum)
        else:
            v_r = self._dfs(mask_r, child_accum)
            v_l = self._dfs(mask_l, child_accum)

        best_child = max(v_l, v_r)
        best = gain + best_child if best_child != float("-inf") else float("-inf")

        # Update incumbent if we got a complete value via children
        if best != float("-inf"):
            self.incumbent = max(self.incumbent, accum + best)

        # Memoize exact value for small subsets
        if k <= STORE_IF_K_LEQ:
            self._memo[mask] = best
            if len(self._memo) > MAX_MEMO:
                self._memo.popitem(last=False)   # evict LRU entry

        # Record which branch gave the better result (for reconstruction)
        if v_l > v_r:
            self._choice[mask] = "L"
        elif v_r > v_l:
            self._choice[mask] = "R"
        else:
            self._choice[mask] = "B" # tie

        return best

    # ------------------------------------------------------------------
    # Public Call Methods
    # ------------------------------------------------------------------

    def solve(self) -> float:
        """
        Compute and return the exact Weitzman diversity W(A) for the full set.

        The result is also stored in self.incumbent.
        """
        # Make mask with all elements
        full = (1 << self.n) - 1

        # Get the first lower bound
        self.incumbent = self.lower_bound(full)
        logger.info("n=%d | initial LB=%.4f", self.n, self.incumbent)

        # Solve
        self._dfs(full, accum=0.0)

        logger.info(
            "n=%d | W(A)=%.4f | nodes=%d | memo=%d",
            self.n, self.incumbent, self._nodes_visited, len(self._memo),
        )
        return self.incumbent

    def reconstruct_optimal_sequence(self) -> list[int]:
        """
        Return one optimal removal sequence (list of vertex indices).

        Requires solve() to have been called first.  When a tie between
        branches exists ('B'), the L-branch (remove g) is chosen arbitrarily.
        """
        sequence: list[int] = []
        mask = (1 << self.n) - 1
        while mask.bit_count() > 1 and mask in self._choice:
            g, h, _ = self._closest_pair(mask)
            ch      = self._choice[mask]

            removed = g if ch in ("L", "B") else h
            sequence.append(removed)
            mask &= ~(1 << removed)
        if mask:
            sequence.append((mask & -mask).bit_length() - 1)

        return sequence
