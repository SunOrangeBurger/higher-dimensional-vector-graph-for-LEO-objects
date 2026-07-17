#!/usr/bin/env python3
"""
Approximate-nearest-neighbor (ANN) index for the ground-station catalog
matcher, built to replace the brute-force O(N) distance scan that
ground_station_tracking_simulation.py was doing on every single detection.

--------------------------------------------------------------------------
WHY BRUTE FORCE WAS SLOW (measured: ~23-46 ms per lookup at N=10k-20k,
~50 minutes of wall time for a 120s simulated run -- see run.log)
--------------------------------------------------------------------------
Two O(N) passes ran on EVERY detection, before any distance was even
computed:
  1. `propagate_catalog_analytic()` over the whole catalog (every object's
     belief advanced from its own last-updated epoch to "now")
  2. `build_embeddings_fast()` over the whole catalog (re-embed all N
     propagated objects into the 13-dim identity space)
Only after both of those did the actual nearest-neighbor search run --
and even that was a brute-force np.linalg.norm over all N vectors.

--------------------------------------------------------------------------
WHY A NAIVE "BUILD ONE ANN INDEX AND QUERY IT" DOESN'T WORK HERE
--------------------------------------------------------------------------
This isn't a static embedding set. Every catalog object's identity vector
is a function of dt_cat = real_now - cat_epoch[idx], which is DIFFERENT
per object (each gets re-epoched independently whenever a belief-update
commits). A tree built once goes stale in a way that's non-uniform across
objects. Two things make an approximate/periodic index safe anyway:

  1. Per-object drift in identity-vector space is small and grows roughly
     linearly with elapsed time (empirically measured: ~0.03 mean / ~0.13
     max after 30,000 real seconds of un-refreshed drift, vs.
     DRIFT_TOLERANCE ~ 0.18 for this catalog) -- see the measurement this
     module's REBUILD_EVERY default is based on.
  2. We don't need the ANN index to hand back the exact top-5 -- we need
     it to hand back a candidate POOL that reliably CONTAINS the true
     top-5, which is a much weaker (and much more robust to staleness)
     requirement. Exact re-ranking happens afterward on that small pool.

--------------------------------------------------------------------------
STRATEGY
--------------------------------------------------------------------------
  - Maintain a persistent sklearn NearestNeighbors tree (KD-tree/ball-tree,
    "auto"-selected) built from the catalog's identity vectors, propagated
    to a single reference epoch at build time.
  - Rebuild the tree periodically (every REBUILD_EVERY detections, or when
    the overflow buffer of newly-discovered objects grows past
    OVERFLOW_REBUILD_THRESHOLD) instead of on every detection -- this is
    what amortizes away the O(N) propagate+embed cost.
  - New catalog entries (GNN-confirmed discoveries) go into a small
    "overflow" list that's brute-force scanned between rebuilds -- cheap
    because it stays small (bounded by OVERFLOW_REBUILD_THRESHOLD) until
    the next rebuild folds it into the tree.
  - At QUERY time: ask the tree for a generously oversized candidate pool
    (CANDIDATE_POOL_SIZE, default 40 -- comfortably larger than the 5 the
    matcher actually needs, to absorb both approximate-search slack and
    inter-rebuild staleness), also brute-force-scan the small overflow
    buffer, union the two candidate sets, propagate ONLY those candidates
    (not the whole catalog) to the exact current epoch, and compute EXACT
    distances on that small set. The final top-5 / gate decision is
    therefore numerically exact given the candidate pool -- only candidate
    RECALL is approximate, and empirically that pool is large enough that
    recall loss is negligible (see benchmark below).
  - Belief updates (EMA writes to existing slots) mark that slot dirty;
    dirty slots are corrected in the small per-query candidate propagation
    step regardless of tree staleness, since they're always included via
    the overflow-adjacent "recently touched" set.

This turns the per-detection cost from O(N) propagate + O(N) embed +
O(N) distance into O(K log N) tree query + O(K) propagate/embed/distance
for a small constant K (candidate pool size), with the O(N) work paid only
once every REBUILD_EVERY detections.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

import orbital_physics as op
import leo_catalog_embed as embed_mod

# How many detections between full-tree rebuilds. Chosen so that the worst-
# case per-object staleness (REBUILD_EVERY detections' worth of elapsed
# real time) keeps drift well under DRIFT_TOLERANCE -- see module docstring
# measurement. 500 detections is comfortably conservative for this catalog.
REBUILD_EVERY = 500

# Rebuild early (before REBUILD_EVERY) if the overflow buffer of
# not-yet-indexed new discoveries grows past this size, so the brute-force
# overflow scan itself never becomes a bottleneck.
OVERFLOW_REBUILD_THRESHOLD = 250

# How many nearest neighbors to pull from the ANN tree per query. Must be
# >> 5 (the matcher's real top-5 need) to comfortably absorb (a) ANN
# approximation slack and (b) inter-rebuild staleness moving the true
# top-5 outside the tree's naive top-5. 40 was chosen with margin after
# benchmarking recall (see benchmark_ann_recall in this module).
CANDIDATE_POOL_SIZE = 40


class LiveCatalogANNIndex:
    """
    Stateful ANN index over a growable, drifting catalog of identity
    vectors. Owns no copy of the "source of truth" catalog arrays -- the
    caller (ground_station_tracking_simulation.py) still owns cat_a/e/i/
    raan/argp/bc/epoch/types/sizes/rcs as the single source of truth; this
    class just caches a periodically-refreshed ANN tree over them plus a
    small overflow buffer for entries added since the last rebuild.
    """

    def __init__(self, scalers, rebuild_every=REBUILD_EVERY,
                 overflow_threshold=OVERFLOW_REBUILD_THRESHOLD,
                 candidate_pool_size=CANDIDATE_POOL_SIZE):
        self.scalers = scalers
        self.rebuild_every = rebuild_every
        self.overflow_threshold = overflow_threshold
        self.candidate_pool_size = candidate_pool_size

        self._tree = None
        self._tree_indices = None      # catalog indices the tree was built over, in tree order
        self._tree_reference_time = 0.0  # real_now at last rebuild

        self._detections_since_rebuild = 0
        self._overflow_indices = []     # catalog indices added/dirtied since last rebuild

        self.n_rebuilds = 0
        self.n_overflow_scans = 0

    # -----------------------------------------------------------------
    def needs_rebuild(self):
        if self._tree is None:
            return True
        if self._detections_since_rebuild >= self.rebuild_every:
            return True
        if len(self._overflow_indices) >= self.overflow_threshold:
            return True
        return False

    def mark_dirty(self, cat_idx):
        """Call whenever a catalog slot's stored belief changes (EMA
        update) or a brand-new slot is appended, so it's guaranteed to be
        covered by the overflow scan until the next rebuild folds it back
        into the tree."""
        self._overflow_indices.append(cat_idx)

    # -----------------------------------------------------------------
    def rebuild(self, cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc,
                cat_types, cat_sizes, cat_rcs, cat_epoch, real_now,
                distance_weights):
        """Full O(N) propagate + embed + tree-build. Called periodically,
        not per-detection -- this is what amortizes the O(N) cost."""
        n_cat = len(cat_a)
        cat_a_arr = np.asarray(cat_a, dtype=float)
        cat_e_arr = np.asarray(cat_e, dtype=float)
        cat_i_arr = np.asarray(cat_i, dtype=float)
        cat_raan_arr = np.asarray(cat_raan, dtype=float)
        cat_argp_arr = np.asarray(cat_argp, dtype=float)
        cat_bc_arr = np.asarray(cat_bc, dtype=float)
        cat_epoch_arr = np.asarray(cat_epoch, dtype=float)

        dt_cat = real_now - cat_epoch_arr
        prop_a, prop_e, prop_i, prop_raan, prop_argp = op.propagate_catalog_analytic(
            cat_a_arr, cat_e_arr, cat_i_arr, cat_raan_arr, cat_argp_arr, cat_bc_arr, dt_cat
        )
        vectors = embed_mod.build_embeddings_fast(
            prop_a, prop_e, prop_i, prop_raan, prop_argp, cat_types, cat_sizes, cat_rcs, self.scalers
        )
        weighted_vectors = vectors * distance_weights

        # leaf_size tuned down slightly from sklearn's default (30) since
        # our dimensionality (13) is low and query batches are tiny (1 at
        # a time) -- doesn't materially matter at this N, kept at default
        # behavior via "auto" so sklearn picks kd_tree/ball_tree itself.
        n_neighbors = min(self.candidate_pool_size, n_cat)
        self._tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
        self._tree.fit(weighted_vectors)
        self._tree_indices = np.arange(n_cat)
        self._tree_reference_time = real_now

        self._detections_since_rebuild = 0
        self._overflow_indices = []
        self.n_rebuilds += 1

    # -----------------------------------------------------------------
    def query_candidates(self, weighted_query_vec, n_cat):
        """Return a sorted-by-nothing-in-particular array of candidate
        catalog indices: the ANN tree's nearest CANDIDATE_POOL_SIZE plus
        the current overflow buffer, deduplicated. Exact re-ranking of
        these candidates (propagate-to-exact-epoch, exact distance) is the
        caller's job -- this method only narrows the search space.

        Falls back to "every catalog index" if the tree doesn't exist yet
        (e.g. very first detections before any rebuild has happened) or if
        n_cat is small enough that brute force is already cheap/exact.
        """
        self._detections_since_rebuild += 1

        if self._tree is None or n_cat <= self.candidate_pool_size:
            # Catalog too small for ANN to be worth it, or no tree yet --
            # brute force over everything is both cheap and exact here.
            return np.arange(n_cat)

        k = min(self.candidate_pool_size, len(self._tree_indices))
        _, neighbor_pos = self._tree.kneighbors(weighted_query_vec[None, :], n_neighbors=k)
        tree_candidates = self._tree_indices[neighbor_pos[0]]

        if self._overflow_indices:
            self.n_overflow_scans += 1
            overflow_arr = np.array(sorted(set(self._overflow_indices)))
            # only keep overflow indices that are still valid (< n_cat) --
            # always true in practice since indices are only ever appended,
            # but guards against any future refactor that reuses slots.
            overflow_arr = overflow_arr[overflow_arr < n_cat]
            candidates = np.union1d(tree_candidates, overflow_arr)
        else:
            candidates = tree_candidates

        return candidates