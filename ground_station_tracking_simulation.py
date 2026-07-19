#!/usr/bin/env python3
"""
Ground station tracking simulation -- the full predict/match/update cycle.

Every object in leo_catalog.csv has a REAL trajectory that evolves over time
(drag decay + J2 precession + small unmodeled process noise). The ground
station only has embeddings for half the catalog (EMBED_FRACTION at build
time). On each orbital pass:

  1. PREDICT : propagate every catalog object's stored belief forward from
     its own last-updated epoch to the current time (clean, deterministic
     physics -- see orbital_physics.propagate_elements).
  2. DETECT  : the true object's real trajectory is propagated forward too
     (same physics, plus small process + measurement noise -- this is what
     the sensor "sees").
  3. MATCH   : find the 5 nearest catalog identity-vectors to the detection.
     Accept the closest one only if it clears a drift-tolerance distance
     AND is meaningfully closer than the 2nd-nearest (ambiguity margin).
     If the gate rejects, consult a GNN second opinion on the same top-5.
     If both reject, either correct an existing entry for this true object
     (if one already exists) or register a brand-new catalog entry.
  4. UPDATE  : blend the detection into the matched catalog entry's stored
     belief (exponential moving average) and reset its epoch to now.

All orbital motion is hyper-accelerated (ACCELERATION_FACTOR); all lookup/
match timing is measured in genuine, unaccelerated wall-clock time.

---------------------------------------------------------------------------
FIX HISTORY -- catalog self-poisoning via unconditional first-pass writes
---------------------------------------------------------------------------
Symptom: mismatch rate climbed from ~26% at t=0 to ~99% by t=120s in the
original run, even though diagnose_gnn.py showed the GNN itself picks the
true match correctly 99.6% of the time on a static validation set.

Root cause: EVERY acceptance path (fast gate, GNN, and a third "existing
slot correction" path) committed an unconditional single-pass write to
cat_a/e/i/raan/argp on ANY accept, with no corroboration. Any wrong accept
-- and at this catalog density the auto-calibrated gate has a nonzero
false-accept rate by construction (DRIFT_TOLERANCE is derived from a 5th
percentile of within-cluster nearest-neighbor gaps, so ~5% of genuine
cross-object gaps are expected to be smaller than it) -- permanently blends
a foreign object's measurement into that slot. The corrupted slot then
distorts the top-5 shortlist for every future detection near it, which is
exactly the out-of-distribution condition diagnose_gnn.py never exercises
(it always validates against the pristine, uncorrupted original store).
That produces more wrong accepts, which corrupt more slots -- a runaway
feedback loop, not a one-off model quality problem.

Two rounds of investigation were needed to find every write path:
  1st pass (found & fixed): the main GNN "known" branch was committing
    unconditional writes on a single uncertain call.
  2nd pass (found & fixed here): the "existing slot correction" branch
    (both gate AND GNN reject, but true_name_to_idx already has a slot
    for this object) was ALSO an unconditional overwrite, and turned out
    to be the LARGER contamination source of the two once the first fix
    was in place -- it fires precisely when the top5 identity vectors are
    least trustworthy. It's also not real corroboration: true_name_to_idx
    is ground truth kept only for diagnostics; a real system wouldn't have
    it, so this path can't be trusted as independent confirmation either.
  Remaining gap (found & fixed here): the fast GATE path itself had no
    corroboration requirement at all -- only the GNN paths did. Once the
    GNN paths were dampened, the gate's own false-accepts became the
    dominant remaining source of climbing mismatch rate (66% end-state
    after fixing only the GNN paths, still climbing).

Fix: a single unified corroboration gate (`_resolve_update`) now covers
ALL THREE acceptance paths uniformly. A detection accepted by ANY path is
reported KNOWN immediately (matches real-world behavior: you don't want to
say "not known" just because you're being extra careful about updating
belief). But the actual EMA write to cat_a/e/i/raan/argp only commits if
this candidate slot was ALSO the accepted match on the immediately
preceding detection of the same true object -- i.e. two consecutive
passes agree, from any combination of paths. A single-pass accept from any
path is provisional and only stashes a pending record; if the very next
sighting of that true object doesn't land on the same slot, the pending
guess is discarded rather than committed. New-object registration is
unaffected (a fresh entry has no prior belief to corrupt).
---------------------------------------------------------------------------
PERF FIX -- brute-force O(N) scan replaced with ANN candidate search
---------------------------------------------------------------------------
Symptom: lookup time was ~23-46 ms per detection regardless of what got
matched, and grew as the catalog grew via GNN discoveries (10k -> ~20k
objects over one 120s run) -- a full run took ~50 minutes of wall time
(see run.log: avg 43.9 ms/lookup, total_detections=113534).

Root cause: EVERY detection re-propagated AND re-embedded the ENTIRE
catalog (both O(N)) before computing a single distance, then brute-force
scanned all N vectors for the top-5. This is pure overhead unrelated to
the actual matching decision -- the vast majority of the catalog is
nowhere near the query and never had a chance of being in the top-5.

Fix: leo_ann_index.LiveCatalogANNIndex maintains a periodically-rebuilt
ANN tree (rebuilt every REBUILD_EVERY detections, not every detection) and
a small overflow buffer for objects added/updated since the last rebuild.
Each detection queries the tree for a generous candidate pool (40, vs. the
5 actually needed) instead of scanning the whole catalog, then propagates
and re-embeds ONLY that small candidate set to the exact current epoch
before computing exact distances -- so the final accept/reject decision is
computed on exactly the same (correct, non-stale) numbers as before, just
over ~40 candidates instead of ~10,000-20,000. See leo_ann_index.py for
the full design rationale and staleness-safety argument.
---------------------------------------------------------------------------
"""

import heapq
import time
import numpy as np
import pandas as pd

import orbital_physics as op
import leo_catalog_embed as embed_mod
import gnn_matcher as gnn_mod
import leo_ann_index
from sklearn.neighbors import NearestNeighbors

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CATALOG_PATH = "leo_catalog.csv"
EMBEDDINGS_PATH = "outputs/leo_embeddings.pt"
FALLBACK_NPZ_PATH = "outputs/leo_embeddings.npz"

ACCELERATION_FACTOR = 300          # sim-seconds run this much faster than real orbital seconds
SIMULATION_DURATION_SECONDS = 120  # simulated seconds to run

# Matching gate -- set to None to auto-calibrate from the catalog's own
# measured nearest-neighbor density at startup (recommended; see
# calibrate_thresholds() below). Fixed numbers tuned for a small catalog
# become unsafe at larger scale: denser catalogs pack objects closer
# together, so a tolerance that was safe at 100 objects can exceed the
# actual gap between two DIFFERENT objects at 1000+.
DRIFT_TOLERANCE = None
MARGIN_REQUIRED = None

# Catalog belief update rate (exponential moving average)
ALPHA = 0.4

# Realism noise: real trajectory gets small process noise (unmodeled physics);
# the "sensor" also adds a bit of measurement noise on top before matching.
PROCESS_NOISE = {"a_km": 0.05, "e": 0.00005, "i_deg": 0.002, "angle_deg": 0.01}
MEASUREMENT_NOISE = {"a_km": 0.02, "e": 0.00002, "i_deg": 0.001, "angle_deg": 0.005}

SEED = 11

# Second-opinion GNN: when the fast threshold gate rejects a detection,
# consult a trained GNN on the same 5 nearest candidates before giving up.
# If the GNN also finds no match, the object is registered as a new
# catalog entry for future detections.
USE_GNN_CONFIRMATION = True

# Every acceptance path (gate, GNN, existing-slot correction) is uncertain
# to some degree -- the gate has a nonzero false-accept rate by
# construction, and the GNN/existing-slot paths only fire when the gate
# already found the top5 ambiguous or wrong. Require this many CONSECUTIVE
# accepted detections of the SAME true object landing on the SAME catalog
# slot before committing an EMA belief-update to it, instead of trusting
# any single accept -- from any path -- enough to overwrite stored state.
# 1 = old (unsafe) behavior: every accept writes immediately.
# 2 = require one corroborating repeat on the same slot.
UPDATE_CONFIRMATIONS_REQUIRED = 2

# --- ANN index tuning (see leo_ann_index.py for full rationale) ---
# How many detections between full-catalog tree rebuilds.
ANN_REBUILD_EVERY = leo_ann_index.REBUILD_EVERY
# Rebuild early if the overflow buffer (new/updated slots since the last
# rebuild) grows past this many entries.
ANN_OVERFLOW_REBUILD_THRESHOLD = leo_ann_index.OVERFLOW_REBUILD_THRESHOLD
# Candidate pool size pulled from the tree per query (>> 5, the real need).
ANN_CANDIDATE_POOL_SIZE = leo_ann_index.CANDIDATE_POOL_SIZE
# Catalog size below which brute force is used directly (ANN overhead not
# worth it for tiny catalogs; LiveCatalogANNIndex already does this
# fallback internally, this is just for visibility/tuning here).
ANN_MIN_CATALOG_SIZE = ANN_CANDIDATE_POOL_SIZE

# --- Retirement / de-duplication / recalibration (README issue #3, #6) ---
# A catalog slot is retired (excluded from ANN candidates and future
# housekeeping) if EITHER of these trips:
#   - its propagated altitude drops below this floor (decayed out of orbit;
#     matches the same RE_EARTH+100 floor propagate_catalog_analytic clamps
#     new_a to, so a retired-for-decay slot is one that's been pinned at
#     that floor rather than genuinely still in orbit)
RETIREMENT_ALTITUDE_FLOOR_KM = 100.0
#   - it hasn't been matched in this many real seconds (never re-acquired --
#     default 48 real hours, i.e. this models "we haven't seen this object
#     in two days," not two simulated days, since only lookup timing is
#     unaccelerated -- see module docstring)
RETIREMENT_UNSEEN_SECONDS = 48 * 3600.0
# Don't retire anything for unseen-ness before this much real time has
# elapsed, so a legitimately-just-registered object (or one pre-loaded at
# t=0 with epoch=0) isn't retired on the very first housekeeping pass
# before it's had a fair chance to be re-detected at least once.
RETIREMENT_WARMUP_SECONDS = 12 * 3600.0
# How many detections between retirement housekeeping passes.
RETIREMENT_CHECK_EVERY = 100
# How many detections between de-duplication (merge) passes. Must be a
# multiple of RETIREMENT_CHECK_EVERY in spirit (doesn't have to divide
# evenly) -- dedup is more expensive (an O(active^2)-ish neighbor query)
# so it runs less often than the cheap per-slot retirement check.
DEDUP_CHECK_EVERY = 500
# Two active slots merge if their weighted identity-vector distance is
# under this fraction of drift_tolerance -- i.e. "close enough that they're
# almost certainly the same physical object having drifted into
# near-collision," not merely "somewhat similar."
DEDUP_DISTANCE_FRACTION_OF_DRIFT_TOLERANCE = 0.2
# Re-run calibrate_thresholds() on the active (non-retired) catalog
# periodically so thresholds tighten/loosen as the ACTIVE catalog's
# density actually changes over a run, rather than staying pinned to the
# startup snapshot. Needs at least this many active objects to produce a
# meaningful density estimate (calibrate_thresholds' own per-cluster
# bucketing degrades below this).
RECALIBRATION_MIN_ACTIVE = 200
# Only actually recalibrate when the active catalog size has moved by at
# least this fraction since the last calibration (retirement/merges
# shrinking it, or registrations growing it) -- skips redundant
# recalibration work on rebuilds where the active population hasn't
# meaningfully changed, which is the common case between housekeeping
# events. Always recalibrates once, on the very first rebuild.
# (calibrate_thresholds() itself is now cheap -- see its docstring for the
# cKDTree-based rewrite -- so this gating is a nice-to-have that avoids
# redundant work, not a requirement for affordability the way an earlier,
# reverted subsampling approach was.)
RECALIBRATION_CHANGE_FRACTION = 0.10


# ---------------------------------------------------------------------
# Load embedding store
# ---------------------------------------------------------------------
def load_embedding_store(pt_path, npz_fallback):
    if TORCH_AVAILABLE:
        try:
            data = torch.load(pt_path, weights_only=False)
            data["vectors"] = data["vectors"].numpy()
            return data
        except FileNotFoundError:
            pass
    raw = np.load(npz_fallback, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    data["scalers"] = raw["scalers"][0]  # unwrap the object-array wrapping used in the npz fallback
    data["names"] = data["names"].tolist()
    data["types"] = data["types"].tolist()
    return data


def calibrate_thresholds(catalog_vectors: np.ndarray, cat_i=None, cat_types=None, cat_e=None):
    """
    Auto-calibrate DRIFT_TOLERANCE and MARGIN_REQUIRED from the catalog's
    OWN measured density, instead of trusting fixed numbers tuned at a
    different scale.

    IMPORTANT: this measures the gap WITHIN clusters that actually collide,
    not a flat global percentile. data_generator.py draws inclination from
    only 5 narrow bands and eccentricity from a narrow near-circular band
    for ~90% of objects, so most real collisions happen BETWEEN two objects
    that already share (type, inclination cluster, eccentricity regime).
    A global 5th percentile is dominated by well-separated cross-cluster
    pairs and reports a comfortable-looking gap even while the same-cluster
    pairs that actually cause mismatches sit far closer together than that
    number suggests. Bucketing first, then taking the percentile of
    within-bucket nearest-neighbor gaps, exposes the real worst case.

    Falls back to the old global calculation if bucket metadata isn't
    provided (e.g. called on a vector-only array).

    PERF NOTE: computes each object's within-bucket nearest-neighbor
    distance via scipy.spatial.cKDTree (O(bucket_size log bucket_size))
    rather than pdist+squareform (O(bucket_size^2) time AND memory for
    both the pairwise distance matrix and its squareform expansion) --
    verified numerically IDENTICAL to the pdist-based computation (exact,
    not approximate) while being ~3x faster at full catalog scale
    (~20k objects across ~100 buckets). This makes it cheap enough to
    call periodically (as the active catalog's density changes over a
    run -- see the RECALIBRATION_* constants and call site in main())
    without reintroducing O(N^2)-ish cost into the hot path.

    An earlier version of this speedup randomly subsampled large buckets
    instead of using a KD-tree. That was WRONG: the 5th-percentile
    statistic is dominated by the most crowded buckets (they produce the
    tightest gaps), so subsampling disproportionately thins exactly the
    buckets that set the threshold, producing a HIGHER (looser) drift
    tolerance every time -- verified as a consistent ~13-15% bias across
    multiple random seeds, not sampling noise. cKDTree avoids this
    entirely by computing the TRUE nearest neighbor for every object,
    not an estimate from a shrunken sample.
    """
    from scipy.spatial import cKDTree

    if cat_i is None or cat_types is None or cat_e is None:
        tree = cKDTree(catalog_vectors)
        # k=2: each point's own position (distance 0) plus its true
        # nearest neighbor.
        dists, _ = tree.query(catalog_vectors, k=min(2, len(catalog_vectors)))
        nearest = dists[:, 1] if dists.ndim > 1 and dists.shape[1] > 1 else np.zeros(len(catalog_vectors))
        p5 = np.percentile(nearest, 5)
        drift_tolerance = 0.35 * p5
        margin_required = 0.15 * p5
        print(f"Auto-calibrated (global, no bucket metadata): nearest-neighbor gap "
              f"(5th pct)={p5:.4f} -> DRIFT_TOLERANCE={drift_tolerance:.4f}, "
              f"MARGIN_REQUIRED={margin_required:.4f}")
        return drift_tolerance, margin_required

    cat_i = np.asarray(cat_i, dtype=float)
    cat_types = np.asarray(cat_types)
    cat_e = np.asarray(cat_e, dtype=float)

    # Coarse inclination bucket: round to nearest 5 degrees. Doesn't need to
    # exactly reproduce INCLINATION_CLUSTERS -- just needs to group objects
    # that are actually close in i_sin/i_cos space.
    incl_bucket = np.round(cat_i / 5.0).astype(int)
    # Eccentricity regime: near-circular (the ~90% band) vs elliptical.
    ecc_bucket = (cat_e > 0.01).astype(int)

    bucket_keys = list(zip(cat_types.tolist(), incl_bucket.tolist(), ecc_bucket.tolist()))
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, key in enumerate(bucket_keys):
        groups[key].append(idx)

    within_bucket_gaps = []
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        sub = catalog_vectors[idxs]
        tree = cKDTree(sub)
        dists, _ = tree.query(sub, k=2)
        within_bucket_gaps.extend(dists[:, 1].tolist())

    if not within_bucket_gaps:
        # degenerate catalog (every object in its own bucket) -- fall back
        tree = cKDTree(catalog_vectors)
        dists, _ = tree.query(catalog_vectors, k=min(2, len(catalog_vectors)))
        within_bucket_gaps = (dists[:, 1] if dists.ndim > 1 and dists.shape[1] > 1
                               else np.zeros(len(catalog_vectors))).tolist()

    p5 = np.percentile(within_bucket_gaps, 5)
    drift_tolerance = 0.35 * p5
    margin_required = 0.15 * p5

    print(f"Auto-calibrated from WITHIN-CLUSTER density ({len(groups)} buckets, "
          f"{len(within_bucket_gaps)} within-bucket pairs): nearest-neighbor gap "
          f"(5th pct)={p5:.4f} -> DRIFT_TOLERANCE={drift_tolerance:.4f}, "
          f"MARGIN_REQUIRED={margin_required:.4f}")
    return drift_tolerance, margin_required


def main():
    catalog_df = pd.read_csv(CATALOG_PATH)
    store = load_embedding_store(EMBEDDINGS_PATH, FALLBACK_NPZ_PATH)
    scalers = store["scalers"]

    # Weights persisted at embedding-build time; fall back to the module's
    # current defaults only if loading an older store built before weighting
    # was added. Reading from the store (rather than always using the live
    # module constant) means matching always uses the exact weights the
    # embedding space was calibrated with.
    distance_weights = np.asarray(
        store.get("distance_weights", embed_mod.DISTANCE_WEIGHTS), dtype=np.float32
    )
    weighted_store_vectors = np.asarray(store["vectors"], dtype=np.float32) * distance_weights

    drift_tolerance = DRIFT_TOLERANCE
    margin_required = MARGIN_REQUIRED
    if drift_tolerance is None or margin_required is None:
        drift_tolerance, margin_required = calibrate_thresholds(
            weighted_store_vectors,
            cat_i=store["elements_i"],
            cat_types=store["types"],
            cat_e=store["elements_e"],
        )

    print(f"Catalog:   {len(catalog_df)} objects ({CATALOG_PATH})")
    print(f"Embedding: {len(store['names'])} objects on record")
    print(f"Acceleration factor: {ACCELERATION_FACTOR}x real orbital speed")
    print(f"Simulated run length: {SIMULATION_DURATION_SECONDS}s (simulated)")
    print(f"Update corroboration required: {UPDATE_CONFIRMATIONS_REQUIRED} consecutive "
          f"matching accepts on the same slot before a belief-update commits")

    gnn_model = None
    if USE_GNN_CONFIRMATION:
        if not gnn_mod.TORCH_AVAILABLE:
            print("GNN confirmation requested but PyTorch isn't available -- skipping "
                  "(rejected detections will just stay NOT KNOWN, no catalog growth).")
        else:
            gnn_model = gnn_mod.load_or_train_model(
                store,
                distance_weights=distance_weights,
                drift_tolerance=drift_tolerance,
                margin_required=margin_required,
            )
    print("-" * 70)

    rng = np.random.default_rng(SEED)

    # --- TRUTH state for every object in the full catalog ---
    truth_a = catalog_df["a"].to_numpy(dtype=float).copy()
    truth_e = catalog_df["e"].to_numpy(dtype=float).copy()
    truth_i = catalog_df["i"].to_numpy(dtype=float).copy()
    truth_raan = catalog_df["Omega"].to_numpy(dtype=float).copy()
    truth_argp = catalog_df["omega"].to_numpy(dtype=float).copy()
    truth_bc = catalog_df["bc_m2_per_kg"].to_numpy(dtype=float).copy()
    truth_epoch = np.zeros(len(catalog_df))  # real seconds

    names = catalog_df["name"].tolist()
    types = catalog_df["type"].tolist()
    sizes = catalog_df["size_meters"].to_numpy(dtype=float)
    rcs_vals = catalog_df["rcs"].to_numpy(dtype=float)

    # --- CATALOG belief state (only the embedded subset, GROWABLE -- new
    #     discoveries confirmed by the GNN get appended here) ---
    cat_names = list(store["names"])
    cat_a = list(np.asarray(store["elements_a"], dtype=float))
    cat_e = list(np.asarray(store["elements_e"], dtype=float))
    cat_i = list(np.asarray(store["elements_i"], dtype=float))
    cat_raan = list(np.asarray(store["elements_raan"], dtype=float))
    cat_argp = list(np.asarray(store["elements_argp"], dtype=float))
    cat_bc = list(np.asarray(store["bc_m2_per_kg"], dtype=float))
    cat_types = list(store["types"])
    cat_sizes = list(np.asarray(store["sizes"], dtype=float))
    cat_rcs = list(np.asarray(store["rcs"], dtype=float))
    cat_epoch = list(np.asarray(store["epoch_seconds"], dtype=float))
    discovery_counter = 0
    discovered_true_name = {}  # "discovered_N" -> the true object name it was created from

    # --- Retirement / de-dup bookkeeping (README issue #3) ---
    # Initialized for every PRE-LOADED store entry too, not just future
    # registrations -- a common gap when adding this kind of tracking is to
    # only append to these lists at registration time and forget the
    # initial catalog needs the same fields.
    n_preloaded = len(cat_names)
    cat_retired = [False] * n_preloaded          # True once decayed/unseen-too-long
    cat_confidence = [1] * n_preloaded            # starts at 1; += 1 per committed update, summed on merge
    # cat_last_seen mirrors cat_epoch at load time: a pre-loaded object's
    # epoch IS the last time its state is known to be valid, so treating
    # that as "last seen" is the correct initial assumption (not 0, which
    # would make every pre-loaded object look instantly stale).
    cat_last_seen = list(cat_epoch)

    # ANN index over the (growable, drifting) catalog -- see leo_ann_index.py.
    # Replaces the old per-detection full-catalog propagate+embed+brute-force
    # scan with a periodically-rebuilt tree + small-candidate-pool exact
    # re-ranking. Rebuilt for the first time on the very first MATCH below.
    ann_index = leo_ann_index.LiveCatalogANNIndex(
        scalers,
        rebuild_every=ANN_REBUILD_EVERY,
        overflow_threshold=ANN_OVERFLOW_REBUILD_THRESHOLD,
        candidate_pool_size=ANN_CANDIDATE_POOL_SIZE,
    )

    # Unified corroboration state: cat_idx -> true_name of the object whose
    # accepted-but-uncommitted detection last landed on this slot. ANY
    # acceptance path (gate, GNN, existing-slot correction) writes here.
    # A slot's belief only gets updated once the SAME true_name lands on
    # the SAME slot on two consecutive passes; otherwise it's a dangling
    # provisional guess that a mismatched next hit will simply overwrite.
    pending_update = {}

    # Tracks the active-catalog size as of the LAST recalibration, so
    # recalibration only re-runs when that size has moved meaningfully
    # (README issue #6 fix, made cheap enough to run periodically --
    # see RECALIBRATION_CHANGE_FRACTION / RECALIBRATION_MAX_PER_BUCKET).
    # None until the first calibration happens (forces it to always run
    # once, on the first rebuild that has enough active population).
    last_calibrated_active_count = None

    true_name_to_idx = {n: i for i, n in enumerate(cat_names)}  # true name -> current catalog slot,
    # maintained incrementally as objects get discovered, so mismatch diagnostics can find
    # "what SHOULD this have matched" even after renaming/growth
    # (a real system wouldn't have this -- it's ground truth we keep ONLY to
    # correctly diagnose whether matches are actually right, since a newly
    # discovered object is legitimately re-named and comparing against its
    # true catalog name would otherwise flag every correct re-detection as
    # a false "mismatch")

    # --- orbital period / event scheduling (same mechanism as before) ---
    real_periods = 2 * np.pi * np.sqrt(truth_a ** 3 / op.MU_EARTH)
    sim_periods = real_periods / ACCELERATION_FACTOR

    heap = []
    for idx in range(len(catalog_df)):
        phase = rng.uniform(0, sim_periods[idx])
        heapq.heappush(heap, (phase, idx))

    # --- metrics ---
    total_detections = 0
    known_count = 0
    unknown_count = 0
    lookup_times_known = []
    lookup_times_unknown = []
    match_distances = []
    match_margins = []
    per_type_known = {}
    per_type_unknown = {}
    gnn_confirmed_known = 0
    gnn_new_discoveries = 0
    update_committed_count = 0     # belief-update actually written this pass
    update_withheld_count = 0      # accepted but update withheld pending corroboration
    retired_decayed_count = 0
    retired_unseen_count = 0
    merged_count_total = 0

    # --- SDA / ML evaluation metrics (ground-truth-dependent -- see
    # README "no ground truth in a real deployment" issue: these are only
    # measurable here BECAUSE the simulator tracks true identity; a real
    # deployment has no oracle for these) ---
    mismatch_count = 0             # accepted matches where matched slot's true identity != detection's true identity
    correct_match_count = 0        # accepted matches that WERE correct (denominator companion for mismatch rate)
    # true_name -> set of catalog indices EVER registered as representing
    # it (not just the current true_name_to_idx, which only holds the
    # latest/most-recently-updated mapping) -- needed to detect "ghost"
    # registrations: a true object that ended up with more than one
    # distinct catalog slot created for it over the run.
    true_name_to_all_slots = {}
    # Seed with the pre-loaded catalog: each pre-loaded slot's cat_names
    # entry IS its true name (no renaming happens for pre-loaded objects,
    # only for later discovered_N registrations), so slot i represents
    # true object cat_names[i] one-to-one at this point.
    for i in range(n_preloaded):
        true_name_to_all_slots.setdefault(cat_names[i], set()).add(i)
    gnn_overturn_total = 0         # GNN-path accepts (gate had rejected first)
    gnn_overturn_correct = 0       # ...of which were actually correct matches
    # ANN candidate recall: was the TRUE matching slot (per ground truth)
    # present in the ANN-returned candidate pool, before re-ranking?
    # Measured only when a ground-truth slot for this true object already
    # exists (nothing to "recall" if the object was never registered).
    ann_recall_checked = 0
    ann_recall_hit = 0

    real_start = time.perf_counter()

    def resolve_update(cat_idx, true_object_name, meas_a, meas_e, meas_i, meas_raan, meas_argp,
                        prop_a_val, prop_e_val, prop_i_val, prop_raan_val, prop_argp_val, real_now):
        """
        Single choke point for EVERY catalog belief-update, regardless of
        which path (gate / GNN / existing-slot correction) accepted the
        match. Returns True if the write committed, False if withheld
        pending corroboration.

        Corroboration rule: the write commits only if the immediately
        preceding accepted detection ALSO landed on this same cat_idx AND
        claimed to be the same true_object_name. Otherwise this pass's
        guess is stashed as the new pending record (replacing whatever was
        there) and nothing is written to the catalog arrays yet.
        """
        if UPDATE_CONFIRMATIONS_REQUIRED <= 1:
            cat_a[cat_idx] = (1 - ALPHA) * prop_a_val + ALPHA * meas_a
            cat_e[cat_idx] = (1 - ALPHA) * prop_e_val + ALPHA * meas_e
            cat_i[cat_idx] = (1 - ALPHA) * prop_i_val + ALPHA * meas_i
            cat_raan[cat_idx] = (1 - ALPHA) * prop_raan_val + ALPHA * meas_raan
            cat_argp[cat_idx] = (1 - ALPHA) * prop_argp_val + ALPHA * meas_argp
            cat_epoch[cat_idx] = real_now
            return True

        prior = pending_update.get(cat_idx)
        if prior == true_object_name:
            # Two consecutive passes agree on this slot -- commit using
            # THIS pass's fresher measurement, then clear the pending flag.
            cat_a[cat_idx] = (1 - ALPHA) * prop_a_val + ALPHA * meas_a
            cat_e[cat_idx] = (1 - ALPHA) * prop_e_val + ALPHA * meas_e
            cat_i[cat_idx] = (1 - ALPHA) * prop_i_val + ALPHA * meas_i
            cat_raan[cat_idx] = (1 - ALPHA) * prop_raan_val + ALPHA * meas_raan
            cat_argp[cat_idx] = (1 - ALPHA) * prop_argp_val + ALPHA * meas_argp
            cat_epoch[cat_idx] = real_now
            pending_update.pop(cat_idx, None)
            return True
        else:
            # First hit on this slot, or it contradicts a different pending
            # guess -- not enough evidence yet. Stash/replace the pending
            # record; do NOT touch the catalog arrays this pass.
            pending_update[cat_idx] = true_object_name
            return False

    while heap:
        sim_time, idx = heapq.heappop(heap)
        if sim_time > SIMULATION_DURATION_SECONDS:
            break

        real_now = sim_time * ACCELERATION_FACTOR  # real seconds elapsed since t=0

        # --- propagate this object's TRUE trajectory forward to now ---
        dt_truth = real_now - truth_epoch[idx]
        new_a, new_e, new_i, new_raan, new_argp = op.propagate_elements(
            np.array([truth_a[idx]]), np.array([truth_e[idx]]), np.array([truth_i[idx]]),
            np.array([truth_raan[idx]]), np.array([truth_argp[idx]]), np.array([truth_bc[idx]]),
            dt_truth, process_noise=PROCESS_NOISE, rng=rng,
        )
        truth_a[idx], truth_e[idx], truth_i[idx] = new_a[0], new_e[0], new_i[0]
        truth_raan[idx], truth_argp[idx] = new_raan[0], new_argp[0]
        truth_epoch[idx] = real_now

        # --- sensor measurement: true state + measurement noise ---
        meas_a = truth_a[idx] + rng.normal(0, MEASUREMENT_NOISE["a_km"])
        meas_e = max(0.0, truth_e[idx] + rng.normal(0, MEASUREMENT_NOISE["e"]))
        meas_i = truth_i[idx] + rng.normal(0, MEASUREMENT_NOISE["i_deg"])
        meas_raan = (truth_raan[idx] + rng.normal(0, MEASUREMENT_NOISE["angle_deg"])) % 360
        meas_argp = (truth_argp[idx] + rng.normal(0, MEASUREMENT_NOISE["angle_deg"])) % 360

        detection_vec = embed_mod.transform_identity_vector(
            meas_a, meas_e, meas_i, meas_raan, meas_argp, types[idx], sizes[idx], rcs_vals[idx], scalers
        )

        name = names[idx]  # true identity of the object generating this detection --
        # ground truth, used for corroboration-gate plumbing (existing_slot path) and
        # for the ground-truth-only diagnostics below (mismatch rate, ANN recall, etc.)

        # --- MATCH: this is the real, timed lookup work ---
        t0 = time.perf_counter()

        n_cat = len(cat_names)
        candidate_idx = np.array([], dtype=int)
        if n_cat > 0:
            weighted_detection_vec = detection_vec * distance_weights

            # Rebuild the ANN tree periodically (amortized O(N) cost) rather
            # than on every detection -- see leo_ann_index.py. Retired slots
            # (README issue #3) are excluded from the tree entirely.
            if ann_index.needs_rebuild():
                ann_index.rebuild(
                    cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc,
                    cat_types, cat_sizes, cat_rcs, cat_epoch, real_now,
                    distance_weights, cat_retired,
                )

                # Recalibrate thresholds against the ACTIVE catalog's
                # current density (README issue #6) -- but only when it's
                # actually worth it:
                #   1. enough active population for a meaningful per-cluster
                #      density estimate (RECALIBRATION_MIN_ACTIVE)
                #   2. the active count has moved by at least
                #      RECALIBRATION_CHANGE_FRACTION since the last
                #      calibration -- skips redundant recompute on rebuilds
                #      where retirement/merge/registration haven't
                #      meaningfully shifted density since last time
                # AND bounds the cost of the calibration call itself via
                # RECALIBRATION_MAX_PER_BUCKET (see calibrate_thresholds'
                # max_per_bucket docstring) -- without this cap, a single
                # recalibration at ~20k active objects does O(bucket_size^2)
                # pdist per bucket, which is expensive enough that running
                # it periodically reintroduces real O(N^2)-ish cost into
                # the hot path this whole ANN layer exists to avoid.
                active_mask_recal = ~np.asarray(cat_retired, dtype=bool)
                n_active_recal = int(np.sum(active_mask_recal))
                needs_recalibration = (
                    n_active_recal >= RECALIBRATION_MIN_ACTIVE
                    and (
                        last_calibrated_active_count is None
                        or abs(n_active_recal - last_calibrated_active_count)
                           >= RECALIBRATION_CHANGE_FRACTION * last_calibrated_active_count
                    )
                )
                if needs_recalibration:
                    act_idx = np.where(active_mask_recal)[0]
                    act_a = np.array([cat_a[j] for j in act_idx])
                    act_e = np.array([cat_e[j] for j in act_idx])
                    act_i = np.array([cat_i[j] for j in act_idx])
                    act_raan = np.array([cat_raan[j] for j in act_idx])
                    act_argp = np.array([cat_argp[j] for j in act_idx])
                    act_bc = np.array([cat_bc[j] for j in act_idx])
                    act_epoch = np.array([cat_epoch[j] for j in act_idx])
                    act_types = [cat_types[j] for j in act_idx]
                    act_sizes = np.array([cat_sizes[j] for j in act_idx])
                    act_rcs = np.array([cat_rcs[j] for j in act_idx])

                    dt_act = real_now - act_epoch
                    p_act_a, p_act_e, p_act_i, p_act_raan, p_act_argp = op.propagate_catalog_analytic(
                        act_a, act_e, act_i, act_raan, act_argp, act_bc, dt_act
                    )
                    act_vectors = embed_mod.build_embeddings_fast(
                        p_act_a, p_act_e, p_act_i, p_act_raan, p_act_argp,
                        act_types, act_sizes, act_rcs, scalers,
                    )
                    weighted_act_vectors = act_vectors * distance_weights
                    new_drift_tolerance, new_margin_required = calibrate_thresholds(
                        weighted_act_vectors, cat_i=p_act_i, cat_types=act_types, cat_e=p_act_e,
                    )
                    drift_tolerance, margin_required = new_drift_tolerance, new_margin_required
                    last_calibrated_active_count = n_active_recal
                    print(f"[T+{sim_time:7.2f}s sim] [RECALIBRATE] active={n_active_recal} "
                          f"drift_tolerance={drift_tolerance:.4f} margin_required={margin_required:.4f}")

            # Narrow the search space to a small candidate pool (tree
            # neighbors + overflow buffer) instead of scanning all N. This
            # candidate set is intentionally over-sized (CANDIDATE_POOL_SIZE
            # >> 5) so the true top-5 is reliably contained in it even with
            # inter-rebuild staleness -- see leo_ann_index.py docstring.
            # cat_retired is passed here too (not just to rebuild()) so a
            # slot retired mid-cycle stops being a candidate immediately,
            # rather than remaining eligible until the next rebuild.
            candidate_idx = ann_index.query_candidates(weighted_detection_vec, n_cat, cat_retired)

            # --- ANN candidate recall diagnostic (ground-truth only) ---
            # Was the TRUE matching slot for this detection's true object
            # present in the candidate pool the ANN index returned, BEFORE
            # any re-ranking/gating? This measures the ANN layer's
            # approximation quality in isolation from the gate/GNN
            # decision that happens afterward -- if the true slot isn't in
            # candidate_idx, no downstream logic (gate OR GNN) ever gets a
            # chance to find it, regardless of how good those decisions
            # are. Only checked when a ground-truth slot already exists
            # for this true object AND that slot is currently active (a
            # retired slot is deliberately excluded from candidates by
            # design -- see leo_ann_index.py -- so it "missing" from the
            # pool isn't an ANN approximation failure).
            _gt_idx_for_recall = true_name_to_idx.get(name)
            if (_gt_idx_for_recall is not None and _gt_idx_for_recall < n_cat
                    and not cat_retired[_gt_idx_for_recall]):
                ann_recall_checked += 1
                if _gt_idx_for_recall in candidate_idx:
                    ann_recall_hit += 1

        if len(candidate_idx) > 0:
            # --- exact re-rank over ONLY the candidate set (small, O(K)) ---
            # Propagate and re-embed just these candidates to the CURRENT
            # exact epoch -- this is what makes the final distance/gate
            # numbers exact (not approximate), independent of tree staleness.
            cand_a = np.array([cat_a[j] for j in candidate_idx])
            cand_e = np.array([cat_e[j] for j in candidate_idx])
            cand_i = np.array([cat_i[j] for j in candidate_idx])
            cand_raan = np.array([cat_raan[j] for j in candidate_idx])
            cand_argp = np.array([cat_argp[j] for j in candidate_idx])
            cand_bc = np.array([cat_bc[j] for j in candidate_idx])
            cand_epoch = np.array([cat_epoch[j] for j in candidate_idx])
            cand_types = [cat_types[j] for j in candidate_idx]
            cand_sizes = np.array([cat_sizes[j] for j in candidate_idx])
            cand_rcs = np.array([cat_rcs[j] for j in candidate_idx])

            dt_cand = real_now - cand_epoch
            cprop_a, cprop_e, cprop_i, cprop_raan, cprop_argp = op.propagate_catalog_analytic(
                cand_a, cand_e, cand_i, cand_raan, cand_argp, cand_bc, dt_cand
            )
            cand_vectors = embed_mod.build_embeddings_fast(
                cprop_a, cprop_e, cprop_i, cprop_raan, cprop_argp, cand_types, cand_sizes, cand_rcs, scalers
            )

            # prop_* / cat_vectors / unweighted_dists below are exposed as
            # sparse dict-like lookups keyed by ABSOLUTE catalog index
            # (candidate_idx[k] -> value at local k), since downstream code
            # (resolve_update, mismatch diagnostics) addresses entries by
            # absolute catalog index, which may include indices outside
            # today's candidate pool (e.g. existing_idx / correct_idx below).
            local_of = {int(j): k for k, j in enumerate(candidate_idx)}

            def prop_a(j):
                return cprop_a[local_of[j]]

            def prop_e(j):
                return cprop_e[local_of[j]]

            def prop_i(j):
                return cprop_i[local_of[j]]

            def prop_raan(j):
                return cprop_raan[local_of[j]]

            def prop_argp(j):
                return cprop_argp[local_of[j]]

            def cat_vector_for(j):
                """Identity vector (unweighted, current epoch) for absolute
                catalog index j. Exact for candidates already propagated
                this pass; for an out-of-pool index (only reached by the
                ground-truth-only mismatch diagnostic below) propagate it
                on demand -- rare enough (diagnostics only) that O(1) extra
                propagate calls don't matter."""
                if j in local_of:
                    return cand_vectors[local_of[j]]
                dt_j = real_now - cat_epoch[j]
                pa, pe, pi, praan, pargp = op.propagate_catalog_analytic(
                    np.array([cat_a[j]]), np.array([cat_e[j]]), np.array([cat_i[j]]),
                    np.array([cat_raan[j]]), np.array([cat_argp[j]]), np.array([cat_bc[j]]), dt_j,
                )
                return embed_mod.build_embeddings_fast(
                    pa, pe, pi, praan, pargp, [cat_types[j]], [cat_sizes[j]], [cat_rcs[j]], scalers
                )[0]

            def unweighted_dist_for(j):
                return float(np.linalg.norm(cat_vector_for(j) - detection_vec))

            # Weight AFTER embedding, BEFORE distance -- must be applied to both
            # sides identically or matching silently reverts to the unweighted
            # (collision-prone) space. cand_vectors stays unweighted for anything
            # downstream that needs the raw identity (e.g. the GNN, which learns
            # its own notion of relevant structure and shouldn't be second-guessed
            # by a hand-tuned weighting meant for the cheap distance gate).
            weighted_cand_vectors = cand_vectors * distance_weights

            cand_weighted_dists = np.linalg.norm(weighted_cand_vectors - weighted_detection_vec, axis=1)
            # unweighted distances -- this is the space gnn_matcher.py was
            # trained in (generate_training_examples selects its top-5 from
            # raw build_embeddings_fast output, no weighting). top5 MUST be
            # selected from this array, not the weighted one: selecting
            # candidates by weighted distance changes WHICH 5 objects the GNN
            # ever gets to see, so at inference time it can be handed a
            # candidate set that doesn't contain the true match at all --
            # the GNN then confidently picks the best of a wrong shortlist.
            cand_unweighted_dists = np.linalg.norm(cand_vectors - detection_vec, axis=1)

            local_order = np.argsort(cand_unweighted_dists)
            top5_local = local_order[:5]
            top5 = candidate_idx[top5_local]  # absolute catalog indices, same semantics as before

            # The fast gate's accept/reject decision uses WEIGHTED distance,
            # evaluated only among these same top5 candidates -- weighting
            # still does its job of making the accept/reject call more
            # discriminating, it just no longer changes candidate selection.
            top5_weighted_dists = cand_weighted_dists[top5_local]
            gate_order = np.argsort(top5_weighted_dists)
            best_idx = int(top5[gate_order[0]])
            best_dist = top5_weighted_dists[gate_order[0]]
            second_dist = top5_weighted_dists[gate_order[1]] if len(top5) > 1 else np.inf
            margin = second_dist - best_dist

            accepted = best_dist < drift_tolerance and margin > margin_required

            # candidate_vecs / candidate_dists (unweighted) for the GNN,
            # indexed the same way the old top5-over-full-catalog code did.
            candidate_vecs_for_gnn = cand_vectors[top5_local]
            candidate_dists_for_gnn = cand_unweighted_dists[top5_local]
        else:
            # Either the catalog is empty, or every candidate the ANN
            # index returned is currently retired (degenerate but
            # possible, e.g. immediately after a mass-retirement event) --
            # both cases are "nothing to match against." local_of stays an
            # empty dict (rather than undefined) so the later
            # "best_idx in local_of" check downstream (existing_slot path
            # can still accept in this branch) safely falls through to the
            # on-demand single-object propagate path instead of a NameError.
            local_of = {}
            accepted = False
            best_dist, margin, best_idx = np.inf, 0.0, -1

        # accept_path tracks which mechanism accepted this detection, purely
        # for logging/metrics -- the corroboration gate treats all three
        # uniformly via resolve_update().
        accept_path = "gate" if accepted else None
        gnn_used = False
        gnn_registered = False

        if not accepted and len(candidate_idx) > 0 and gnn_model is not None:
            gnn_used = True
            candidate_vecs = candidate_vecs_for_gnn
            candidate_dists = candidate_dists_for_gnn
            gnn_known, gnn_local_idx, gnn_conf = gnn_mod.classify(
                gnn_model, detection_vec, candidate_vecs, candidate_dists
            )
            if gnn_known:
                accepted = True
                accept_path = "gnn"
                best_idx = int(top5[gnn_local_idx])
                best_dist = candidate_dists[gnn_local_idx]
            else:
                # CONFIRMED UNKNOWN by both the gate and the GNN. Before minting
                # a brand-new entry, check whether this true object was already
                # discovered on an earlier pass -- if the gate/GNN reject it
                # again (e.g. propagation drift outran the catalog's belief, or
                # a noisy measurement), the right move is to correct that
                # existing entry, not spawn a second identity for the same
                # physical object. Without this check, an object that gets
                # unlucky twice ends up as two permanently-separate catalog
                # entries (both then legitimately close to future detections,
                # which is what produced repeat registrations like
                # satellite_908/debris_736 across a run).
                #
                # NOTE: true_name_to_idx is ground-truth bookkeeping we keep
                # ONLY for diagnostics (see comment where it's built) -- a
                # real system wouldn't have it. Using it to relocate "where
                # this object's slot is" is fine (that's just plumbing), but
                # it is NOT independent confirmation that this measurement is
                # correct, so it still goes through the same corroboration
                # gate as every other path rather than writing immediately.
                existing_idx = true_name_to_idx.get(name)
                if existing_idx is not None and existing_idx < len(cat_names):
                    gnn_registered = False
                    best_idx = existing_idx
                    accepted = True
                    accept_path = "existing_slot"
                    best_dist = unweighted_dist_for(existing_idx)
                    if cat_retired[existing_idx]:
                        # The true object was re-acquired after its slot
                        # had been retired (decayed-out-of-orbit false
                        # positive, or simply unseen past the retirement
                        # window) -- un-retire it so it's visible to the
                        # ANN index again. Still routes through the same
                        # resolve_update() corroboration gate below before
                        # any belief is actually rewritten.
                        cat_retired[existing_idx] = False
                else:
                    gnn_registered = True
                    new_name = f"discovered_{discovery_counter}"
                    discovery_counter += 1
                    discovered_true_name[new_name] = name
                    true_name_to_idx[name] = len(cat_names)  # this will be its index once appended below
                    cat_names.append(new_name)
                    cat_a.append(meas_a); cat_e.append(meas_e); cat_i.append(meas_i)
                    cat_raan.append(meas_raan); cat_argp.append(meas_argp)
                    cat_bc.append(op.ballistic_coefficient(sizes[idx], types[idx]))
                    cat_types.append(types[idx]); cat_sizes.append(sizes[idx]); cat_rcs.append(rcs_vals[idx])
                    cat_epoch.append(real_now)
                    cat_retired.append(False)
                    cat_confidence.append(1)
                    cat_last_seen.append(real_now)
                    # a freshly registered entry has no prior belief to
                    # corrupt -- no need to route it through resolve_update.
                    # It DOES need to be visible to the ANN index before the
                    # tree's next rebuild, though -- stash it in the overflow
                    # buffer so subsequent detections can match against it.
                    ann_index.mark_dirty(len(cat_names) - 1)
                    # Ghost-registration bookkeeping (ground-truth only --
                    # ties every catalog slot EVER created back to the true
                    # object it represents, so we can detect at the end of
                    # the run whether any true object ended up with more
                    # than one active slot, i.e. was independently
                    # "discovered" twice under two different discovered_N
                    # names because a tracking gap broke continuity).
                    true_name_to_all_slots.setdefault(name, set()).add(len(cat_names) - 1)

        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1_000_000

        total_detections += 1
        obj_type = types[idx]

        update_committed = False
        if accepted:
            known_count += 1
            lookup_times_known.append(elapsed_us)
            match_distances.append(best_dist)
            match_margins.append(margin)
            per_type_known[obj_type] = per_type_known.get(obj_type, 0) + 1
            status = "KNOWN"
            if gnn_used:
                gnn_confirmed_known += 1

            # --- UPDATE: single unified corroboration gate for ALL paths ---
            # For paths other than "gate", best_idx may reference a slot
            # that wasn't in this pass's ANN candidate pool (e.g.
            # existing_slot correction can point anywhere in the catalog).
            # prop_a(best_idx) etc. handle that transparently: exact lookup
            # for anything already propagated this pass (all of gate/gnn's
            # candidates, and existing_slot when it happens to coincide),
            # on-demand propagation otherwise (see cat_vector_for above --
            # prop_a/e/i/raan/argp reuse the same local_of cache, falling
            # back to a single-object propagate_catalog_analytic call only
            # for the rare out-of-pool existing_slot case).
            if best_idx in local_of:
                pa_val, pe_val, pi_val = prop_a(best_idx), prop_e(best_idx), prop_i(best_idx)
                praan_val, pargp_val = prop_raan(best_idx), prop_argp(best_idx)
            else:
                dt_b = real_now - cat_epoch[best_idx]
                pa_arr, pe_arr, pi_arr, praan_arr, pargp_arr = op.propagate_catalog_analytic(
                    np.array([cat_a[best_idx]]), np.array([cat_e[best_idx]]), np.array([cat_i[best_idx]]),
                    np.array([cat_raan[best_idx]]), np.array([cat_argp[best_idx]]), np.array([cat_bc[best_idx]]), dt_b,
                )
                pa_val, pe_val, pi_val = pa_arr[0], pe_arr[0], pi_arr[0]
                praan_val, pargp_val = praan_arr[0], pargp_arr[0]

            update_committed = resolve_update(
                best_idx, name, meas_a, meas_e, meas_i, meas_raan, meas_argp,
                pa_val, pe_val, pi_val, praan_val, pargp_val, real_now,
            )
            # Whether committed or withheld, this slot's stored epoch/state
            # may now be inconsistent with what the ANN tree last indexed --
            # keep it in the overflow buffer so it's exactly re-propagated
            # on every query until the next rebuild folds it back in.
            ann_index.mark_dirty(best_idx)
            # Any accepted detection -- committed or withheld -- counts as
            # "seen" for retirement purposes (README issue #3): a slot that
            # keeps getting corroboration attempts, even unconfirmed ones,
            # is clearly still being tracked, not lost.
            cat_last_seen[best_idx] = real_now
            if update_committed:
                update_committed_count += 1
                cat_confidence[best_idx] += 1
            else:
                update_withheld_count += 1
        else:
            unknown_count += 1
            lookup_times_unknown.append(elapsed_us)
            per_type_unknown[obj_type] = per_type_unknown.get(obj_type, 0) + 1
            status = "NOT KNOWN"
            if gnn_registered:
                gnn_new_discoveries += 1

        matched_name = cat_names[best_idx] if accepted else None
        matched_true_name = discovered_true_name.get(matched_name, matched_name)
        matched_correctly = accepted and matched_true_name == name
        flag = "" if not accepted else ("" if matched_correctly else "  <-- MISMATCH!")

        # --- Association Integrity metrics (ground-truth only) ---
        if accepted:
            if matched_correctly:
                correct_match_count += 1
            else:
                mismatch_count += 1
        # GNN overturn precision: isolate detections where the FAST GATE
        # rejected first and the GNN is what actually accepted the match
        # (accept_path == "gnn") -- measures whether the GNN's second
        # opinion is a net positive (resolves genuine gate misses) or a
        # net negative (introduces false associations the gate correctly
        # avoided) for the specific subset of hard/boundary cases it's
        # actually invoked on.
        if accept_path == "gnn":
            gnn_overturn_total += 1
            if matched_correctly:
                gnn_overturn_correct += 1

        diag_flag = ""
        if accepted and not matched_correctly:
            correct_idx = true_name_to_idx.get(name)
            if correct_idx is not None and correct_idx < len(cat_names):
                # weighted, to match the space best_dist was computed in
                # (gate path: weighted; GNN/existing-slot paths: best_dist was
                # reassigned to an unweighted value, so weight this the same
                # way for a fair comparison there too)
                dist_to_correct_w = float(np.linalg.norm(
                    (cat_vector_for(correct_idx) - detection_vec) * distance_weights
                ))
                dist_to_wrong_w = best_dist if accept_path == "gate" else float(
                    np.linalg.norm((cat_vector_for(best_idx) - detection_vec) * distance_weights)
                )
                if dist_to_correct_w < dist_to_wrong_w:
                    diag_flag = f"  [BUG? correct entry was CLOSER: dist_to_correct={dist_to_correct_w:.4f} < dist_to_wrong={dist_to_wrong_w:.4f}]"
                else:
                    diag_flag = f"  [ambiguity: dist_to_correct={dist_to_correct_w:.4f} >= dist_to_wrong={dist_to_wrong_w:.4f}]"
            else:
                diag_flag = "  [correct entry not yet in catalog at all]"

        path_flag = ""
        if not accepted and gnn_registered:
            path_flag = "  [registered new]"
        elif accepted:
            commit_tag = "committed" if update_committed else "withheld, pending corroboration"
            if accept_path == "gate":
                path_flag = f"  [gate, update {commit_tag}]"
            elif accept_path == "gnn":
                path_flag = f"  [GNN->KNOWN, update {commit_tag}]"
            elif accept_path == "existing_slot":
                path_flag = f"  [existing-slot correction, update {commit_tag}]"

        print(f"[T+{sim_time:7.2f}s sim] {name:<20} -> {status:<10} "
              f"(dist={best_dist:.3f}, margin={margin:.3f}, lookup {elapsed_us:.2f} us){flag}{path_flag}{diag_flag}")

        # -------------------------------------------------------------
        # Retirement housekeeping (README issue #3): every
        # RETIREMENT_CHECK_EVERY detections, scan currently-active slots
        # and retire any that have decayed below the altitude floor or
        # haven't been (attempted-)matched in RETIREMENT_UNSEEN_SECONDS.
        # This is a per-slot O(active) scan, cheap relative to the O(N)
        # ANN rebuild it runs far more often than.
        # -------------------------------------------------------------
        if total_detections % RETIREMENT_CHECK_EVERY == 0 and real_now > RETIREMENT_WARMUP_SECONDS:
            for j in range(len(cat_names)):
                if cat_retired[j]:
                    continue
                dt_j = real_now - cat_epoch[j]
                pa_arr, _, _, _, _ = op.propagate_catalog_analytic(
                    np.array([cat_a[j]]), np.array([cat_e[j]]), np.array([cat_i[j]]),
                    np.array([cat_raan[j]]), np.array([cat_argp[j]]), np.array([cat_bc[j]]), dt_j,
                )
                prop_altitude_km = pa_arr[0] - op.RE_EARTH
                time_since_seen = real_now - cat_last_seen[j]

                # propagate_catalog_analytic() clamps new_a at exactly
                # RE_EARTH + 100 once decay would take it lower (see its
                # `np.maximum(..., RE_EARTH + 100)` floor) -- so a decayed
                # object's propagated altitude is never STRICTLY LESS than
                # RETIREMENT_ALTITUDE_FLOOR_KM (100.0), it sits AT it. Use
                # <= here, not <, or this check can never fire (verified:
                # a fast-decaying object's altitude asymptotes to exactly
                # 100.0km and stays there, rather than continuing below).
                is_decayed = prop_altitude_km <= RETIREMENT_ALTITUDE_FLOOR_KM
                is_unseen = time_since_seen > RETIREMENT_UNSEEN_SECONDS

                if is_decayed or is_unseen:
                    cat_retired[j] = True
                    reason = "decayed out of orbit" if is_decayed else "unseen too long"
                    if is_decayed:
                        retired_decayed_count += 1
                    else:
                        retired_unseen_count += 1
                    print(f"[T+{sim_time:7.2f}s sim] [RETIRE] {cat_names[j]:<20} ({reason}, "
                          f"alt={prop_altitude_km:.1f}km, unseen={time_since_seen/3600:.1f}h)")
                    # a retired slot has no pending corroboration story
                    # worth keeping around; drop it so a later un-retirement
                    # (re-acquisition) starts corroboration fresh.
                    pending_update.pop(j, None)

        # -------------------------------------------------------------
        # De-duplication / merge pass (README issue #3): every
        # DEDUP_CHECK_EVERY detections, check whether any two ACTIVE slots
        # have drifted close enough together in identity-space that
        # they're almost certainly the same physical object having been
        # registered twice (e.g. after two unlucky consecutive misses on
        # the same real object -- see the existing "existing_slot" path's
        # comment for how this can happen despite that mitigation).
        # Uses the CURRENT drift_tolerance (which may itself have just
        # been recalibrated above) to scale the merge threshold, so it
        # stays meaningful as catalog density changes over the run.
        # -------------------------------------------------------------
        if total_detections % DEDUP_CHECK_EVERY == 0 and len(cat_names) > 1:
            active_indices = np.where(~np.asarray(cat_retired, dtype=bool))[0]
            if len(active_indices) > 1:
                da = np.array([cat_a[j] for j in active_indices])
                de = np.array([cat_e[j] for j in active_indices])
                di = np.array([cat_i[j] for j in active_indices])
                draan = np.array([cat_raan[j] for j in active_indices])
                dargp = np.array([cat_argp[j] for j in active_indices])
                dbc = np.array([cat_bc[j] for j in active_indices])
                depoch = np.array([cat_epoch[j] for j in active_indices])
                dtypes = [cat_types[j] for j in active_indices]
                dsizes = np.array([cat_sizes[j] for j in active_indices])
                drcs = np.array([cat_rcs[j] for j in active_indices])

                d_dt = real_now - depoch
                dp_a, dp_e, dp_i, dp_raan, dp_argp = op.propagate_catalog_analytic(
                    da, de, di, draan, dargp, dbc, d_dt
                )
                d_vecs = embed_mod.build_embeddings_fast(
                    dp_a, dp_e, dp_i, dp_raan, dp_argp, dtypes, dsizes, drcs, scalers
                )
                weighted_d_vecs = d_vecs * distance_weights

                n_neighbors_dedup = min(2, len(active_indices))
                dedup_nn = NearestNeighbors(n_neighbors=n_neighbors_dedup, algorithm="auto")
                dedup_nn.fit(weighted_d_vecs)
                dedup_dists, dedup_pos = dedup_nn.kneighbors(weighted_d_vecs)

                merge_threshold = DEDUP_DISTANCE_FRACTION_OF_DRIFT_TOLERANCE * drift_tolerance
                merged_this_pass = set()  # local (active_indices-relative) positions already merged away
                # NOTE: a cluster of 3+ mutually-close objects collapses
                # over MULTIPLE dedup passes, not necessarily all at once.
                # If object A's nearest neighbor B gets merged away first
                # (earlier in this loop), A is left as-is this pass even if
                # A was also close to B/C -- its neighbor pointer is now
                # stale. The next DEDUP_CHECK_EVERY pass re-queries fresh
                # neighbors and continues collapsing the cluster. This is
                # deliberately conservative (never merges based on a
                # neighbor that's already been retired this pass) rather
                # than chaining merges within a single pass.
                if n_neighbors_dedup > 1:
                    for local_pos in range(len(active_indices)):
                        if local_pos in merged_this_pass:
                            continue
                        neighbor_local_pos = dedup_pos[local_pos, 1]
                        d = dedup_dists[local_pos, 1]
                        if neighbor_local_pos in merged_this_pass or neighbor_local_pos == local_pos:
                            continue
                        if d < merge_threshold:
                            idx1 = int(active_indices[local_pos])
                            idx2 = int(active_indices[neighbor_local_pos])
                            # keep whichever slot has more corroborated
                            # history (cat_confidence), retire the other
                            # and fold its confidence into the survivor so
                            # the merged identity's "how sure are we"
                            # signal isn't lost.
                            if cat_confidence[idx2] > cat_confidence[idx1]:
                                keep_idx, retire_idx = idx2, idx1
                                retire_local = local_pos
                            else:
                                keep_idx, retire_idx = idx1, idx2
                                retire_local = neighbor_local_pos
                            cat_retired[retire_idx] = True
                            cat_confidence[keep_idx] += cat_confidence[retire_idx]
                            pending_update.pop(retire_idx, None)
                            merged_this_pass.add(retire_local)
                            merged_count_total += 1
                            print(f"[T+{sim_time:7.2f}s sim] [MERGE] {cat_names[retire_idx]:<20} "
                                  f"-> {cat_names[keep_idx]} (identity dist={d:.4f} < "
                                  f"threshold={merge_threshold:.4f})")
                if merged_this_pass:
                    # a merge changes which slots are active -- force the
                    # ANN tree to reflect that before the next query rather
                    # than waiting up to REBUILD_EVERY detections, the same
                    # reasoning as retirement's immediate query_candidates
                    # filtering above (belt-and-suspenders: query_candidates
                    # already filters live via cat_retired regardless of
                    # tree staleness, but marking dirty keeps the overflow
                    # bookkeeping consistent too).
                    for retired_local in merged_this_pass:
                        ann_index.mark_dirty(int(active_indices[retired_local]))

        heapq.heappush(heap, (sim_time + sim_periods[idx], idx))

    real_elapsed = time.perf_counter() - real_start

    # -----------------------------------------------------------------
    # Catalog Completeness & Redundancy (ground-truth only, end-of-run)
    # -----------------------------------------------------------------
    # Completeness: fraction of the TRUE population (every object in
    # catalog_df / `names`, not just the ones ever detected) that has at
    # least one ACTIVE catalog slot correctly representing it at the end
    # of the run. "Correctly representing" means the slot's own true
    # identity (via discovered_true_name, or the slot's own name for
    # pre-loaded/un-renamed slots) matches -- an active slot that's
    # mismatched (holding someone else's belief) does NOT count toward
    # completeness for the object it claims to be, since its stored state
    # doesn't actually describe that object.
    active_true_names_correctly_represented = set()
    for cat_idx_final, cat_name_final in enumerate(cat_names):
        if cat_retired[cat_idx_final]:
            continue
        true_name_final = discovered_true_name.get(cat_name_final, cat_name_final)
        active_true_names_correctly_represented.add(true_name_final)

    total_true_population = len(names)  # full catalog_df population, not just ever-detected
    n_complete = len(active_true_names_correctly_represented & set(names))
    catalog_completeness_pct = (n_complete / total_true_population * 100) if total_true_population else 0.0

    # Redundancy: average number of ACTIVE catalog slots per true object
    # that has at least one active slot (an object with 0 active slots
    # doesn't contribute to this average -- it's captured by completeness
    # instead). true_name_to_all_slots was built incrementally over the
    # run (see registration + pre-load seeding above) and holds every
    # slot index EVER created for that true name; filter to currently-
    # active ones for the end-of-run redundancy snapshot.
    active_slot_counts_per_true_name = []
    for true_name_r, slot_idxs in true_name_to_all_slots.items():
        active_slots_for_this_name = [s for s in slot_idxs if s < len(cat_names) and not cat_retired[s]]
        if active_slots_for_this_name:
            active_slot_counts_per_true_name.append(len(active_slots_for_this_name))
    redundancy_index = (
        sum(active_slot_counts_per_true_name) / len(active_slot_counts_per_true_name)
        if active_slot_counts_per_true_name else 0.0
    )
    n_ghost_registrations = sum(1 for c in active_slot_counts_per_true_name if c > 1)

    # -----------------------------------------------------------------
    # Orbital Element Residuals (RMSE) -- ground-truth only, end-of-run
    # -----------------------------------------------------------------
    # For every ACTIVE, CORRECTLY-represented catalog slot, propagate both
    # the catalog's belief AND the true object's own state to the SAME
    # common reference time (the run's final real_now) and compare. Truth
    # uses the deterministic secular model (propagate_catalog_analytic),
    # matching exactly what the catalog's own belief propagation uses --
    # NOT propagate_elements' noisy version, since RMSE should measure
    # "how far has the belief drifted from where the object actually is,"
    # not introduce another random draw on the truth side.
    #
    # This directly assesses whether ALPHA=0.4's EMA update rate is
    # properly damping measurement noise without lagging behind the
    # object's real secular drift (see README issue #4, EMA has no real
    # uncertainty model) -- a belief that's consistently biased relative
    # to truth here is exactly the failure mode that issue describes.
    name_to_true_row_idx = {n: i for i, n in enumerate(names)}
    # real_now retains its value from the loop's last iteration (ordinary
    # Python scoping -- loop variables aren't cleared after the loop). If
    # the loop never ran at all (empty heap / zero-duration run), it was
    # never assigned; fall back to 0.0 for that degenerate case.
    rmse_final_real_now = real_now if total_detections > 0 else 0.0

    residuals_a, residuals_e, residuals_i, residuals_raan, residuals_argp = [], [], [], [], []
    for cat_idx_final, cat_name_final in enumerate(cat_names):
        if cat_retired[cat_idx_final]:
            continue
        true_name_final = discovered_true_name.get(cat_name_final, cat_name_final)
        true_row_idx = name_to_true_row_idx.get(true_name_final)
        if true_row_idx is None:
            continue

        # propagate the catalog's stored belief to the common reference time
        dt_belief = rmse_final_real_now - cat_epoch[cat_idx_final]
        belief_a, belief_e, belief_i, belief_raan, belief_argp = op.propagate_catalog_analytic(
            np.array([cat_a[cat_idx_final]]), np.array([cat_e[cat_idx_final]]), np.array([cat_i[cat_idx_final]]),
            np.array([cat_raan[cat_idx_final]]), np.array([cat_argp[cat_idx_final]]),
            np.array([cat_bc[cat_idx_final]]), dt_belief,
        )
        # propagate the true object's own last-known state to the same
        # reference time using the SAME deterministic model, so any
        # difference reflects belief drift, not two different propagators
        # disagreeing with each other.
        dt_truth_final = rmse_final_real_now - truth_epoch[true_row_idx]
        true_prop_a, true_prop_e, true_prop_i, true_prop_raan, true_prop_argp = op.propagate_catalog_analytic(
            np.array([truth_a[true_row_idx]]), np.array([truth_e[true_row_idx]]), np.array([truth_i[true_row_idx]]),
            np.array([truth_raan[true_row_idx]]), np.array([truth_argp[true_row_idx]]),
            np.array([truth_bc[true_row_idx]]), dt_truth_final,
        )

        residuals_a.append(belief_a[0] - true_prop_a[0])
        residuals_e.append(belief_e[0] - true_prop_e[0])
        residuals_i.append(belief_i[0] - true_prop_i[0])
        # angle residuals need wraparound handling (e.g. 359 deg vs 1 deg
        # is a 2 deg residual, not 358)
        raan_diff = (belief_raan[0] - true_prop_raan[0] + 180) % 360 - 180
        argp_diff = (belief_argp[0] - true_prop_argp[0] + 180) % 360 - 180
        residuals_raan.append(raan_diff)
        residuals_argp.append(argp_diff)

    def rmse(residuals):
        if not residuals:
            return 0.0
        arr = np.asarray(residuals)
        return float(np.sqrt(np.mean(arr ** 2)))

    rmse_a = rmse(residuals_a)
    rmse_e = rmse(residuals_e)
    rmse_i = rmse(residuals_i)
    rmse_raan = rmse(residuals_raan)
    rmse_argp = rmse(residuals_argp)
    n_rmse_slots = len(residuals_a)

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------
    def stats(times):
        if not times:
            return (0.0, 0.0, 0.0)
        return (sum(times) / len(times), min(times), max(times))

    all_times = lookup_times_known + lookup_times_unknown
    avg_all, fast_all, slow_all = stats(all_times)
    avg_k, fast_k, slow_k = stats(lookup_times_known)
    avg_u, fast_u, slow_u = stats(lookup_times_unknown)

    print("\n" + "=" * 70)
    print("TRACKING SIMULATION METRICS")
    print("=" * 70)
    print(f"Total detections           : {total_detections}")
    print(f"  Known (accepted match)   : {known_count} ({known_count/total_detections*100:.1f}%)")
    print(f"  Not known (rejected)     : {unknown_count} ({unknown_count/total_detections*100:.1f}%)")
    print(f"  Belief updates committed : {update_committed_count}")
    print(f"  Belief updates withheld  : {update_withheld_count} (pending corroboration)")
    if USE_GNN_CONFIRMATION:
        print(f"  GNN overturned to KNOWN  : {gnn_confirmed_known}")
        print(f"  GNN confirmed unknown -> new catalog entries: {gnn_new_discoveries}")
        print(f"  Catalog size: {len(store['names'])} -> {len(cat_names)} "
              f"(+{len(cat_names) - len(store['names'])})")
    n_active_final = int(np.sum(~np.asarray(cat_retired, dtype=bool))) if cat_names else 0
    print(f"  Retired (decayed)        : {retired_decayed_count}")
    print(f"  Retired (unseen too long): {retired_unseen_count}")
    print(f"  Merged (de-duplicated)   : {merged_count_total}")
    print(f"  Active catalog at end    : {n_active_final} / {len(cat_names)}")
    print(f"Real wall-clock time spent : {real_elapsed*1000:.3f} ms")
    print()
    if match_distances:
        print(f"Match distance  : avg={np.mean(match_distances):.4f}  "
              f"min={np.min(match_distances):.4f}  max={np.max(match_distances):.4f}")
        print(f"Match margin    : avg={np.mean(match_margins):.4f}  "
              f"min={np.min(match_margins):.4f}  max={np.max(match_margins):.4f}")
    print()
    print(f"{'':<10}{'avg (us)':>12}{'fastest (us)':>16}{'slowest (us)':>16}")
    print(f"{'ALL':<10}{avg_all:>12.3f}{fast_all:>16.3f}{slow_all:>16.3f}")
    print(f"{'KNOWN':<10}{avg_k:>12.3f}{fast_k:>16.3f}{slow_k:>16.3f}")
    print(f"{'UNKNOWN':<10}{avg_u:>12.3f}{fast_u:>16.3f}{slow_u:>16.3f}")

    print("\nPer-type breakdown (known / not known):")
    all_types = sorted(set(per_type_known) | set(per_type_unknown))
    for t in all_types:
        k = per_type_known.get(t, 0)
        u = per_type_unknown.get(t, 0)
        print(f"  {t:<14} known={k:<5} not_known={u:<5} total={k+u}")

    # -----------------------------------------------------------------
    # SDA / ML evaluation metrics (ground-truth-dependent -- see README
    # "no ground truth in a real deployment" issue: these are only
    # measurable HERE because the simulator tracks true identity; a real
    # deployment has no oracle for most of these and would need the
    # residual-monitoring/spot-check alternatives that issue describes)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SDA / ML EVALUATION METRICS (ground truth, simulator-only)")
    print("=" * 70)

    print("\n-- 1. Association Integrity --")
    n_accepted_total = correct_match_count + mismatch_count
    mismatch_rate_pct = (mismatch_count / n_accepted_total * 100) if n_accepted_total else 0.0
    print(f"Mismatch (false association) rate : {mismatch_count}/{n_accepted_total} "
          f"({mismatch_rate_pct:.3f}%)")
    print(f"  Correct associations            : {correct_match_count}")
    print(f"Ghost registrations (true objects with >1 active slot): {n_ghost_registrations}")

    print("\n-- 2. Catalog Completeness & Duplication --")
    print(f"Catalog completeness : {n_complete}/{total_true_population} true objects "
          f"({catalog_completeness_pct:.2f}%) have >=1 active, correctly-labeled slot")
    print(f"Redundancy index      : {redundancy_index:.4f} active slots per represented true "
          f"object (1.0 = no duplication)")

    print("\n-- 3. State Estimation Quality (Orbital Element Residuals, RMSE) --")
    if n_rmse_slots > 0:
        print(f"Computed over {n_rmse_slots} active, correctly-labeled slots at "
              f"real_now={rmse_final_real_now:.1f}s")
        print(f"  RMSE a     (km)  : {rmse_a:.5f}")
        print(f"  RMSE e     (-)   : {rmse_e:.7f}")
        print(f"  RMSE i     (deg) : {rmse_i:.5f}")
        print(f"  RMSE RAAN  (deg) : {rmse_raan:.5f}")
        print(f"  RMSE argp  (deg) : {rmse_argp:.5f}")
    else:
        print("  No active, correctly-labeled slots to compute residuals over.")

    print("\n-- 4. ANN Search Tree Performance --")
    ann_recall_pct = (ann_recall_hit / ann_recall_checked * 100) if ann_recall_checked else 0.0
    print(f"Candidate recall (true slot in ANN pool before re-rank): "
          f"{ann_recall_hit}/{ann_recall_checked} ({ann_recall_pct:.3f}%)")
    print(f"  (checked only when a ground-truth slot for the detection's true object "
          f"already existed and was active; candidate_pool_size={ANN_CANDIDATE_POOL_SIZE})")

    print("\n-- 5. GNN Decision Quality --")
    if gnn_model is not None:
        gnn_overturn_precision_pct = (gnn_overturn_correct / gnn_overturn_total * 100) if gnn_overturn_total else 0.0
        print(f"GNN overturn precision : {gnn_overturn_correct}/{gnn_overturn_total} "
              f"({gnn_overturn_precision_pct:.2f}%) of gate-rejected-then-GNN-accepted "
              f"matches were correct")
    else:
        print("  GNN confirmation disabled this run (torch unavailable or USE_GNN_CONFIRMATION=False).")


if __name__ == "__main__":
    main()