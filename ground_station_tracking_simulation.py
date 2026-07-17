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

# ---------------------------------------------------------------------
# Cold start (see ISSUES_AND_UPGRADE_PATH.md #1)
# ---------------------------------------------------------------------
# A true cold-start deployment has no pre-known catalog at all -- every
# object starts "unknown" and has to earn its way in via repeated
# corroborated sightings. load_embedding_store() below transparently
# accepts either a normally-built store (some fraction pre-known) or the
# empty scaffold leo_catalog_embed.py writes when run with --cold-start /
# EMBED_FRACTION=0.0 -- both have the identical dict shape, so nothing
# downstream needs to branch on which one it got.

# New-object REGISTRATION requires this many corroborating detections
# landing on the same pending candidate before a catalog slot is actually
# minted -- the same "wait for a second agreeing observation" idea already
# used for belief UPDATES (UPDATE_CONFIRMATIONS_REQUIRED), just applied one
# step earlier in the pipeline so a cold-start flood of first-pass
# registrations can't poison the catalog with noise-driven duplicates.
# 1 = old (unsafe at cold start) behavior: any GNN-confirmed "unknown"
#     immediately mints a new slot.
REGISTRATION_CONFIRMATIONS_REQUIRED = 2

# Below this many registered catalog objects, auto-calibration
# (calibrate_thresholds) is skipped in favor of a wide, deliberately
# permissive fallback -- with only a handful of objects registered, most
# "clusters" have 0-1 members and the percentile calculation falls back to
# the noisy global case, producing thresholds that don't reflect the
# catalog's eventual (post-warm-up) density. Chosen so that at least a few
# objects land in most of the catalog's inclination/type/eccentricity
# buckets before trusting a measured percentile.
MIN_CATALOG_SIZE_FOR_CALIBRATION = 200
# Fallback (DRIFT_TOLERANCE, MARGIN_REQUIRED) used below
# MIN_CATALOG_SIZE_FOR_CALIBRATION -- deliberately wide/permissive so early
# objects aren't wrongly rejected while the catalog is too sparse to
# measure real density. These are generous relative to the typical
# calibrated values (~0.05-0.15 / ~0.02-0.06) seen once the catalog warms
# up, so early false-rejects are far more likely than early false-accepts.
SPARSE_CATALOG_FALLBACK_DRIFT_TOLERANCE = 0.5
SPARSE_CATALOG_FALLBACK_MARGIN_REQUIRED = 0.05

# Refit real scalers (StandardScaler fit on the growing catalog) once this
# many objects have been registered, replacing the domain-default cold-
# start prior. See leo_catalog_embed.refit_scalers_in_place.
MIN_CATALOG_SIZE_FOR_SCALER_REFIT = 50


# ---------------------------------------------------------------------
# Load embedding store
# ---------------------------------------------------------------------
def load_embedding_store(pt_path, npz_fallback):
    """
    Loads a pre-built store, if one exists. This covers BOTH:
      - a normally-built store (some EMBED_FRACTION of the catalog
        pre-known), and
      - the empty scaffold leo_catalog_embed.py writes when run with
        --cold-start / EMBED_FRACTION=0.0 (identical dict shape, just
        zero-length catalog-sized arrays).
    Both come back from this function in exactly the same shape, so no
    downstream consumer needs to branch on which one it got.

    If NEITHER file exists (no store has ever been built), fall back to
    constructing the empty scaffold directly in-process rather than
    failing -- this is the true "never ran leo_catalog_embed.py at all"
    cold-start entry point (see ISSUES_AND_UPGRADE_PATH.md #1).
    """
    if TORCH_AVAILABLE:
        try:
            data = torch.load(pt_path, weights_only=False)
            data["vectors"] = data["vectors"].numpy()
            return data
        except FileNotFoundError:
            pass
    try:
        raw = np.load(npz_fallback, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}
        data["scalers"] = raw["scalers"][0]  # unwrap the object-array wrapping used in the npz fallback
        data["names"] = data["names"].tolist()
        data["types"] = data["types"].tolist()
        return data
    except FileNotFoundError:
        print(f"No embedding store found at {pt_path} or {npz_fallback} -- "
              f"starting from an in-process cold-start scaffold "
              f"(0 pre-known objects, domain-default scalers).")
        return embed_mod.build_empty_scaffold_store()


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
    """
    from scipy.spatial.distance import pdist, squareform

    if cat_i is None or cat_types is None or cat_e is None:
        D = squareform(pdist(catalog_vectors))
        np.fill_diagonal(D, np.inf)
        nearest = D.min(axis=1)
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
        Dsub = squareform(pdist(sub))
        np.fill_diagonal(Dsub, np.inf)
        within_bucket_gaps.extend(Dsub.min(axis=1).tolist())

    if not within_bucket_gaps:
        # degenerate catalog (every object in its own bucket) -- fall back
        D = squareform(pdist(catalog_vectors))
        np.fill_diagonal(D, np.inf)
        within_bucket_gaps = D.min(axis=1).tolist()

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
        if len(store["names"]) < MIN_CATALOG_SIZE_FOR_CALIBRATION:
            # Sparse/empty catalog (cold start, or early warm-up): a
            # measured percentile here is noise, not signal -- most
            # buckets have 0-1 members. Use a deliberately wide/permissive
            # fallback instead so early objects aren't wrongly rejected
            # while there isn't enough density to measure yet. See
            # ISSUES_AND_UPGRADE_PATH.md #1.
            drift_tolerance = SPARSE_CATALOG_FALLBACK_DRIFT_TOLERANCE
            margin_required = SPARSE_CATALOG_FALLBACK_MARGIN_REQUIRED
            print(f"Catalog has only {len(store['names'])} pre-known objects "
                  f"(< {MIN_CATALOG_SIZE_FOR_CALIBRATION}) -- skipping auto-calibration, "
                  f"using sparse-catalog fallback thresholds: "
                  f"DRIFT_TOLERANCE={drift_tolerance}, MARGIN_REQUIRED={margin_required}")
        else:
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
    # Minimum pre-known catalog size needed before the GNN has anything
    # meaningful to train on -- generate_training_examples samples "known"
    # examples FROM the catalog itself (rng.integers(0, n_cat)) and needs a
    # non-trivial top5 to build training graphs from. At true cold start
    # (n_cat=0) there's nothing to train on yet; defer training until
    # enough objects have been registered via corroborated GNN-confirmed
    # discoveries during warm-up (see ISSUES_AND_UPGRADE_PATH.md #1). Until
    # then, rejected detections simply register as new catalog entries
    # (subject to REGISTRATION_CONFIRMATIONS_REQUIRED) without a GNN second
    # opinion -- exactly matching what happens today when PyTorch isn't
    # available at all.
    MIN_CATALOG_SIZE_FOR_GNN_TRAINING = 50
    gnn_deferred = False
    if USE_GNN_CONFIRMATION:
        if not gnn_mod.TORCH_AVAILABLE:
            print("GNN confirmation requested but PyTorch isn't available -- skipping "
                  "(rejected detections will just stay NOT KNOWN, no catalog growth).")
        elif len(store["names"]) < MIN_CATALOG_SIZE_FOR_GNN_TRAINING:
            gnn_deferred = True
            print(f"Catalog has only {len(store['names'])} pre-known objects "
                  f"(< {MIN_CATALOG_SIZE_FOR_GNN_TRAINING}) -- deferring GNN training until "
                  f"warm-up registers enough objects to train on. Gate-rejected detections "
                  f"will register as new entries directly (fast-gate-only second opinion) "
                  f"until then.")
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
    # Confidence/age field (see ISSUES_AND_UPGRADE_PATH.md #3): number of
    # corroborated updates this slot has received. Pre-known store entries
    # start at a nominal confidence of 1 (they were seeded, not
    # corroborated in-run, but they're not "brand new and unproven"
    # either); freshly-registered entries also start at 1 (their one
    # corroborating registration). Incremented on every committed
    # resolve_update(); used by the de-duplication/retirement housekeeping
    # pass to deprioritize/expire low-confidence entries.
    cat_confidence = [1] * len(cat_names)
    # Real-time (real_now) this slot was last associated with ANY detection
    # (accepted match OR fresh registration) -- distinct from cat_epoch,
    # which only advances on a COMMITTED belief-update. Used by the
    # retirement rule (#3) and the maneuver/missed-pass detector (#2) to
    # tell "not seen in a while" apart from "seen but update withheld
    # pending corroboration."
    cat_last_seen = list(cat_epoch)
    # Retired slots (decayed out of orbit, or unseen too long -- see #3):
    # moved out of the active working set, excluded from ann_index.rebuild()
    # and from candidate search, but kept around (not deleted) so indices
    # stay stable and diagnostics/logs can still reference them.
    cat_retired = [False] * len(cat_names)
    discovery_counter = 0
    discovered_true_name = {}  # "discovered_N" -> the true object name it was created from
    registration_committed_count = 0   # new-object registration actually minted this pass
    registration_withheld_count = 0    # candidate registration pending corroboration

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

    # Corroborated REGISTRATION state (see ISSUES_AND_UPGRADE_PATH.md #1):
    # a GNN-confirmed "unknown" doesn't immediately mint a catalog entry --
    # it stashes a candidate registration here, keyed by a coarse identity
    # bucket (object type + a coarse quantization of the measured elements)
    # so a second, independent detection that's plausibly the SAME physical
    # object (not just any new unknown) can corroborate it. Only on that
    # second agreeing observation does the candidate actually get appended
    # to cat_names/cat_a/etc. This reuses the exact same "wait for a second
    # agreeing observation" idea already proven out for belief updates
    # (pending_update / resolve_update) -- applied one step earlier in the
    # pipeline, before a slot even exists to update.
    #
    # Entry shape: bucket_key -> dict(meas_a, meas_e, meas_i, meas_raan,
    # meas_argp, obj_type, size, rcs, real_now, true_name) -- the true_name
    # field is diagnostics-only (ground truth), used purely to log whether
    # corroboration correctly matched the same physical object or
    # accidentally collided two different-but-nearby unknowns into one
    # registration (a real system has no true_name to check this with, but
    # WOULD still functionally corroborate correctly since the bucket key
    # and identity-space proximity are derived from real measurements).
    pending_registration = {}

    def registration_bucket_key(obj_type, meas_a, meas_i, meas_raan):
        """Coarse identity bucket for candidate-registration corroboration:
        groups detections that are plausibly the same physical object
        (same type, close semi-major axis, close inclination) without
        needing an already-built ANN tree over not-yet-registered objects.
        Quantization width chosen deliberately coarse relative to
        measurement noise (MEASUREMENT_NOISE) so two detections of the
        same true object reliably land in the same bucket, but far finer
        than the gap between physically distinct objects -- RAAN is
        excluded from the key itself (it precesses between passes) and
        instead checked via a tolerance at corroboration time."""
        a_bucket = round(meas_a / 5.0)      # 5 km bins
        i_bucket = round(meas_i / 0.5)      # 0.5 deg bins
        return (obj_type, a_bucket, i_bucket)

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

    # Warm-up tracking (see ISSUES_AND_UPGRADE_PATH.md #1): registrations
    # per detection, binned by sim-time, to sanity check that cold start
    # actually behaves as expected -- lots of registrations early, dropping
    # off as the catalog saturates -- rather than continuously registering
    # duplicates throughout the run.
    WARMUP_BIN_SECONDS = 10.0  # sim-seconds per reporting bin
    warmup_bin_detections = {}     # bin_idx -> detection count
    warmup_bin_registrations = {}  # bin_idx -> committed-registration count

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
        registration_committed_this_pass = False

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

        # --- MATCH: this is the real, timed lookup work ---
        t0 = time.perf_counter()

        n_cat = len(cat_names)
        if n_cat > 0:
            weighted_detection_vec = detection_vec * distance_weights

            # Rebuild the ANN tree periodically (amortized O(N) cost) rather
            # than on every detection -- see leo_ann_index.py.
            if ann_index.needs_rebuild():
                # Cold-start scaler refit (see ISSUES_AND_UPGRADE_PATH.md
                # #1): domain-default scalers are a placeholder prior, not
                # a measurement. Once enough objects have been registered
                # to fit real StandardScalers on, swap them in -- every
                # consumer (transform_identity_vector, build_embeddings_fast,
                # the ANN index via `scalers`) reads the `scalers` dict
                # fresh each call, so replacing this one reference is all
                # that's needed; no matching code changes.
                if scalers.get("source") == "domain_default" and len(cat_a) >= MIN_CATALOG_SIZE_FOR_SCALER_REFIT:
                    refitted = embed_mod.refit_scalers_in_place(cat_a, cat_e, cat_sizes, cat_rcs,
                                                                  min_n=MIN_CATALOG_SIZE_FOR_SCALER_REFIT)
                    if refitted is not None:
                        scalers.clear()
                        scalers.update(refitted)
                        ann_index.scalers = scalers
                        print(f"[T+{sim_time:7.2f}s sim] Refit scalers from {len(cat_a)} registered "
                              f"objects (domain_default -> fitted).")

                ann_index.rebuild(
                    cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc,
                    cat_types, cat_sizes, cat_rcs, cat_epoch, real_now,
                    distance_weights,
                )

            # Narrow the search space to a small candidate pool (tree
            # neighbors + overflow buffer) instead of scanning all N. This
            # candidate set is intentionally over-sized (CANDIDATE_POOL_SIZE
            # >> 5) so the true top-5 is reliably contained in it even with
            # inter-rebuild staleness -- see leo_ann_index.py docstring.
            candidate_idx = ann_index.query_candidates(weighted_detection_vec, n_cat)

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
            accepted = False
            best_dist, margin, best_idx = np.inf, 0.0, -1

        # accept_path tracks which mechanism accepted this detection, purely
        # for logging/metrics -- the corroboration gate treats all three
        # uniformly via resolve_update().
        accept_path = "gate" if accepted else None
        gnn_used = False
        gnn_registered = False
        name = names[idx]

        # confirmed_unknown becomes True once every available second-opinion
        # mechanism has had its say and none of them accepted -- this is
        # the trigger for "try existing_slot correction, then registration"
        # below. Two ways to get here:
        #   (a) the GNN ran (n_cat > 0 and a trained model is available)
        #       and it also rejected the detection, or
        #   (b) there's no GNN second opinion to consult at all -- either
        #       the catalog is empty (n_cat == 0, true cold start) or
        #       training hasn't happened yet (gnn_model is None, deferred
        #       per MIN_CATALOG_SIZE_FOR_GNN_TRAINING). In that case the
        #       fast gate's reject is the only opinion there is, and
        #       "not accepted" already means "confirmed unknown" -- this
        #       must still flow into existing_slot/registration or the
        #       catalog can never grow past its seed size while the GNN is
        #       unavailable (a real bug: previously this whole branch was
        #       gated on `gnn_model is not None`, so cold start's 0-object
        #       catalog and any GNN-unavailable run never registered
        #       anything at all).
        confirmed_unknown = False
        if not accepted and n_cat > 0 and gnn_model is not None:
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
                confirmed_unknown = True
        elif not accepted:
            # No GNN second opinion available this pass (empty catalog, or
            # training deferred/unavailable) -- the gate's reject is final.
            confirmed_unknown = True

        if confirmed_unknown:
                # CONFIRMED UNKNOWN by every available check. Before minting
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
                # existing_idx can only be non-None if this true object was
                # already registered on an earlier pass, which requires
                # n_cat > 0 (and therefore unweighted_dist_for to exist) --
                # n_cat == 0 (true cold start, nothing registered yet)
                # always has existing_idx is None, so the n_cat > 0 guard
                # here is a defensive invariant check, not a normal-path
                # branch.
                if existing_idx is not None and existing_idx < len(cat_names) and n_cat > 0:
                    gnn_registered = False
                    best_idx = existing_idx
                    accepted = True
                    accept_path = "existing_slot"
                    best_dist = unweighted_dist_for(existing_idx)
                else:
                    # CONFIRMED UNKNOWN by both gate and GNN, and no
                    # existing slot for this true object -- this is a
                    # genuine new-object candidate. Route it through the
                    # SAME corroboration idea used for belief updates
                    # (REGISTRATION_CONFIRMATIONS_REQUIRED, default 2):
                    # don't mint a catalog slot on a single uncertain call
                    # -- stash it as a pending registration, and only
                    # commit once a second detection lands in the same
                    # coarse identity bucket AND close enough in full
                    # identity-space to plausibly be the same physical
                    # object. See ISSUES_AND_UPGRADE_PATH.md #1: at cold
                    # start, essentially 100% of early detections take this
                    # path, so this is exactly the mechanism that needs
                    # corroboration most.
                    bucket_key = registration_bucket_key(types[idx], meas_a, meas_i, meas_raan)

                    if REGISTRATION_CONFIRMATIONS_REQUIRED <= 1:
                        commit_registration = True
                    else:
                        prior = pending_registration.get(bucket_key)
                        if prior is not None:
                            # Same bucket seen before -- check the two
                            # detections are actually close in identity
                            # space too (the coarse bucket alone isn't
                            # enough to rule out two distinct objects that
                            # happen to share a bucket). Reuse the same
                            # DRIFT_TOLERANCE the fast gate trusts for "is
                            # this plausibly one object" -- no new
                            # threshold needed.
                            prior_vec = embed_mod.transform_identity_vector(
                                prior["meas_a"], prior["meas_e"], prior["meas_i"],
                                prior["meas_raan"], prior["meas_argp"],
                                prior["obj_type"], prior["size"], prior["rcs"], scalers,
                            )
                            this_vec = embed_mod.transform_identity_vector(
                                meas_a, meas_e, meas_i, meas_raan, meas_argp,
                                types[idx], sizes[idx], rcs_vals[idx], scalers,
                            )
                            corroboration_dist = float(np.linalg.norm(
                                (prior_vec - this_vec) * distance_weights
                            ))
                            commit_registration = corroboration_dist < drift_tolerance
                        else:
                            commit_registration = False

                    if commit_registration:
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
                        cat_confidence.append(1)  # first corroborated hit -- see #3/#4
                        cat_last_seen.append(real_now)
                        # a freshly registered entry has no prior belief to
                        # corrupt -- no need to route it through resolve_update.
                        # It DOES need to be visible to the ANN index before the
                        # tree's next rebuild, though -- stash it in the overflow
                        # buffer so subsequent detections can match against it.
                        ann_index.mark_dirty(len(cat_names) - 1)
                        pending_registration.pop(bucket_key, None)
                        registration_committed_count += 1
                        registration_committed_this_pass = True
                    else:
                        gnn_registered = False
                        pending_registration[bucket_key] = {
                            "meas_a": meas_a, "meas_e": meas_e, "meas_i": meas_i,
                            "meas_raan": meas_raan, "meas_argp": meas_argp,
                            "obj_type": types[idx], "size": sizes[idx], "rcs": rcs_vals[idx],
                            "real_now": real_now, "true_name": name,
                        }
                        registration_withheld_count += 1

        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1_000_000

        total_detections += 1
        obj_type = types[idx]

        # Warm-up bin bookkeeping (see ISSUES_AND_UPGRADE_PATH.md #1).
        warmup_bin = int(sim_time // WARMUP_BIN_SECONDS)
        warmup_bin_detections[warmup_bin] = warmup_bin_detections.get(warmup_bin, 0) + 1
        if registration_committed_this_pass:
            warmup_bin_registrations[warmup_bin] = warmup_bin_registrations.get(warmup_bin, 0) + 1

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
            if update_committed:
                update_committed_count += 1
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
        elif not accepted and gnn_used and not gnn_registered and n_cat >= 0 and accept_path is None:
            # GNN confirmed unknown but registration is pending corroboration
            # (see REGISTRATION_CONFIRMATIONS_REQUIRED) -- distinguish this
            # from a plain "not known" so warm-up behavior is legible in logs.
            path_flag = "  [pending registration, awaiting corroboration]"
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

        heapq.heappush(heap, (sim_time + sim_periods[idx], idx))

    real_elapsed = time.perf_counter() - real_start

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
        print(f"  Registrations committed : {registration_committed_count}")
        print(f"  Registrations withheld  : {registration_withheld_count} (pending corroboration)")
        print(f"  Catalog size: {len(store['names'])} -> {len(cat_names)} "
              f"(+{len(cat_names) - len(store['names'])})")
    if len(store["names"]) < MIN_CATALOG_SIZE_FOR_CALIBRATION or gnn_deferred:
        print(f"  Cold-start / warm-up run: started from {len(store['names'])} pre-known "
              f"objects (scalers source={scalers.get('source', 'fitted')}).")
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

    if warmup_bin_detections:
        print("\nWarm-up phase (registrations per detection over time, "
              f"{WARMUP_BIN_SECONDS:.0f}s bins):")
        for b in sorted(warmup_bin_detections):
            det = warmup_bin_detections[b]
            reg = warmup_bin_registrations.get(b, 0)
            rate = reg / det * 100 if det else 0.0
            bar = "#" * min(int(rate / 2), 50)
            print(f"  t=[{b*WARMUP_BIN_SECONDS:6.0f}s,{(b+1)*WARMUP_BIN_SECONDS:6.0f}s)  "
                  f"detections={det:<6} registrations={reg:<5} rate={rate:5.1f}%  {bar}")

    print("\nPer-type breakdown (known / not known):")
    all_types = sorted(set(per_type_known) | set(per_type_unknown))
    for t in all_types:
        k = per_type_known.get(t, 0)
        u = per_type_unknown.get(t, 0)
        print(f"  {t:<14} known={k:<5} not_known={u:<5} total={k+u}")


if __name__ == "__main__":
    main()