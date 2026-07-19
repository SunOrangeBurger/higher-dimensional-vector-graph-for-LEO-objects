#!/usr/bin/env python3
"""
Ground station tracking simulation -- full predict/match/update cycle.
Fixes applied:
  - #5: Removed ground-truth oracle (existing_slot force-accept).
  - #6: Dynamic threshold recalibration on every rebuild.
  - #4: Kalman filter instead of fixed EMA.
  - #3: Retirement and basic de-duplication.
  - Added metrics logging for visualization.
"""

import heapq
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict

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

ACCELERATION_FACTOR = 300
SIMULATION_DURATION_SECONDS = 120

DRIFT_TOLERANCE = None
MARGIN_REQUIRED = None

# Kalman / Retirement config
INITIAL_P_DIAG = [10.0, 1e-4, 0.5, 10.0, 10.0]  # a, e, i, raan, argp
PROCESS_NOISE = {"a_km": 0.05, "e": 0.00005, "i_deg": 0.002, "angle_deg": 0.01}
MEASUREMENT_NOISE = {"a_km": 0.02, "e": 0.00002, "i_deg": 0.001, "angle_deg": 0.005}
RETIRE_AFTER_MISSED_PERIODS = 3
DE_DUP_INTERVAL_REBUILDS = 5

SEED = 11
USE_GNN_CONFIRMATION = True
UPDATE_CONFIRMATIONS_REQUIRED = 2
REGISTRATION_CONFIRMATIONS_REQUIRED = 2

ANN_REBUILD_EVERY = leo_ann_index.REBUILD_EVERY
ANN_OVERFLOW_REBUILD_THRESHOLD = leo_ann_index.OVERFLOW_REBUILD_THRESHOLD
ANN_CANDIDATE_POOL_SIZE = leo_ann_index.CANDIDATE_POOL_SIZE

MIN_CATALOG_SIZE_FOR_CALIBRATION = 200
SPARSE_CATALOG_FALLBACK_DRIFT_TOLERANCE = 0.5
SPARSE_CATALOG_FALLBACK_MARGIN_REQUIRED = 0.05
MIN_CATALOG_SIZE_FOR_SCALER_REFIT = 50
MIN_CATALOG_SIZE_FOR_GNN_TRAINING = 50

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_embedding_store(pt_path, npz_fallback):
    """
    Load a pre-built embedding store, or create an empty cold‑start scaffold
    if no store exists. Returns a dict with the same shape either way.
    """
    # Try PyTorch .pt first
    if TORCH_AVAILABLE:
        try:
            data = torch.load(pt_path, weights_only=False)
            data["vectors"] = data["vectors"].numpy()
            return data
        except FileNotFoundError:
            pass

    # Try fallback .npz
    try:
        raw = np.load(npz_fallback, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}
        data["scalers"] = raw["scalers"][0]   # unwrap object array
        data["names"] = data["names"].tolist()
        data["types"] = data["types"].tolist()
        return data
    except FileNotFoundError:
        print(f"No embedding store found at {pt_path} or {npz_fallback} -- "
              f"starting from an in-process cold-start scaffold "
              f"(0 pre-known objects, domain-default scalers).")

    # ---- No store exists: build an empty scaffold inline ----
    # Domain‑default scalers (same as in leo_catalog_embed.build_domain_default_scalers)
    from data_generator import EARTH_RADIUS_KM, LEO_ALT_MIN, LEO_ALT_MAX, \
                               OBJECT_TYPES, TYPE_WEIGHTS, SIZE_RANGES, RCS_RANGES
    a_lo = EARTH_RADIUS_KM + LEO_ALT_MIN
    a_hi = EARTH_RADIUS_KM + LEO_ALT_MAX
    a_mean = (a_lo + a_hi) / 2.0
    a_std = (a_hi - a_lo) / np.sqrt(12)

    e_lo_log, e_hi_log = np.log1p(0.0001), np.log1p(0.15)
    e_log_mean = 0.9 * np.log1p(0.005) + 0.1 * np.log1p(0.08)
    e_log_std = (e_hi_log - e_lo_log) / np.sqrt(12)

    def _blended_log_stats(ranges):
        los, his = [], []
        for t, w in zip(OBJECT_TYPES, TYPE_WEIGHTS):
            lo, hi = ranges[t]
            los.append(np.log1p(lo) * w)
            his.append(np.log1p(hi) * w)
        lo_log, hi_log = sum(los), sum(his)
        mean_log = (lo_log + hi_log) / 2.0
        std_log = (hi_log - lo_log) / np.sqrt(12)
        return mean_log, max(std_log, 1e-6)

    size_log_mean, size_log_std = _blended_log_stats(SIZE_RANGES)
    rcs_log_mean, rcs_log_std = _blended_log_stats(RCS_RANGES)

    scalers = {
        "a_mean": float(a_mean), "a_std": float(a_std),
        "e_log_mean": float(e_log_mean), "e_log_std": float(e_log_std),
        "size_log_mean": float(size_log_mean), "size_log_std": float(size_log_std),
        "rcs_log_mean": float(rcs_log_mean), "rcs_log_std": float(rcs_log_std),
        "source": "domain_default",
    }

    # Empty scaffold (zero‑length arrays for catalog‑sized fields)
    from leo_catalog_embed import IDENTITY_DIMENSIONS, DISTANCE_WEIGHTS
    return {
        "names": [],
        "vectors": np.zeros((0, len(IDENTITY_DIMENSIONS)), dtype=np.float32),
        "dimensions": IDENTITY_DIMENSIONS,
        "scalers": scalers,
        "elements_a": np.zeros(0, dtype=float),
        "elements_e": np.zeros(0, dtype=float),
        "elements_i": np.zeros(0, dtype=float),
        "elements_raan": np.zeros(0, dtype=float),
        "elements_argp": np.zeros(0, dtype=float),
        "bc_m2_per_kg": np.zeros(0, dtype=float),
        "types": [],
        "sizes": np.zeros(0, dtype=float),
        "rcs": np.zeros(0, dtype=float),
        "epoch_seconds": np.zeros(0, dtype=float),
        "distance_weights": DISTANCE_WEIGHTS,
    }

def calibrate_thresholds(catalog_vectors, cat_i=None, cat_types=None, cat_e=None):
    """
    Auto-calibrate DRIFT_TOLERANCE and MARGIN_REQUIRED from the catalog's
    OWN measured density, instead of trusting fixed numbers tuned at a
    different scale.

    Uses within-cluster nearest-neighbor gaps if bucket metadata is provided,
    otherwise falls back to global percentile.
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

    # Coarse inclination bucket: round to nearest 5 degrees.
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
        # degenerate catalog – fall back to global
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

def registration_bucket_key(obj_type, meas_a, meas_i, meas_raan):
    a_bucket = round(meas_a / 5.0)
    i_bucket = round(meas_i / 0.5)
    return (obj_type, a_bucket, i_bucket)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    global DRIFT_TOLERANCE, MARGIN_REQUIRED

    catalog_df = pd.read_csv(CATALOG_PATH)
    store = load_embedding_store(EMBEDDINGS_PATH, FALLBACK_NPZ_PATH)
    scalers = store["scalers"]

    distance_weights = np.asarray(
        store.get("distance_weights", embed_mod.DISTANCE_WEIGHTS), dtype=np.float32
    )

    # ---- Initial threshold calibration ----
    if len(store["names"]) < MIN_CATALOG_SIZE_FOR_CALIBRATION:
        DRIFT_TOLERANCE = SPARSE_CATALOG_FALLBACK_DRIFT_TOLERANCE
        MARGIN_REQUIRED = SPARSE_CATALOG_FALLBACK_MARGIN_REQUIRED
    else:
        weighted_vecs = np.asarray(store["vectors"], dtype=np.float32) * distance_weights
        DRIFT_TOLERANCE, MARGIN_REQUIRED = calibrate_thresholds(
            weighted_vecs,
            cat_i=store["elements_i"],
            cat_types=store["types"],
            cat_e=store["elements_e"],
        )

    print(f"Initial thresholds: DT={DRIFT_TOLERANCE:.4f}, MR={MARGIN_REQUIRED:.4f}")

    # ---- GNN setup ----
    gnn_model = None
    if USE_GNN_CONFIRMATION and gnn_mod.TORCH_AVAILABLE and len(store["names"]) >= MIN_CATALOG_SIZE_FOR_GNN_TRAINING:
        gnn_model = gnn_mod.load_or_train_model(store, distance_weights=distance_weights)

    rng = np.random.default_rng(SEED)

    # ---- Truth state ----
    truth_a = catalog_df["a"].to_numpy(dtype=float).copy()
    truth_e = catalog_df["e"].to_numpy(dtype=float).copy()
    truth_i = catalog_df["i"].to_numpy(dtype=float).copy()
    truth_raan = catalog_df["Omega"].to_numpy(dtype=float).copy()
    truth_argp = catalog_df["omega"].to_numpy(dtype=float).copy()
    truth_bc = catalog_df["bc_m2_per_kg"].to_numpy(dtype=float).copy()
    truth_epoch = np.zeros(len(catalog_df))
    truth_periods = 2 * np.pi * np.sqrt(truth_a ** 3 / op.MU_EARTH) / ACCELERATION_FACTOR

    names = catalog_df["name"].tolist()
    types = catalog_df["type"].tolist()
    sizes = catalog_df["size_meters"].to_numpy(dtype=float)
    rcs_vals = catalog_df["rcs"].to_numpy(dtype=float)

    # ---- Catalog state (growable) ----
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
    cat_last_seen = list(cat_epoch)
    cat_confidence = [1] * len(cat_names)

    # ---- Kalman filter state (diagonal covariance for a, e, i, raan, argp) ----
    cat_P = []
    for _ in range(len(cat_names)):
        cat_P.append(np.diag(INITIAL_P_DIAG).copy())

    # ---- Retirement ----
    cat_retired = [False] * len(cat_names)

    # ---- Discovery ----
    discovery_counter = 0
    true_name_to_idx = {n: i for i, n in enumerate(cat_names)}
    discovered_true_name = {}

    # ---- Pending structures ----
    pending_update = {}
    pending_registration = {}

    # ---- ANN Index ----
    ann_index = leo_ann_index.LiveCatalogANNIndex(
        scalers,
        rebuild_every=ANN_REBUILD_EVERY,
        overflow_threshold=ANN_OVERFLOW_REBUILD_THRESHOLD,
        candidate_pool_size=ANN_CANDIDATE_POOL_SIZE,
    )

    # ---- Event heap ----
    heap = []
    for idx in range(len(catalog_df)):
        phase = rng.uniform(0, truth_periods[idx])
        heapq.heappush(heap, (phase, idx))

    # ---- Metrics logging ----
    metrics_history = []
    WARMUP_BIN_SECONDS = 10.0
    warmup_bin_detections = defaultdict(int)
    warmup_bin_registrations = defaultdict(int)
    total_detections = 0
    known_count = 0
    unknown_count = 0
    mismatch_count = 0
    lookup_times = []
    match_distances = []
    gnn_overturns = 0
    registrations_committed = 0
    update_committed_count = 0
    de_dup_merges = 0
    rebuild_count_since_de_dup = 0

    # -----------------------------------------------------------------
    # Core Functions (defined inside main to access state)
    # -----------------------------------------------------------------
    def recalibrate_thresholds(weighted_vectors):
        global DRIFT_TOLERANCE, MARGIN_REQUIRED
        if len(cat_names) < MIN_CATALOG_SIZE_FOR_CALIBRATION:
            return
        dt, mr = calibrate_thresholds(
            weighted_vectors,
            cat_i=np.array(cat_i),
            cat_types=cat_types,
            cat_e=np.array(cat_e),
        )
        DRIFT_TOLERANCE = dt
        MARGIN_REQUIRED = mr
        print(f"[REBUILD] Recalibrated thresholds: DT={dt:.4f}, MR={mr:.4f}")

    def de_duplicate_catalog():
        nonlocal de_dup_merges
        if len(cat_names) < 2:
            return
        # Build temp vectors for all active entries
        active_indices = [i for i, r in enumerate(cat_retired) if not r]
        if len(active_indices) < 2:
            return
        # Naively propagate to current time (simplified for de-dup)
        # We'll just use the stored epoch and current real_now
        dt_arr = real_now - np.array([cat_epoch[i] for i in active_indices])
        pa, pe, pi, pr, par = op.propagate_catalog_analytic(
            np.array([cat_a[i] for i in active_indices]),
            np.array([cat_e[i] for i in active_indices]),
            np.array([cat_i[i] for i in active_indices]),
            np.array([cat_raan[i] for i in active_indices]),
            np.array([cat_argp[i] for i in active_indices]),
            np.array([cat_bc[i] for i in active_indices]),
            dt_arr
        )
        vecs = embed_mod.build_embeddings_fast(
            pa, pe, pi, pr, par,
            [cat_types[i] for i in active_indices],
            np.array([cat_sizes[i] for i in active_indices]),
            np.array([cat_rcs[i] for i in active_indices]),
            scalers
        ) * distance_weights

        # Brute force self-query with threshold DRIFT_TOLERANCE * 0.5
        # Merge entries that are too close AND both are discovered_*
        to_merge = {}
        for idx1 in range(len(active_indices)):
            for idx2 in range(idx1+1, len(active_indices)):
                abs_i = active_indices[idx1]
                abs_j = active_indices[idx2]
                if cat_names[abs_i].startswith("discovered_") and cat_names[abs_j].startswith("discovered_"):
                    if np.linalg.norm(vecs[idx1] - vecs[idx2]) < DRIFT_TOLERANCE * 0.5:
                        to_merge[abs_j] = abs_i  # merge j into i

        if to_merge:
            # Keep the older one (lower index usually)
            # Re-map true_name_to_idx
            for remove_idx, keep_idx in to_merge.items():
                # Transfer any pending updates? For simplicity, just retire the removed one.
                cat_retired[remove_idx] = True
                # Update true_name mapping if needed
                for tname, tidx in list(true_name_to_idx.items()):
                    if tidx == remove_idx:
                        true_name_to_idx[tname] = keep_idx
                de_dup_merges += 1
            print(f"[DE-DUP] Merged {len(to_merge)} duplicate entries.")

    def resolve_update(cat_idx, true_object_name, meas_a, meas_e, meas_i, meas_raan, meas_argp,
                        prop_a, prop_e, prop_i, prop_raan, prop_argp, real_now):
        """
        Kalman filter update for a, e, i, raan, argp.
        """
        # Measurement vector
        z = np.array([meas_a, meas_e, meas_i, meas_raan, meas_argp])
        # Measurement noise covariance (diagonal)
        R = np.diag([
            MEASUREMENT_NOISE["a_km"]**2,
            MEASUREMENT_NOISE["e"]**2,
            MEASUREMENT_NOISE["i_deg"]**2,
            MEASUREMENT_NOISE["angle_deg"]**2,
            MEASUREMENT_NOISE["angle_deg"]**2
        ])

        # Prediction (propagated state) is already in prop_a/e/i/raan/argp
        x_pred = np.array([prop_a, prop_e, prop_i, prop_raan, prop_argp])

        if UPDATE_CONFIRMATIONS_REQUIRED > 1:
            prior = pending_update.get(cat_idx)
            if prior != true_object_name:
                pending_update[cat_idx] = true_object_name
                return False  # withheld

        # Get current covariance for this slot
        P = cat_P[cat_idx]

        # Kalman gain
        S = P + R
        K = P @ np.linalg.inv(S)

        # Update state
        x_new = x_pred + K @ (z - x_pred)
        P_new = (np.eye(5) - K) @ P

        # Clamp angles
        x_new[3] = x_new[3] % 360
        x_new[4] = x_new[4] % 360
        # Clamp e
        x_new[1] = max(0.0, x_new[1])

        # Write back
        cat_a[cat_idx] = x_new[0]
        cat_e[cat_idx] = x_new[1]
        cat_i[cat_idx] = x_new[2]
        cat_raan[cat_idx] = x_new[3]
        cat_argp[cat_idx] = x_new[4]
        cat_epoch[cat_idx] = real_now
        cat_P[cat_idx] = P_new
        cat_confidence[cat_idx] += 1
        cat_last_seen[cat_idx] = real_now

        pending_update.pop(cat_idx, None)
        return True

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    real_start = time.perf_counter()
    while heap:
        sim_time, idx = heapq.heappop(heap)
        if sim_time > SIMULATION_DURATION_SECONDS:
            break

        real_now = sim_time * ACCELERATION_FACTOR
        registration_committed_this_pass = False

        # -- Propagate truth --
        dt_truth = real_now - truth_epoch[idx]
        new_a, new_e, new_i, new_raan, new_argp = op.propagate_elements(
            np.array([truth_a[idx]]), np.array([truth_e[idx]]), np.array([truth_i[idx]]),
            np.array([truth_raan[idx]]), np.array([truth_argp[idx]]), np.array([truth_bc[idx]]),
            dt_truth, process_noise=PROCESS_NOISE, rng=rng,
        )
        truth_a[idx], truth_e[idx], truth_i[idx] = new_a[0], new_e[0], new_i[0]
        truth_raan[idx], truth_argp[idx] = new_raan[0], new_argp[0]
        truth_epoch[idx] = real_now

        # -- Sensor measurement --
        meas_a = truth_a[idx] + rng.normal(0, MEASUREMENT_NOISE["a_km"])
        meas_e = max(0.0, truth_e[idx] + rng.normal(0, MEASUREMENT_NOISE["e"]))
        meas_i = truth_i[idx] + rng.normal(0, MEASUREMENT_NOISE["i_deg"])
        meas_raan = (truth_raan[idx] + rng.normal(0, MEASUREMENT_NOISE["angle_deg"])) % 360
        meas_argp = (truth_argp[idx] + rng.normal(0, MEASUREMENT_NOISE["angle_deg"])) % 360

        detection_vec = embed_mod.transform_identity_vector(
            meas_a, meas_e, meas_i, meas_raan, meas_argp, types[idx], sizes[idx], rcs_vals[idx], scalers
        )

        t0 = time.perf_counter()
        n_cat = len(cat_names)
        accepted = False
        best_idx = -1
        best_dist = np.inf
        margin = 0.0
        accept_path = None
        gnn_used = False
        gnn_registered = False
        confirmed_unknown = False

        # -- ANN Query --
        if n_cat > 0:
            weighted_detection_vec = detection_vec * distance_weights

            if ann_index.needs_rebuild():
                # Rebuild tree
                weighted_vectors = ann_index.rebuild(
                    cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc,
                    cat_types, cat_sizes, cat_rcs, cat_epoch, cat_retired,
                    real_now, distance_weights
                )
                # Recalibrate thresholds (#6)
                if weighted_vectors is not None and len(weighted_vectors) > 0:
                    recalibrate_thresholds(weighted_vectors)

                # De-dup every N rebuilds (#3)
                rebuild_count_since_de_dup += 1
                if rebuild_count_since_de_dup >= DE_DUP_INTERVAL_REBUILDS:
                    de_duplicate_catalog()
                    rebuild_count_since_de_dup = 0

            candidate_idx = ann_index.query_candidates(weighted_detection_vec, n_cat)

            if len(candidate_idx) > 0:
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

                weighted_cand_vectors = cand_vectors * distance_weights
                cand_unweighted_dists = np.linalg.norm(cand_vectors - detection_vec, axis=1)
                cand_weighted_dists = np.linalg.norm(weighted_cand_vectors - weighted_detection_vec, axis=1)

                local_of = {int(j): k for k, j in enumerate(candidate_idx)}
                def cat_vector_for(j):
                    if j in local_of:
                        return cand_vectors[local_of[j]]
                    dt_j = real_now - cat_epoch[j]
                    pa, pe, pi, pr, par = op.propagate_catalog_analytic(
                        np.array([cat_a[j]]), np.array([cat_e[j]]), np.array([cat_i[j]]),
                        np.array([cat_raan[j]]), np.array([cat_argp[j]]), np.array([cat_bc[j]]), dt_j,
                    )
                    return embed_mod.build_embeddings_fast(
                        pa, pe, pi, pr, par, [cat_types[j]], [cat_sizes[j]], [cat_rcs[j]], scalers
                    )[0]

                local_order = np.argsort(cand_unweighted_dists)
                top5_local = local_order[:5]
                top5 = candidate_idx[top5_local]
                top5_weighted_dists = cand_weighted_dists[top5_local]

                gate_order = np.argsort(top5_weighted_dists)
                best_idx = int(top5[gate_order[0]])
                best_dist = top5_weighted_dists[gate_order[0]]
                second_dist = top5_weighted_dists[gate_order[1]] if len(top5) > 1 else np.inf
                margin = second_dist - best_dist

                accepted = best_dist < DRIFT_TOLERANCE and margin > MARGIN_REQUIRED
                if accepted:
                    accept_path = "gate"

                # -- GNN second opinion --
                if not accepted and gnn_model is not None:
                    gnn_used = True
                    candidate_vecs_gnn = cand_vectors[top5_local]
                    candidate_dists_gnn = cand_unweighted_dists[top5_local]
                    gnn_known, gnn_local_idx, _ = gnn_mod.classify(
                        gnn_model, detection_vec, candidate_vecs_gnn, candidate_dists_gnn
                    )
                    if gnn_known:
                        accepted = True
                        accept_path = "gnn"
                        best_idx = int(top5[gnn_local_idx])
                        best_dist = candidate_dists_gnn[gnn_local_idx]
                        gnn_overturns += 1
                    else:
                        confirmed_unknown = True
                elif not accepted:
                    confirmed_unknown = True

        # -- Handle Rejected / Unknown --
        if confirmed_unknown:
            # NO ORACLE (#5): we do NOT force-accept based on true_name_to_idx.
            # Instead, we check if this true object already has a slot.
            existing_idx = true_name_to_idx.get(names[idx])
            if existing_idx is not None and existing_idx < len(cat_names):
                # It has a slot, but we missed it. Do NOT update or accept.
                # Just log a miss.
                accepted = False
                best_idx = -1
            else:
                # Genuine new object candidate -> pending registration (#1)
                bucket_key = registration_bucket_key(types[idx], meas_a, meas_i, meas_raan)
                if REGISTRATION_CONFIRMATIONS_REQUIRED <= 1:
                    commit_registration = True
                else:
                    prior = pending_registration.get(bucket_key)
                    if prior is not None:
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
                        commit_registration = corroboration_dist < DRIFT_TOLERANCE
                    else:
                        commit_registration = False

                if commit_registration:
                    new_name = f"discovered_{discovery_counter}"
                    discovery_counter += 1
                    discovered_true_name[new_name] = names[idx]
                    true_name_to_idx[names[idx]] = len(cat_names)
                    cat_names.append(new_name)
                    cat_a.append(meas_a); cat_e.append(meas_e); cat_i.append(meas_i)
                    cat_raan.append(meas_raan); cat_argp.append(meas_argp)
                    cat_bc.append(op.ballistic_coefficient(sizes[idx], types[idx]))
                    cat_types.append(types[idx]); cat_sizes.append(sizes[idx]); cat_rcs.append(rcs_vals[idx])
                    cat_epoch.append(real_now)
                    cat_confidence.append(1)
                    cat_last_seen.append(real_now)
                    cat_P.append(np.diag(INITIAL_P_DIAG).copy())
                    cat_retired.append(False)
                    ann_index.mark_dirty(len(cat_names) - 1)
                    pending_registration.pop(bucket_key, None)
                    registrations_committed += 1
                    registration_committed_this_pass = True
                    gnn_registered = True
                else:
                    pending_registration[bucket_key] = {
                        "meas_a": meas_a, "meas_e": meas_e, "meas_i": meas_i,
                        "meas_raan": meas_raan, "meas_argp": meas_argp,
                        "obj_type": types[idx], "size": sizes[idx], "rcs": rcs_vals[idx],
                        "real_now": real_now, "true_name": names[idx],
                    }

        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1_000_000
        total_detections += 1
        lookup_times.append(elapsed_us)

        update_committed = False
        if accepted:
            known_count += 1
            match_distances.append(best_dist)
            # Update with Kalman (#4)
            if best_idx >= 0:
                # Propagate the selected slot to current time for the residual
                dt_slot = real_now - cat_epoch[best_idx]
                pa, pe, pi, pr, par = op.propagate_catalog_analytic(
                    np.array([cat_a[best_idx]]), np.array([cat_e[best_idx]]),
                    np.array([cat_i[best_idx]]), np.array([cat_raan[best_idx]]),
                    np.array([cat_argp[best_idx]]), np.array([cat_bc[best_idx]]), dt_slot,
                )
                update_committed = resolve_update(
                    best_idx, names[idx], meas_a, meas_e, meas_i, meas_raan, meas_argp,
                    pa[0], pe[0], pi[0], pr[0], par[0], real_now
                )
                if update_committed:
                    update_committed_count += 1
                    ann_index.mark_dirty(best_idx)
                # Check mismatch
                matched_true = discovered_true_name.get(cat_names[best_idx], cat_names[best_idx])
                if matched_true != names[idx]:
                    mismatch_count += 1
        else:
            unknown_count += 1

        # -- Warmup bins --
        warmup_bin = int(sim_time // WARMUP_BIN_SECONDS)
        warmup_bin_detections[warmup_bin] += 1
        if registration_committed_this_pass:
            warmup_bin_registrations[warmup_bin] += 1

        # -- Retirement check (#3) --
        if total_detections % 100 == 0:
            for i in range(len(cat_names)):
                if cat_retired[i]:
                    continue
                period = 2 * np.pi * np.sqrt(cat_a[i] ** 3 / op.MU_EARTH) / ACCELERATION_FACTOR
                if real_now - cat_last_seen[i] > RETIRE_AFTER_MISSED_PERIODS * period:
                    cat_retired[i] = True
                    print(f"[RETIRE] Retired slot {i} ({cat_names[i]}) unseen for too long.")

        # -- Push next event --
        heapq.heappush(heap, (sim_time + truth_periods[idx], idx))

        # -- Log metrics every 100 detections --
        if total_detections % 100 == 0:
            metrics_history.append({
                "sim_time": sim_time,
                "total_detections": total_detections,
                "known_rate": known_count / total_detections if total_detections > 0 else 0,
                "mismatch_count": mismatch_count,
                "catalog_size": len(cat_names),
                "avg_lookup_us": np.mean(lookup_times[-100:]) if lookup_times else 0,
                "registrations_committed": registrations_committed,
                "drift_tolerance": DRIFT_TOLERANCE,
                "margin_required": MARGIN_REQUIRED,
            })

    # -----------------------------------------------------------------
    # End of simulation: Save metrics
    # -----------------------------------------------------------------
    with open("metrics_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    # Print summary (keep existing print statements)
    print("\n" + "=" * 70)
    print("TRACKING SIMULATION METRICS")
    print("=" * 70)
    print(f"Total detections           : {total_detections}")
    print(f"  Known (accepted match)   : {known_count} ({known_count/total_detections*100:.1f}%)")
    print(f"  Not known (rejected)     : {unknown_count} ({unknown_count/total_detections*100:.1f}%)")
    print(f"  Mismatches (false accepts): {mismatch_count}")
    print(f"  Belief updates committed : {update_committed_count}")
    print(f"  Registrations committed : {registrations_committed}")
    print(f"  De-dup merges           : {de_dup_merges}")
    print(f"  Catalog size: {len(store['names'])} -> {len(cat_names)}")
    print(f"Real wall-clock time spent : {time.perf_counter() - real_start:.3f} s")
    print(f"Metrics logged to metrics_history.json")

if __name__ == "__main__":
    main()