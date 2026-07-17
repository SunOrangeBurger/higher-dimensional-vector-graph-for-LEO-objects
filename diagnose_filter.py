#!/usr/bin/env python3
"""Diagnose why generate_training_examples produces 0 known examples."""
import numpy as np
import gnn_matcher as gnn_mod
import ground_station_tracking_simulation as sim_mod
import leo_catalog_embed as embed_mod
import orbital_physics as op

store = sim_mod.load_embedding_store(sim_mod.EMBEDDINGS_PATH, sim_mod.FALLBACK_NPZ_PATH)
distance_weights = np.asarray(store.get("distance_weights", embed_mod.DISTANCE_WEIGHTS), dtype=np.float32)
weighted_store_vectors = np.asarray(store["vectors"], dtype=np.float32) * distance_weights
drift_tolerance, margin_required = sim_mod.calibrate_thresholds(
    weighted_store_vectors, cat_i=store["elements_i"], cat_types=store["types"], cat_e=store["elements_e"]
)

rng = np.random.default_rng(999)
scalers = store["scalers"]
cat_a = store["elements_a"]; cat_e = store["elements_e"]; cat_i = store["elements_i"]
cat_raan = store["elements_raan"]; cat_argp = store["elements_argp"]; cat_bc = store["bc_m2_per_kg"]
cat_types = list(store["types"]); cat_sizes = store["sizes"]; cat_rcs = store["rcs"]
n_cat = len(cat_types)

PROCESS_NOISE = {"a_km": 0.05, "e": 0.00005, "i_deg": 0.002, "angle_deg": 0.01}
MEAS_NOISE = {"a_km": 0.02, "e": 0.00002, "i_deg": 0.001, "angle_deg": 0.005}

n_known_drawn = 0
n_known_in_top5 = 0
n_known_gate_rejected = 0
n_known_kept = 0

for _ in range(300):
    dt = rng.uniform(0, 48 * 3600)
    prop_a, prop_e, prop_i, prop_raan, prop_argp = op.propagate_catalog_analytic(cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc, dt)
    cat_vectors = embed_mod.build_embeddings_fast(prop_a, prop_e, prop_i, prop_raan, prop_argp, cat_types, cat_sizes, cat_rcs, scalers)

    src_idx = rng.integers(0, n_cat)
    n_known_drawn += 1
    new_a, new_e, new_i, new_raan, new_argp = op.propagate_elements(
        np.array([cat_a[src_idx]]), np.array([cat_e[src_idx]]), np.array([cat_i[src_idx]]),
        np.array([cat_raan[src_idx]]), np.array([cat_argp[src_idx]]), np.array([cat_bc[src_idx]]),
        dt, process_noise=PROCESS_NOISE, rng=rng,
    )
    meas_a = new_a[0] + rng.normal(0, MEAS_NOISE["a_km"])
    meas_e = max(0.0, new_e[0] + rng.normal(0, MEAS_NOISE["e"]))
    meas_i = new_i[0] + rng.normal(0, MEAS_NOISE["i_deg"])
    meas_raan = (new_raan[0] + rng.normal(0, MEAS_NOISE["angle_deg"])) % 360
    meas_argp = (new_argp[0] + rng.normal(0, MEAS_NOISE["angle_deg"])) % 360
    q_type, q_size, q_rcs = cat_types[src_idx], cat_sizes[src_idx], cat_rcs[src_idx]

    query_vec = embed_mod.transform_identity_vector(meas_a, meas_e, meas_i, meas_raan, meas_argp, q_type, q_size, q_rcs, scalers)
    dists = np.linalg.norm(cat_vectors - query_vec, axis=1)
    top5 = np.argsort(dists)[:5]

    if src_idx in top5:
        n_known_in_top5 += 1

        w_cat = cat_vectors[top5] * distance_weights
        w_query = query_vec * distance_weights
        w_dists = np.linalg.norm(w_cat - w_query, axis=1)
        w_order = np.argsort(w_dists)
        w_best = w_dists[w_order[0]]
        w_second = w_dists[w_order[1]] if len(w_order) > 1 else np.inf
        w_margin = w_second - w_best
        gate_would_accept = w_best < drift_tolerance and w_margin > margin_required
        if gate_would_accept:
            n_known_kept += 0  # discarded by filter (gate would've handled it)
        else:
            n_known_gate_rejected += 1
            n_known_kept += 1  # this is what SHOULD survive the filter and be kept as training data
    # else: true match not even in top5 -- can't be a positive label regardless of gate

print(f"Known cases drawn: {n_known_drawn}")
print(f"  True match landed in top5 (unweighted): {n_known_in_top5} ({n_known_in_top5/n_known_drawn*100:.1f}%)")
print(f"  Of those, gate-rejected (should survive filter): {n_known_gate_rejected}")
print(f"  Of those, gate-accepted (correctly discarded): {n_known_in_top5 - n_known_gate_rejected}")