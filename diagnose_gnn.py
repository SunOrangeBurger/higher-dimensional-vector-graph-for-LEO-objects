#!/usr/bin/env python3
"""
Standalone diagnostic: evaluate the currently-saved GNN matcher on a fresh
validation batch, independent of the tracking simulation, to see whether
the model itself learned a useful decision boundary or not.

Run this from your project directory (same place you run
ground_station_tracking_simulation.py from) so relative paths resolve.
"""
import numpy as np
import torch

import gnn_matcher as gnn_mod
import ground_station_tracking_simulation as sim_mod
import leo_catalog_embed as embed_mod

store = sim_mod.load_embedding_store(sim_mod.EMBEDDINGS_PATH, sim_mod.FALLBACK_NPZ_PATH)
distance_weights = np.asarray(store.get("distance_weights", embed_mod.DISTANCE_WEIGHTS), dtype=np.float32)
weighted_store_vectors = np.asarray(store["vectors"], dtype=np.float32) * distance_weights
drift_tolerance, margin_required = sim_mod.calibrate_thresholds(
    weighted_store_vectors, cat_i=store["elements_i"], cat_types=store["types"], cat_e=store["elements_e"]
)

print(f"\nUsing drift_tolerance={drift_tolerance:.4f} margin_required={margin_required:.4f}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = gnn_mod.load_or_train_model(
    store,
    distance_weights=distance_weights,
    drift_tolerance=drift_tolerance,
    margin_required=margin_required,
)
model.eval()

print("\nGenerating 500 fresh validation examples (same filtered distribution as training)...")
node_feats, edge_dist, labels = gnn_mod.generate_training_examples(
    store, n_examples=500, seed=999,  # different seed than training (123)
    distance_weights=distance_weights, drift_tolerance=drift_tolerance, margin_required=margin_required,
)

nf = torch.from_numpy(node_feats).to(device)
ed = torch.from_numpy(edge_dist).to(device)
lb = torch.from_numpy(labels).to(device)

with torch.no_grad():
    logits = model(nf, ed)
    probs = torch.sigmoid(logits).cpu().numpy()

true_known = (labels.sum(axis=1) > 0)
n_known = true_known.sum()
n_unknown = (~true_known).sum()
print(f"Validation set: {n_known} 'known' examples (true match among top5), {n_unknown} 'unknown' (novel object)")

# Does argmax pick the TRUE match, when there is one?
argmax_idx = probs.argmax(axis=1)
correct_pick = np.zeros(len(labels), dtype=bool)
for i in range(len(labels)):
    if true_known[i]:
        correct_pick[i] = labels[i, argmax_idx[i]] == 1.0

print(f"\nAmong 'known' examples: argmax picked the TRUE match {correct_pick[true_known].sum()}/{n_known} "
      f"({correct_pick[true_known].mean()*100:.1f}%)")

# What's argmax's confidence distribution?
best_conf = probs.max(axis=1)
print(f"\nBest-candidate confidence (all examples): min={best_conf.min():.3f} max={best_conf.max():.3f} "
      f"avg={best_conf.mean():.3f}")
print(f"Best-candidate confidence (known only):    avg={best_conf[true_known].mean():.3f}")
print(f"Best-candidate confidence (unknown only):  avg={best_conf[~true_known].mean():.3f}")

above_thresh = best_conf > gnn_mod.GNN_CONFIRM_THRESHOLD
print(f"\nFraction clearing GNN_CONFIRM_THRESHOLD={gnn_mod.GNN_CONFIRM_THRESHOLD}: "
      f"{above_thresh.mean()*100:.1f}% overall, "
      f"{above_thresh[true_known].mean()*100:.1f}% of known, "
      f"{above_thresh[~true_known].mean()*100:.1f}% of unknown (false positive rate if high)")