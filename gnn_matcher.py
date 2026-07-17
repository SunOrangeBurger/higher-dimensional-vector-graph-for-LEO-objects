#!/usr/bin/env python3
"""
GNN-based second-opinion matcher for detections the fast threshold gate
rejected.

The fast gate (distance + margin threshold) handles clear-cut cases cheaply.
For rejected/borderline cases, this module builds a small local graph --
the query (rejected detection) plus its 5 nearest catalog candidates, fully
connected (15 edges: 5 query-candidate + 10 candidate-candidate) -- and uses
a message-passing GNN to make a more informed decision than raw distance
alone. Unlike a threshold, the GNN can learn things like "this candidate
sits in a dense cluster, be more skeptical of a close-ish match" or "these
5 candidates are all far from each other, so if the query is roughly
equidistant to two of them, that's a genuinely different signal than being
close to one and far from the rest."

If the GNN also says "not a match to any of these 5", the object is
CONFIRMED UNKNOWN and gets registered as a new catalog entry.

Training data is generated for free from our own physics simulator (we
control ground truth): "known" examples are real catalog objects observed
with realistic drift+noise at a random elapsed time; "novel" examples are
freshly-sampled random objects unrelated to anything in the catalog.
"""

import numpy as np

import orbital_physics as op
import leo_catalog_embed as embed_mod
import data_generator as gen

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

FEAT_DIM = 14   # 13-dim identity vector + is_query flag
HIDDEN = 32
N_NODES = 6     # 1 query + 5 candidates

MODEL_PATH = "outputs/gnn_matcher.pt"
GNN_CONFIRM_THRESHOLD = 0.5  # score above this -> GNN says "this candidate is a real match"


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
if TORCH_AVAILABLE:
    class GNNMatcher(nn.Module):
        """Two rounds of message passing over the fully-connected 6-node
        graph, then an edge-scoring head over the 5 query-candidate edges.
        Batched: node_feats (B,6,14), edge_dist (B,6,6) -> logits (B,5)."""

        def __init__(self, feat_dim=FEAT_DIM, hidden=HIDDEN):
            super().__init__()
            self.msg1 = nn.Linear(feat_dim + 1, hidden)
            self.upd1 = nn.Linear(feat_dim + hidden, hidden)
            self.msg2 = nn.Linear(hidden + 1, hidden)
            self.upd2 = nn.Linear(hidden + hidden, hidden)
            self.score = nn.Linear(hidden * 2 + 1, 1)

        def _message_pass(self, h, edge_dist, msg_layer, upd_layer):
            B, N, D = h.shape
            h_j = h.unsqueeze(1).expand(B, N, N, D)               # h_j[b,i,j,:] = h[b,j,:]
            edge_feat = edge_dist.unsqueeze(-1)                    # (B,N,N,1)
            msg_in = torch.cat([h_j, edge_feat], dim=-1)           # (B,N,N,D+1)
            msg = torch.tanh(msg_layer(msg_in))                    # (B,N,N,H)
            diag_mask = torch.eye(N, device=h.device).view(1, N, N, 1)
            msg = msg * (1 - diag_mask)
            agg = msg.sum(dim=2) / (N - 1)                         # mean over N-1 neighbors -> (B,N,H)
            upd_in = torch.cat([h, agg], dim=-1)                   # (B,N,D+H)
            return torch.tanh(upd_layer(upd_in))

        def forward(self, node_feats, edge_dist):
            h = self._message_pass(node_feats, edge_dist, self.msg1, self.upd1)
            h = self._message_pass(h, edge_dist, self.msg2, self.upd2)

            h_query = h[:, 0:1, :].expand(-1, 5, -1)               # (B,5,H)
            h_cand = h[:, 1:6, :]                                  # (B,5,H)
            dist_qc = edge_dist[:, 0, 1:6].unsqueeze(-1)           # (B,5,1)
            score_in = torch.cat([h_query, h_cand, dist_qc], dim=-1)
            logits = self.score(score_in).squeeze(-1)              # (B,5)
            return logits


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------
def build_graph(query_vec, candidate_vecs, query_candidate_dists):
    """query_vec: (13,), candidate_vecs: (5,13), query_candidate_dists: (5,)
    -> node_feats (6,14), edge_dist (6,6)"""
    all_vecs = np.vstack([query_vec[None, :], candidate_vecs])
    is_query = np.zeros((N_NODES, 1), dtype=np.float32)
    is_query[0, 0] = 1.0
    node_feats = np.concatenate([all_vecs, is_query], axis=1).astype(np.float32)

    edge_dist = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    edge_dist[0, 1:] = query_candidate_dists
    edge_dist[1:, 0] = query_candidate_dists
    diffs = candidate_vecs[:, None, :] - candidate_vecs[None, :, :]
    edge_dist[1:, 1:] = np.linalg.norm(diffs, axis=-1)
    return node_feats, edge_dist


# ---------------------------------------------------------------------
# Training data generation (free, synthetic, from our own physics)
# ---------------------------------------------------------------------
def rng_py_random(np_rng):
    """Adapter so we can reuse data_generator's sampling functions (which
    expect Python's random.Random interface) with a numpy Generator seed."""
    import random
    return random.Random(int(np_rng.integers(0, 2**31 - 1)))


def generate_training_examples(catalog_store, n_examples=4000, max_dt_hours=48, seed=123,
                                distance_weights=None, drift_tolerance=None, margin_required=None):
    """Returns node_feats (N,6,14), edge_dist (N,6,6), labels (N,5) float32.

    Class balance: ALL known/positive examples are kept unconditionally --
    an earlier version of this function filtered out known cases the fast
    gate would have accepted on its own, on the assumption that the GNN is
    only ever invoked on gate-rejected detections. That assumption was
    wrong: the gate accepts the large majority of TRUE re-detections
    correctly (matched distance collapses close to 0 for a real match), so
    that filter discarded almost the entire positive class and the model
    trained on ~100% novel-object examples, learning to always predict
    "unknown" regardless of input.

    For NOVEL (unknown) examples, a majority of easy negatives (novel
    objects that land far from every candidate, which the gate rejects
    trivially and which teach the model little) are downsampled in favor
    of hard negatives (novel objects that happen to land close to a real
    catalog entry) -- these hard negatives are what the GNN actually needs
    practice distinguishing from true matches.
    """
    rng = np.random.default_rng(seed)
    scalers = catalog_store["scalers"]

    cat_a = catalog_store["elements_a"]
    cat_e = catalog_store["elements_e"]
    cat_i = catalog_store["elements_i"]
    cat_raan = catalog_store["elements_raan"]
    cat_argp = catalog_store["elements_argp"]
    cat_bc = catalog_store["bc_m2_per_kg"]
    cat_types = list(catalog_store["types"])
    cat_sizes = catalog_store["sizes"]
    cat_rcs = catalog_store["rcs"]
    n_cat = len(cat_types)

    all_node_feats, all_edge_dist, all_labels = [], [], []

    PROCESS_NOISE = {"a_km": 0.05, "e": 0.00005, "i_deg": 0.002, "angle_deg": 0.01}
    MEAS_NOISE = {"a_km": 0.02, "e": 0.00002, "i_deg": 0.001, "angle_deg": 0.005}

    attempts = 0
    max_attempts = n_examples * 20  # gate-reject filtering means some draws get discarded
    while len(all_labels) < n_examples and attempts < max_attempts:
        attempts += 1
        dt = rng.uniform(0, max_dt_hours * 3600)

        # propagate the whole catalog to this elapsed time (matches real inference conditions)
        prop_a, prop_e, prop_i, prop_raan, prop_argp = op.propagate_catalog_analytic(
            cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc, dt
        )
        cat_vectors = embed_mod.build_embeddings_fast(
            prop_a, prop_e, prop_i, prop_raan, prop_argp, cat_types, cat_sizes, cat_rcs, scalers
        )

        is_known_case = rng.random() < 0.5

        if is_known_case:
            src_idx = rng.integers(0, n_cat)
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
            true_name_idx = src_idx
        else:
            py_rng = rng_py_random(rng)
            obj_type = gen.sample_type(py_rng)
            size, rcs = gen.sample_size_and_rcs(py_rng, obj_type)
            meas_a = gen.sample_semi_major_axis(py_rng)
            meas_e = gen.sample_eccentricity(py_rng)
            meas_i = gen.sample_inclination(py_rng)
            meas_raan = gen.sample_angle(py_rng)
            meas_argp = gen.sample_angle(py_rng)
            q_type, q_size, q_rcs = obj_type, size, rcs
            true_name_idx = None

        query_vec = embed_mod.transform_identity_vector(
            meas_a, meas_e, meas_i, meas_raan, meas_argp, q_type, q_size, q_rcs, scalers
        )

        # top5 selected by UNWEIGHTED distance -- must match inference's
        # candidate selection (ground_station_tracking_simulation.py picks
        # top5 from unweighted_dists so the GNN always sees the same kind
        # of shortlist it was trained on).
        dists = np.linalg.norm(cat_vectors - query_vec, axis=1)
        top5 = np.argsort(dists)[:5]
        candidate_vecs = cat_vectors[top5]
        candidate_dists = dists[top5]

        if is_known_case and distance_weights is not None and drift_tolerance is not None and margin_required is not None:
            # Diagnostic check confirmed: the fast gate accepts the vast
            # majority of true re-detections correctly (candidate distance
            # collapses to ~0.01 for a real match vs DRIFT_TOLERANCE~0.3),
            # so filtering "discard known cases the gate would accept" was
            # wrong -- it discarded essentially ALL known examples, leaving
            # the GNN training on ~100% novel-object examples. It learned
            # to always say "unknown" (confirmed by near-zero confidence
            # across the board). Known/positive examples are NOT filtered:
            # the GNN needs to see real matches regardless of what the gate
            # would have done with them.
            pass
        elif not is_known_case and distance_weights is not None and drift_tolerance is not None and margin_required is not None:
            # For NOVEL objects, keep the full mix but bias sampling toward
            # the genuinely hard negatives: a random novel object usually
            # sits far from everything and the gate rejects it easily (an
            # "easy" negative that teaches the GNN little). Novel objects
            # that happen to land close to a real catalog entry (the gate's
            # margin requirement nearly satisfied, or its distance
            # requirement nearly satisfied) are the hard negatives that
            # actually cause false "known" calls at inference -- keep those
            # preferentially by discarding a portion of the easy negatives
            # instead of discarding based on gate accept/reject, which
            # would (as with known cases) risk skewing the class balance
            # in an untested direction. Keep ~40% of easy negatives so
            # novel examples aren't over-represented relative to known ones.
            w_cat_vectors = cat_vectors[top5] * distance_weights
            w_query_vec = query_vec * distance_weights
            w_dists = np.linalg.norm(w_cat_vectors - w_query_vec, axis=1)
            w_order = np.argsort(w_dists)
            w_best = w_dists[w_order[0]]
            w_second = w_dists[w_order[1]] if len(w_order) > 1 else np.inf
            w_margin = w_second - w_best
            is_easy_negative = w_best > 2 * drift_tolerance and w_margin > 2 * margin_required
            if is_easy_negative and rng.random() > 0.4:
                continue

        node_feats, edge_dist = build_graph(query_vec, candidate_vecs, candidate_dists)
        labels = np.zeros(5, dtype=np.float32)
        if true_name_idx is not None and true_name_idx in top5:
            labels[list(top5).index(true_name_idx)] = 1.0

        all_node_feats.append(node_feats)
        all_edge_dist.append(edge_dist)
        all_labels.append(labels)

    if len(all_labels) < n_examples:
        print(f"WARNING: only generated {len(all_labels)}/{n_examples} training examples "
              f"after {attempts} attempts. Consider raising max_attempts or n_examples.")

    return (np.stack(all_node_feats), np.stack(all_edge_dist), np.stack(all_labels))


# ---------------------------------------------------------------------
# Train / save / load
# ---------------------------------------------------------------------
def train_model(catalog_store, n_examples=4000, epochs=60, lr=1e-3, seed=123,
                 distance_weights=None, drift_tolerance=None, margin_required=None, gate_stamp=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to train the GNN matcher.")

    print(f"Generating {n_examples} synthetic training examples...")
    node_feats, edge_dist, labels = generate_training_examples(
        catalog_store, n_examples, seed=seed,
        distance_weights=distance_weights, drift_tolerance=drift_tolerance, margin_required=margin_required,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    node_feats_t = torch.from_numpy(node_feats).to(device)
    edge_dist_t = torch.from_numpy(edge_dist).to(device)
    labels_t = torch.from_numpy(labels).to(device)

    model = GNNMatcher().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # positive class (true match) is rare (~1-in-6 edges); weight it up
    pos_weight = torch.tensor([5.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Training GNN matcher on {device} for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(node_feats_t, edge_dist_t)
        loss = loss_fn(logits, labels_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > GNN_CONFIRM_THRESHOLD).float()
                # accuracy on whether "any edge predicted positive" matches "any true positive"
                pred_known = (preds.sum(dim=1) > 0).float()
                true_known = (labels_t.sum(dim=1) > 0).float()
                acc = (pred_known == true_known).float().mean().item()
            print(f"  epoch {epoch+1:>3}  loss={loss.item():.4f}  known/unknown_acc={acc:.4f}")

    model.eval()
    torch.save({"state_dict": model.state_dict(), "gate_stamp": gate_stamp}, MODEL_PATH)
    print(f"Saved trained GNN matcher to {MODEL_PATH}")
    return model


def load_or_train_model(catalog_store, distance_weights=None, drift_tolerance=None, margin_required=None,
                         force_retrain=False):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for the GNN matcher.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GNNMatcher().to(device)

    # TRAINING_LOGIC_VERSION: bump this whenever generate_training_examples'
    # sampling/filtering logic changes, so a stale checkpoint trained under
    # old (possibly broken) logic can't be silently reused just because the
    # gate settings happen to match. v2 fixes a bug in v1 where known/
    # positive examples were almost entirely filtered out, so the model
    # trained on ~0 positive examples and always predicted "unknown."
    TRAINING_LOGIC_VERSION = 2

    # Stamp the gate settings + training logic version the checkpoint was
    # trained under, so a model trained before DRIFT_TOLERANCE/weights/
    # training logic changed doesn't get silently reused against a now-
    # mismatched deployment distribution or fixed data-generation bug.
    gate_stamp = {
        "training_logic_version": TRAINING_LOGIC_VERSION,
        "drift_tolerance": None if drift_tolerance is None else round(float(drift_tolerance), 6),
        "margin_required": None if margin_required is None else round(float(margin_required), 6),
        "distance_weights": None if distance_weights is None else np.asarray(distance_weights, dtype=np.float32).tolist(),
    }

    if not force_retrain:
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            ckpt_stamp = ckpt.get("gate_stamp")
            if ckpt_stamp == gate_stamp:
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                print(f"Loaded existing GNN matcher from {MODEL_PATH} (gate settings match)")
                return model
            else:
                print(f"Existing GNN matcher at {MODEL_PATH} was trained under different gate "
                      f"settings than the current run -- retraining instead of using a stale model.")
        except FileNotFoundError:
            pass

    return train_model(catalog_store, distance_weights=distance_weights,
                        drift_tolerance=drift_tolerance, margin_required=margin_required,
                        gate_stamp=gate_stamp)


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
def classify(model, query_vec, candidate_vecs, candidate_dists):
    """Returns (is_known: bool, matched_local_idx: int|None, confidence: float)."""
    node_feats, edge_dist = build_graph(query_vec, candidate_vecs, candidate_dists)
    device = next(model.parameters()).device
    nf = torch.from_numpy(node_feats[None]).to(device)
    ed = torch.from_numpy(edge_dist[None]).to(device)
    with torch.no_grad():
        logits = model(nf, ed)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    best_local = int(np.argmax(probs))
    best_conf = float(probs[best_local])
    if best_conf > GNN_CONFIRM_THRESHOLD:
        return True, best_local, best_conf
    return False, None, best_conf   