#!/usr/bin/env python3
"""
Embed the catalog into a normalized IDENTITY vector space for ground-station
tracking, and persist everything needed to propagate + re-match objects later.

Identity vector = 13 dims, deliberately EXCLUDING true anomaly (nu):
    a_z, e_log_z, i_sin, i_cos, Omega_sin, Omega_cos, omega_sin, omega_cos,
    type_satellite, type_debris, type_rocket_body, size_log_z, rcs_log_z

Why nu is excluded: nu is the object's position ALONG its orbit, which
changes completely on every single pass (~90-130 min period). It's not
identity, it's fast-changing ephemeral state -- including it would make the
same physical object look like a totally different vector on every
detection, breaking re-identification entirely.

Scalers are FIT ONCE on the embedded catalog and their parameters (mean/std)
are persisted in the output file -- this is essential for a tracking system:
every future detection must be transformed with the SAME scale, or "distance"
between two vectors measured at different times becomes meaningless.

Also persisted: each embedded object's raw physical elements (a, e, i,
Omega, omega), its ballistic coefficient, and its epoch (simulation time
the stored elements are valid at, starts at 0) -- the tracker needs these
to actually PROPAGATE the catalog's belief forward in time, not just
compare static vectors.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import orbital_physics as op

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CATALOG_INPUT_PATH = "leo_catalog.csv"
OUTPUT_PATH = "outputs/leo_embeddings.pt"
FALLBACK_NPZ_PATH = "outputs/leo_embeddings.npz"

N_OBJECTS = None
EMBED_FRACTION = 0.5     # fraction of the catalog the ground station "knows about"
SEED = 42

OBJECT_TYPES = ["satellite", "debris", "rocket_body"]

IDENTITY_DIMENSIONS = [
    "a_z", "e_log_z",
    "i_sin", "i_cos",
    "Omega_sin", "Omega_cos",
    "omega_sin", "omega_cos",
    "type_satellite", "type_debris", "type_rocket_body",
    "size_log_z", "rcs_log_z",
]

# ---------------------------------------------------------------------
# Per-dimension distance weights.
#
# Several raw dims carry far less discriminating power than others once
# z-scored, because the underlying catalog distribution is clustered/
# narrow, not because the dimension is unimportant:
#   - i_sin/i_cos: inclination is drawn from only 5 tight clusters
#     (INCLINATION_CLUSTERS in data_generator.py), so same-cluster objects
#     are near-identical on these 2 dims regardless of true identity.
#   - e_log_z: ~90% of the catalog is sampled from a narrow near-circular
#     band, so this dim is close to constant for most objects.
#   - Omega/omega sin-cos: these precess under J2 at a rate that depends
#     only on (a, i), so two objects sharing an (a, i) cluster precess in
#     lockstep -- their *relative* separation barely changes over time,
#     meaning an initially-close pair stays close indefinitely. With ~400
#     objects per inclination cluster, close initial pairs are common
#     (birthday-paradox effect), so these dims are LESS reliable than raw
#     spread would suggest.
#   - a_z: semi-major axis is close to uniform across the full LEO band
#     and is NOT shared/aliased by cluster membership -- empirically the
#     strongest discriminator, so it's upweighted.
#
# These weights multiply each dimension AFTER z-scoring, i.e. they scale
# each axis's contribution to Euclidean distance. Apply them consistently
# everywhere (embedding build, single-vector transform, fast batch
# transform) via `apply_distance_weights()` so nothing goes out of sync.
DISTANCE_WEIGHTS = np.array([
    2.2,   # a_z            -- strongest, unclustered discriminator
    0.5,   # e_log_z        -- mostly collapsed for 90% of catalog
    0.6,   # i_sin          -- clustered into 5 bands
    0.6,   # i_cos
    1.0,   # Omega_sin      -- precesses in lockstep within a cluster
    1.0,   # Omega_cos
    1.0,   # omega_sin
    1.0,   # omega_cos
    1.0,   # type_satellite
    1.0,   # type_debris
    1.0,   # type_rocket_body
    1.3,   # size_log_z     -- independent of orbital cluster, helps split ties
    1.3,   # rcs_log_z
], dtype=np.float32)


def apply_distance_weights(vectors: np.ndarray) -> np.ndarray:
    """Scale identity vector(s) by DISTANCE_WEIGHTS. Works on a single
    (13,) vector or a batch (N,13). Call this on BOTH sides of any distance
    comparison (catalog vectors and the query vector) -- weighting only one
    side silently breaks matching."""
    return vectors * DISTANCE_WEIGHTS


def cyclical_encode(degrees: np.ndarray):
    radians = np.deg2rad(degrees.astype(float))
    return np.sin(radians), np.cos(radians)


def fit_scalers(df: pd.DataFrame) -> dict:
    """Fit once on the embedded catalog; parameters get persisted and reused
    for every future single-vector transform (see transform_identity_vector)."""
    a_scaler = StandardScaler().fit(df[["a"]])
    e_log = np.log1p(df["e"].to_numpy(dtype=float)).reshape(-1, 1)
    e_scaler = StandardScaler().fit(e_log)
    size_log = np.log1p(df["size_meters"].to_numpy(dtype=float)).reshape(-1, 1)
    size_scaler = StandardScaler().fit(size_log)
    rcs_log = np.log1p(df["rcs"].to_numpy(dtype=float)).reshape(-1, 1)
    rcs_scaler = StandardScaler().fit(rcs_log)

    return {
        "a_mean": float(a_scaler.mean_[0]), "a_std": float(a_scaler.scale_[0]),
        "e_log_mean": float(e_scaler.mean_[0]), "e_log_std": float(e_scaler.scale_[0]),
        "size_log_mean": float(size_scaler.mean_[0]), "size_log_std": float(size_scaler.scale_[0]),
        "rcs_log_mean": float(rcs_scaler.mean_[0]), "rcs_log_std": float(rcs_scaler.scale_[0]),
    }


def transform_identity_vector(a, e, i_deg, raan_deg, argp_deg, obj_type, size_m, rcs, scalers: dict):
    """Transform ONE object's raw elements into the 13-dim identity vector,
    using FIXED (already-fitted) scaler parameters. This is what both the
    initial catalog build and every later detection during tracking must
    call, so everything stays on the same consistent scale."""
    a_z = (a - scalers["a_mean"]) / scalers["a_std"]
    e_log_z = (np.log1p(e) - scalers["e_log_mean"]) / scalers["e_log_std"]
    i_sin, i_cos = np.sin(np.radians(i_deg)), np.cos(np.radians(i_deg))
    O_sin, O_cos = np.sin(np.radians(raan_deg)), np.cos(np.radians(raan_deg))
    w_sin, w_cos = np.sin(np.radians(argp_deg)), np.cos(np.radians(argp_deg))
    type_onehot = [1.0 if obj_type == t else 0.0 for t in OBJECT_TYPES]
    size_log_z = (np.log1p(size_m) - scalers["size_log_mean"]) / scalers["size_log_std"]
    rcs_log_z = (np.log1p(rcs) - scalers["rcs_log_mean"]) / scalers["rcs_log_std"]

    return np.array([a_z, e_log_z, i_sin, i_cos, O_sin, O_cos, w_sin, w_cos,
                      *type_onehot, size_log_z, rcs_log_z], dtype=np.float32)


def build_embeddings_fast(a, e, i_deg, raan_deg, argp_deg, obj_types, size_m, rcs, scalers: dict) -> np.ndarray:
    """Pure-numpy vectorized identity transform for N objects at once, with
    NO pandas/sklearn object construction -- this is the hot path, called
    once per detection event for the entire live catalog, so overhead here
    matters a lot at scale (toward ~100k objects)."""
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)

    a_z = (a - scalers["a_mean"]) / scalers["a_std"]
    e_log_z = (np.log1p(e) - scalers["e_log_mean"]) / scalers["e_log_std"]

    i_sin, i_cos = cyclical_encode(np.asarray(i_deg, dtype=float))
    O_sin, O_cos = cyclical_encode(np.asarray(raan_deg, dtype=float))
    w_sin, w_cos = cyclical_encode(np.asarray(argp_deg, dtype=float))

    obj_types_arr = np.asarray(obj_types)
    type_onehot = np.column_stack([(obj_types_arr == t).astype(np.float32) for t in OBJECT_TYPES])

    size_log_z = (np.log1p(np.asarray(size_m, dtype=float)) - scalers["size_log_mean"]) / scalers["size_log_std"]
    rcs_log_z = (np.log1p(np.asarray(rcs, dtype=float)) - scalers["rcs_log_mean"]) / scalers["rcs_log_std"]

    return np.column_stack([
        a_z, e_log_z, i_sin, i_cos, O_sin, O_cos, w_sin, w_cos,
        type_onehot, size_log_z, rcs_log_z,
    ]).astype(np.float32)


def build_embeddings(df: pd.DataFrame, scalers: dict) -> np.ndarray:
    """Vectorized batch version of transform_identity_vector, for building
    the initial catalog (equivalent result to calling it row-by-row)."""
    a_z = (df["a"].to_numpy(dtype=float) - scalers["a_mean"]) / scalers["a_std"]
    e_log_z = (np.log1p(df["e"].to_numpy(dtype=float)) - scalers["e_log_mean"]) / scalers["e_log_std"]

    i_sin, i_cos = cyclical_encode(df["i"].to_numpy())
    O_sin, O_cos = cyclical_encode(df["Omega"].to_numpy())
    w_sin, w_cos = cyclical_encode(df["omega"].to_numpy())

    ohe = OneHotEncoder(categories=[OBJECT_TYPES], sparse_output=False, handle_unknown="ignore")
    type_onehot = ohe.fit_transform(df[["type"]])

    size_log_z = (np.log1p(df["size_meters"].to_numpy(dtype=float)) - scalers["size_log_mean"]) / scalers["size_log_std"]
    rcs_log_z = (np.log1p(df["rcs"].to_numpy(dtype=float)) - scalers["rcs_log_mean"]) / scalers["rcs_log_std"]

    return np.column_stack([
        a_z, e_log_z, i_sin, i_cos, O_sin, O_cos, w_sin, w_cos,
        type_onehot, size_log_z, rcs_log_z,
    ]).astype(np.float32)


def main():
    catalog_df = pd.read_csv(CATALOG_INPUT_PATH)
    full_count = len(catalog_df)

    if N_OBJECTS is not None:
        catalog_df = catalog_df.head(N_OBJECTS).reset_index(drop=True)
    elif EMBED_FRACTION is not None and EMBED_FRACTION < 1.0:
        catalog_df = (
            catalog_df.sample(frac=EMBED_FRACTION, random_state=SEED)
            .sort_index()
            .reset_index(drop=True)
        )

    print(f"Catalog has {full_count} objects; embedding {len(catalog_df)} of them "
          f"({len(catalog_df) / full_count * 100:.0f}%).")

    scalers = fit_scalers(catalog_df)
    identity_vectors = build_embeddings(catalog_df, scalers)

    print(f"Embedded {identity_vectors.shape[0]} objects into a "
          f"{identity_vectors.shape[1]}-dimensional IDENTITY space (nu excluded).")

    names = catalog_df["name"].tolist()
    payload = {
        "names": names,
        "vectors": identity_vectors,             # (N, 13) identity vectors at epoch 0
        "dimensions": IDENTITY_DIMENSIONS,
        "scalers": scalers,                       # fixed, reused for every future transform
        # raw physical state needed to PROPAGATE each catalog object forward:
        "elements_a": catalog_df["a"].to_numpy(dtype=float),
        "elements_e": catalog_df["e"].to_numpy(dtype=float),
        "elements_i": catalog_df["i"].to_numpy(dtype=float),
        "elements_raan": catalog_df["Omega"].to_numpy(dtype=float),
        "elements_argp": catalog_df["omega"].to_numpy(dtype=float),
        "bc_m2_per_kg": catalog_df["bc_m2_per_kg"].to_numpy(dtype=float),
        "types": catalog_df["type"].tolist(),
        "sizes": catalog_df["size_meters"].to_numpy(dtype=float),
        "rcs": catalog_df["rcs"].to_numpy(dtype=float),
        "epoch_seconds": np.zeros(len(catalog_df)),  # sim-time each object's stored state is valid at
        "distance_weights": DISTANCE_WEIGHTS,  # persisted so matching always uses the weights this
                                                # embedding was built/calibrated with, never a stale copy
    }

    if TORCH_AVAILABLE:
        vectors_t = torch.from_numpy(identity_vectors)
        payload["vectors"] = vectors_t
        torch.save(payload, OUTPUT_PATH)
        print(f"Wrote PyTorch tensor file to {OUTPUT_PATH}  (vectors shape: {tuple(vectors_t.shape)})")
    else:
        np.savez(FALLBACK_NPZ_PATH, **{k: (v if not isinstance(v, dict) else np.array([v], dtype=object))
                                        for k, v in payload.items()})
        print(f"torch not available in this environment -- wrote fallback {FALLBACK_NPZ_PATH} instead.")


if __name__ == "__main__":
    main()