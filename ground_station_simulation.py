#!/usr/bin/env python3
"""
Ground station orbital-pass simulation.

Every object in leo_catalog.csv has a real orbital period, derived from its
semi-major axis via Kepler's third law. A ground station "sees" an object
once per orbit (one pass per period). We simulate this on a HYPER-ACCELERATED
clock (real orbital periods are ~90-130 minutes -- far too slow to sit and
wait for), while the actual embedding lookup performed on each pass is timed
with genuine wall-clock precision, since that's real compute work that isn't
being sped up.

Only ACCELERATION_FACTOR-times faster than reality on the ORBITAL side.
The LOOKUP side is never accelerated -- it's measured as-is.

Pipeline:
    1. Load full catalog (leo_catalog.csv)              <- "all objects seen by the sensor"
    2. Load embedding store (outputs/leo_embeddings.pt) <- "what the ground station has on record"
       (only half the catalog, per the earlier embedding step)
    3. For every object: period = 2*pi*sqrt(a^3 / mu)   <- Kepler's third law
    4. Schedule recurring appearance events on a compressed clock
    5. Run the event queue for a fixed simulated duration
    6. On each appearance: look the object up against the embedding,
       print KNOWN / NOT KNOWN with the actual (real, unaccelerated) lookup time
    7. Report aggregate metrics at the end
"""

import heapq
import time
import numpy as np
import pandas as pd

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
FALLBACK_NPZ_PATH = "outputs/leo_embeddings.npz"   # used only if torch/.pt isn't available

MU_EARTH = 398600.4418   # standard gravitational parameter of Earth, km^3/s^2

# How much faster than real orbital motion the simulation clock runs.
# e.g. 300 -> a 95-minute real orbit becomes a ~19-second simulated interval.
ACCELERATION_FACTOR = 300

# How long to run the simulation, in SIMULATED seconds (post-acceleration).
SIMULATION_DURATION_SECONDS = 120

SEED = 7


# ---------------------------------------------------------------------
# 1. Load catalog + embedding store
# ---------------------------------------------------------------------
def load_embedding_names(pt_path: str, npz_fallback: str) -> set:
    if TORCH_AVAILABLE:
        try:
            data = torch.load(pt_path, weights_only=False)
            return set(data["names"])
        except FileNotFoundError:
            pass
    data = np.load(npz_fallback, allow_pickle=True)
    return set(data["names"].tolist())


def orbital_period_seconds(a_km: float) -> float:
    """Kepler's third law: T = 2*pi*sqrt(a^3 / mu), real seconds."""
    return 2 * np.pi * np.sqrt(a_km ** 3 / MU_EARTH)


# ---------------------------------------------------------------------
# 2. Simulation
# ---------------------------------------------------------------------
def main():
    catalog_df = pd.read_csv(CATALOG_PATH)
    embedded_names = load_embedding_names(EMBEDDINGS_PATH, FALLBACK_NPZ_PATH)

    print(f"Catalog:   {len(catalog_df)} objects ({CATALOG_PATH})")
    print(f"Embedding: {len(embedded_names)} objects on record")
    print(f"Acceleration factor: {ACCELERATION_FACTOR}x real orbital speed")
    print(f"Simulated run length: {SIMULATION_DURATION_SECONDS}s (simulated)")
    print("-" * 70)

    rng = np.random.default_rng(SEED)

    # Real orbital period per object, and its compressed (simulated) period
    real_periods = orbital_period_seconds(catalog_df["a"].to_numpy())
    sim_periods = real_periods / ACCELERATION_FACTOR

    names = catalog_df["name"].tolist()
    types = catalog_df["type"].tolist()

    # Event queue: (next_sim_time, object_index). Random initial phase so
    # objects don't all appear synchronized at t=0, just like real passes.
    heap = []
    for idx in range(len(catalog_df)):
        phase = rng.uniform(0, sim_periods[idx])
        heapq.heappush(heap, (phase, idx))

    # --- metrics ---
    total_detections = 0
    known_count = 0
    unknown_count = 0
    lookup_times_known = []     # real microseconds
    lookup_times_unknown = []   # real microseconds
    per_type_known = {}
    per_type_unknown = {}

    sim_clock = 0.0
    real_start = time.perf_counter()

    while heap:
        sim_time, idx = heapq.heappop(heap)
        if sim_time > SIMULATION_DURATION_SECONDS:
            break
        sim_clock = sim_time

        name = names[idx]
        obj_type = types[idx]

        # --- the actual embedding lookup: real wall-clock time, not accelerated ---
        t0 = time.perf_counter()
        found = name in embedded_names
        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1_000_000

        total_detections += 1
        if found:
            known_count += 1
            lookup_times_known.append(elapsed_us)
            per_type_known[obj_type] = per_type_known.get(obj_type, 0) + 1
            status = "KNOWN"
        else:
            unknown_count += 1
            lookup_times_unknown.append(elapsed_us)
            per_type_unknown[obj_type] = per_type_unknown.get(obj_type, 0) + 1
            status = "NOT KNOWN"

        print(f"[T+{sim_clock:7.2f}s sim] {name:<20} -> {status:<10} (lookup {elapsed_us:.2f} us)")

        # schedule the object's next pass
        heapq.heappush(heap, (sim_time + sim_periods[idx], idx))

    real_elapsed = time.perf_counter() - real_start

    # ---------------------------------------------------------------------
    # 3. Metrics
    # ---------------------------------------------------------------------
    def stats(times):
        if not times:
            return (0.0, 0.0, 0.0)
        return (sum(times) / len(times), min(times), max(times))

    all_times = lookup_times_known + lookup_times_unknown
    avg_all, fast_all, slow_all = stats(all_times)
    avg_k, fast_k, slow_k = stats(lookup_times_known)
    avg_u, fast_u, slow_u = stats(lookup_times_unknown)

    equivalent_real_hours = (SIMULATION_DURATION_SECONDS * ACCELERATION_FACTOR) / 3600

    print("\n" + "=" * 70)
    print("SIMULATION METRICS")
    print("=" * 70)
    print(f"Simulated duration        : {SIMULATION_DURATION_SECONDS} s "
          f"(~{equivalent_real_hours:.2f} real hours of orbital tracking, "
          f"{ACCELERATION_FACTOR}x acceleration)")
    print(f"Real wall-clock time spent: {real_elapsed * 1000:.3f} ms "
          f"(script + all lookups combined)")
    print(f"Total detections (passes) : {total_detections}")
    print(f"  Known                   : {known_count} ({known_count/total_detections*100:.1f}%)")
    print(f"  Not known               : {unknown_count} ({unknown_count/total_detections*100:.1f}%)")
    print(f"Detection throughput      : {total_detections / real_elapsed:.0f} detections/sec (real time)")
    print()
    print(f"Orbital period range      : {real_periods.min()/60:.1f} - {real_periods.max()/60:.1f} min "
          f"(mean {real_periods.mean()/60:.1f} min)")
    print(f"Compressed period range   : {sim_periods.min():.2f} - {sim_periods.max():.2f} s "
          f"(mean {sim_periods.mean():.2f} s)")
    print()
    print(f"{'':<10}{'avg (us)':>12}{'fastest (us)':>16}{'slowest (us)':>16}")
    print(f"{'ALL':<10}{avg_all:>12.3f}{fast_all:>16.3f}{slow_all:>16.3f}")
    print(f"{'KNOWN':<10}{avg_k:>12.3f}{fast_k:>16.3f}{slow_k:>16.3f}")
    print(f"{'UNKNOWN':<10}{avg_u:>12.3f}{fast_u:>16.3f}{slow_u:>16.3f}")
    if lookup_times_known and lookup_times_unknown:
        delta_pct = (avg_u - avg_k) / avg_k * 100
        direction = "slower" if delta_pct > 0 else "faster"
        print(f"\nUnknown-object lookups were {abs(delta_pct):.1f}% {direction} than known-object lookups, on average.")

    print("\nPer-type breakdown (known / not known):")
    all_types = sorted(set(per_type_known) | set(per_type_unknown))
    for t in all_types:
        k = per_type_known.get(t, 0)
        u = per_type_unknown.get(t, 0)
        print(f"  {t:<14} known={k:<5} not_known={u:<5} total={k+u}")


if __name__ == "__main__":
    main()