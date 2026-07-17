#!/usr/bin/env python3
"""
Cross-check every object in leo_catalog.csv against the embedded object
set stored in leo_embeddings.pt.

For each row in the catalog:
    - print which object is being checked
    - look it up (by name) in the embedding file
    - time how long the lookup took
    - tally it as recognized / unrecognized

At the end, print a summary: counts, hit rate, and timing stats.
"""

import time
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------
# Config (relative paths — run from the same directory as the files)
# ---------------------------------------------------------------------
CATALOG_PATH = "leo_catalog.csv"
EMBEDDINGS_PATH = "outputs/leo_embeddings.pt"
FALLBACK_NPZ_PATH = "leo_embeddings.npz"  # used only if torch/.pt isn't available


def load_embedding_names(pt_path: str, npz_fallback: str) -> set:
    """Load just the object names out of the embedding file, as a set
    for O(1) membership checks."""
    if TORCH_AVAILABLE:
        try:
            data = torch.load(pt_path, weights_only=False)
            return set(data["names"])
        except FileNotFoundError:
            pass

    # fallback path (used here in sandbox / torch-less environments)
    import numpy as np
    data = np.load(npz_fallback, allow_pickle=True)
    return set(data["names"].tolist())


def main():
    embedded_names = load_embedding_names(EMBEDDINGS_PATH, FALLBACK_NPZ_PATH)
    print(f"Loaded {len(embedded_names)} objects from embedding store.\n")

    catalog_df = pd.read_csv(CATALOG_PATH)

    recognized = 0
    unrecognized = 0
    found_times = []      # seconds->us, lookups that were FOUND
    notfound_times = []   # seconds->us, lookups that were NOT FOUND

    overall_start = time.perf_counter()

    for _, row in catalog_df.iterrows():
        name = row["name"]

        t0 = time.perf_counter()
        found = name in embedded_names
        t1 = time.perf_counter()

        elapsed_us = (t1 - t0) * 1_000_000

        if found:
            recognized += 1
            found_times.append(elapsed_us)
            status = "FOUND"
        else:
            unrecognized += 1
            notfound_times.append(elapsed_us)
            status = "NOT FOUND"

        print(f"Checking {name:<20} -> {status:<10} ({elapsed_us:.2f} us)")

    overall_elapsed = time.perf_counter() - overall_start

    total = recognized + unrecognized
    all_times = found_times + notfound_times

    def stats(times):
        if not times:
            return (0.0, 0.0, 0.0)
        return (sum(times) / len(times), min(times), max(times))

    avg_us, fastest_us, slowest_us = stats(all_times)
    hit_avg, hit_fastest, hit_slowest = stats(found_times)
    miss_avg, miss_fastest, miss_slowest = stats(notfound_times)

    print("\n" + "=" * 50)
    print("CHECK SUMMARY")
    print("=" * 50)
    print(f"Total objects checked : {total}")
    print(f"Recognized (hits)     : {recognized}")
    print(f"Unrecognized (misses) : {unrecognized}")
    print(f"Recognition rate      : {(recognized / total * 100 if total else 0):.1f}%")
    print(f"Miss rate             : {(unrecognized / total * 100 if total else 0):.1f}%")
    print(f"Total wall time       : {overall_elapsed * 1000:.3f} ms")
    print()
    print(f"{'':<10}{'avg (us)':>12}{'fastest (us)':>16}{'slowest (us)':>16}")
    print(f"{'ALL':<10}{avg_us:>12.3f}{fastest_us:>16.3f}{slowest_us:>16.3f}")
    print(f"{'HITS':<10}{hit_avg:>12.3f}{hit_fastest:>16.3f}{hit_slowest:>16.3f}")
    print(f"{'MISSES':<10}{miss_avg:>12.3f}{miss_fastest:>16.3f}{miss_slowest:>16.3f}")
    if found_times and notfound_times:
        delta_pct = (miss_avg - hit_avg) / hit_avg * 100
        direction = "slower" if delta_pct > 0 else "faster"
        print(f"\nMisses were {abs(delta_pct):.1f}% {direction} than hits, on average.")


if __name__ == "__main__":
    main()