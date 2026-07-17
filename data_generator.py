#!/usr/bin/env python3
"""
Generate synthetic Low Earth Orbit (LEO) satellite catalog data
in Keplerian element form, with realistic value ranges.

Columns:
    name          - unique identifier
    a             - semi-major axis (km)
    e             - eccentricity (unitless, 0-1)
    i             - inclination (degrees)
    Omega         - RAAN, right ascension of ascending node (degrees)
    omega         - argument of periapsis (degrees)
    nu            - true anomaly (degrees)
    type          - object type (satellite / debris / rocket_body)
    size_meters   - characteristic size (m)
    rcs           - radar cross section (m^2)
    bc_m2_per_kg  - ballistic coefficient Cd*A/m, physically derived from
                    size_meters and type (see orbital_physics.ballistic_coefficient)
                    -- this drives how fast the object's orbit decays from drag
"""

import csv
import random
import argparse

import orbital_physics as op

EARTH_RADIUS_KM = 6378.137

# Realistic LEO altitude band (km above surface)
LEO_ALT_MIN = 160
LEO_ALT_MAX = 2000

# Common inclination clusters seen in real LEO populations
# (sun-synchronous ~97-99, ISS-like ~51.6, polar ~90, low-inclination ~0-30, Starlink-like ~53/70/97)
INCLINATION_CLUSTERS = [
    (0, 30),      # low-inclination / equatorial
    (51.0, 53.5), # ISS / Starlink-like
    (60, 75),     # mid-inclination constellations
    (82, 90),     # near-polar
    (96, 99.5),   # sun-synchronous
]

OBJECT_TYPES = ["satellite", "debris", "rocket_body"]
TYPE_WEIGHTS = [0.55, 0.35, 0.10]

# Size / RCS ranges per object type (rough real-world scale)
SIZE_RANGES = {
    "satellite": (0.3, 8.0),
    "debris": (0.05, 2.0),
    "rocket_body": (5.0, 15.0),
}
RCS_RANGES = {
    "satellite": (0.05, 12.0),
    "debris": (0.01, 3.0),
    "rocket_body": (5.0, 30.0),
}


def sample_semi_major_axis(rng):
    alt = rng.uniform(LEO_ALT_MIN, LEO_ALT_MAX)
    return EARTH_RADIUS_KM + alt


def sample_eccentricity(rng):
    # Most LEO objects are near-circular; occasional elliptical orbits
    if rng.random() < 0.9:
        return round(rng.uniform(0.0001, 0.01), 10)
    else:
        return round(rng.uniform(0.01, 0.15), 10)


def sample_inclination(rng):
    lo, hi = rng.choice(INCLINATION_CLUSTERS)
    # add slight spread beyond the cluster edges
    val = rng.uniform(lo, hi) + rng.gauss(0, 1.0)
    return max(0.0, min(180.0, val))


def sample_angle(rng):
    """Uniform angle 0-360 degrees (RAAN, arg. periapsis, true anomaly)."""
    return rng.uniform(0.0, 360.0)


def sample_type(rng):
    return rng.choices(OBJECT_TYPES, weights=TYPE_WEIGHTS, k=1)[0]


def sample_size_and_rcs(rng, obj_type):
    smin, smax = SIZE_RANGES[obj_type]
    rmin, rmax = RCS_RANGES[obj_type]
    size = rng.uniform(smin, smax)
    # RCS loosely correlated with size, with noise
    frac = (size - smin) / (smax - smin + 1e-9)
    rcs = rmin + frac * (rmax - rmin)
    rcs *= rng.uniform(0.6, 1.4)  # noise
    rcs = max(rmin * 0.5, min(rmax * 1.2, rcs))
    return size, rcs


def generate_row(idx, rng):
    obj_type = sample_type(rng)
    size, rcs = sample_size_and_rcs(rng, obj_type)
    bc = op.ballistic_coefficient(size, obj_type)  # physically consistent with size + type
    return {
        "name": f"{obj_type}_{idx}",
        "a": sample_semi_major_axis(rng),
        "e": sample_eccentricity(rng),
        "i": sample_inclination(rng),
        "Omega": sample_angle(rng),
        "omega": sample_angle(rng),
        "nu": sample_angle(rng),
        "type": obj_type,
        "size_meters": size,
        "rcs": rcs,
        "bc_m2_per_kg": bc,
    }


def generate_catalog(n, seed=None):
    rng = random.Random(seed)
    return [generate_row(i, rng) for i in range(n)]


def write_csv(rows, path):
    fieldnames = ["name", "a", "e", "i", "Omega", "omega", "nu", "type",
                  "size_meters", "rcs", "bc_m2_per_kg"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LEO Keplerian catalog data.")
    parser.add_argument("-n", "--num", type=int, default=20000, help="Number of objects to generate")
    parser.add_argument("-o", "--output", type=str, default="leo_catalog.csv", help="Output CSV path")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    rows = generate_catalog(args.num, seed=args.seed)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()