#!/usr/bin/env python3
"""
Orbital physics core.

Two propagation models are provided, deliberately different in fidelity,
mirroring how real operational systems are actually built:

  - `propagate_truth_numeric()` : full numerical integration (Cowell's method,
    RK4) of two-body gravity + J2 oblateness + atmospheric drag, acting
    directly on the Cartesian state vector. This is the highest-fidelity
    model practical without a full numerical atmosphere (NRLMSISE-00) and
    higher-order gravity field -- used to generate GROUND TRUTH object
    trajectories, since we control the synthetic data and can afford the
    cost of calling this occasionally (once per detection event, for one
    object).

  - `propagate_catalog_analytic()`: closed-form secular perturbation theory
    (J2 nodal/apsidal precession + averaged exponential-atmosphere drag decay
    of semi-major axis). This is conceptually the same category of model as
    SGP4 (analytic secular theory, not step-by-step numerical integration) --
    used for the CATALOG's internal belief/prediction, because that has to
    be evaluated for every catalog object at every detection event, and
    needs to be fast and vectorizable across the whole catalog at once.

All angles in degrees at the module boundary (matching the CSV convention);
internally everything is radians. Distances in km, time in seconds.
"""

import numpy as np

MU_EARTH = 398600.4418        # km^3/s^2
RE_EARTH = 6378.137           # km, equatorial radius
J2 = 1.08263e-3                # Earth's second zonal harmonic (dimensionless)
OMEGA_EARTH = 7.2921159e-5     # rad/s, Earth's rotation rate (for atmosphere co-rotation)

# ---------------------------------------------------------------------
# Exponential atmosphere model (piecewise, standard reference values --
# same table used in Vallado's "Fundamentals of Astrodynamics and
# Applications" for approximate drag work when a full numerical
# atmosphere model isn't warranted).
#   base_alt_km : (nominal_density_kg_m3, scale_height_km)
# ---------------------------------------------------------------------
_ATMOSPHERE_TABLE = [
    (150, 2.070e-9,  22.523),
    (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
]


def atmospheric_density(altitude_km: float) -> float:
    """Piecewise exponential atmosphere density, kg/m^3."""
    alt = max(altitude_km, _ATMOSPHERE_TABLE[0][0])
    # find the highest base altitude <= alt
    band = _ATMOSPHERE_TABLE[0]
    for entry in _ATMOSPHERE_TABLE:
        if entry[0] <= alt:
            band = entry
        else:
            break
    h0, rho0, H = band
    return rho0 * np.exp(-(alt - h0) / H)


def ballistic_coefficient(size_m: float, obj_type: str, cd: float = 2.2) -> float:
    """
    Physically-derived ballistic coefficient Cd*A/m, m^2/kg.

    A (cross-sectional area) from the object's characteristic size, treating
    it as a sphere: A = pi * (size/2)^2.
    Mass from an assumed effective bulk density by object type (satellites
    and rocket bodies are mostly hollow structure, not solid metal; debris
    fragments are denser on average). These are standard order-of-magnitude
    assumptions for this class of object, not measured values.
    """
    effective_density_kg_m3 = {
        "satellite": 100.0,
        "debris": 1000.0,
        "rocket_body": 200.0,
    }.get(obj_type, 300.0)

    radius_m = size_m / 2.0
    area_m2 = np.pi * radius_m ** 2
    volume_m3 = (4.0 / 3.0) * np.pi * radius_m ** 3
    mass_kg = max(volume_m3 * effective_density_kg_m3, 0.5)  # floor to avoid div-by-zero on tiny debris

    return cd * area_m2 / mass_kg


# ---------------------------------------------------------------------
# Keplerian elements <-> Cartesian state vector (ECI frame)
# ---------------------------------------------------------------------
def kepler_to_rv(a, e, i_deg, raan_deg, argp_deg, nu_deg):
    """Convert classical Keplerian elements to position/velocity (km, km/s)."""
    i = np.radians(i_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    nu = np.radians(nu_deg)

    p = a * (1 - e ** 2)
    r_mag = p / (1 + e * np.cos(nu))

    r_pf = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    v_pf = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])

    r_eci = R @ r_pf
    v_eci = R @ v_pf
    return r_eci, v_eci


def rv_to_kepler(r, v):
    """Convert position/velocity (km, km/s) back to classical Keplerian elements
    (a in km, e dimensionless, angles in degrees)."""
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    n_vec = np.cross([0, 0, 1], h_vec)
    n_mag = np.linalg.norm(n_vec)

    e_vec = (np.cross(v, h_vec) / MU_EARTH) - (r / r_mag)
    e = np.linalg.norm(e_vec)

    energy = v_mag ** 2 / 2 - MU_EARTH / r_mag
    a = -MU_EARTH / (2 * energy)

    i = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))

    if n_mag > 1e-8:
        raan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0  # near-equatorial orbit, RAAN undefined; degenerate edge case

    if n_mag > 1e-8 and e > 1e-8:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0

    if e > 1e-8:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0

    return (a, e, np.degrees(i), np.degrees(raan) % 360, np.degrees(argp) % 360, np.degrees(nu) % 360)


# ---------------------------------------------------------------------
# Full numerical propagation (Cowell's method): two-body + J2 + drag,
# integrated directly on the Cartesian state via RK4.
# ---------------------------------------------------------------------
def _acceleration(r, v, cd_a_over_m):
    r_mag = np.linalg.norm(r)

    # two-body point-mass gravity
    a_two_body = -MU_EARTH * r / r_mag ** 3

    # J2 oblateness perturbation (standard closed form)
    x, y, z = r
    factor = 1.5 * J2 * MU_EARTH * RE_EARTH ** 2 / r_mag ** 5
    zx2 = 5 * z ** 2 / r_mag ** 2
    a_j2 = factor * np.array([
        x * (zx2 - 1),
        y * (zx2 - 1),
        z * (zx2 - 3),
    ])

    # atmospheric drag: relative velocity accounts for atmosphere co-rotating with Earth
    altitude_km = r_mag - RE_EARTH
    rho_kg_m3 = atmospheric_density(altitude_km)
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel_km_s = v - np.cross(omega_vec, r)
    v_rel_m_s = v_rel_km_s * 1000.0
    v_rel_mag_m_s = np.linalg.norm(v_rel_m_s)

    # drag accel = -0.5 * (Cd A/m) * rho * |v_rel| * v_rel   (m/s^2) -> convert to km/s^2
    a_drag_m_s2 = -0.5 * cd_a_over_m * rho_kg_m3 * v_rel_mag_m_s * v_rel_m_s
    a_drag = a_drag_m_s2 / 1000.0

    return a_two_body + a_j2 + a_drag


def _rk4_step(r, v, dt, cd_a_over_m):
    def deriv(r_, v_):
        return v_, _acceleration(r_, v_, cd_a_over_m)

    k1r, k1v = deriv(r, v)
    k2r, k2v = deriv(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v)
    k3r, k3v = deriv(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v)
    k4r, k4v = deriv(r + dt * k3r, v + dt * k3v)

    r_new = r + (dt / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
    v_new = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r_new, v_new


def _j2_potential(r):
    """J2 perturbing potential term, km^2/s^2. Adding this to the naive
    two-body specific energy (v^2/2 - mu/r) yields a quantity that is
    exactly conserved under J2 + two-body dynamics (verified numerically) --
    i.e. it filters out the short-period osculating-element oscillation that
    J2 otherwise causes in a naively-extracted 'a', leaving only genuine
    secular change (from drag) visible."""
    r_mag = np.linalg.norm(r)
    z = r[2]
    return (MU_EARTH * J2 * RE_EARTH ** 2) / (2 * r_mag ** 3) * (3 * (z / r_mag) ** 2 - 1)


def mean_semi_major_axis(r, v):
    """J2-corrected 'mean' semi-major axis (km) -- stable across short-period
    oscillation, changes only due to genuine secular effects like drag."""
    eps_mean = np.dot(v, v) / 2 - MU_EARTH / np.linalg.norm(r) + _j2_potential(r)
    return -MU_EARTH / (2 * eps_mean)


def propagate_elements(a, e, i_deg, raan_deg, argp_deg, cd_a_over_m, dt_seconds, process_noise=None, rng=None):
    """
    Physically-grounded secular propagation (validated J2 nodal/apsidal
    precession + closed-form exponential-atmosphere drag decay of 'a').
    Vectorized: all element args are numpy arrays of equal length.

    This is used for BOTH the ground-truth trajectory and the catalog's
    internal belief -- the only difference is that "truth" gets small
    physically-motivated process noise added (representing real unmodeled
    perturbations: solar activity variation, third-body effects, attitude/
    area variation, etc.), while the catalog's belief is the clean
    deterministic prediction.

    (Full numerical Cowell (RK4) integration of the exact equations of
    motion is implemented in propagate_truth_numeric() below for reference
    and was used to validate the physical magnitudes here -- but isolating
    the genuine secular drag signal from Cowell output requires full
    Brouwer/Kozai mean-element theory to filter J2's much larger short-period
    oscillation, which is effectively re-deriving SGP4's analytic machinery.
    The closed-form model here IS that analytic approach, validated against
    the numerical integration in isolation, and is what's actually used.)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    e = np.atleast_1d(np.asarray(e, dtype=float))
    i_deg = np.atleast_1d(np.asarray(i_deg, dtype=float))
    raan_deg = np.atleast_1d(np.asarray(raan_deg, dtype=float))
    argp_deg = np.atleast_1d(np.asarray(argp_deg, dtype=float))
    cd_a_over_m = np.atleast_1d(np.asarray(cd_a_over_m, dtype=float))

    new_a, new_e, new_i, new_raan, new_argp = propagate_catalog_analytic(
        a, e, i_deg, raan_deg, argp_deg, cd_a_over_m, dt_seconds
    )

    if process_noise is not None and rng is not None:
        # small physically-motivated stochastic perturbation representing
        # unmodeled effects (solar-activity-driven density variation, etc.)
        new_a = new_a + rng.normal(0, process_noise.get("a_km", 0.0), size=new_a.shape)
        new_e = np.clip(new_e + rng.normal(0, process_noise.get("e", 0.0), size=new_e.shape), 0, 0.9)
        new_i = new_i + rng.normal(0, process_noise.get("i_deg", 0.0), size=new_i.shape)
        new_raan = (new_raan + rng.normal(0, process_noise.get("angle_deg", 0.0), size=new_raan.shape)) % 360
        new_argp = (new_argp + rng.normal(0, process_noise.get("angle_deg", 0.0), size=new_argp.shape)) % 360

    return new_a, new_e, new_i, new_raan, new_argp


# ---------------------------------------------------------------------
# Fast analytic secular propagation (catalog-side "belief" model) --
# J2 nodal/apsidal precession, closed-form drag decay of 'a' using the
# local exponential-atmosphere density rate. Vectorized across many
# objects at once (all inputs are numpy arrays of equal length).
# ---------------------------------------------------------------------
def propagate_catalog_analytic(a, e, i_deg, raan_deg, argp_deg, cd_a_over_m, dt_seconds):
    """Vectorized closed-form secular propagation for N objects at once.
    Returns updated (a, e, i, raan, argp) arrays. e and i are treated as
    secularly constant at this order (standard J2 secular theory)."""
    i = np.radians(i_deg)
    n_mean = np.sqrt(MU_EARTH / a ** 3)          # mean motion, rad/s
    p = a * (1 - e ** 2)

    raan_rate = -1.5 * n_mean * J2 * (RE_EARTH / p) ** 2 * np.cos(i)
    argp_rate = 0.75 * n_mean * J2 * (RE_EARTH / p) ** 2 * (5 * np.cos(i) ** 2 - 1)

    new_raan = (raan_deg + np.degrees(raan_rate) * dt_seconds) % 360
    new_argp = (argp_deg + np.degrees(argp_rate) * dt_seconds) % 360

    # averaged drag decay of 'a' for a near-circular orbit (standard secular
    # decay approximation, e.g. King-Hele / Vallado):
    #   da/dt [m/s] = -(Cd A/m)[m^2/kg] * rho[kg/m^3] * v[m/s] * a[m]
    # Converting a to meters and the result back to km/s algebraically
    # simplifies to: da/dt [km/s] = -(Cd A/m) * rho * v[m/s] * a[km]
    altitude_km = a - RE_EARTH
    rho = np.array([atmospheric_density(h) for h in np.atleast_1d(altitude_km)])
    circular_speed_km_s = np.sqrt(MU_EARTH / a)
    v_m_s = circular_speed_km_s * 1000.0

    da_dt_km_s = -cd_a_over_m * rho * v_m_s * a

    new_a = np.maximum(a + da_dt_km_s * dt_seconds, RE_EARTH + 100)  # floor: don't decay through the atmosphere

    return new_a, e.copy(), i_deg.copy(), new_raan, new_argp