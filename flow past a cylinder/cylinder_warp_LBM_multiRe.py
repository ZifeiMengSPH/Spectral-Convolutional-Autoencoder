r"""
Differentiable cylinder flow: NVIDIA Warp, D2Q9 MRT-LBM, and a smooth PSM body.

Numerics
--------
The fused pull-streaming/collision kernel stores post-collision populations.
MRT uses the Lallemand-Luo basis. The "magic" option sets the odd-moment rate
to 8*(2-s_v)/(8-s_v); "reg" relaxes the ghost moments to equilibrium. This
ghost relaxation is not a general equivalence to every regularized LBM scheme.
Optional Smagorinsky viscosity is computed from the deviatoric non-equilibrium
stress. PSM blends the fluid collision with a moving-solid collision, using
either Noble-Torczynski or linear volume-fraction weighting.

The radial Fourier shape has four harmonics. Rotation prescribes a tangential
velocity about the center; jets prescribe velocity along the shape normal. The
rotation rate is clipped by omega_limit(): a rotational surface-speed bound combined with a
resolution heuristic, because a fast-spinning body under-resolves its own
boundary layer. That bound is a stability guard, not a validated envelope.
For a noncircular stationary shape, rotation is a prescribed velocity field,
not a simulation of a rigid body whose geometry rotates in time. Jet forcing
does not impose a separate mass source or enforce zero integrated mass flux.
The diffuse interface has a smooth, compact exterior tail. The deep solid
center uses a constant fraction to avoid the undefined polar angle at r=0.

Boundaries: Zou-He velocity inlet, an approximate zero-gradient outlet with a
viscosity sponge, and far/slip/noslip/periodic top and bottom conditions. The
outlet is not an exact nonreflecting boundary. Macroscopic output uses the
stored post-collision populations; values inside the PSM band depend on this
time convention.

Objectives and differentiation
------------------------------
All 23 design variables are stored in params (see PAR_NAMES and VAR_GROUPS).
drag = mean(Cd); liftvar = mean((Cl-mean(Cl))**2); combo = mean(Cd)+w*liftvar.
series matches a Cd/Cl history; field matches the final sampled velocity.
Force coefficients use the fixed baseline U and D, including during inference.
All derivatives hold the spin-up state fixed. They do not include derivatives
through spin-up or describe an infinite-time statistical sensitivity.

Checkpointed BPTT with T steps and sub-window W stores O(T/W) checkpoints and
O(W) replay states, plus O(T) force samples. Total state storage is O(W+T/W),
not independent of T. Including replay gradients, variable storage is proportional
to 2*(W+1)+ceil(T/W), minimized near W = sqrt(T/2). Before spin-up, commands
estimate storage and reduce or rebalance W if the CUDA budget requires it.
The physical horizon T is unchanged. Checkpoints default to host memory on
a GPU run, which costs two transfers per sub-window; --checkpoint-device device
keeps them resident when device memory allows. CPU runs use ordinary host
copies. Forward-only objective evaluations store no checkpoints.

Force coefficients are normalized by the FIXED baseline U and D. A drag-type
objective therefore falls if U_in is left in the active design set, without any
improvement to the body. Use --vars design (geometry and control only) for
design studies; --vars all is for gradient testing and joint inference.

Validation and limitations
--------------------------
nd/sqrt(Re) is a resolution heuristic, not an accuracy certificate. Check grid,
domain, interface-width, and time-window convergence for each application.
No universal blockage correction is applied. The bundled REF_2D ranges are
legacy orientation values without a reproducible benchmark provenance here.
Earlier embedded numerical benchmark claims were not independently reproduced
and are not validation of this version. High-Re 2D runs do not resolve 3D
cylinder wakes. Long chaotic windows may produce rapidly growing adjoints.
Boundary perturbations also enter from the far-field edges; an inlet-to-body
acoustic travel estimate is not a strict criterion for exactly zero gradients.
Run fp64 finite differences on a small case after changing numerical kernels.
When gradcheck fails it repeats the worst variable at h/2 and 2h: a gap that
shrinks with h is finite-difference truncation, a gap that does not is the
derivative. Field health (nonfinite populations, density floor) is reduced on
the device, so the check costs one kernel and two scalars per call.

6. Usage
--------
  python cylinder_warp_lbm.py table
  python cylinder_warp_lbm.py run --re 100 --device cuda --save-field
  python cylinder_warp_lbm.py gradcheck --device cpu --fp64 --nd 8 --L 8 --H 6 \
      --xc 2 --spinup 30 --adj-steps 20 --window 7 --vars all
  python cylinder_warp_lbm.py optimize --re 200 --vars design --loss drag
  python cylinder_warp_lbm.py inverse --re 200 --adj-steps 3000 --window 100

Errors from configuration, validation and divergence all exit with a single
short message; --traceback restores the full stack.

For gradcheck/optimize --loss series, pass --target forces.npz containing
target of shape (adj_steps, 2), in baseline-normalized Cd/Cl units.
For --loss field, pass --target field.npz containing target of shape (2,ni,nj)
and integer scalar keys x0, y0, stride defining the probe grid.
Optional --cache-dir selects the Warp compilation cache directory.

Requires warp-lang and numpy; matplotlib is optional for plotting. This
revision is regression-tested with Warp 1.12.1; other versions need checking.
The health kernel uses explicit signed bounds to avoid unary-negation issues
with closed-over float64 constants in that version.
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import warp as wp

# ==============================================================================
# 0. Design-variable layout (the params array)
# ==============================================================================
NFOUR = 4      # number of shape Fourier harmonics (a_k, b_k), k = 1..NFOUR
NJET  = 4      # number of blowing/suction harmonics (c_k, d_k), k = 1..NJET

P_UX  = 0      # inlet x velocity (lattice units)
P_UY  = 1      # inlet y velocity (angle of attack / transverse forcing)
P_TAU = 2      # base relaxation time tau0 -> nu = (tau0-0.5)/3 -> Re
P_XC  = 3      # cylinder centre x (lattice)
P_YC  = 4      # cylinder centre y (lattice)
P_R   = 5      # cylinder radius   (lattice)
P_OMG = 6      # rotation rate (lattice units, positive = counter-clockwise)
P_A   = 7                    # a_1..a_NFOUR
P_B   = P_A + NFOUR          # b_1..b_NFOUR
P_JC  = P_B + NFOUR          # c_1..c_NJET  (normal blowing/suction, cos terms)
P_JS  = P_JC + NJET          # d_1..d_NJET  (normal blowing/suction, sin terms)
NPAR  = P_JS + NJET

PAR_NAMES = (["U_in_x", "U_in_y", "tau0", "xc", "yc", "R", "omega"]
             + [f"a{k+1}" for k in range(NFOUR)]
             + [f"b{k+1}" for k in range(NFOUR)]
             + [f"jc{k+1}" for k in range(NJET)]
             + [f"js{k+1}" for k in range(NJET)])

VAR_GROUPS = {
    "inlet":   [P_UX, P_UY],
    "visc":    [P_TAU],
    "params":  [P_UX, P_TAU],                      # usual pair for inference
    "pos":     [P_XC, P_YC],
    "size":    [P_R],
    "shape":   [P_R] + list(range(P_A, P_A + 2 * NFOUR)),
    "control": [P_OMG] + list(range(P_JC, P_JC + 2 * NJET)),
    "rot":     [P_OMG],
    "jet":     list(range(P_JC, P_JC + 2 * NJET)),
    "all":     list(range(NPAR)),
    # Geometry and control only. Prefer this over "all" for design studies: the force
    # coefficients are normalized by the FIXED baseline U, so leaving U_in active lets a
    # drag objective fall simply by reducing the inflow.
    "design":  [P_XC, P_YC, P_R, P_OMG] + list(range(P_A, NPAR)),
}

WALL_SLIP, WALL_NOSLIP, WALL_PERIODIC, WALL_FAR = 0, 1, 2, 3
WALL_MAP = {"far": WALL_FAR, "slip": WALL_SLIP, "noslip": WALL_NOSLIP,
            "periodic": WALL_PERIODIC}
GHOST_MAGIC, GHOST_REG = 0, 1

# ==============================================================================
# 1. Resolution / relaxation settings per Reynolds number
# ==============================================================================
# nd    : lattice nodes per diameter (D = nd)
# u     : free-stream lattice velocity (<< cs = 0.577 to limit compressibility error)
# L,H   : domain size / D
# xc    : inlet-to-centre distance / D
# cs    : Smagorinsky constant Cs (0 disables LES)
# ghost : MRT ghost-moment strategy
# tconv : total physical time / (D/U)
# Resolution heuristic: nd/sqrt(Re) estimates nodes across a laminar boundary layer.
# It does not replace grid, domain, and interface-width convergence studies.
RE_TABLE = {
    100:   dict(nd=48,  u=0.060, L=32, H=20, xc=8, cs=0.00, ghost="magic", tconv=250),
    200:   dict(nd=64,  u=0.060, L=32, H=20, xc=8, cs=0.00, ghost="magic", tconv=250),
    300:   dict(nd=80,  u=0.070, L=30, H=18, xc=8, cs=0.00, ghost="magic", tconv=200),
    400:   dict(nd=88,  u=0.070, L=30, H=18, xc=8, cs=0.00, ghost="magic", tconv=200),
    500:   dict(nd=96,  u=0.080, L=28, H=16, xc=8, cs=0.00, ghost="magic", tconv=180),
    1000:  dict(nd=128, u=0.080, L=28, H=16, xc=8, cs=0.10, ghost="reg",   tconv=150),
    2000:  dict(nd=160, u=0.090, L=26, H=15, xc=7, cs=0.12, ghost="reg",   tconv=120),
    3000:  dict(nd=176, u=0.090, L=26, H=15, xc=7, cs=0.12, ghost="reg",   tconv=100),
    10000: dict(nd=224, u=0.100, L=24, H=14, xc=7, cs=0.14, ghost="reg",   tconv=80),
}

# Legacy reference ranges for orientation only; provenance is not recorded here.
REF_2D = {
    100:   dict(cd=(1.32, 1.40), st=(0.163, 0.168), note="2D and 3D agree, laminar periodic vortex street"),
    200:   dict(cd=(1.30, 1.42), st=(0.190, 0.202), note="onset of 3D transition (mode A), 2D slightly high"),
    300:   dict(cd=(1.35, 1.45), st=(0.205, 0.218), note="real flow already 3D, 2D is a numerical test case"),
    400:   dict(cd=(1.38, 1.50), st=(0.210, 0.225), note="as above"),
    500:   dict(cd=(1.40, 1.55), st=(0.215, 0.232), note="as above"),
    1000:  dict(cd=(1.45, 1.70), st=(0.225, 0.245), note="2D clearly overpredicts (experiment Cd~1.0)"),
    2000:  dict(cd=(1.50, 1.90), st=(0.230, 0.255), note="shear-layer instability dominates, LES required"),
    3000:  dict(cd=(1.55, 2.00), st=(0.230, 0.260), note="as above"),
    10000: dict(cd=(1.60, 2.30), st=(0.200, 0.260), note="experiment (3D) Cd~1.2, St~0.20"),
}


def make_config(re, scale=1.0, override=None):
    """Build the full discrete configuration from Re. scale < 1 shrinks nd for quick tests."""
    _positive(re, "re")
    _positive(scale, "scale")
    if re in RE_TABLE:
        base = dict(RE_TABLE[re])
    else:                                   # Re outside the table: extrapolate from the nearest entry
        key = min(RE_TABLE, key=lambda k: abs(math.log(k) - math.log(max(re, 1))))
        base = dict(RE_TABLE[key])
        base["nd"] = int(round(base["nd"] * math.sqrt(re / key)))
    if override:
        base.update({k: v for k, v in override.items() if v is not None})

    for name in ("nd", "u", "L", "H", "xc", "tconv"):
        _positive(base[name], name)
    _positive(base["cs"], "cs", allow_zero=True)
    if base["u"] >= 1.0 / math.sqrt(3.0):
        raise ValueError("u must be below the lattice sound speed (1/sqrt(3))")
    if base["ghost"] not in ("magic", "reg"):
        raise ValueError("ghost must be 'magic' or 'reg'")

    nd = max(8, int(round(base["nd"] * scale)))
    u  = float(base["u"])
    nu = u * nd / float(re)                 # lattice viscosity
    tau0 = 3.0 * nu + 0.5

    nx = int(round(base["L"] * nd))
    ny = int(round(base["H"] * nd))
    cx = float(base["xc"] * nd)
    cy = float(ny) * 0.5 + 0.5              # half-cell offset breaks symmetry, speeds up shedding
    steps = int(round(base["tconv"] * nd / u))
    if nx < 4 or ny < 4 or steps < 1:
        raise ValueError("the grid must be at least 4x4 and the run must contain a step")
    if min(cx, nx - 1 - cx, cy, ny - 1 - cy) <= 0.5 * nd + 1.0:
        raise ValueError("the cylinder must fit inside the domain with a one-cell margin")

    n_bl = nd / math.sqrt(max(re, 1.0))
    # As tau approaches 0.5 the magic odd-moment rate approaches zero.
    # Use equilibrium ghost relaxation on coarse grids unless explicitly overridden.
    ghost_auto = False
    if (not override or override.get("ghost") is None) \
            and base["ghost"] == "magic" and n_bl < 3.0:
        base["ghost"] = "reg"
        ghost_auto = True
    cfg = dict(re=float(re), nd=nd, D=float(nd), u=u, nu=nu, tau0=tau0,
               n_bl=n_bl, ghost_auto=ghost_auto,
               nx=nx, ny=ny, cx=cx, cy=cy, R=0.5 * nd,
               cs=float(base["cs"]), ghost=base["ghost"], steps=steps,
               tconv=base["tconv"], L=base["L"], H=base["H"])
    if tau0 <= 0.5:
        raise ValueError(f"tau0={tau0} <= 0.5; increase nd or u, or reduce Re")
    return cfg


def _positive(value, name, allow_zero=False):
    """Reject nonfinite or out-of-range scalar settings before allocation."""
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {bound}")


def _integer(value, name, minimum=1):
    if not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def print_table(scale=1.0):
    hdr = (f"{'Re':>7} {'nd':>5} {'u_lb':>6} {'nu_lb':>9} {'tau0':>8} {'Ma':>6} "
           f"{'N_bl':>5} {'nx':>6} {'ny':>6} {'Mcell':>7} {'GB':>5} {'steps':>8} "
           f"{'Gupd':>6} {'LES':>5} {'ghost':>6}")
    print(hdr)
    print("-" * len(hdr))
    tot = 0.0
    for re in sorted(RE_TABLE):
        c = make_config(re, scale)
        ma = c["u"] / math.sqrt(1.0 / 3.0)
        mc = c["nx"] * c["ny"] / 1e6
        gb = mc * 1e6 * 9 * 4 * 2 / 1024 ** 3
        gu = mc * c["steps"] / 1e3
        tot += gu
        print(f"{re:>7d} {c['nd']:>5d} {c['u']:>6.3f} {c['nu']:>9.5f} {c['tau0']:>8.5f} "
              f"{ma:>6.3f} {c['n_bl']:>5.1f} {c['nx']:>6d} {c['ny']:>6d} {mc:>7.2f} "
              f"{gb:>5.2f} {c['steps']:>8d} {gu:>6.0f} {c['cs']:>5.2f} {c['ghost']:>6s}")
    print("\n  N_bl = nd/sqrt(Re) = nodes across the boundary layer "
          "(resolution heuristic; verify convergence)")
    print("  GB   = double-buffered population memory (fp32);  Gupd = billions of node updates")
    print(f"  All nine Re total {tot:.0f} Gupd -- wall time from your measured MLUPS: "
          f"1000 MLUPS ~= {tot/1000/3.6:.1f} h, 2000 MLUPS ~= {tot/2000/3.6:.1f} h")
    print("\n2D reference values:")
    for re in sorted(REF_2D):
        r = REF_2D[re]
        print(f"  Re={re:<6d} Cd ~ {r['cd'][0]:.2f}-{r['cd'][1]:.2f}  "
              f"St ~ {r['st'][0]:.3f}-{r['st'][1]:.3f}   {r['note']}")


# ==============================================================================
# 2. Warp kernels (built per floating-point type, so fp64 gradient checks are possible)
# ==============================================================================
_KERNEL_CACHE = {}


def build_kernels(dt):
    """Build and cache every kernel. dt = wp.float32 / wp.float64.

    Top/bottom boundary (wall):
      0 slip     specular reflection -- an infinite row of mirror images, max blockage
      1 noslip   half-way bounce-back -- a real wind-tunnel wall
      2 periodic
      3 far      free-stream equilibrium for the missing populations; lets mass leave
                 laterally, far less confining than slip, closest to unbounded flow (default)
    """
    if dt in _KERNEL_CACHE:
        return _KERNEL_CACHE[dt]

    v9 = wp.types.vector(length=9, dtype=dt)

    # ---- typed constants (Warp does not mix float32/float64 implicitly) ----
    F0, F05, F1, F15 = dt(0.0), dt(0.5), dt(1.0), dt(1.5)
    F2, F3, F4, F45 = dt(2.0), dt(3.0), dt(4.0), dt(4.5)
    F8, F18 = dt(8.0), dt(18.0)
    R9, R36, R6, R12, R4 = dt(1.0/9.0), dt(1.0/36.0), dt(1.0/6.0), dt(1.0/12.0), dt(0.25)
    W0, W1, W2 = dt(4.0/9.0), dt(1.0/9.0), dt(1.0/36.0)
    TT, SX = dt(2.0/3.0), dt(1.0/6.0)
    EPS = dt(1.0e-12)
    RHOMIN = dt(1.0e-3)
    HEALTH_LOWER, HEALTH_UPPER = dt(-1.0e30), dt(1.0e30)
    SHAPE_MAX = dt(1.45)  # guaranteed by parameter validation
    TAIL_START, TAIL_END = dt(6.0), dt(8.0)

    # --------------------------------------------------------------------
    # D2Q9 equilibrium
    #   i : 0(0,0) 1(1,0) 2(0,1) 3(-1,0) 4(0,-1) 5(1,1) 6(-1,1) 7(-1,-1) 8(1,-1)
    #   opp: 0     3      4      1       2       7      8       5        6
    # --------------------------------------------------------------------
    @wp.func
    def feq(rho: dt, ux: dt, uy: dt) -> v9:
        usq = F15 * (ux * ux + uy * uy)
        e = v9()
        e[0] = W0 * rho * (F1 - usq)
        c = ux
        e[1] = W1 * rho * (F1 + F3 * c + F45 * c * c - usq)
        c = uy
        e[2] = W1 * rho * (F1 + F3 * c + F45 * c * c - usq)
        c = -ux
        e[3] = W1 * rho * (F1 + F3 * c + F45 * c * c - usq)
        c = -uy
        e[4] = W1 * rho * (F1 + F3 * c + F45 * c * c - usq)
        c = ux + uy
        e[5] = W2 * rho * (F1 + F3 * c + F45 * c * c - usq)
        c = -ux + uy
        e[6] = W2 * rho * (F1 + F3 * c + F45 * c * c - usq)
        c = -ux - uy
        e[7] = W2 * rho * (F1 + F3 * c + F45 * c * c - usq)
        c = ux - uy
        e[8] = W2 * rho * (F1 + F3 * c + F45 * c * c - usq)
        return e

    # --------------------------------------------------------------------
    # Solid volume fraction + solid velocity (smooth => differentiable w.r.t. geometry)
    #   phi = r - R[1 + sum_k a_k cos(k*th) + b_k sin(k*th)]
    #   eps = 0.5 [1 - tanh(phi/delta)]
    #   u_s = omega x r  +  4 eps (1-eps) v_n(th) n_hat
    # --------------------------------------------------------------------
    @wp.func
    def solid_at(par: wp.array(dtype=dt), xf: dt, yf: dt, delta: dt):
        dx = xf - par[P_XC]
        dy = yf - par[P_YC]
        Rb = par[P_R]
        r2 = dx * dx + dy * dy
        rmax = Rb * SHAPE_MAX + TAIL_END * delta
        epsS = F0
        usx = F0
        usy = F0
        if r2 < rmax * rmax:
            om = par[P_OMG]
            usx = -om * dy
            usy = om * dx
            epsS = F1
            core_radius = dt(0.1) * Rb
            if r2 > core_radius * core_radius:
                rr = wp.sqrt(r2)
                th = wp.atan2(dy, dx)
                shp = F1
                dshp = F0
                for k in range(NFOUR):
                    kk = dt(k + 1)
                    co, si = wp.cos(kk * th), wp.sin(kk * th)
                    shp += par[P_A + k] * co + par[P_B + k] * si
                    dshp += kk * (-par[P_A + k] * si + par[P_B + k] * co)
                phi = rr - Rb * shp
                z = phi / delta
                fraction = F05 * (F1 - wp.tanh(z))
                taper = F1
                if z >= TAIL_END:
                    taper = F0
                elif z > TAIL_START:
                    a = (z - TAIL_START) / (TAIL_END - TAIL_START)
                    taper = F1 - a*a*a * (dt(10.0) - dt(15.0)*a + dt(6.0)*a*a)
                # Smoothly saturate the inner core, where polar coordinates are singular.
                core_blend = F1
                if rr < dt(0.25) * Rb:
                    a = (rr / Rb - dt(0.1)) / dt(0.15)
                    core_blend = a*a*a * (dt(10.0) - dt(15.0)*a + dt(6.0)*a*a)
                epsS = F1 + core_blend * (fraction * taper - F1)
                wj = F4 * epsS * (F1 - epsS)
                vn = F0
                for k in range(NJET):
                    kk = dt(k + 1)
                    vn += par[P_JC + k] * wp.cos(kk * th) + par[P_JS + k] * wp.sin(kk * th)
                # For r=R*s(theta), the outward normal is proportional to s*e_r-s'*e_theta.
                norm = wp.sqrt(shp * shp + dshp * dshp)
                usx += wj * vn * (shp * dx + dshp * dy) / (rr * norm)
                usy += wj * vn * (shp * dy - dshp * dx) / (rr * norm)
        return epsS, usx, usy

    # --------------------------------------------------------------------
    # Main kernel: fused pull-streaming + boundaries + MRT collision + LES + PSM + force
    #   g_in / g_out hold the **post-collision** populations, layout (9, nx, ny)
    # --------------------------------------------------------------------
    @wp.kernel
    def k_step(g_in:  wp.array3d(dtype=dt),
               g_out: wp.array3d(dtype=dt),
               par:   wp.array(dtype=dt),
               tau_sp: wp.array(dtype=dt),        # extra outlet-sponge viscosity (length nx)
               force: wp.array2d(dtype=dt),       # (T, 2) aerodynamic force per step
               tstep: int,
               nx: int, ny: int,
               wall: int, ghost: int,
               cs2: dt, se: dt, seps: dt,
               delta: dt, uy_kick: dt, psm_lin: int):

        x, y = wp.tid()

        # ---- outlet: zero gradient (last column copies the upstream column's inflow set) ----
        xg = x
        if x >= nx - 1:
            xg = nx - 2
        xm = xg - 1
        if xm < 0:
            xm = 0
        xp = xg + 1
        if xp > nx - 1:
            xp = nx - 1

        # ------------------------------------------------------------------
        # !! Warp autodiff pitfall !!
        # If a vector component is first assigned a differentiable expression and is then
        # overwritten inside a branch, reverse mode does not clear the adjoint path of the old
        # value and a spurious gradient appears (scalars are safe -- Warp renames them in SSA
        # form). The Zou-He inlet is exactly this "stream in, then overwrite" pattern, so the
        # populations are collected in scalars f0..f8 and written into the v9 only once.
        # ------------------------------------------------------------------
        f0 = g_in[0, xg, y]
        f1 = g_in[1, xm, y]
        f3 = g_in[3, xp, y]

        # ---- bottom wall y=0: populations 2,5,6 are missing ----
        f2 = F0
        f5 = F0
        f6 = F0
        if y == 0:
            if wall == 0:                       # free slip (specular reflection)
                f2 = g_in[4, xg, 0]
                f5 = g_in[8, xm, 0]
                f6 = g_in[7, xp, 0]
            elif wall == 1:                     # no slip (half-way bounce-back)
                f2 = g_in[4, xg, 0]
                f5 = g_in[7, xg, 0]
                f6 = g_in[8, xg, 0]
            elif wall == 2:                     # periodic
                f2 = g_in[2, xg, ny - 1]
                f5 = g_in[5, xm, ny - 1]
                f6 = g_in[6, xp, ny - 1]
            else:                               # far field (free-stream equilibrium)
                ef = feq(F1, par[P_UX], par[P_UY] + uy_kick)
                f2 = ef[2]
                f5 = ef[5]
                f6 = ef[6]
        else:
            f2 = g_in[2, xg, y - 1]
            f5 = g_in[5, xm, y - 1]
            f6 = g_in[6, xp, y - 1]

        # ---- top wall y=ny-1: populations 4,7,8 are missing ----
        f4 = F0
        f7 = F0
        f8 = F0
        if y == ny - 1:
            if wall == 0:
                f4 = g_in[2, xg, ny - 1]
                f7 = g_in[6, xp, ny - 1]
                f8 = g_in[5, xm, ny - 1]
            elif wall == 1:
                f4 = g_in[2, xg, ny - 1]
                f7 = g_in[5, xg, ny - 1]
                f8 = g_in[6, xg, ny - 1]
            elif wall == 2:
                f4 = g_in[4, xg, 0]
                f7 = g_in[7, xp, 0]
                f8 = g_in[8, xm, 0]
            else:                               # far field
                ef = feq(F1, par[P_UX], par[P_UY] + uy_kick)
                f4 = ef[4]
                f7 = ef[7]
                f8 = ef[8]
        else:
            f4 = g_in[4, xg, y + 1]
            f7 = g_in[7, xp, y + 1]
            f8 = g_in[8, xm, y + 1]

        # ---- inlet x=0: Zou-He velocity BC (differentiable w.r.t. U_in) ----
        if x == 0:
            ux0 = par[P_UX]
            uy0 = par[P_UY] + uy_kick
            rw = (f0 + f2 + f4 + F2 * (f3 + f6 + f7)) / (F1 - ux0)
            f1 = f3 + TT * rw * ux0
            f5 = f7 - F05 * (f2 - f4) + SX * rw * ux0 + F05 * rw * uy0
            f8 = f6 + F05 * (f2 - f4) + SX * rw * ux0 - F05 * rw * uy0

        f = v9()
        f[0] = f0
        f[1] = f1
        f[2] = f2
        f[3] = f3
        f[4] = f4
        f[5] = f5
        f[6] = f6
        f[7] = f7
        f[8] = f8

        # ---- macroscopic quantities ----
        rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
        if rho < RHOMIN:
            rho = RHOMIN
        jx = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
        jy = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
        ux = jx / rho
        uy = jy / rho

        # ---- moments (Lallemand-Luo) ----
        s45 = f[5] + f[6] + f[7] + f[8]
        s14 = f[1] + f[2] + f[3] + f[4]
        m1 = -F4 * f[0] - s14 + F2 * s45
        m2 = F4 * f[0] - F2 * s14 + s45
        m4 = -F2 * (f[1] - f[3]) + (f[5] - f[6] - f[7] + f[8])
        m6 = -F2 * (f[2] - f[4]) + (f[5] + f[6] - f[7] - f[8])
        m7 = (f[1] + f[3]) - (f[2] + f[4])
        m8 = f[5] - f[6] + f[7] - f[8]

        j2 = (jx * jx + jy * jy) / rho
        m1e = -F2 * rho + F3 * j2
        m2e = rho - F3 * j2
        m4e = -jx
        m6e = -jy
        m7e = (jx * jx - jy * jy) / rho
        m8e = jx * jy / rho

        # ---- Smagorinsky LES (from the non-equilibrium moments) ----
        tau0 = par[P_TAU] + tau_sp[x]
        tau = tau0
        if cs2 > F0:
            d7 = m7 - m7e
            d8 = m8 - m8e
            Qn = wp.sqrt(d7 * d7 + F4 * d8 * d8 + EPS)
            tau = F05 * (tau0 + wp.sqrt(tau0 * tau0 + F18 * cs2 * Qn / rho))

        sv = F1 / tau
        s_e = se
        s_ep = seps
        s_q = F1
        if ghost == 0:                          # magic: Lambda = 3/16
            s_q = F8 * (F2 - sv) / (F8 - sv)
        else:                                   # regularized: all ghost moments -> equilibrium
            s_e = F1
            s_ep = F1

        m1p = m1 - s_e * (m1 - m1e)
        m2p = m2 - s_ep * (m2 - m2e)
        m4p = m4 - s_q * (m4 - m4e)
        m6p = m6 - s_q * (m6 - m6e)
        m7p = m7 - sv * (m7 - m7e)
        m8p = m8 - sv * (m8 - m8e)

        # ---- inverse transform M^{-1} = M^T diag(1/||row||^2) ----
        n0 = rho * R9
        n1 = m1p * R36
        n2 = m2p * R36
        n3 = jx * R6
        n4 = m4p * R12
        n5 = jy * R6
        n6 = m6p * R12
        n7 = m7p * R4
        n8 = m8p * R4

        gc = v9()
        gc[0] = n0 - F4 * n1 + F4 * n2
        gc[1] = n0 - n1 - F2 * n2 + n3 - F2 * n4 + n7
        gc[2] = n0 - n1 - F2 * n2 + n5 - F2 * n6 - n7
        gc[3] = n0 - n1 - F2 * n2 - n3 + F2 * n4 + n7
        gc[4] = n0 - n1 - F2 * n2 - n5 + F2 * n6 - n7
        gc[5] = n0 + F2 * n1 + n2 + n3 + n4 + n5 + n6 + n8
        gc[6] = n0 + F2 * n1 + n2 - n3 - n4 + n5 + n6 - n8
        gc[7] = n0 + F2 * n1 + n2 - n3 - n4 - n5 - n6 + n8
        gc[8] = n0 + F2 * n1 + n2 + n3 + n4 - n5 - n6 - n8

        # ---- PSM immersed boundary ----
        epsS, usx, usy = solid_at(par, dt(x), dt(y), delta)
        B = F0
        om = v9()
        if epsS > F0:
            # Noble-Torczynski weight: at low viscosity (tau -> 0.5) the interface coupling
            # weakens and the wall gets slightly permeable; the linear weight B = eps is
            # viscosity independent and stiffer at high Re, at a small cost in accuracy.
            if psm_lin == 1:
                B = epsS
            else:
                B = epsS * (tau - F05) / ((F1 - epsS) + (tau - F05))
            es = feq(rho, usx, usy)
            eu = feq(rho, ux, uy)
            om[0] = es[0] - eu[0]
            om[1] = f[3] - f[1] + es[1] - eu[3]
            om[2] = f[4] - f[2] + es[2] - eu[4]
            om[3] = f[1] - f[3] + es[3] - eu[1]
            om[4] = f[2] - f[4] + es[4] - eu[2]
            om[5] = f[7] - f[5] + es[5] - eu[7]
            om[6] = f[8] - f[6] + es[6] - eu[8]
            om[7] = f[5] - f[7] + es[7] - eu[5]
            om[8] = f[6] - f[8] + es[8] - eu[6]
            # fluid momentum gain = B * sum Omega^s c  =>  force on the solid takes a minus sign
            Sx = om[1] - om[3] + om[5] - om[6] - om[7] + om[8]
            Sy = om[2] - om[4] + om[5] + om[6] - om[7] - om[8]
            wp.atomic_add(force, tstep, 0, -B * Sx)
            wp.atomic_add(force, tstep, 1, -B * Sy)

        for i in range(9):
            g_out[i, x, y] = f[i] + (F1 - B) * (gc[i] - f[i]) + B * om[i]

    # --------------------------------------------------------------------
    # Initialization / macroscopic extraction
    # --------------------------------------------------------------------
    @wp.kernel
    def k_init(g: wp.array3d(dtype=dt), par: wp.array(dtype=dt)):
        x, y = wp.tid()
        e = feq(F1, par[P_UX], par[P_UY])
        for i in range(9):
            g[i, x, y] = e[i]

    @wp.kernel
    def k_macro(g: wp.array3d(dtype=dt), out: wp.array3d(dtype=dt)):
        x, y = wp.tid()
        rho = F0
        for i in range(9):
            rho += g[i, x, y]
        if rho < RHOMIN:
            rho = RHOMIN
        jx = g[1, x, y] - g[3, x, y] + g[5, x, y] - g[6, x, y] - g[7, x, y] + g[8, x, y]
        jy = g[2, x, y] - g[4, x, y] + g[5, x, y] + g[6, x, y] - g[7, x, y] - g[8, x, y]
        out[0, x, y] = rho
        out[1, x, y] = jx / rho
        out[2, x, y] = jy / rho

    @wp.kernel
    def k_solidmap(par: wp.array(dtype=dt), out: wp.array2d(dtype=dt), delta: dt):
        x, y = wp.tid()
        e, a, b = solid_at(par, dt(x), dt(y), delta)
        out[x, y] = e

    # --------------------------------------------------------------------
    # Loss kernels (all atomic_add into loss[0] so the tape can reverse through them)
    # --------------------------------------------------------------------
    @wp.kernel
    def k_loss_drag(force: wp.array2d(dtype=dt), loss: wp.array(dtype=dt), sc: dt):
        t = wp.tid()
        wp.atomic_add(loss, 0, sc * force[t, 0])

    @wp.kernel
    def k_loss_liftvar(force: wp.array2d(dtype=dt), loss: wp.array(dtype=dt),
                       sc: dt, qi: dt, mean_cl: dt):
        t = wp.tid()
        cl = force[t, 1] * qi - mean_cl
        wp.atomic_add(loss, 0, sc * cl * cl)

    @wp.kernel
    def k_loss_combo(force: wp.array2d(dtype=dt), loss: wp.array(dtype=dt),
                     sc: dt, qi: dt, wl: dt, mean_cl: dt):
        t = wp.tid()
        cd = force[t, 0] * qi
        cl = force[t, 1] * qi - mean_cl
        wp.atomic_add(loss, 0, sc * (cd + wl * cl * cl))

    @wp.kernel
    def k_loss_series(force: wp.array2d(dtype=dt), tgt: wp.array2d(dtype=dt),
                      loss: wp.array(dtype=dt), sc: dt, qi: dt, off: int):
        t = wp.tid()
        ex = force[t, 0] * qi - tgt[t + off, 0]
        ey = force[t, 1] * qi - tgt[t + off, 1]
        wp.atomic_add(loss, 0, sc * (ex * ex + ey * ey))

    @wp.kernel
    def k_loss_field(g: wp.array3d(dtype=dt), tgt: wp.array3d(dtype=dt),
                     loss: wp.array(dtype=dt), sc: dt,
                     x0: int, y0: int, st: int):
        i, j = wp.tid()
        x = x0 + i * st
        y = y0 + j * st
        rho = F0
        for k in range(9):
            rho += g[k, x, y]
        if rho < RHOMIN:
            rho = RHOMIN
        jx = g[1, x, y] - g[3, x, y] + g[5, x, y] - g[6, x, y] - g[7, x, y] + g[8, x, y]
        jy = g[2, x, y] - g[4, x, y] + g[5, x, y] + g[6, x, y] - g[7, x, y] - g[8, x, y]
        ex = jx / rho - tgt[0, i, j]
        ey = jy / rho - tgt[1, i, j]
        wp.atomic_add(loss, 0, sc * (ex * ex + ey * ey))

    @wp.kernel
    def k_probe_field(g: wp.array3d(dtype=dt), out: wp.array3d(dtype=dt),
                      x0: int, y0: int, st: int):
        i, j = wp.tid()
        x = x0 + i * st
        y = y0 + j * st
        rho = F0
        for k in range(9):
            rho += g[k, x, y]
        if rho < RHOMIN:
            rho = RHOMIN
        out[0, i, j] = (g[1, x, y] - g[3, x, y] + g[5, x, y]
                        - g[6, x, y] - g[7, x, y] + g[8, x, y]) / rho
        out[1, i, j] = (g[2, x, y] - g[4, x, y] + g[5, x, y]
                        + g[6, x, y] - g[7, x, y] - g[8, x, y]) / rho

    @wp.kernel
    def k_health(g: wp.array3d(dtype=dt), out: wp.array(dtype=dt)):
        # Device-side reduction so a health check costs one kernel and two scalars
        # instead of nine host transfers. Comparison-based nonfinite detection keeps
        # this working on Warp versions without wp.isfinite.
        x, y = wp.tid()
        s = F0
        bad = F0
        for i in range(9):
            v = g[i, x, y]
            # Use distinct signed constants: -UPPER can miscompile for fp64 in Warp 1.12.1.
            if not (v > HEALTH_LOWER and v < HEALTH_UPPER):
                bad = F1
            s += v
        if bad > F0:
            wp.atomic_max(out, 1, F1)
        else:
            wp.atomic_min(out, 0, s)

    K = dict(step=k_step, init=k_init, macro=k_macro, solidmap=k_solidmap,
             health=k_health,
             loss_drag=k_loss_drag, loss_liftvar=k_loss_liftvar,
             loss_combo=k_loss_combo, loss_series=k_loss_series,
             loss_field=k_loss_field, probe_field=k_probe_field, dtype=dt)
    _KERNEL_CACHE[dt] = K
    return K


# ==============================================================================
# 3. Solver
# ==============================================================================
class CylinderLBM:
    """D2Q9 MRT-LBM solver for flow past a cylinder (differentiable)."""

    def __init__(self, cfg, device=None, fp64=False, wall="far",
                 delta=0.50, sponge_len=3.0, sponge_max=0.7,
                 se=1.19, seps=1.4, kick_amp=0.05, kick_periods=2.0,
                 psm="nt", checkpoint_device="auto", verbose=True):
        _positive(delta, "delta")
        _positive(sponge_len, "sponge_len", allow_zero=True)
        _positive(sponge_max, "sponge_max", allow_zero=True)
        _positive(kick_amp, "kick_amp", allow_zero=True)
        _positive(kick_periods, "kick_periods", allow_zero=True)
        if wall not in WALL_MAP or psm not in ("nt", "linear"):
            raise ValueError("unsupported wall or PSM option")
        if not (0 < se < 2 and 0 < seps < 2):
            raise ValueError("MRT relaxation rates must be in (0, 2)")
        if delta > 0.25 * cfg["R"]:
            raise ValueError("delta must not exceed one quarter of the baseline radius")
        if checkpoint_device not in ("auto", "host", "device"):
            raise ValueError("checkpoint_device must be auto, host or device")
        self.cfg = cfg
        self.dev = wp.get_device(device) if device else wp.get_device()
        self.dt = wp.float64 if fp64 else wp.float32
        self.np_dt = np.float64 if fp64 else np.float32
        self.K = build_kernels(self.dt)
        self.nx, self.ny = cfg["nx"], cfg["ny"]
        self.wall = WALL_MAP[wall]
        self.ghost = GHOST_MAGIC if cfg["ghost"] == "magic" else GHOST_REG
        self.cs2 = cfg["cs"] ** 2
        self.delta = delta
        self.se, self.seps = se, seps
        self.psm_lin = 1 if psm == "linear" else 0
        self.verbose = verbose
        # BPTT checkpoints: keeping them on the host frees device memory but adds two
        # transfers per sub-window. CPU runs still copy into independent host arrays.
        if checkpoint_device == "device" or (checkpoint_device == "auto" and self.dev.is_cpu):
            self.cp_dev = self.dev
        else:
            self.cp_dev = wp.get_device("cpu")

        # design variables
        p = np.zeros(NPAR, dtype=self.np_dt)
        p[P_UX] = cfg["u"]
        p[P_UY] = 0.0
        p[P_TAU] = cfg["tau0"]
        p[P_XC] = cfg["cx"]
        p[P_YC] = cfg["cy"]
        p[P_R] = cfg["R"]
        self.par = wp.zeros(NPAR, dtype=self.dt, device=self.dev, requires_grad=True)
        self.set_params(p)          # the baseline goes through the same validation
        self.p0 = p.copy()

        # Outlet sponge: over the last sponge_len*D the viscosity is raised smoothly so that
        # vortices are absorbed instead of reflecting back into the domain.
        sp = np.zeros(self.nx, dtype=self.np_dt)
        if sponge_len > 0:
            n_sp = min(self.nx - 1, max(2, int(sponge_len * cfg["nd"])))
            x0 = self.nx - n_sp
            s = np.linspace(0.0, 1.0, n_sp)
            sp[x0:] = sponge_max * s ** 2
        self.tau_sp = wp.array(sp, dtype=self.dt, device=self.dev)

        # shedding trigger (a deterministic function of t, not differentiated)
        self.kick_amp = kick_amp * cfg["u"]
        self.kick_T = cfg["nd"] / cfg["u"]
        self.kick_steps = int(kick_periods * self.kick_T)

        # normalization 0.5*rho*U^2*D, frozen at the initial U so that the definition of Cd
        # does not drift while U itself is being optimized
        self.qref = 0.5 * cfg["u"] ** 2 * cfg["D"]
        self.qinv = 1.0 / self.qref

        if verbose:
            print(f"[cfg] Re={cfg['re']:.0f} grid={self.nx}x{self.ny} "
                  f"({self.nx*self.ny/1e6:.2f} Mcells) nd={cfg['nd']} u={cfg['u']:.3f} "
                  f"tau0={cfg['tau0']:.5f} Cs={cfg['cs']:.2f} ghost={cfg['ghost']} "
                  f"device={self.dev} dtype={'fp64' if fp64 else 'fp32'}")
            nb = cfg["n_bl"]
            q = ("check grid and domain convergence" if nb >= 4 else
                 "marginally resolved" if nb >= 2.5 else
                 "severely under-resolved, qualitative / autodiff demo only")
            print(f"[res] N_bl = nd/sqrt(Re) = {nb:.2f} cells ({q});  "
                  f"PSM interface width ~= {2*delta:.1f} cells")
            if cfg.get("ghost_auto"):
                print("[bc ] low resolution: switched automatically from magic to "
                      "regularized ghost for stability (force it with --ghost magic)")

    # ---------------- state management ----------------
    def new_state(self, requires_grad=False):
        return wp.zeros((9, self.nx, self.ny), dtype=self.dt,
                        device=self.dev, requires_grad=requires_grad)

    def init_state(self, g):
        wp.launch(self.K["init"], dim=(self.nx, self.ny),
                  inputs=[g, self.par], device=self.dev)

    def set_params(self, p_np):
        p = np.asarray(p_np, dtype=self.np_dt)
        if p.shape != (NPAR,) or not np.all(np.isfinite(p)):
            raise ValueError(f"params must contain {NPAR} finite scalars")
        if not 0 < p[P_UX] < 1 / math.sqrt(3) or abs(p[P_UY]) >= 1 / math.sqrt(3):
            raise ValueError("inlet components must be subsonic, with U_in_x > 0")
        if p[P_TAU] <= 0.5 or p[P_R] <= 0:
            raise ValueError("tau0 must exceed 0.5 and R must be positive")
        shape_norm = float(np.sum(np.abs(p[P_A:P_JC])))
        if shape_norm > 0.450001:
            raise ValueError("the sum of absolute shape coefficients must not exceed 0.45")
        extent = float(p[P_R]) * (1.0 + shape_norm) + 1.0
        if min(p[P_XC], self.nx-1-p[P_XC], p[P_YC], self.ny-1-p[P_YC]) <= extent:
            raise ValueError("the shape must fit inside the domain with a one-cell margin")
        self.par.assign(p)

    def get_params(self):
        return self.par.numpy().astype(np.float64)

    def kick(self, t):
        if t >= self.kick_steps:
            return 0.0
        return self.kick_amp * math.sin(2.0 * math.pi * t / self.kick_T)

    # ---------------- single step ----------------
    def step(self, gin, gout, force, tslot, t_global):
        wp.launch(self.K["step"], dim=(self.nx, self.ny),
                  inputs=[gin, gout, self.par, self.tau_sp, force, tslot,
                          self.nx, self.ny, self.wall, self.ghost,
                          self.cs2, self.se, self.seps,
                          self.delta, self.kick(t_global), self.psm_lin],
                  device=self.dev)

    # ---------------- forward run (no gradient, ping-pong double buffer) ----------------
    def run(self, nsteps, g=None, t0=0, report_every=None, snap_every=0,
            snap_cb=None):
        nsteps = _integer(nsteps, "nsteps")
        _integer(t0, "t0", minimum=0)
        _integer(snap_every, "snap_every", minimum=0)
        if g is None:
            g = self.new_state()
            self.init_state(g)
        gb = self.new_state()
        force = wp.zeros((nsteps, 2), dtype=self.dt, device=self.dev)
        if report_every is None:
            report_every = max(1, nsteps // 20)
        _integer(report_every, "report_every")

        t_start = time.perf_counter()
        for t in range(nsteps):
            self.step(g, gb, force, t, t0 + t)
            g, gb = gb, g
            if (t + 1) % report_every == 0 or t + 1 == nsteps:
                fr = force[max(0, t - 200):t + 1].numpy()
                if not np.all(np.isfinite(fr)):
                    raise FloatingPointError("nonfinite force; increase resolution or viscosity")
                cd = float(np.mean(fr[:, 0])) * self.qinv
                cl = float(fr[-1, 1]) * self.qinv
                el = max(time.perf_counter() - t_start, 1e-9)
                mlups = (t + 1) * self.nx * self.ny / el / 1e6
                if self.verbose:
                    print(f"  step {t+1:>8d}/{nsteps}  t*={(t0+t+1)*self.cfg['u']/self.cfg['nd']:7.1f}"
                          f"  Cd={cd:7.4f}  Cl={cl:+7.4f}  {mlups:7.1f} MLUPS  {el:6.1f}s",
                          flush=True)
            if snap_every and snap_cb and (t + 1) % snap_every == 0:
                snap_cb(t + 1, g)
        result = force.numpy().astype(np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite force history")
        self._check_state(g)
        return g, result

    def _check_state(self, g):
        """Reduce the field on the device and transfer only (min density, bad flag)."""
        h = wp.array(np.array([1.0e30, 0.0], dtype=self.np_dt), dtype=self.dt,
                     device=self.dev)
        wp.launch(self.K["health"], dim=(self.nx, self.ny), inputs=[g, h],
                  device=self.dev)
        rho_min, bad = (float(v) for v in h.numpy())
        if bad > 0.0:
            raise FloatingPointError("nonfinite populations in the flow field")
        if rho_min <= float(self.np_dt(1.0e-3)):
            raise FloatingPointError(
                f"density reached the numerical floor (min rho = {rho_min:.3e}); "
                "the flow is invalid")

    # ---------------- macroscopic fields ----------------
    def macro(self, g):
        out = wp.zeros((3, self.nx, self.ny), dtype=self.dt, device=self.dev)
        wp.launch(self.K["macro"], dim=(self.nx, self.ny),
                  inputs=[g, out], device=self.dev)
        return out.numpy().astype(np.float64)

    def solid_map(self):
        out = wp.zeros((self.nx, self.ny), dtype=self.dt, device=self.dev)
        wp.launch(self.K["solidmap"], dim=(self.nx, self.ny),
                  inputs=[self.par, out, self.delta], device=self.dev)
        return out.numpy().astype(np.float64)

    def probe(self, g, x0, y0, stride, ni, nj):
        self._validate_probe(dict(x0=x0, y0=y0, stride=stride, ni=ni, nj=nj))
        out = wp.zeros((2, ni, nj), dtype=self.dt, device=self.dev)
        wp.launch(self.K["probe_field"], dim=(ni, nj),
                  inputs=[g, out, x0, y0, stride], device=self.dev)
        return out

    def _validate_probe(self, pg):
        for name in ("x0", "y0"):
            _integer(pg[name], name, minimum=0)
        for name in ("stride", "ni", "nj"):
            _integer(pg[name], name)
        if (pg["x0"] + (pg["ni"] - 1) * pg["stride"] >= self.nx or
                pg["y0"] + (pg["nj"] - 1) * pg["stride"] >= self.ny):
            raise ValueError("probe grid extends outside the flow domain")

    def _prepare_loss(self, spec, nsteps):
        spec = dict(spec)
        kind = spec.get("kind", "drag")
        if kind not in ("drag", "liftvar", "combo", "series", "field"):
            raise ValueError(f"unknown objective {kind}")
        if kind == "combo":
            _positive(spec.get("w", 1.0), "w", allow_zero=True)
        if kind in ("series", "field"):
            if "target" not in spec:
                raise ValueError(f"{kind} loss requires target data")
            if kind == "field":
                if "probe" not in spec:
                    raise ValueError("field loss requires a probe grid")
                pg = spec["probe"]
                self._validate_probe(pg)
                shape = (2, pg["ni"], pg["nj"])
            else:
                shape = (nsteps, 2)
            target = np.asarray(spec["target"], dtype=self.np_dt)
            if target.shape != shape or not np.all(np.isfinite(target)):
                raise ValueError(f"{kind} target must be finite with shape {shape}")
            # Derive both representations from the same rounded data. Ignore stale target_wp.
            spec["target"] = target
            spec["target_wp"] = wp.array(target, dtype=self.dt, device=self.dev)
        return spec

    # ==================================================================
    # 4. Adjoint: checkpointed BPTT
    # ==================================================================
    def loss_grad(self, g0, nsteps, loss_spec, t0=0, window=None,
                  need_grad=True):
        """
        Integrate nsteps steps from the fixed initial state g0; return the scalar objective
        J together with dJ/dparams.

        loss_spec = dict(kind=..., ...)
          kind='drag'    : J = <Cd>_t
          kind='liftvar' : J = <(Cl-<Cl>)^2>_t
          kind='combo'   : J = <Cd>_t + w*Var(Cl)       (w = loss_spec['w'])
          kind='series'  : J = <(Cd-Cd*)^2+(Cl-Cl*)^2>_t (target=(T,2) ndarray)
          kind='field'   : J = mean square error of the final velocity field (probe grid)

        Returns (J, grad[NPAR]); grad is None when need_grad=False.
        Note: J is defined on a **frozen initial state** over a finite time window, so finite
        differences and the adjoint measure the same quantity and can be compared exactly.
        """
        nsteps = _integer(nsteps, "nsteps")
        _integer(t0, "t0", minimum=0)
        window = nsteps if window is None else _integer(window, "window")
        window = min(window, nsteps)
        nchunk = int(math.ceil(nsteps / window))
        loss_spec = self._prepare_loss(loss_spec, nsteps)

        # ---- forward pass: record one checkpoint per sub-window (no tape) ----
        # Checkpoint storage is O(T/W) and replay storage is O(W); the total is
        # Include replay gradients when budgeting: 2*(W+1)+ceil(T/W) state buffers.
        cps = ([wp.empty((9, self.nx, self.ny), dtype=self.dt, device=self.cp_dev)
                for _ in range(nchunk)] if need_grad else [])
        gA, gB = self.new_state(), self.new_state()
        gA.assign(g0)
        force_fw = wp.zeros((nsteps, 2), dtype=self.dt, device=self.dev)
        for k in range(nchunk):
            if need_grad:
                wp.copy(cps[k], gA)
            lo, hi = k * window, min((k + 1) * window, nsteps)
            for t in range(lo, hi):
                self.step(gA, gB, force_fw, t, t0 + t)
                gA, gB = gB, gA
        g_end = gA
        self._check_state(g_end)

        # ---- objective value (forward) ----
        scale = 1.0 / nsteps
        fnp = force_fw.numpy().astype(np.float64)
        if not np.all(np.isfinite(fnp)):
            raise FloatingPointError("nonfinite force during objective evaluation")
        # The derivative through the global mean cancels because sum(Cl-mean(Cl))=0.
        # Reuse this same mean for every chunk; chunk-local variances are a different loss.
        loss_spec["mean_cl"] = float(np.mean(fnp[:, 1]) * self.qinv)
        Jval = self._loss_value(fnp, g_end, loss_spec, scale)
        if not math.isfinite(Jval):
            raise FloatingPointError("nonfinite objective")
        if not need_grad:
            return Jval, None

        # ---- reverse pass: replay one sub-window at a time ----
        self.par.grad.zero_()
        states = [self.new_state(requires_grad=True) for _ in range(window + 1)]
        adj_next = wp.zeros((9, self.nx, self.ny), dtype=self.dt, device=self.dev)

        for k in range(nchunk - 1, -1, -1):
            lo, hi = k * window, min((k + 1) * window, nsteps)
            n = hi - lo
            fchunk = wp.zeros((n, 2), dtype=self.dt, device=self.dev,
                              requires_grad=True)
            loss = wp.zeros(1, dtype=self.dt, device=self.dev, requires_grad=True)
            for s in states[:n + 1]:
                s.grad.zero_()
            wp.copy(states[0], cps[k])

            tape = wp.Tape()
            with tape:
                for t in range(n):
                    self.step(states[t], states[t + 1], fchunk, t, t0 + lo + t)
                self._loss_launch(fchunk, states[n], loss, loss_spec,
                                  scale, offset=lo, last=(k == nchunk - 1))

            loss.grad.fill_(self.np_dt(1.0))
            states[n].grad.assign(adj_next)
            tape.backward()
            adj_next.assign(states[0].grad)
            # NOTE: never call tape.reset()/tape.zero() -- they also zero par.grad, and
            # multi-window BPTT relies on par.grad accumulating. A fresh Tape per window suffices.

        grad = self.par.grad.numpy().astype(np.float64).copy()
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("nonfinite gradient; shorten the adjoint window")
        return Jval, grad

    # ---- objective: kernel launches (recorded on the tape) ----
    def _loss_launch(self, force, g_end, loss, spec, scale, offset, last):
        kind = spec.get("kind", "drag")
        n = force.shape[0]
        K = self.K
        if kind == "drag":
            wp.launch(K["loss_drag"], dim=n,
                      inputs=[force, loss, scale * self.qinv], device=self.dev)
        elif kind == "liftvar":
            wp.launch(K["loss_liftvar"], dim=n,
                      inputs=[force, loss, scale, self.qinv, spec["mean_cl"]], device=self.dev)
        elif kind == "combo":
            wp.launch(K["loss_combo"], dim=n,
                      inputs=[force, loss, scale, self.qinv,
                              float(spec.get("w", 1.0)), spec["mean_cl"]], device=self.dev)
        elif kind == "series":
            wp.launch(K["loss_series"], dim=n,
                      inputs=[force, spec["target_wp"], loss, scale,
                              self.qinv, offset], device=self.dev)
        elif kind == "field":
            if last:
                pg = spec["probe"]
                wp.launch(K["loss_field"], dim=(pg["ni"], pg["nj"]),
                          inputs=[g_end, spec["target_wp"], loss,
                                  1.0 / (pg["ni"] * pg["nj"]),
                                  pg["x0"], pg["y0"], pg["stride"]],
                          device=self.dev)
        else:
            raise ValueError(f"unknown objective {kind}")

    # ---- objective: numpy recomputation (returns the value, identical to the kernel) ----
    def _loss_value(self, fnp, g_end, spec, scale):
        kind = spec.get("kind", "drag")
        cd = fnp[:, 0] * self.qinv
        cl = fnp[:, 1] * self.qinv
        if kind == "drag":
            return float(np.sum(cd) * scale)
        if kind == "liftvar":
            return float(np.sum((cl - np.mean(cl)) ** 2) * scale)
        if kind == "combo":
            return float(np.sum(cd + spec.get("w", 1.0) * (cl - np.mean(cl)) ** 2) * scale)
        if kind == "series":
            tg = spec["target"]
            m = len(cd)
            return float(np.sum((cd - tg[:m, 0]) ** 2 + (cl - tg[:m, 1]) ** 2) * scale)
        if kind == "field":
            pg = spec["probe"]
            uv = self.probe(g_end, pg["x0"], pg["y0"], pg["stride"],
                            pg["ni"], pg["nj"]).numpy().astype(np.float64)
            tg = spec["target"]
            return float(np.mean((uv[0] - tg[0]) ** 2 + (uv[1] - tg[1]) ** 2))
        raise ValueError(kind)


# ==============================================================================
# 5. Post-processing (Cd/Cl statistics, Strouhal number, plots)
# ==============================================================================
def analyse_forces(fnp, cfg, qinv, frac=0.5):
    fnp = np.asarray(fnp, dtype=np.float64)
    if fnp.ndim != 2 or fnp.shape[1] != 2 or not len(fnp) or not np.all(np.isfinite(fnp)):
        raise ValueError("force history must be a nonempty finite (T, 2) array")
    if not math.isfinite(frac) or not 0 < frac <= 1:
        raise ValueError("stat_frac must be in (0, 1]")
    _positive(qinv, "qinv")
    cd = fnp[:, 0] * qinv
    cl = fnp[:, 1] * qinv
    n = len(cd)
    i0 = int(n * (1.0 - frac))
    scd, scl = cd[i0:], cl[i0:]
    res = dict(cd_mean=float(np.mean(scd)), cd_std=float(np.std(scd)),
               cl_mean=float(np.mean(scl)), cl_rms=float(np.sqrt(np.mean(scl ** 2))),
               cl_std=float(np.std(scl)), stat_start=i0,
               cl_amp=float(np.max(np.abs(scl - np.mean(scl)))) if len(scl) else 0.0,
               st=float("nan"), f_lat=float("nan"))
    m = len(scl)
    if m > 64 and np.std(scl) > 1e-12 * max(1.0, float(np.max(np.abs(scl)))):
        y = (scl - np.mean(scl)) * np.hanning(m)
        sp = np.abs(np.fft.rfft(y))
        fr = np.fft.rfftfreq(m, d=1.0)          # cycles per lattice time step
        k = int(np.argmax(sp[1:])) + 1
        if 1 <= k < len(sp) - 1:                # parabolic interpolation of the peak
            a, b, c = sp[k - 1], sp[k], sp[k + 1]
            den = (a - 2 * b + c)
            dk = 0.5 * (a - c) / den if abs(den) > 1e-30 else 0.0
        else:
            dk = 0.0
        f = (k + dk) * (fr[1] - fr[0])
        res["f_lat"] = float(f)
        res["st"] = float(f * cfg["nd"] / cfg["u"])
    return res


def vorticity(mac, cfg):
    ux, uy = mac[1], mac[2]
    duy_dx = np.gradient(uy, axis=0)
    dux_dy = np.gradient(ux, axis=1)
    return (duy_dx - dux_dy) * cfg["nd"] / cfg["u"]      # non-dimensionalized omega*D/U


def make_plots(outdir, tag, fnp, mac, solid, cfg, stats, qinv):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                # pragma: no cover
        print(f"  [warn] matplotlib not available, skipping plots ({e})")
        return []
    files = []
    nd, u = cfg["nd"], cfg["u"]
    tstar = (np.arange(len(fnp)) + 1) * u / nd
    cd, cl = fnp[:, 0] * qinv, fnp[:, 1] * qinv

    # --- vorticity field ---
    w = vorticity(mac, cfg)
    sub = max(1, cfg["nx"] // 1600)
    ws = w[::sub, ::sub].T
    ss = solid[::sub, ::sub].T
    ws = np.ma.masked_where(ss > 0.5, ws)
    fig, ax = plt.subplots(figsize=(13, max(3.0, 12.2 * cfg["ny"] / cfg["nx"] + 0.7)),
                           constrained_layout=True)
    lim = float(np.nanpercentile(np.abs(ws.compressed()), 99.0)) if ws.count() else 5.0
    lim = max(lim, 1e-6)
    ex = [0, cfg["nx"] / nd, 0, cfg["ny"] / nd]
    im = ax.imshow(ws, origin="lower", extent=ex, cmap="RdBu_r",
                   vmin=-lim, vmax=lim, interpolation="bilinear")
    ax.contour(np.linspace(ex[0], ex[1], ss.shape[1]),
               np.linspace(ex[2], ex[3], ss.shape[0]), ss, levels=[0.5],
               colors="k", linewidths=1.0)
    ax.set_xlabel("x/D"); ax.set_ylabel("y/D")
    ax.set_title(f"Re={cfg['re']:.0f}  vorticity $\\omega D/U$   "
                 f"$\\overline{{C_d}}$={stats['cd_mean']:.3f}  St={stats['st']:.3f}")
    fig.colorbar(im, ax=ax, shrink=0.85, label="$\\omega D/U$")
    p = os.path.join(outdir, f"vorticity_{tag}.png")
    fig.savefig(p, dpi=130); plt.close(fig); files.append(p)

    # --- force history + spectrum ---
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    skip = int(2.0 * nd / u)
    # Retain short histories instead of reducing them to a single invisible point.
    k0 = skip if len(tstar) > skip + 64 else 0
    axs[0].plot(tstar[k0:], cd[k0:], lw=0.8, label="$C_d$")
    axs[0].plot(tstar[k0:], cl[k0:], lw=0.8, label="$C_l$")
    axs[0].axhline(stats["cd_mean"], color="k", ls="--", lw=0.8,
                   label=f"$\\overline{{C_d}}$={stats['cd_mean']:.3f}")
    axs[0].set_xlabel("$t U/D$"); axs[0].set_ylabel("force coefficient")
    seg = slice(stats["stat_start"], None)
    lo = min(cl[seg].min(), cd[seg].min()) - 0.4
    hi = max(cl[seg].max(), cd[seg].max()) + 0.4
    axs[0].set_ylim(lo, hi)
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    i0 = stats["stat_start"]
    y = cl[i0:] - cl[i0:].mean()
    if len(y) > 64:
        sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        fr = np.fft.rfftfreq(len(y), d=1.0) * nd / u
        axs[1].semilogy(fr[1:], sp[1:] + 1e-30, lw=0.8)
        if np.isfinite(stats["st"]):
            axs[1].axvline(stats["st"], color="r", ls="--", lw=0.8,
                           label=f"St={stats['st']:.4f}")
            axs[1].legend(fontsize=8)
        axs[1].set_xlim(0, 1.2)
    else:
        axs[1].text(0.5, 0.5, "Not enough samples for a spectrum",
                    transform=axs[1].transAxes, ha="center", va="center")
    axs[1].set_xlabel("St = f D / U"); axs[1].set_ylabel("|FFT($C_l$)|")
    axs[1].grid(alpha=.3)
    fig.suptitle(f"Re = {cfg['re']:.0f}")
    fig.tight_layout()
    p = os.path.join(outdir, f"forces_{tag}.png")
    fig.savefig(p, dpi=130); plt.close(fig); files.append(p)
    return files


# ==============================================================================
# 6. Optimization utilities (Adam + variable scaling + analytic constraint terms)
# ==============================================================================
def param_scales(cfg):
    s = np.ones(NPAR)
    s[P_UX] = cfg["u"]
    s[P_UY] = cfg["u"]
    s[P_TAU] = max(cfg["tau0"] - 0.5, 1e-4)
    s[P_XC] = 0.25 * cfg["nd"]
    s[P_YC] = 0.25 * cfg["nd"]
    s[P_R] = 0.15 * cfg["R"]
    s[P_OMG] = cfg["u"] / cfg["R"]
    s[P_A:P_A + 2 * NFOUR] = 0.10
    s[P_JC:P_JC + 2 * NJET] = 0.05 * cfg["u"]
    return s


class Adam:
    def __init__(self, n, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(n); self.v = np.zeros(n); self.t = 0

    def step(self, g, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return -lr * mh / (np.sqrt(vh) + self.eps)


def shape_area_penalty(p, R0, weight):
    """Shape area constraint: A = pi R^2 (1 + 0.5 sum(a_k^2 + b_k^2));
    penalty (A/A0 - 1)^2 with an analytic gradient."""
    R = p[P_R]
    a = p[P_A:P_A + NFOUR]
    b = p[P_B:P_B + NFOUR]
    s2 = float(np.sum(a * a + b * b))
    A = (R ** 2) * (1.0 + 0.5 * s2)
    A0 = R0 ** 2
    r = A / A0 - 1.0
    val = weight * r * r
    g = np.zeros(NPAR)
    c = weight * 2.0 * r / A0
    g[P_R] = c * 2.0 * R * (1.0 + 0.5 * s2)
    g[P_A:P_A + NFOUR] = c * (R ** 2) * a
    g[P_B:P_B + NFOUR] = c * (R ** 2) * b
    return val, g


def control_energy_penalty(p, cfg, weight):
    """Amplitude regularizer: (omega R/U_ref)^2 + sum((jet coefficients/U_ref)^2)."""
    R, U = p[P_R], cfg["u"]
    om = p[P_OMG] * R / U
    jc = p[P_JC:P_JC + NJET] / U
    js = p[P_JS:P_JS + NJET] / U
    val = weight * (om * om + float(np.sum(jc * jc + js * js)))
    g = np.zeros(NPAR)
    g[P_OMG] = weight * 2.0 * om * R / U
    g[P_R] = weight * 2.0 * om * p[P_OMG] / U
    g[P_JC:P_JC + NJET] = weight * 2.0 * jc / U
    g[P_JS:P_JS + NJET] = weight * 2.0 * js / U
    return val, g


def clip_params(p, cfg, active=None):
    """Keep the geometry star-shaped and the control amplitudes physically sensible."""
    p = np.asarray(p, dtype=np.float64).copy()
    active = set(range(NPAR) if active is None else active)
    lim = 0.45
    s = np.abs(p[P_A:P_A + NFOUR]).sum() + np.abs(p[P_B:P_B + NFOUR]).sum()
    if s > lim and active.intersection(range(P_A, P_JC)):
        p[P_A:P_A + 2 * NFOUR] *= lim / s
    if P_R in active:
        p[P_R] = float(np.clip(p[P_R], 0.4 * cfg["R"], 2.0 * cfg["R"]))
    jm = 0.35 * cfg["u"]
    for i in range(P_JC, NPAR):
        if i in active:
            p[i] = float(np.clip(p[i], -jm, jm))
    if P_TAU in active:
        p[P_TAU] = max(p[P_TAU], min(cfg["tau0"], 0.5005))
    if P_UX in active:
        p[P_UX] = float(np.clip(p[P_UX], 0.2 * cfg["u"], min(3.0 * cfg["u"], 0.3)))
    if P_UY in active:
        p[P_UY] = float(np.clip(p[P_UY], -0.3, 0.3))
    shape_bound = 1.0 + np.abs(p[P_A:P_JC]).sum()
    # Reserve two cells and limit size before moving the center.
    max_radius = (min(cfg["nx"]-1, cfg["ny"]-1) / 2.0 - 2.0) / shape_bound
    if P_XC not in active:
        max_radius = min(max_radius, (min(p[P_XC], cfg["nx"]-1-p[P_XC])-2.0)/shape_bound)
    if P_YC not in active:
        max_radius = min(max_radius, (min(p[P_YC], cfg["ny"]-1-p[P_YC])-2.0)/shape_bound)
    if P_R in active:
        p[P_R] = min(p[P_R], max_radius)
    extent = p[P_R] * shape_bound + 2.0
    for i, n in ((P_XC, cfg["nx"]), (P_YC, cfg["ny"])):
        if i in active:
            p[i] = float(np.clip(p[i], extent, n - 1 - extent))
    if P_OMG in active:
        # Bound the rotational speed at the longest possible radius of the final shape.
        om_max = omega_limit(cfg, p[P_R] * shape_bound)
        p[P_OMG] = float(np.clip(p[P_OMG], -om_max, om_max))
    return p


def design_penalty(p, cfg, args, idx):
    """Apply regularizers whenever their variables are active, including --vars all."""
    val, grad = 0.0, np.zeros(NPAR)
    if set(idx).intersection(VAR_GROUPS["shape"]) and args.area_w > 0:
        v, g = shape_area_penalty(p, cfg["R"], args.area_w)
        val += v
        grad += g
    if set(idx).intersection(VAR_GROUPS["control"]) and args.ctrl_w > 0:
        v, g = control_energy_penalty(p, cfg, args.ctrl_w)
        val += v
        grad += g
    return val, grad


# ==============================================================================
# 7. Command-line sub-commands
# ==============================================================================
def _mkout(d):
    os.makedirs(d, exist_ok=True)
    return d


def _re_tag(re):
    """Preserve distinct float Reynolds numbers while keeping legacy integer names."""
    re = float(re)
    _positive(re, "re")
    label = str(int(re)) if re.is_integer() else repr(re)
    return f"Re{label}"


def _sim_from_args(args, re, verbose=True):
    ov = dict(nd=args.nd, u=args.u, cs=args.cs, L=args.L, H=args.H,
              xc=args.xc, ghost=args.ghost, tconv=args.tconv)
    cfg = make_config(re, scale=args.scale, override=ov)
    if args.steps is not None:
        cfg["steps"] = args.steps
    sim = CylinderLBM(cfg, device=args.device, fp64=args.fp64, wall=args.wall,
                      delta=args.delta, psm=args.psm, verbose=verbose,
                      checkpoint_device=args.checkpoint_device)
    return cfg, sim


def _loss_from_args(args, sim):
    spec = dict(kind=args.loss, w=args.w)
    if args.loss in ("series", "field"):
        if args.target is None:
            raise ValueError(f"--loss {args.loss} requires --target observations.npz")
        with np.load(args.target, allow_pickle=False) as data:
            spec["target"] = data["target"]
            if args.loss == "field":
                if spec["target"].ndim != 3:
                    raise ValueError("field target must have shape (2, ni, nj)")
                pg = {}
                for name in ("x0", "y0", "stride"):
                    value = data[name]
                    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
                        raise ValueError(f"target {name} must be an integer scalar")
                    pg[name] = int(value)
                pg.update(ni=spec["target"].shape[1], nj=spec["target"].shape[2])
                spec["probe"] = pg
    return sim._prepare_loss(spec, args.adj_steps)


def cmd_run(args, re=None, outdir=None):
    re = re if re is not None else args.re
    outdir = _mkout(outdir or args.out)
    cfg, sim = _sim_from_args(args, re)
    tag = _re_tag(re)

    t0 = time.perf_counter()
    g, fnp = sim.run(cfg["steps"])
    wall = max(time.perf_counter() - t0, 1e-9)

    stats = analyse_forces(fnp, cfg, sim.qinv, frac=args.stat_frac)
    mac = sim.macro(g)
    solid = sim.solid_map()

    np.savetxt(os.path.join(outdir, f"forces_{tag}.csv"),
               np.column_stack([(np.arange(len(fnp)) + 1) * cfg["u"] / cfg["nd"],
                                fnp[:, 0] * sim.qinv, fnp[:, 1] * sim.qinv]),
               delimiter=",", header="t_star,Cd,Cl", comments="")
    if args.save_field:
        np.savez_compressed(os.path.join(outdir, f"field_{tag}.npz"),
                            rho=mac[0].astype(np.float32),
                            ux=mac[1].astype(np.float32),
                            uy=mac[2].astype(np.float32),
                            solid=solid.astype(np.float32), **{k: v for k, v in cfg.items()})
    if not args.no_plot:
        make_plots(outdir, tag, fnp, mac, solid, cfg, stats, sim.qinv)

    ref = REF_2D.get(int(re))
    print(f"\n=== Re = {float(re):g} ===")
    print(f"  grid {cfg['nx']}x{cfg['ny']}  steps {cfg['steps']}  wall time {wall:.1f}s "
          f"({cfg['steps']*cfg['nx']*cfg['ny']/wall/1e6:.1f} MLUPS)")
    print(f"  Cd_mean = {stats['cd_mean']:.4f}   Cd_std = {stats['cd_std']:.4f}")
    print(f"  Cl_rms  = {stats['cl_rms']:.4f}   Cl_std = {stats['cl_std']:.4f}   "
          f"Cl_amp = {stats['cl_amp']:.4f}")
    print(f"    (Cl_rms is taken about zero, Cl_std about the sample mean; statistics "
          f"use the last {100*args.stat_frac:.0f}% of the history)")
    print(f"  St      = {stats['st']:.4f}")
    if ref:
        print(f"  2D reference: Cd {ref['cd'][0]:.2f}-{ref['cd'][1]:.2f}, "
              f"St {ref['st'][0]:.3f}-{ref['st'][1]:.3f}   [{ref['note']}]")
    stats.update(re=float(re), nx=cfg["nx"], ny=cfg["ny"], nd=cfg["nd"],
                 u=cfg["u"], tau0=cfg["tau0"], steps=cfg["steps"], wall_s=wall)
    return stats


def cmd_sweep(args):
    outdir = _mkout(args.out)
    res = []
    for re in [float(x) for x in args.re_list.split(",")]:
        try:
            res.append(cmd_run(args, re=re, outdir=outdir))
        except Exception as e:
            print(f"  [Re={re}] failed: {e}")
            res.append(dict(re=re, cd_mean=float("nan"), st=float("nan"), error=str(e)))
    with open(os.path.join(outdir, "sweep_summary.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("\n" + "=" * 74)
    print(f"{'Re':>8} {'grid':>13} {'Cd_mean':>9} {'Cd_std':>8} {'Cl_rms':>8} "
          f"{'St':>8} {'Cd_ref':>12}")
    print("-" * 74)
    for r in res:
        ref = REF_2D.get(int(r["re"]))
        rs = f"{ref['cd'][0]:.2f}-{ref['cd'][1]:.2f}" if ref else "-"
        gr = f"{r.get('nx','-')}x{r.get('ny','-')}"
        print(f"{r['re']:>8.0f} {gr:>13} {r['cd_mean']:>9.4f} "
              f"{r.get('cd_std',float('nan')):>8.4f} {r.get('cl_rms',float('nan')):>8.4f} "
              f"{r['st']:>8.4f} {rs:>12}")
    print("=" * 74)
    print(f"results written to {outdir}")
    failed = sum("error" in result for result in res)
    if failed:
        raise RuntimeError(f"{failed} of {len(res)} sweep cases failed; "
                           f"see {os.path.join(outdir, 'sweep_summary.json')}")
    return res


def _spinup(sim, cfg, n):
    if n <= 0:
        g = sim.new_state()
        sim.init_state(g)
        return g, 0
    v, sim.verbose = sim.verbose, False
    try:
        g, _ = sim.run(n)
    finally:
        sim.verbose = v
    return g, n


def checkpoint_storage(sim, nsteps, window):
    """Estimate state and force-array storage; exclude compiler/allocator overhead."""
    _integer(nsteps, "nsteps")
    window = min(_integer(window, "window"), nsteps)
    nchunk = (nsteps + window - 1) // window
    itemsize = np.dtype(sim.np_dt).itemsize
    per = 9 * sim.nx * sim.ny * itemsize
    checkpoint_bytes = nchunk * per
    replay_bytes = 2 * (window + 1) * per
    # Fixed state, two forward buffers, and the carried state adjoint.
    working_bytes = 4 * per
    force_bytes = (2 * nsteps + 8 * window + 4) * itemsize
    compute_bytes = replay_bytes + working_bytes + force_bytes
    if sim.cp_dev == sim.dev:
        compute_bytes += checkpoint_bytes
    return dict(window=window, nchunk=nchunk, state_bytes=per,
                checkpoint_bytes=checkpoint_bytes, replay_bytes=replay_bytes,
                compute_bytes=compute_bytes,
                state_units=2 * (window + 1) + nchunk)


def _best_checkpoint_window(nsteps):
    """Minimize weighted state storage, including both replay values and gradients."""
    _integer(nsteps, "nsteps")
    # Beyond sqrt(T), two added replay buffers outweigh the saved checkpoints.
    return min(range(1, min(nsteps, math.isqrt(nsteps) + 1) + 1),
               key=lambda w: 2 * (w + 1) + (nsteps + w - 1) // w)


def checkpoint_report(sim, nsteps, window):
    """Preflight BPTT before spin-up; return the chosen replay window."""
    storage = checkpoint_storage(sim, nsteps, window)
    window = storage["window"]
    best = _best_checkpoint_window(nsteps)
    if sim.dev.is_cuda:
        # Leave room for compilation, the allocator, and diagnostic temporaries.
        budget = int(0.85 * sim.dev.free_memory)
        if storage["compute_bytes"] > budget:
            candidate = best
            trial = checkpoint_storage(sim, nsteps, candidate)
            if sim.cp_dev != sim.dev:
                while candidate > 1 and trial["compute_bytes"] > budget:
                    candidate = max(1, candidate // 2)
                    trial = checkpoint_storage(sim, nsteps, candidate)
            if trial["compute_bytes"] > budget:
                raise RuntimeError(
                    f"BPTT cannot fit the CUDA budget ({budget/1024**3:.2f} GiB) "
                    "before spin-up; reduce --nd, use --checkpoint-device host, "
                    "or choose --device cpu")
            print(f"  [memory] adjusted --window {window} -> {candidate} to fit CUDA; "
                  f"--adj-steps remains {nsteps}")
            storage, window = trial, candidate
    print(f"  BPTT storage: {storage['nchunk']} checkpoints on {sim.cp_dev} "
          f"({storage['checkpoint_bytes']/1024**3:.2f} GiB) + "
          f"{window+1} replay states with gradients on {sim.dev} "
          f"({storage['replay_bytes']/1024**3:.2f} GiB)")
    print(f"  Estimated compute-device arrays including working buffers: "
          f"{storage['compute_bytes']/1024**3:.2f} GiB; runtime overhead is additional")
    optimal = checkpoint_storage(sim, nsteps, best)
    if optimal["state_units"] < 0.8 * storage["state_units"]:
        print(f"  [hint] --window {best} reduces total state storage from "
              f"{storage['state_units']} to {optimal['state_units']} state-equivalents; "
              "check host and device budgets separately")
    if sim.cp_dev != sim.dev:
        print(f"  [hint] every sub-window transfers "
              f"{2*storage['state_bytes']/1024**2:.0f} MiB across the host/device link")
    return window


def omega_limit(cfg, radius=None):
    """Cap on the rotation rate.

    radius is an upper bound on the final shape radius, not its nominal Fourier R.
    Two limits combine: a rotational speed limit (|omega| radius <= 0.2) and a
    resolution heuristic, because a fast-spinning body under-resolves its own boundary
    layer. This bounds the rotation component only, not the combined jet velocity.
    """
    R = max(float(cfg["R"] if radius is None else radius), 1e-12)
    by_res = min(4.0, max(0.4, 0.4 * cfg["n_bl"])) * cfg["u"] / R
    return min(by_res, 0.2 / R)


def _shedding_hint(cfg, adj_steps, need=2.0):
    """Estimate the shedding period (steps) and check the gradient window covers enough of them."""
    st = 0.20
    for re, r in sorted(REF_2D.items()):
        if cfg["re"] <= re:
            st = 0.5 * (r["st"][0] + r["st"][1]); break
    T = cfg["nd"] / (st * cfg["u"])
    print(f"  shedding period ~= {T:.0f} steps (St ~= {st:.2f});  gradient window "
          f"{adj_steps} steps = {adj_steps/T:.2f} periods")
    if adj_steps < need * T:
        print(f"  [hint] the window covers less than {need:g} shedding period(s), which weakens\n"
              f"         the sensitivity to viscosity and shape. Suggest --adj-steps >= "
              f"{int(need*T)} (host checkpoint memory also grows with the total).")
    return T


def cmd_gradcheck(args):
    outdir = _mkout(args.out)
    cfg, sim = _sim_from_args(args, args.re)
    spec = _loss_from_args(args, sim)
    if not args.fp64:
        print("  [hint] add --fp64: in fp32 the finite-difference noise limits the "
              "comparison to ~1e-2.")

    nsp = args.spinup if args.spinup is not None else int(20 * cfg["nd"] / cfg["u"])
    args.window = checkpoint_report(sim, args.adj_steps, args.window)
    print(f"\n  spin-up {nsp} steps -> frozen initial state g0;  gradient window "
          f"{args.adj_steps} steps (checkpoint sub-window {args.window})")
    g0, t0 = _spinup(sim, cfg, nsp)
    g0 = wp.clone(g0)

    idx = sorted(set(VAR_GROUPS[args.vars]))
    sc = param_scales(cfg)
    p0 = sim.get_params()

    t = time.perf_counter()
    J, gA = sim.loss_grad(g0, args.adj_steps, spec, t0=t0, window=args.window)
    t_adj = time.perf_counter() - t
    print(f"  J = {J:.10g}   adjoint took {t_adj:.1f}s "
          f"(one reverse pass returns all {NPAR} gradients)")

    gscale = max(float(np.max(np.abs(gA[idx] * sc[idx]))), 1e-30)
    print(f"\n{'param':>8} {'adjoint dJ/dp':>16} {'FD dJ/dp':>16} "
          f"{'rel.err':>10} {'norm.err':>10} {'h':>10}")
    print("-" * 78)
    rows = []
    for i in idx:
        h = args.fd_rel * sc[i]
        try:
            pp = p0.copy(); pp[i] += h
            sim.set_params(pp)
            Jp, _ = sim.loss_grad(g0, args.adj_steps, spec, t0=t0,
                                  window=args.window, need_grad=False)
            pm = p0.copy(); pm[i] -= h
            sim.set_params(pm)
            Jm, _ = sim.loss_grad(g0, args.adj_steps, spec, t0=t0,
                                  window=args.window, need_grad=False)
        finally:
            sim.set_params(p0)
        gfd = (Jp - Jm) / (2 * h)
        den = max(abs(gfd), abs(gA[i]), 1e-30)
        err = abs(gfd - gA[i]) / den
        # Compare dimensionless derivatives so position units cannot hide other errors.
        nerr = abs(gfd - gA[i]) * sc[i] / gscale
        print(f"{PAR_NAMES[i]:>8} {gA[i]:>16.8g} {gfd:>16.8g} "
              f"{err:>10.2e} {nerr:>10.2e} {h:>10.2e}")
        rows.append(dict(name=PAR_NAMES[i], adj=float(gA[i]), fd=float(gfd),
                         rel=float(err), norm=float(nerr), h=float(h)))
    print("-" * 78)
    worst = max((r["norm"] for r in rows), default=0.0)
    tol = 1e-6 if args.fp64 else 1e-2
    print(f"  normalized max deviation {worst:.2e}  (tolerance {tol:.0e}: "
          f"{'PASS' if worst < tol else 'FAIL: check step-size convergence and kernel derivatives'})")
    print("  Very small gradients can indicate weak sensitivity over this frozen-state window.")
    if worst >= tol:
        _fd_step_study(sim, g0, t0, args, spec, p0, sc, rows, gA)
    with open(os.path.join(outdir, f"gradcheck_{_re_tag(args.re)}.json"), "w") as fh:
        json.dump(dict(J=J, loss=args.loss, adj_steps=args.adj_steps, fd_rel=args.fd_rel,
                       fp64=args.fp64, passed=bool(worst < tol), tolerance=tol,
                       rows=rows), fh, indent=2)
    if worst >= tol:
        raise RuntimeError("gradient check failed; see the saved comparison")
    return rows


def _fd_step_study(sim, g0, t0, args, spec, p0, sc, rows, gA):
    """Re-measure the worst offender at h/2 and 2h.

    If the finite difference marches towards the adjoint as the step shrinks, the
    mismatch is finite-difference truncation and the step should be reduced. If it
    stalls at the same wrong value, the derivative itself is wrong.
    """
    worst_row = max(rows, key=lambda r: r["norm"])
    i = PAR_NAMES.index(worst_row["name"])
    print(f"\n  step-size study for {worst_row['name']} (adjoint {gA[i]:.8g}):")
    prev = None
    for factor in (2.0, 1.0, 0.5):
        h = args.fd_rel * sc[i] * factor
        try:
            pp = p0.copy(); pp[i] += h; sim.set_params(pp)
            Jp, _ = sim.loss_grad(g0, args.adj_steps, spec, t0=t0,
                                  window=args.window, need_grad=False)
            pm = p0.copy(); pm[i] -= h; sim.set_params(pm)
            Jm, _ = sim.loss_grad(g0, args.adj_steps, spec, t0=t0,
                                  window=args.window, need_grad=False)
        except (ValueError, FloatingPointError) as exc:
            print(f"    h={h:.2e}  unusable ({exc})")
            continue
        finally:
            sim.set_params(p0)
        gfd = (Jp - Jm) / (2 * h)
        trend = "" if prev is None else \
            ("  -> closer" if abs(gfd - gA[i]) < abs(prev - gA[i]) else "  -> not closer")
        print(f"    h={h:.2e}  FD={gfd:.8g}  |FD-adj|={abs(gfd-gA[i]):.3e}{trend}")
        prev = gfd
    print("    Shrinking h should reduce |FD-adj| if the gap is truncation error; a flat\n"
          "    gap points at the derivative. In fp32 the trend reverses once round-off\n"
          "    dominates -- use --fp64 before concluding anything.")


def cmd_optimize(args):
    outdir = _mkout(args.out)
    cfg, sim = _sim_from_args(args, args.re)
    spec = _loss_from_args(args, sim)
    nsp = args.spinup if args.spinup is not None else int(60 * cfg["nd"] / cfg["u"])
    args.window = checkpoint_report(sim, args.adj_steps, args.window)
    print(f"\n  spin-up {nsp} steps to develop the wake -> frozen common initial state g0")
    g0, t0 = _spinup(sim, cfg, nsp)
    g0 = wp.clone(g0)
    _shedding_hint(cfg, args.adj_steps, need=1.0)

    idx = np.array(sorted(set(VAR_GROUPS[args.vars])), dtype=int)
    if args.loss in ("drag", "combo") and (P_UX in idx or P_UY in idx):
        print("  [warn] U_in is an active design variable while the objective is a drag-type\n"
              "         coefficient normalized by the FIXED baseline U. Reducing the inflow\n"
              "         lowers the objective without improving the design. Use --vars design\n"
              "         (geometry and control only) for a design study.")
    sc = param_scales(cfg)
    p = sim.get_params()
    opt = Adam(len(idx), lr=args.lr)
    hist = []

    J0 = None
    for it in range(args.iters):
        sim.set_params(p)
        J, g = sim.loss_grad(g0, args.adj_steps, spec, t0=t0, window=args.window)
        pen, gpen = design_penalty(p, cfg, args, idx)
        Jt = J + pen
        gt = g + gpen
        if J0 is None:
            J0 = Jt
        gz = gt[idx] * sc[idx]                       # gradient in normalized variables
        hist.append(dict(it=it, J=Jt, J_phys=J, pen=pen,
                         gnorm=float(np.linalg.norm(gz)),
                         p={PAR_NAMES[i]: float(p[i]) for i in idx}))
        print(f"  it {it:>3d}  J={Jt:.6f} (physical {J:.6f} + penalty {pen:.6f})  "
              f"|g|={np.linalg.norm(gz):.3e}")
        if it % 5 == 0 or it == args.iters - 1:
            print("        " + "  ".join(f"{PAR_NAMES[i]}={p[i]:+.5g}" for i in idx))
        dz = opt.step(gz, lr=args.lr / (1.0 + args.lr_decay * it))
        p[idx] += dz * sc[idx]
        p = clip_params(p, cfg, active=idx)

    sim.set_params(p)
    J_final, _ = sim.loss_grad(g0, args.adj_steps, spec, t0=t0,
                               window=args.window, need_grad=False)
    pen_final, _ = design_penalty(p, cfg, args, idx)
    total_final = J_final + pen_final
    hist.append(dict(it=args.iters, J=total_final, J_phys=J_final, pen=pen_final,
                     p={PAR_NAMES[i]: float(p[i]) for i in idx}))
    print(f"\n  optimization finished: J {J0:.6f} -> {total_final:.6f} "
          f"(change {total_final-J0:+.6g})")
    with open(os.path.join(outdir, f"optimize_{args.vars}_{_re_tag(args.re)}.json"), "w") as fh:
        json.dump(dict(cfg={k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in cfg.items()},
                       vars=args.vars, loss=args.loss, hist=hist, J_final=total_final,
                       stat_frac=args.stat_frac,
                       p_final={PAR_NAMES[i]: float(p[i]) for i in range(NPAR)}),
                  fh, indent=2, ensure_ascii=False)

    if args.final_eval:
        print("\n  re-running full forward simulations for the optimized design ...")
        for lbl, pp in (("baseline", sim.p0.astype(np.float64)), ("optimised", p)):
            sim.set_params(pp)
            old_verbose, sim.verbose = sim.verbose, False
            try:
                gg, fn = sim.run(cfg["steps"])
            finally:
                sim.verbose = old_verbose
            st = analyse_forces(fn, cfg, sim.qinv, frac=args.stat_frac)
            print(f"    {lbl:>10}: Cd={st['cd_mean']:.4f}  Cl_rms={st['cl_rms']:.4f}  "
                  f"St={st['st']:.4f}")
        sim.set_params(p)
    return hist


def cmd_inverse(args):
    """Infer the inflow velocity U_in and the kinematic viscosity nu (i.e. Re)
    from an observed Cd/Cl history."""
    outdir = _mkout(args.out)
    cfg, sim = _sim_from_args(args, args.re)
    nsp = args.spinup if args.spinup is not None else int(40 * cfg["nd"] / cfg["u"])
    args.window = checkpoint_report(sim, args.adj_steps, args.window)
    print(f"\n  spin-up {nsp} steps -> g0 (shared by the truth and the initial guess)")
    g0, t0 = _spinup(sim, cfg, nsp)
    g0 = wp.clone(g0)
    _shedding_hint(cfg, args.adj_steps, need=2.0)
    print(f"  causal travel time for inlet variables ~= {cfg['cx']*math.sqrt(3.0):.0f} steps "
          f"(acoustic propagation over xc={cfg['cx']:.0f} cells)")

    # ---- synthesize the "measured" data ----
    p_true = sim.p0.astype(np.float64).copy()
    p_true[P_UX] *= (1.0 + args.true_du)
    p_true[P_TAU] = 0.5 + (p_true[P_TAU] - 0.5) * (1.0 + args.true_dnu)
    sim.set_params(p_true)
    old_verbose, sim.verbose = sim.verbose, False
    try:
        _, ftrue = sim.run(args.adj_steps, g=wp.clone(g0), t0=t0)
    finally:
        sim.verbose = old_verbose
    obs = np.column_stack([ftrue[:, 0] * sim.qinv, ftrue[:, 1] * sim.qinv])
    if args.noise > 0:
        rng = np.random.default_rng(0)
        obs += rng.normal(0.0, args.noise * np.std(obs, axis=0), obs.shape)
    re_true = cfg["u"] * (1 + args.true_du) * cfg["nd"] / ((p_true[P_TAU] - 0.5) / 3.0)
    print(f"  truth: U={p_true[P_UX]:.6f}  nu={(p_true[P_TAU]-0.5)/3:.6f}  "
          f"Re={re_true:.1f}   (observation noise {args.noise*100:.0f}%)")

    spec = dict(kind="series", target=obs)
    idx = np.array(sorted(set(VAR_GROUPS[args.vars])), dtype=int)
    sc = param_scales(cfg)
    p = sim.p0.astype(np.float64).copy()          # start from the "wrong" initial guess
    opt = Adam(len(idx), lr=args.lr)
    hist = []
    for it in range(args.iters):
        sim.set_params(p)
        J, g = sim.loss_grad(g0, args.adj_steps, spec, t0=t0, window=args.window)
        gz = g[idx] * sc[idx]
        nu = (p[P_TAU] - 0.5) / 3.0
        re_i = p[P_UX] * cfg["nd"] / nu
        hist.append(dict(it=it, J=J, U=float(p[P_UX]), nu=float(nu), Re=float(re_i),
                         p={PAR_NAMES[i]: float(p[i]) for i in idx}))
        print(f"  it {it:>3d}  J={J:.6e}  U={p[P_UX]:.6f} "
              f"({100*(p[P_UX]/p_true[P_UX]-1):+6.2f}%)  nu={nu:.6f} "
              f"({100*(nu/((p_true[P_TAU]-0.5)/3)-1):+6.2f}%)  Re={re_i:.1f}")
        p[idx] += opt.step(gz, lr=args.lr / (1.0 + args.lr_decay * it)) * sc[idx]
        p = clip_params(p, cfg, active=idx)
    sim.set_params(p)
    J_final, _ = sim.loss_grad(g0, args.adj_steps, spec, t0=t0,
                               window=args.window, need_grad=False)
    nu = (p[P_TAU] - 0.5) / 3.0
    hist.append(dict(it=args.iters, J=J_final, U=float(p[P_UX]), nu=float(nu),
                     Re=float(p[P_UX] * cfg["nd"] / nu),
                     p={PAR_NAMES[i]: float(p[i]) for i in idx}))
    print(f"\n  inferred Re={hist[-1]['Re']:.1f}  true Re={re_true:.1f}  "
          f"error {100*(hist[-1]['Re']/re_true-1):+.2f}%")
    with open(os.path.join(outdir, f"inverse_{_re_tag(args.re)}.json"), "w") as fh:
        json.dump(dict(re_true=re_true, hist=hist, J_final=J_final,
                       p_final={PAR_NAMES[i]: float(p[i]) for i in range(NPAR)}), fh, indent=2)
    return hist


# ==============================================================================
# 8. main
# ==============================================================================
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Differentiable flow past a cylinder (Warp / D2Q9 MRT-LBM + PSM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("6. Usage")[-1])
    ap.add_argument("cmd", choices=["run", "sweep", "gradcheck", "optimize",
                                    "inverse", "table"])
    # general
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--re-list", default="100,200,300,400,500,1000,2000,3000,10000")
    ap.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    ap.add_argument("--cache-dir", default=None, help="Warp compilation cache directory")
    ap.add_argument("--traceback", action="store_true",
                    help="re-raise errors with the full traceback instead of a short message")
    ap.add_argument("--fp64", action="store_true", help="double precision (recommended for gradcheck)")
    ap.add_argument("--scale", type=float, default=1.0, help="resolution scaling, for quick tests")
    ap.add_argument("--nd", type=int, default=None, help="override lattice nodes per diameter")
    ap.add_argument("--u", type=float, default=None, help="override the lattice inflow velocity")
    ap.add_argument("--cs", type=float, default=None, help="override the Smagorinsky Cs")
    ap.add_argument("--L", type=float, default=None, help="override streamwise domain length / D")
    ap.add_argument("--H", type=float, default=None, help="override domain height / D (blockage = 1/H)")
    ap.add_argument("--xc", type=float, default=None, help="override inlet-to-centre distance / D")
    ap.add_argument("--ghost", default=None, choices=["magic", "reg"])
    ap.add_argument("--tconv", type=float, default=None, help="override total time / (D/U)")
    ap.add_argument("--steps", type=int, default=None, help="override the total number of steps")
    ap.add_argument("--wall", default="far", choices=list(WALL_MAP),
                    help="top/bottom BC: far (default) / slip / noslip / periodic")
    ap.add_argument("--delta", type=float, default=0.50,
                help="PSM interface width in cells: smaller = more accurate, "
                     "larger = smoother geometry gradients")
    ap.add_argument("--psm", default="nt", choices=["nt", "linear"],
                    help="PSM weight: nt (Noble-Torczynski) / linear (B=eps, stiffer at high Re)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--save-field", action="store_true")
    ap.add_argument("--stat-frac", type=float, default=0.5, help="trailing fraction used for statistics")
    # adjoint / optimization
    ap.add_argument("--adj-steps", type=int, default=400, help="total steps in the gradient window")
    ap.add_argument("--window", type=int, default=100,
                    help="checkpoint sub-window length; adjusted before spin-up if CUDA memory requires it")
    ap.add_argument("--checkpoint-device", default="auto",
                    choices=["auto", "host", "device"],
                    help="where BPTT checkpoints live: auto (host on GPU runs), host, device")
    ap.add_argument("--spinup", type=int, default=None, help="spin-up steps before freezing the initial state")
    ap.add_argument("--loss", default="drag",
                    choices=["drag", "liftvar", "combo", "series", "field"])
    ap.add_argument("--target", default=None, help="NPZ observation data for series/field losses")
    ap.add_argument("--w", type=float, default=2.0, help="weight of the lift term in combo")
    ap.add_argument("--vars", default="shape", choices=sorted(VAR_GROUPS))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--lr-decay", type=float, default=0.06,
                    help="learning-rate decay: lr_t = lr/(1+decay*t)")
    ap.add_argument("--area-w", type=float, default=20.0, help="weight of the shape area constraint")
    ap.add_argument("--ctrl-w", type=float, default=0.02, help="weight of the control-energy regularizer")
    ap.add_argument("--final-eval", action="store_true", help="re-run a full forward simulation after optimizing")
    ap.add_argument("--fd-rel", type=float, default=1e-4,
                    help="relative finite-difference step; on failure the check is repeated "
                         "at h/2 and 2h to separate truncation error from a real mismatch")
    # inference
    ap.add_argument("--true-du", type=float, default=0.15, help="relative offset of the true U")
    ap.add_argument("--true-dnu", type=float, default=-0.20, help="relative offset of the true nu")
    ap.add_argument("--noise", type=float, default=0.0, help="observation noise (relative std-dev)")
    return ap


def main(argv=None):
    ap = build_argparser()
    args = ap.parse_args(argv)
    # Remap before validation so the argparse default reaches inverse as a legal group.
    if args.cmd == "inverse" and args.vars == "shape":
        args.vars = "params"
    try:
        validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))
    if args.cmd == "table":
        print_table(args.scale); return
    if args.cache_dir:
        wp.config.kernel_cache_dir = os.path.abspath(args.cache_dir)
    wp.init()
    handler = dict(run=cmd_run, sweep=cmd_sweep, gradcheck=cmd_gradcheck,
                   optimize=cmd_optimize, inverse=cmd_inverse)[args.cmd]
    # One error surface: configuration, validation and divergence all report the same
    # way. --traceback restores the full stack for debugging.
    try:
        handler(args)
    except (ValueError, FloatingPointError, RuntimeError, FileNotFoundError,
            KeyError) as exc:
        if args.traceback:
            raise
        print(f"{ap.prog}: error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def validate_args(args):
    for name in ("re", "scale", "delta", "adj_steps", "window", "iters", "lr", "fd_rel"):
        _positive(getattr(args, name), name)
    for name in ("cs", "w", "area_w", "ctrl_w", "noise", "lr_decay", "spinup"):
        value = getattr(args, name)
        if value is not None:
            _positive(value, name, allow_zero=True)
    for name in ("nd", "u", "L", "H", "xc", "tconv", "steps"):
        value = getattr(args, name)
        if value is not None:
            _positive(value, name)
    if not math.isfinite(args.stat_frac) or not 0 < args.stat_frac <= 1:
        raise ValueError("stat-frac must be in (0, 1]")
    if args.cmd in ("gradcheck", "optimize") and args.loss in ("series", "field") and not args.target:
        raise ValueError(f"--loss {args.loss} requires --target observations.npz")
    if args.cmd == "inverse":
        _positive(1 + args.true_du, "1 + true-du")
        _positive(1 + args.true_dnu, "1 + true-dnu")
        if args.vars not in ("params", "inlet", "visc"):
            raise ValueError("inverse supports --vars params, inlet or visc "
                             "(the default 'shape' is remapped to 'params')")
    if args.cmd == "sweep":
        for value in args.re_list.split(","):
            _positive(float(value), "re-list entry")


if __name__ == "__main__":
    main()
