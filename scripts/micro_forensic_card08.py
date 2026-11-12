#!/usr/bin/env python3
"""Micro-scale forensic test of Candidate #08 (16.719521, -91.168757):
Tests pure raw .laz point cloud (ZERO kriging/TIN interpolation!) vs 0.33m G1/DEM
on F06, F10, F07, F12, the NE linear feature, and the southern F16 ridge.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import Delaunay

OUT_DIR = Path("/agent/lidar/work/lacandon_deep_dive")
d = np.load(OUT_DIR / "card08_bundle_600m.npz")
cx, cy, res_m = float(d["cx"]), float(d["cy"]), float(d["res_m"])
dem, g1, lrm = d["dem33"], d["g1_33"], d["lrm33"]
ot_x, ot_y, ot_z, ot_cls = d["ot_x"], d["ot_y"], d["ot_z"], d["ot_cls"]
na_x, na_y, na_z, na_cls = d["na_x"], d["na_y"], d["na_z"], d["na_cls"]


def crop_grid(arr: np.ndarray, hm: float) -> np.ndarray:
    r0 = int(round((600.0 - hm) / res_m))
    r1 = int(round((600.0 + hm) / res_m))
    return arr[r0:r1, r0:r1]


def slice_profile(p0: tuple[float, float], p1: tuple[float, float], half_w: float = 9.0) -> dict:
    """p0, p1 in local meters relative to (cx, cy)."""
    x0, y0 = cx + p0[0], cy + p0[1]
    x1, y1 = cx + p1[0], cy + p1[1]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L

    s_ot = (ot_x - x0) * ux + (ot_y - y0) * uy
    d_ot = np.abs(-(ot_x - x0) * uy + (ot_y - y0) * ux)
    m_ot = (s_ot >= 0) & (s_ot <= L) & (d_ot <= half_w)

    s_na = (na_x - x0) * ux + (na_y - y0) * uy
    d_na = np.abs(-(na_x - x0) * uy + (na_y - y0) * ux)
    m_na = (s_na >= 0) & (s_na <= L) & (d_na <= half_w)

    s_grid = np.linspace(0, L, 400)
    gx = x0 + s_grid * ux
    gy = y0 + s_grid * uy
    H, W = dem.shape
    cols = np.clip(((gx - (cx - 600.0)) / 1200.0 * W).astype(int), 0, W - 1)
    rows = np.clip((((cy + 600.0) - gy) / 1200.0 * H).astype(int), 0, H - 1)
    dem_prof = dem[rows, cols]

    valid_p = np.isfinite(dem_prof)
    na_s_sel = s_na[m_na]
    na_z_sel = na_z[m_na]
    na_c_sel = na_cls[m_na]
    g_at_na = np.interp(na_s_sel, s_grid[valid_p], dem_prof[valid_p])
    agl_na = na_z_sel - g_at_na
    veg = (na_c_sel == 1) & (agl_na >= 1.2)

    return {
        "L": L,
        "swath_m": half_w * 2,
        "s_grid": s_grid,
        "dem_prof": dem_prof,
        "ot_s": s_ot[m_ot],
        "ot_z": ot_z[m_ot],
        "ot_d": d_ot[m_ot],
        "ot_cls": ot_cls[m_ot],
        "veg_s": na_s_sel[veg],
        "veg_z": na_z_sel[veg],
        "veg_agl": agl_na[veg],
    }


def main() -> None:
    hm = 140.0  # 280x280m exact match to eye-scan card!
    g1_c = crop_grid(g1, hm)
    dem_c = crop_grid(dem, hm)
    lrm_c = crop_grid(lrm, hm)

    m = (np.abs(ot_x - cx) <= hm) & (np.abs(ot_y - cy) <= hm)
    px = ot_x[m] - cx
    py = ot_y[m] - cy
    pz = ot_z[m]
    pc = ot_cls[m]

    # Build a ZERO-INTERPOLATION 2.0m gridded raw .laz ground elevation map
    # (cells with 0 raw ground points are left NaN / black so we see ONLY real measured ground!)
    cell_m = 2.0
    nc = int(round(2 * hm / cell_m))
    ix = np.clip(((px + hm) / cell_m).astype(int), 0, nc - 1)
    iy = np.clip(((hm - py) / cell_m).astype(int), 0, nc - 1)
    sum_z = np.zeros((nc, nc), dtype=np.float64)
    cnt_z = np.zeros((nc, nc), dtype=np.int32)
    np.add.at(sum_z, (iy, ix), pz)
    np.add.at(cnt_z, (iy, ix), 1)
    raw_2m_z = np.where(cnt_z > 0, sum_z / np.maximum(cnt_z, 1), np.nan)

    # Define 3 forensic transects (18m wide = +/-9m swath):
    # A-A': WSW -> ENE right across F10 mound -> central depression/courtyard -> F06 main rectangular platform -> E shelf
    pA0, pA1 = (-95.0, -35.0), (105.0, 55.0)
    # B-B': NNW -> SSE across northern mounds -> F06 main platform -> F12 -> southern steep ridge F16!
    pB0, pB1 = (-10.0, 130.0), (25.0, -135.0)
    # C-C': NW -> SE specifically across the NE linear feature in upper-right quadrant (dx=+30..+120, dy=+40..+120)
    pC0, pC1 = (25.0, 115.0), (115.0, 25.0)

    sA = slice_profile(pA0, pA1, half_w=9.0)
    sB = slice_profile(pB0, pB1, half_w=9.0)
    sC = slice_profile(pC0, pC1, half_w=9.0)

    for tag, sl in [("A-A' (WSW->ENE across F10 + Plaza + F06)", sA), ("B-B' (NNW->SSE across F06 -> F16 ridge)", sB), ("C-C' (Across NE linear feature)", sC)]:
        print(f"{tag}: length={sl['L']:.1f}m (18m swath) | .laz ground pts={len(sl['ot_s'])} | .las veg pts={len(sl['veg_s'])}")

    # Also check what is in upper-right quadrant (dx in [20, 135], dy in [30, 135])
    m_ne = (px >= 20) & (px <= 135) & (py >= 30) & (py <= 135)
    print(f"Upper-right quadrant [20..135, 30..135] (115x105m = 12,075 m2): {m_ne.sum()} ground pts ({m_ne.sum()/12075:.3f} pts/m2)")

    fig = plt.figure(figsize=(22, 16), dpi=150, facecolor="#0d1117")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.9, 0.9], hspace=0.28, wspace=0.18)
    ext = [-hm, hm, -hm, hm]

    # Top-left: 0.33m G1 with A-A', B-B', C-C' transects
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(g1_c, extent=ext, origin="upper")
    for (q0, q1, col, lbl) in [(pA0, pA1, "#00e5ff", "A"), (pB0, pB1, "#ffd166", "B"), (pC0, pC1, "#ff6b6b", "C")]:
        ax0.plot([q0[0], q1[0]], [q0[1], q1[1]], ls="--", lw=2.0, color=col)
        ax0.text(q0[0], q0[1], lbl, color=col, fontsize=11, fontweight="bold", bbox=dict(boxstyle="circle,pad=0.15", facecolor="black", edgecolor=col))
        ax0.text(q1[0], q1[1], lbl + "'", color=col, fontsize=11, fontweight="bold", bbox=dict(boxstyle="circle,pad=0.15", facecolor="black", edgecolor=col))
    ax0.set_title("1. 0.33m G1 (280x280m) + 3 Transects (18m Swath)", color="white", fontweight="bold")

    # Top-middle: ZERO-INTERPOLATION Raw .LAZ Points Colored by Local Elevation (435-456m clip!)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#05070a")
    sc = ax1.scatter(px, py, c=pz, cmap="turbo", vmin=435.0, vmax=454.0, s=8.0, linewidths=0)
    plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="Raw .LAZ Point Z (m ASL, 435-454m)")
    ax1.set_xlim(-hm, hm)
    ax1.set_ylim(-hm, hm)
    ax1.set_aspect("equal")
    ax1.set_title(f"2. PURE RAW .LAZ POINTS ONLY (ZERO INTERPOLATION, n={len(px):,})", color="#4ade80", fontweight="bold")

    # Top-right: 0.33m G1 + Delaunay triangles colored by edge length (>8m highlighted!)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(g1_c, extent=ext, origin="upper", alpha=0.85)
    tri = Delaunay(np.column_stack([px, py]))
    edges = set()
    for simp in tri.simplices:
        for i1, i2 in [(0, 1), (1, 2), (2, 0)]:
            a, b_i = min(simp[i1], simp[i2]), max(simp[i1], simp[i2])
            edges.add((a, b_i))
    sx_mid, sy_mid, sx_hi, sy_hi = [], [], [], []
    for a, b_i in edges:
        el = math.hypot(px[a] - px[b_i], py[a] - py[b_i])
        if 8.0 <= el < 14.0:
            sx_mid.extend([px[a], px[b_i], np.nan])
            sy_mid.extend([py[a], py[b_i], np.nan])
        elif el >= 14.0:
            sx_hi.extend([px[a], px[b_i], np.nan])
            sy_hi.extend([py[a], py[b_i], np.nan])
    ax2.plot(sx_mid, sy_mid, color="#facc15", lw=0.45, alpha=0.5, label="TIN edge 8-14m")
    ax2.plot(sx_hi, sy_hi, color="#ff2d55", lw=0.85, alpha=0.8, label="TIN edge >=14m (void)")
    ax2.scatter(px, py, s=2.5, c="#00ffcc", alpha=0.75, linewidths=0, label=".laz ground pt")
    ax2.set_xlim(-hm, hm)
    ax2.set_ylim(-hm, hm)
    ax2.legend(loc="lower right", fontsize=8, facecolor="#161b22", labelcolor="white")
    ax2.set_title("3. TIN Facet Audit: Yellow (8-14m) & Red (>=14m) Void Edges", color="white", fontweight="bold")

    # Row 2 & 3: Plot the 3 bare-earth + canopy profiles AND zoomed bare-earth-only profiles!
    for idx, (sl, ttl, col) in enumerate([
        (sA, "Transect A → A' (WSW to ENE: F10 Mound → Courtyard → F06 Main Platform)", "#00e5ff"),
        (sB, "Transect B → B' (NNW to SSE: N Bench → F06 Main Platform → Southern F16 Ridge)", "#ffd166"),
        (sC, "Transect C → C' (NW to SE: Across Upper-Right Linear Ridge / Feature)", "#ff6b6b"),
    ]):
        # Full canopy + ground panel
        ax_full = fig.add_subplot(gs[1, idx])
        ax_full.set_facecolor("#0f141b")
        vp = np.isfinite(sl["dem_prof"])
        ax_full.plot(sl["s_grid"][vp], sl["dem_prof"][vp], color=col, lw=1.8, label="0.33m DEM")
        ax_full.scatter(sl["veg_s"], sl["veg_z"], c=sl["veg_agl"], cmap="Greens", vmin=0, vmax=32, s=2.5, alpha=0.55, label=f"Canopy ({len(sl['veg_s']):,})")
        ax_full.scatter(sl["ot_s"], sl["ot_z"], c="#ff595e", s=8, alpha=0.9, zorder=5, label=f"Ground .laz ({len(sl['ot_s']):,})")
        ax_full.set_title(ttl + " [Full Canopy]", color="white", fontsize=9.5, fontweight="bold")
        ax_full.legend(loc="upper right", fontsize=7.5, facecolor="#161b22", labelcolor="white")
        ax_full.grid(True, alpha=0.25)

        # Bare-earth zoomed vertical scale panel (ONLY ground points + DEM curve!)
        ax_gnd = fig.add_subplot(gs[2, idx])
        ax_gnd.set_facecolor("#0f141b")
        ax_gnd.plot(sl["s_grid"][vp], sl["dem_prof"][vp], color=col, lw=2.0, label="0.33m DEM curve")
        c2 = sl["ot_cls"] == 2
        c8 = sl["ot_cls"] == 8
        ax_gnd.scatter(sl["ot_s"][c2], sl["ot_z"][c2], c="#ff9f1c", s=14, edgecolors="black", linewidths=0.3, zorder=5, label=f"Class 2 ({c2.sum()})")
        ax_gnd.scatter(sl["ot_s"][c8], sl["ot_z"][c8], c="#2ec4b6", s=18, marker="D", edgecolors="black", linewidths=0.3, zorder=6, label=f"Class 8 ({c8.sum()})")
        if len(sl["ot_z"]) > 0:
            zmin = min(float(np.nanmin(sl["dem_prof"][vp])), float(sl["ot_z"].min())) - 1.5
            zmax = max(float(np.nanmax(sl["dem_prof"][vp])), float(sl["ot_z"].max())) + 2.0
            ax_gnd.set_ylim(zmin, zmax)
        ax_gnd.set_title(ttl + " [Bare-Earth Zoom]", color=col, fontsize=9.5, fontweight="bold")
        ax_gnd.legend(loc="upper right", fontsize=7.5, facecolor="#161b22", labelcolor="white")
        ax_gnd.grid(True, alpha=0.3)

    for ax in fig.axes:
        ax.tick_params(colors="#cbd5e1", labelsize=8)

    out_p = OUT_DIR / "diag_card08_micro_profiles.png"
    fig.savefig(out_p, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_p)


if __name__ == "__main__":
    main()
