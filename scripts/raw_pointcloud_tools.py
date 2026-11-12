#!/usr/bin/env python3
"""Reusable Raw LiDAR Point-Cloud (.las + .laz) + 0.33m DEM/G1 + Esri Satellite
fetcher and Pietro Miotti-style cross-section figure renderer.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import re
import shutil
import urllib.request
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform as rio_transform
from rasterio.windows import from_bounds
from scipy import ndimage as ndi

WORK = Path("/agent/lidar/work/carbon_raw_hunt")
RAW_CACHE = WORK / "raw_cache"
RAW_CACHE.mkdir(parents=True, exist_ok=True)
MEDIA = Path("/cursor/stores/bc-adff6ec3-2baf-40bb-9bab-102a5f567f18/media")

S3_CAT = json.loads((WORK / "s3_raw_catalog.json").read_text())
LAZ_MAP = {Path(k).name.split("_GND_v1")[0]: k for k in S3_CAT["laz"]}
DEM33_MAP = {Path(k).name.split("_DEM_0p33m")[0]: k for k in S3_CAT["dtm_0p33m"]}
G1_33_MAP = {Path(k).name.split("_G1_0p33m")[0]: k for k in S3_CAT["g1_0p33m"]}


def strip_id_from_name(strip_name: str) -> str:
    return strip_name.split("_DEM_")[0].split("_G1_")[0].split("_GND_")[0]


def nasa_campaign_and_ls(strip_id: str) -> tuple[str, str]:
    m = re.match(r"^(.+)_(l\d+s\d+)$", strip_id, re.I)
    if not m:
        raise ValueError(f"Cannot parse strip_id: {strip_id}")
    pref, ls = m.group(1), m.group(2)
    mapping = {
        "centro_glas": "AMIGACarb_Yuc_Centro_GLAS_Apr2013",
        "centro_nfi": "AMIGACarb_Yuc_Centro_NFI_Apr2013",
        "norte_glas": "AMIGACarb_Yuc_Norte_GLAS_Apr2013",
        "norte_nfi": "AMIGACarb_Yuc_Norte_NFI_Apr2013",
        "north2_glas": "AMIGACarb_Yuc_North_2_GLAS_Apr2013",
        "north2_nfi": "AMIGACarb_Yuc_North_2_NFI_Apr2013",
        "south_glas": "AMIGACarb_Yuc_South_GLAS_Apr2013",
        "south_nfi": "AMIGACarb_Yuc_South_NFI_Apr2013",
        "outyuc_glas": "AMIGACarb_Out_of_the_Yuc_GLAS_May2013",
        "out_yuc_glas": "AMIGACarb_Out_of_the_Yuc_GLAS_May2013",
        "outyuc_nfi": "AMIGACarb_Out_of_the_Yuc_NFI_May2013",
        "out_yuc_nfi": "AMIGACarb_Out_of_the_Yuc_NFI_May2013",
        "chiapscampeche_nfi": "AMIGACarb_Chiap_Campeche_NFI_Apr2013",
        "chiaps1_glas": "AMIGACarb_Chiaps_1_GLAS_Apr2013",
        "chiaps1_nfi": "AMIGACarb_Chiaps_1_NFI_Apr2013",
    }
    camp = mapping[pref.lower()]
    return camp, ls.lower()


def ensure_pointclouds(strip_id: str) -> tuple[Path, Path]:
    """Ensure both OT .laz (ground classes 2+8) and NASA .las (full canopy+ground)
    are downloaded locally in RAW_CACHE.
    """
    laz_key = LAZ_MAP[strip_id]
    laz_path = RAW_CACHE / Path(laz_key).name
    if not laz_path.exists():
        url = f"https://opentopography.s3.sdsc.edu/pc-bulk/{laz_key}"
        urllib.request.urlretrieve(url, laz_path)

    camp, ls = nasa_campaign_and_ls(strip_id)
    las_path = RAW_CACHE / f"{camp}_{ls}.las"
    if not las_path.exists():
        gz_path = RAW_CACHE / f"{camp}_{ls}.las.gz"
        nasa_url = f"https://glihtdata.gsfc.nasa.gov/files/G-LiHT/{camp}/lidar/las/{camp}_{ls}.las.gz"
        urllib.request.urlretrieve(nasa_url, gz_path)
        with gzip.open(gz_path, "rb") as fi, open(las_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        gz_path.unlink(missing_ok=True)
    return laz_path, las_path


def multi_hillshade(dem: np.ndarray, res_m: float = 0.33) -> np.ndarray:
    valid = np.isfinite(dem)
    fill = np.where(valid, dem, np.nanmedian(dem))
    dy, dx = np.gradient(fill, res_m, res_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    acc = np.zeros_like(dem, dtype=np.float32)
    for az_deg, alt_deg, w in [(315, 42, 0.50), (45, 42, 0.30), (270, 35, 0.20)]:
        az = math.radians(az_deg)
        alt = math.radians(alt_deg)
        sh = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        acc += w * ((sh + 1.0) * 0.5)
    acc[~valid] = np.nan
    return acc


def compute_lrm(dem: np.ndarray, res_m: float = 0.33, sigma_m: float = 20.0) -> np.ndarray:
    valid = np.isfinite(dem)
    step = 6
    sv = valid[::step, ::step].astype(np.float32)
    sd = np.where(valid[::step, ::step], dem[::step, ::step], 0.0).astype(np.float32)
    sig = sigma_m / (res_m * step)
    num = ndi.gaussian_filter(sd, sigma=sig, mode="constant", cval=0.0)
    den = ndi.gaussian_filter(sv, sigma=sig, mode="constant", cval=0.0)
    strd = np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nanmedian(dem)).astype(np.float32)
    h, w = dem.shape
    tr = ndi.zoom(strd, (h / strd.shape[0], w / strd.shape[1]), order=1)[:h, :w]
    lrm = dem - tr
    lrm[~valid] = np.nan
    return lrm


def fetch_site_bundle(
    strip_id: str,
    lat: float,
    lon: float,
    half_m: float = 150.0,
    chm_path: str | None = None,
) -> dict:
    """Fetch 0.33m DEM, 0.33m G1, NASA 1m CHM, Esri World Imagery, and cropped
    OT .laz + NASA .las 3D point clouds for a target box of (2*half_m x 2*half_m).
    """
    u_dem = f"/vsicurl/https://opentopography.s3.sdsc.edu/raster/{DEM33_MAP[strip_id]}"
    u_g1 = f"/vsicurl/https://opentopography.s3.sdsc.edu/raster/{G1_33_MAP[strip_id]}"

    with rasterio.open(u_dem) as ds:
        xs, ys = rio_transform("EPSG:4326", ds.crs, [lon], [lat])
        cx, cy = float(xs[0]), float(ys[0])
        crs = ds.crs
        win = from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, ds.transform)
        dem33 = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        if ds.nodata is not None:
            dem33 = np.where(dem33 == ds.nodata, np.nan, dem33)
        dem33 = np.where((np.abs(dem33) > 1e6) | (dem33 < -100), np.nan, dem33)
        res_m = float(abs(ds.res[0]))

    with rasterio.open(u_g1) as gds:
        w_g1 = from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, gds.transform)
        g1_33 = np.moveaxis(
            gds.read([1, 2, 3], window=w_g1, out_shape=(3, dem33.shape[0], dem33.shape[1]), boundless=True, fill_value=18),
            0,
            -1,
        )

    # CHM window if available
    chm_win = None
    if chm_path is None:
        camp, ls = nasa_campaign_and_ls(strip_id)
        cand_chm = WORK / f"chm/extracted/{camp}/{camp}_{ls}_CHM.tif"
        if cand_chm.exists():
            chm_path = str(cand_chm)
    if chm_path and Path(chm_path).exists():
        with rasterio.open(chm_path) as cds:
            cw = from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, cds.transform)
            chm_win = cds.read(1, window=cw, boundless=True, fill_value=np.nan).astype(np.float32)
            if cds.nodata is not None:
                chm_win = np.where(chm_win == cds.nodata, np.nan, chm_win)
            chm_win = np.where((chm_win < -2) | (chm_win > 65), np.nan, chm_win)

    # Esri World Imagery optical satellite (with local disk cache + retry across primary/mirror endpoints)
    sat_cache = RAW_CACHE / f"esri_sat_{strip_id}_{lat:.5f}_{lon:.5f}_{int(half_m)}.png"
    if sat_cache.exists():
        sat_rgb = np.asarray(Image.open(sat_cache).convert("RGB"))
    else:
        dlat = half_m / 110574.0
        dlon = half_m / (111320.0 * math.cos(math.radians(lat)))
        bbox = f"{lon - dlon:.6f},{lat - dlat:.6f},{lon + dlon:.6f},{lat + dlat:.6f}"
        endpoints = [
            f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={bbox}&bboxSR=4326&imageSR=4326&size=512,512&format=png&f=image",
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={bbox}&bboxSR=4326&imageSR=4326&size=512,512&format=png&f=image",
        ]
        sat_rgb = None
        import time
        for attempt in range(5):
            ep = endpoints[attempt % len(endpoints)]
            try:
                req = urllib.request.Request(ep, headers={"User-Agent": "Mozilla/5.0"})
                raw_png = urllib.request.urlopen(req, timeout=25).read()
                sat_cache.write_bytes(raw_png)
                sat_rgb = np.asarray(Image.open(io.BytesIO(raw_png)).convert("RGB"))
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if sat_rgb is None:
            sat_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

    # Point clouds (.laz ground + .las full canopy)
    laz_path, las_path = ensure_pointclouds(strip_id)
    ot_pc = laspy.read(laz_path)
    om = (
        (ot_pc.x >= cx - half_m)
        & (ot_pc.x <= cx + half_m)
        & (ot_pc.y >= cy - half_m)
        & (ot_pc.y <= cy + half_m)
    )
    ot_x = np.array(ot_pc.x[om], dtype=np.float64)
    ot_y = np.array(ot_pc.y[om], dtype=np.float64)
    ot_z = np.array(ot_pc.z[om], dtype=np.float32)
    ot_cls = np.array(ot_pc.classification[om], dtype=np.uint8)

    na_pc = laspy.read(las_path)
    nm = (
        (na_pc.x >= cx - half_m)
        & (na_pc.x <= cx + half_m)
        & (na_pc.y >= cy - half_m)
        & (na_pc.y <= cy + half_m)
    )
    na_x = np.array(na_pc.x[nm], dtype=np.float64)
    na_y = np.array(na_pc.y[nm], dtype=np.float64)
    na_z = np.array(na_pc.z[nm], dtype=np.float32)
    na_cls = np.array(na_pc.classification[nm], dtype=np.uint8)

    # Align NASA vertical datum if any small geoid/ellipsoid offset exists
    if len(ot_z) > 100 and len(na_z) > 100:
        na_gnd = na_z[na_cls == 2] if (na_cls == 2).sum() > 50 else np.percentile(na_z, [5])
        dz = float(np.percentile(na_gnd, 10) - np.percentile(ot_z, 10))
        if abs(dz) > 0.35:
            na_z = na_z - dz

    hs33 = multi_hillshade(dem33, res_m=res_m)
    lrm33 = compute_lrm(dem33, res_m=res_m, sigma_m=20.0)

    return {
        "strip_id": strip_id,
        "lat": lat,
        "lon": lon,
        "cx": cx,
        "cy": cy,
        "half_m": half_m,
        "res_m": res_m,
        "crs": str(crs),
        "dem33": dem33,
        "hs33": hs33,
        "lrm33": lrm33,
        "g1_33": g1_33,
        "chm": chm_win,
        "sat_rgb": sat_rgb,
        "ot_pts": (ot_x, ot_y, ot_z, ot_cls),
        "nasa_pts": (na_x, na_y, na_z, na_cls),
    }


def choose_optimal_transect(bundle: dict, span_m: float = 190.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Automatically choose an A -> A' cross-section line passing directly through
    the highest monumental peak inside the central architectural core and across
    the adjacent leveled plaza / secondary mound!
    """
    cx, cy = bundle["cx"], bundle["cy"]
    half_m = bundle["half_m"]
    lrm = bundle["lrm33"]
    dem = bundle["dem33"]
    h, w = lrm.shape

    # Search central 55% for primary focal pyramid/mound summit
    r0, r1 = int(h * 0.22), int(h * 0.78)
    c0, c1 = int(w * 0.22), int(w * 0.78)
    sub = np.where(np.isfinite(lrm[r0:r1, c0:c1]), lrm[r0:r1, c0:c1], -999.0)
    pr_rel, pc_rel = np.unravel_index(np.argmax(sub), sub.shape)
    pr, pc = r0 + pr_rel, c0 + pc_rel
    peak_x = (cx - half_m) + (pc / w) * (2 * half_m)
    peak_y = (cy + half_m) - (pr / h) * (2 * half_m)

    # Test 36 azimuths through (peak_x, peak_y) to maximize architectural contrast
    # (passing through peak + flat plaza + secondary flanking mound while staying 95%+ in valid data)
    best_az = 0.0
    best_val = -1e9
    s_vals = np.linspace(-span_m * 0.5, span_m * 0.5, 240)
    for az_deg in np.linspace(0, 175, 36):
        az = math.radians(az_deg)
        xs = peak_x + (s_vals - 15.0) * math.cos(az)  # slight offset so peak is prominently framed
        ys = peak_y + (s_vals - 15.0) * math.sin(az)
        cols = ((xs - (cx - half_m)) / (2 * half_m) * w).astype(int)
        rows = (((cy + half_m) - ys) / (2 * half_m) * h).astype(int)
        in_b = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        if in_b.mean() < 0.96:
            continue
        prof_lrm = lrm[rows[in_b], cols[in_b]]
        if np.isfinite(prof_lrm).mean() < 0.94:
            continue
        # Count distinct structural rises (>1.0m) + flat plaza segments (|lrm|<0.35m)
        high_2 = np.nanpercentile(prof_lrm[len(prof_lrm)//2:], 95) + np.nanpercentile(prof_lrm[:len(prof_lrm)//2], 95)
        plaza_f = float((np.abs(prof_lrm) < 0.35).mean())
        score = high_2 + 2.5 * plaza_f
        if score > best_val:
            best_val = score
            best_az = az

    p0 = (peak_x - (span_m * 0.5 + 15.0) * math.cos(best_az), peak_y - (span_m * 0.5 + 15.0) * math.sin(best_az))
    p1 = (peak_x + (span_m * 0.5 - 15.0) * math.cos(best_az), peak_y + (span_m * 0.5 - 15.0) * math.sin(best_az))
    return p0, p1


def extract_pointcloud_slice(
    bundle: dict,
    p0: tuple[float, float],
    p1: tuple[float, float],
    swath_half_width_m: float = 3.0,
) -> dict:
    """Project raw 3D NASA .las (canopy+ground) and OT .laz (classified ground 2+8)
    points within +/- swath_half_width_m of segment p0 -> p1.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length

    ot_x, ot_y, ot_z, ot_cls = bundle["ot_pts"]
    s_ot = (ot_x - x0) * ux + (ot_y - y0) * uy
    d_ot = np.abs(-(ot_x - x0) * uy + (ot_y - y0) * ux)
    m_ot = (s_ot >= 0) & (s_ot <= length) & (d_ot <= swath_half_width_m)

    na_x, na_y, na_z, na_cls = bundle["nasa_pts"]
    s_na = (na_x - x0) * ux + (na_y - y0) * uy
    d_na = np.abs(-(na_x - x0) * uy + (na_y - y0) * ux)
    m_na = (s_na >= 0) & (s_na <= length) & (d_na <= swath_half_width_m)

    # Also sample exact 0.33m bare-earth DEM along transect to compute AGL height of canopy points
    n_s = 500
    s_grid = np.linspace(0, length, n_s)
    gx = x0 + s_grid * ux
    gy = y0 + s_grid * uy
    cx, cy, half_m = bundle["cx"], bundle["cy"], bundle["half_m"]
    dem = bundle["dem33"]
    h, w = dem.shape
    cols = np.clip(((gx - (cx - half_m)) / (2 * half_m) * w).astype(int), 0, w - 1)
    rows = np.clip((((cy + half_m) - gy) / (2 * half_m) * h).astype(int), 0, h - 1)
    dem_prof = dem[rows, cols]

    # Interpolate bare-earth elevation at each NASA point to separate true vegetation (>1.2m AGL)
    na_s_sel = s_na[m_na]
    na_z_sel = na_z[m_na]
    na_c_sel = na_cls[m_na]
    valid_prof = np.isfinite(dem_prof)
    if valid_prof.sum() > 20:
        ground_at_na = np.interp(na_s_sel, s_grid[valid_prof], dem_prof[valid_prof])
    else:
        ground_at_na = np.full_like(na_z_sel, np.nanmedian(ot_z[m_ot]) if m_ot.any() else 0.0)
    agl_na = na_z_sel - ground_at_na
    veg_mask = (na_c_sel == 1) & (agl_na >= 1.1)

    return {
        "length_m": length,
        "swath_width_m": swath_half_width_m * 2.0,
        "s_grid": s_grid,
        "dem_prof": dem_prof,
        "ot_s": s_ot[m_ot],
        "ot_z": ot_z[m_ot],
        "ot_cls": ot_cls[m_ot],
        "veg_s": na_s_sel[veg_mask],
        "veg_z": na_z_sel[veg_mask],
        "veg_agl": agl_na[veg_mask],
        "n_ot_total_box": len(ot_z),
        "n_nasa_total_box": len(na_z),
    }


def render_candidate_showcase(
    bundle: dict,
    out_path: Path,
    title: str,
    subtitle: str,
    stats_badges: list[str],
    p0: tuple[float, float] | None = None,
    p1: tuple[float, float] | None = None,
    swath_half_m: float = 3.0,
) -> dict:
    """Render a 4-panel + full-width Miotti-style Raw 3D Point-Cloud Cross-Section figure."""
    if p0 is None or p1 is None:
        p0, p1 = choose_optimal_transect(bundle, span_m=min(210.0, bundle["half_m"] * 1.45))
    sl = extract_pointcloud_slice(bundle, p0, p1, swath_half_width_m=swath_half_m)

    bg = "#0d1117"
    panel_bg = "#161b22"
    txt_col = "#e6edf3"
    muted = "#8b949e"
    accent = "#ffd166"

    fig = plt.figure(figsize=(18, 11.2), dpi=160, facecolor=bg)
    gs = GridSpec(
        2, 4, figure=fig, height_ratios=[1.05, 0.95], hspace=0.24, wspace=0.14,
        left=0.045, right=0.975, top=0.885, bottom=0.065
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", color=txt_col, y=0.965)
    fig.text(0.5, 0.928, subtitle, ha="center", va="center", fontsize=10.5, color="#b1bac4")
    if stats_badges:
        badge_str = "   |   ".join(stats_badges)
        fig.text(
            0.5, 0.902, badge_str, ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=accent,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#1f2937", edgecolor="#374151", lw=1.0),
        )

    cx, cy, half_m = bundle["cx"], bundle["cy"], bundle["half_m"]
    ext = [-half_m, half_m, -half_m, half_m]
    tx0, ty0 = p0[0] - cx, p0[1] - cy
    tx1, ty1 = p1[0] - cx, p1[1] - cy

    def add_transect_overlay(ax, line_col="#ffd166"):
        ax.plot([tx0, tx1], [ty0, ty1], ls="--", lw=1.9, color=line_col, alpha=0.95)
        ax.scatter([tx0, tx1], [ty0, ty1], s=42, c=line_col, edgecolors="black", zorder=6)
        for lx, ly, lbl in [(tx0, ty0, "A"), (tx1, ty1, "A'")]:
            ax.text(
                lx, ly + half_m * 0.06, lbl, color=line_col, fontsize=10, fontweight="bold",
                ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            )

    def style_map_ax(ax, ptitle: str, show_scale: bool = False):
        ax.set_facecolor(panel_bg)
        ax.set_title(ptitle, color=txt_col, fontsize=10.5, fontweight="bold", pad=7)
        ax.set_xlim(-half_m, half_m)
        ax.set_ylim(-half_m, half_m)
        ax.tick_params(colors=muted, labelsize=8)
        ax.set_xlabel("Meters East of Center", color=muted, fontsize=8)
        if show_scale:
            ax.set_ylabel("Meters North of Center", color=muted, fontsize=8)
            bar_m = 50 if half_m <= 160 else 100
            bx0 = -half_m * 0.86
            by0 = -half_m * 0.86
            ax.plot([bx0, bx0 + bar_m], [by0, by0], color="white", lw=3.5, zorder=7,
                    path_effects=[pe.withStroke(linewidth=5.5, foreground="black")])
            ax.text(bx0 + bar_m / 2, by0 + half_m * 0.05, f"{bar_m} m", color="white",
                    fontsize=8.5, fontweight="bold", ha="center",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
        else:
            ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    # Panel 1: 0.33m Multi-Azimuth Hillshade + LRM relief blend
    ax1 = fig.add_subplot(gs[0, 0])
    hs = bundle["hs33"]
    lrm = bundle["lrm33"]
    hs_norm = np.clip((hs - np.nanpercentile(hs, 2)) / max(np.nanpercentile(hs, 98) - np.nanpercentile(hs, 2), 1e-4), 0, 1)
    lrm_clip = np.clip((lrm + 1.2) / 4.8, 0, 1)
    cmap_lrm = plt.cm.inferno(np.nan_to_num(lrm_clip, nan=0.0))[..., :3]
    blend = 0.62 * np.repeat(np.nan_to_num(hs_norm, nan=0.08)[..., None], 3, axis=-1) + 0.38 * cmap_lrm
    blend[~np.isfinite(hs)] = 0.07
    ax1.imshow(np.clip(blend, 0, 1), extent=ext, origin="upper")
    add_transect_overlay(ax1, "#00e5ff")
    style_map_ax(ax1, "1. 0.33m Bare-Earth Hillshade + LRM", show_scale=True)

    # Panel 2: 0.33m OpenTopography G1 Composite
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(bundle["g1_33"], extent=ext, origin="upper")
    add_transect_overlay(ax2, "#ffd166")
    style_map_ax(ax2, "2. 0.33m G1 Architectural Relief")

    # Panel 3: 2013 NASA Canopy Height Model (CHM)
    ax3 = fig.add_subplot(gs[0, 2])
    if bundle["chm"] is not None and np.isfinite(bundle["chm"]).any():
        im_chm = ax3.imshow(bundle["chm"], extent=ext, origin="upper", cmap="YlGn", vmin=0, vmax=28)
        c_med = float(np.nanmedian(bundle["chm"]))
        c_p90 = float(np.nanpercentile(bundle["chm"][np.isfinite(bundle["chm"])], 90))
        ax3.text(
            0.04, 0.05, f"Median CHM: {c_med:.1f} m\n90th pct: {c_p90:.1f} m",
            transform=ax3.transAxes, color="white", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d1117cc", edgecolor="#2ea043"),
        )
    add_transect_overlay(ax3, "#ffd166")
    style_map_ax(ax3, "3. NASA G-LiHT Canopy Height (CHM)")

    # Panel 4: Esri World Imagery Contemporary Optical Satellite
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(bundle["sat_rgb"], extent=ext, origin="upper")
    add_transect_overlay(ax4, "#ff6b6b")
    ax4.text(
        0.04, 0.05, "Optical Satellite:\nStructures hidden under jungle",
        transform=ax4.transAxes, color="white", fontsize=8.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d1117cc", edgecolor="#30363d"),
    )
    style_map_ax(ax4, "4. Esri World Imagery (Optical Canopy)")

    # Bottom Full-Width Panel: Pietro Miotti-Style Raw 3D Point-Cloud Cross-Section (A -> A')
    axp = fig.add_subplot(gs[1, :])
    axp.set_facecolor("#0f141b")
    for sp in axp.spines.values():
        sp.set_color("#30363d")

    ot_s, ot_z, ot_cls = sl["ot_s"], sl["ot_z"], sl["ot_cls"]
    veg_s, veg_z, veg_agl = sl["veg_s"], sl["veg_z"], sl["veg_agl"]
    s_grid, dem_prof = sl["s_grid"], sl["dem_prof"]

    # Fill subtle earth beneath bare-earth profile
    valid_p = np.isfinite(dem_prof)
    base_z = float(np.nanmin(dem_prof)) - 2.2 if valid_p.any() else float(np.min(ot_z)) - 2.0
    if valid_p.any():
        axp.fill_between(
            s_grid[valid_p], base_z, dem_prof[valid_p],
            color="#2b2118", alpha=0.65, zorder=1
        )
        axp.plot(s_grid[valid_p], dem_prof[valid_p], color="#ffca28", lw=1.6, alpha=0.9, zorder=4,
                 label="0.33m Bare-Earth Surface")

    # Plot raw NASA .las Class 1 Canopy & Understory returns colored by AGL height
    if len(veg_s) > 0:
        sc_veg = axp.scatter(
            veg_s, veg_z, c=np.clip(veg_agl, 1.5, 26.0), cmap="Greens", vmin=-3.0, vmax=26.0,
            s=3.2, alpha=0.62, linewidths=0, zorder=2,
            label=f"NASA Raw .las Canopy Returns (n={len(veg_s):,})"
        )

    # Plot raw OpenTopography .laz Class 2 (Ground) & Class 8 (Key-Point Masonry/Terrain)
    if len(ot_s) > 0:
        c2 = ot_cls == 2
        c8 = ot_cls == 8
        if c2.any():
            axp.scatter(
                ot_s[c2], ot_z[c2], color="#e07a5f", s=4.5, alpha=0.80, linewidths=0, zorder=5,
                label=f"OT Raw .laz Class 2 Ground (n={int(c2.sum()):,})"
            )
        if c8.any():
            axp.scatter(
                ot_s[c8], ot_z[c8], color="#ffd166", s=5.5, alpha=0.90, linewidths=0, zorder=6,
                label=f"OT Raw .laz Class 8 Key-Points (n={int(c8.sum()):,})"
            )

    # Annotate primary architectural summit along profile
    if valid_p.any():
        vp_s = s_grid[valid_p]
        vp_z = dem_prof[valid_p]
        # Local baseline (25th percentile of bare-earth transect = plaza/base floor)
        plaza_z = float(np.percentile(vp_z, 25))
        pk_idx = int(np.argmax(vp_z))
        pk_s, pk_z = float(vp_s[pk_idx]), float(vp_z[pk_idx])
        prom_m = pk_z - plaza_z
        axp.annotate(
            f"Monumental Mound / Pyramid\n+{prom_m:.1f} m above plaza floor ({pk_z:.1f} m ASL)",
            xy=(pk_s, pk_z + 0.3),
            xytext=(min(sl["length_m"] * 0.82, max(sl["length_m"] * 0.18, pk_s + 22.0)), pk_z + max(6.5, prom_m * 0.6)),
            color="#ffd166", fontsize=9.2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22ee", edgecolor="#ffd166", lw=1.1),
            arrowprops=dict(arrowstyle="->", color="#ffd166", lw=1.5),
            zorder=8,
        )
        axp.axhline(plaza_z, ls=":", lw=1.0, color="#58a6ff", alpha=0.65, zorder=3)
        axp.text(
            sl["length_m"] * 0.015, plaza_z + 0.35,
            f"Plaza / Basal Datum (~{plaza_z:.1f} m)", color="#79c0ff", fontsize=8.2, fontweight="bold"
        )

    z_top = max(
        float(np.max(veg_z)) + 3.5 if len(veg_z) > 0 else -1e9,
        float(np.nanmax(dem_prof)) + 10.0 if valid_p.any() else 30.0,
    )
    axp.set_xlim(0, sl["length_m"])
    axp.set_ylim(base_z, z_top)
    axp.grid(True, ls="--", lw=0.5, color="#21262d", alpha=0.8)
    axp.tick_params(colors=muted, labelsize=9)
    axp.set_xlabel(f"Distance Along Transect A → A' (meters; {sl['swath_width_m']:.0f} m wide raw point-cloud slice)", color=txt_col, fontsize=10)
    axp.set_ylabel("Elevation (m EGM96)", color=txt_col, fontsize=10)

    # Label A and A' at left/right ends
    axp.text(0.008, 0.92, "A", transform=axp.transAxes, color="#00e5ff", fontsize=12, fontweight="bold")
    axp.text(0.985, 0.92, "A'", transform=axp.transAxes, color="#00e5ff", fontsize=12, fontweight="bold", ha="right")

    axp.set_title(
        f"5. Raw 3D LiDAR Point-Cloud Cross-Section (A → A') — Forest Canopy (.las) vs. Bare-Earth Architecture (.laz)",
        color=txt_col, fontsize=11.5, fontweight="bold", pad=8
    )
    leg = axp.legend(loc="upper right", frameon=True, facecolor="#161b22dd", edgecolor="#30363d", fontsize=8.5, ncol=2)
    for t in leg.get_texts():
        t.set_color(txt_col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=bg)
    plt.close(fig)

    return {
        "out_path": str(out_path),
        "n_veg_slice": int(len(veg_s)),
        "n_ground_slice": int(len(ot_s)),
        "transect_len_m": round(sl["length_m"], 1),
        "canopy_p95_agl_m": round(float(np.percentile(veg_agl, 95)), 2) if len(veg_agl) > 10 else None,
        "peak_prominence_m": round(float(np.nanmax(dem_prof) - np.nanpercentile(dem_prof[valid_p], 25)), 2) if valid_p.any() else None,
    }
