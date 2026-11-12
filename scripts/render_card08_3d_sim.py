#!/usr/bin/env python3
"""Render flat-plinth 0.33 m 3D Canopy-Strip & 360° Bare-Earth Orbit GIF + MP4
for Candidate #08 (Nuevo Jalisco Acropolis & Gran Plaza, ChiapsCampeche_NFI_l0s473)
and LC-08 (Top Unmapped 17-Mound Lacandon Courtyard, ChiapsCampeche_NFI_l0s474).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, "/agent/maya-lidar-discovery/scripts")
sys.path.insert(0, "/agent/maya-lidar-discovery/scripts/open_lidar_hunt_v2")
import raw_pointcloud_tools as rpt
import render_3d_video_simulations as r3d
import v2_scene_tools as vst
from render_nj_unbroken_jungle_v3 import _worker_chunk, prepare_flat_plinth_scene

OUT_STORE = Path("/cursor/stores/bc-adff6ec3-2baf-40bb-9bab-102a5f567f18/media/open-lidar-hunt-v3")
OUT_REPO = Path("/agent/maya-lidar-discovery/media/open-lidar-hunt-v3")
VID_TMP = Path("/agent/lidar/work/lacandon_deep_dive/video_sims")
VID_TMP.mkdir(parents=True, exist_ok=True)

SIM_SITES = [
    {
        "code": "CARD-08",
        "slug_base": "sim-card08-nuevo-jalisco-acropolis",
        "strip_id": "ChiapsCampeche_NFI_l0s473",
        # Centered +22m N of 16.719521 with half_m=118m so the entire 80x70m Gran Plaza,
        # SW/NW/SE range platforms, and 70x25m NE Acropolis sit centered on a flat plinth
        # while cropping out the steep southern natural karst ridge!
        "lat": 16.71972,
        "lon": -91.16876,
        "half_m": 118.0,
        "z_exag": 2.15,
        "peak_label": "NE Acropolis (+8.7 m above Gran Plaza)",
        "title": "CANDIDATE #08: Classic Maya NE Acropolis & 80×70 m Gran Plaza (Nuevo Jalisco)",
        "region": "Lacandón Rainforest, Chiapas (16.71952° N, 91.16876° W) • 11.2 km NW of Bonampak",
        "stats": "1,425 .LAZ Ground Pts on Acropolis (45,277 in 320m) • 99.0% Virgin Jungle (25–38 m Canopy)",
        "lit_status": "Matches Tovalín 2004 Ground Map Within 19 m • Missing from EAAMS, INAH SIC, OSM & MARI-GISLAB",
    },
    {
        "code": "LC-08",
        "slug_base": "sim-card08-lc08-unmapped-lacandon-courtyard",
        "strip_id": "ChiapsCampeche_NFI_l0s474",
        "lat": 16.76310,
        "lon": -91.21789,
        "half_m": 135.0,
        "z_exag": 2.45,
        "peak_label": "Orthogonal Courtyard Pyramid (+3.2 m)",
        "title": "LC-08: Unmapped 17-Mound Orthogonal Courtyard Complex (7.1 km NW of #08)",
        "region": "Lacandón Rainforest Corridor (16.76310° N, 91.21789° W) • 5.57 km W of Ojos de Agua",
        "stats": "17 Mounds (6 Rectangular, 4/4 Quadrants) • 1,165 m² Plaza • 91,887 Raw .LAZ Ground Pts (1.41/m²)",
        "lit_status": "0 IN MARI-GISLAB, EAAMS, INAH SIC, OSM & TOVALÍN SURVEYS • 98% Unbroken Jungle (21–31 m Canopy)",
    },
]


def prepare_lacandon_flat_plinth(site: dict) -> dict:
    """Identical to prepare_flat_plinth_scene, with canopy height extended up to 42 m
    for towering Lacandón mahogany/ceiba rainforest!
    """
    b = rpt.fetch_site_bundle(site["strip_id"], site["lat"], site["lon"], half_m=site["half_m"])
    dem = b["dem33"].copy()
    valid = np.isfinite(dem)
    res_m = b["res_m"]

    base_z = float(np.nanpercentile(dem[valid], 10))
    dem_fill = np.where(valid, dem, base_z)
    dem_s = ndi.gaussian_filter(dem_fill, sigma=0.85)

    # Detrend broad >42m karst valley tilt so the plaza sits completely level on the 3D stone plinth
    broad_trend = ndi.gaussian_filter(dem_s, sigma=42.0 / res_m)
    local_topo = np.clip((dem_s - broad_trend) + 1.0, 0.0, 9.5)

    lrm_s = ndi.gaussian_filter(np.nan_to_num(b["lrm33"], nan=0.0), sigma=0.85)
    lrm_norm = np.clip((lrm_s + 0.35) / 5.5, 0.0, 1.0)

    z_struct_boost = np.clip(lrm_s, 0.0, 9.5) * 1.15
    z_surf_grid = 1.6 + 0.32 * local_topo + z_struct_boost

    dz_drow, dz_dcol = np.gradient(z_surf_grid * site["z_exag"], res_m, res_m)
    dz_dx = dz_dcol
    dz_dy = -dz_drow
    norm_len = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
    nx_grid = (-dz_dx / norm_len).astype(np.float32)
    ny_grid = (-dz_dy / norm_len).astype(np.float32)
    nz_grid = (1.0 / norm_len).astype(np.float32)

    albedo_grid = r3d.base_albedo_from_lrm(lrm_norm, b["g1_33"])
    hs_f = np.clip(np.nan_to_num(b["hs33"], nan=0.5), 0.0, 1.0)[..., None]
    hud_bare_arr = np.clip(albedo_grid * (0.35 + 0.80 * hs_f) * 255.0, 0, 255).astype(np.uint8)
    hud_bare_arr[~valid] = np.array([14, 18, 24], dtype=np.uint8)

    dlat = site["half_m"] / 110574.0
    dlon = site["half_m"] / (111320.0 * np.cos(np.radians(site["lat"])))
    sat_raw = vst.fetch_esri_bbox_rgb(
        site["lon"] - dlon, site["lat"] - dlat, site["lon"] + dlon, site["lat"] + dlat,
        out_w=512, out_h=512, zoom_level=16,
    )
    sat_img = Image.fromarray(sat_raw).resize((dem.shape[1], dem.shape[0]), Image.Resampling.BILINEAR)
    sat_rgb = np.asarray(sat_img, dtype=np.uint8)

    H, W = dem.shape
    step = max(1, int(round(min(H, W) / 360.0)))
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    half_m = site["half_m"]
    wx = ((xx.astype(np.float32) / max(W - 1, 1)) * 2.0 - 1.0) * half_m
    wy = (1.0 - (yy.astype(np.float32) / max(H - 1, 1)) * 2.0) * half_m
    wz = z_surf_grid[yy, xx].astype(np.float32)
    v_mask = valid[yy, xx]

    surf_x = wx[v_mask].ravel()
    surf_y = wy[v_mask].ravel()
    surf_z = wz[v_mask].ravel()
    surf_nx = nx_grid[yy, xx][v_mask].ravel()
    surf_ny = ny_grid[yy, xx][v_mask].ravel()
    surf_nz = nz_grid[yy, xx][v_mask].ravel()
    surf_alb = albedo_grid[yy, xx][v_mask].reshape(-1, 3)
    surf_sat_col = sat_rgb[yy, xx][v_mask].reshape(-1, 3)

    skirt_x, skirt_y, skirt_z, skirt_nx, skirt_ny, skirt_nz, skirt_alb = [], [], [], [], [], [], []
    n_z_steps = 14
    for r_idx, c_idx, enx, eny in [
        (H - 1 - step, slice(0, W, step), 0.0, -1.0),
        (0, slice(0, W, step), 0.0, 1.0),
        (slice(0, H, step), 0, -1.0, 0.0),
        (slice(0, H, step), W - 1 - step, 1.0, 0.0),
    ]:
        ex = (
            (np.arange(W)[c_idx] if isinstance(c_idx, slice) else np.full(len(range(0, H, step)), c_idx)).astype(np.float32)
            / max(W - 1, 1) * 2.0 - 1.0
        ) * half_m
        ey = (
            1.0
            - (np.arange(H)[r_idx] if isinstance(r_idx, slice) else np.full(len(range(0, W, step)), r_idx)).astype(np.float32)
            / max(H - 1, 1) * 2.0
        ) * half_m
        ez_top = z_surf_grid[r_idx, c_idx].astype(np.float32)
        ev = valid[r_idx, c_idx]
        if not np.any(ev):
            continue
        ex, ey, ez_top = ex[ev], ey[ev], ez_top[ev]
        for k in range(n_z_steps):
            frac = k / float(n_z_steps - 1)
            zk = ez_top * frac
            skirt_x.append(ex)
            skirt_y.append(ey)
            skirt_z.append(zk)
            skirt_nx.append(np.full_like(ex, enx))
            skirt_ny.append(np.full_like(ey, eny))
            skirt_nz.append(np.zeros_like(ex))
            band_shade = 0.26 + 0.14 * frac + 0.04 * np.sin(zk * 2.5)
            scol = np.column_stack([band_shade * 0.95, band_shade * 0.88, band_shade * 0.76]).astype(np.float32)
            skirt_alb.append(scol)

    sk_x = np.concatenate(skirt_x)
    sk_y = np.concatenate(skirt_y)
    sk_z = np.concatenate(skirt_z)
    sk_nx = np.concatenate(skirt_nx)
    sk_ny = np.concatenate(skirt_ny)
    sk_nz = np.concatenate(skirt_nz)
    sk_alb = np.vstack(skirt_alb)

    r0, r1 = int(H * 0.20), int(H * 0.80)
    c0, c1 = int(W * 0.20), int(W * 0.80)
    sub_lrm = np.where(valid[r0:r1, c0:c1], lrm_s[r0:r1, c0:c1], -999.0)
    pr, pc = np.unravel_index(int(np.argmax(sub_lrm)), sub_lrm.shape)
    pr += r0
    pc += c0
    peak_x = float(((pc / max(W - 1, 1)) * 2.0 - 1.0) * half_m)
    peak_y = float((1.0 - (pr / max(H - 1, 1)) * 2.0) * half_m)
    peak_z = float(z_surf_grid[pr, pc])

    cx, cy = b["cx"], b["cy"]
    raw_nx, raw_ny, raw_nz, _ = b["nasa_pts"]
    na_x = (raw_nx - cx).astype(np.float32)
    na_y = (raw_ny - cy).astype(np.float32)
    na_z_asl = raw_nz.astype(np.float32)

    gx = np.clip(((na_x / half_m + 1.0) * 0.5 * (W - 1)).astype(np.int32), 0, W - 1)
    gy = np.clip(((1.0 - (na_y / half_m + 1.0) * 0.5) * (H - 1)).astype(np.int32), 0, H - 1)
    agl = na_z_asl - dem_s[gy, gx]
    can_mask = (agl >= 2.0) & (agl <= 42.0) & valid[gy, gx]

    can_x = na_x[can_mask]
    can_y = na_y[can_mask]
    can_agl = agl[can_mask]
    # Scale tall 38m Lacandon canopy slightly (0.58x) so canopy fits gracefully inside 3D viewport
    can_z = z_surf_grid[gy[can_mask], gx[can_mask]].astype(np.float32) + can_agl * 0.58

    if len(can_x) > 85000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(can_x), size=85000, replace=False)
        can_x, can_y, can_z, can_agl = can_x[idx], can_y[idx], can_z[idx], can_agl[idx]

    c_gx = np.clip(((can_x / half_m + 1.0) * 0.5 * (W - 1)).astype(np.int32), 0, W - 1)
    c_gy = np.clip(((1.0 - (can_y / half_m + 1.0) * 0.5) * (H - 1)).astype(np.int32), 0, H - 1)
    sat_c = sat_rgb[c_gy, c_gx].astype(np.float32)
    h_norm = np.clip((can_agl - 2.0) / 30.0, 0.0, 1.0)[:, None]
    emerald = np.column_stack(
        [20.0 + 52.0 * h_norm[:, 0], 86.0 + 118.0 * h_norm[:, 0], 30.0 + 46.0 * h_norm[:, 0]]
    ).astype(np.float32)
    can_col = np.clip(0.40 * sat_c + 0.60 * emerald, 0, 255).astype(np.uint8)

    hud_bare = Image.fromarray(hud_bare_arr).resize((250, 250), Image.Resampling.BILINEAR)
    hud_sat = Image.fromarray(sat_rgb).resize((250, 250), Image.Resampling.BILINEAR)

    return {
        "site": site,
        "base_z": base_z,
        "peak": (peak_x, peak_y, peak_z),
        "surf_x": surf_x,
        "surf_y": surf_y,
        "surf_z": surf_z,
        "surf_nx": surf_nx,
        "surf_ny": surf_ny,
        "surf_nz": surf_nz,
        "surf_alb": surf_alb,
        "surf_sat_col": surf_sat_col,
        "sk_x": sk_x,
        "sk_y": sk_y,
        "sk_z": sk_z,
        "sk_nx": sk_nx,
        "sk_ny": sk_ny,
        "sk_nz": sk_nz,
        "sk_alb": sk_alb,
        "can_x": can_x,
        "can_y": can_y,
        "can_z": can_z,
        "can_col": can_col,
        "hud_bare": np.asarray(hud_bare),
        "hud_sat": np.asarray(hud_sat),
    }


def main() -> None:
    total_frames = 96
    fps_mp4 = 20

    for site in SIM_SITES:
        print(f"Building flat-plinth 0.33m 3D orbit for {site['code']}...", flush=True)
        scene = prepare_lacandon_flat_plinth(site)
        fdir = VID_TMP / f"frames_{site['code']}"
        if fdir.exists():
            shutil.rmtree(fdir)
        fdir.mkdir(parents=True, exist_ok=True)

        n_workers = min(8, os.cpu_count() or 4)
        chunks = [list(range(w, total_frames, n_workers)) for w in range(n_workers)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_worker_chunk, [(scene, ch, fdir, total_frames) for ch in chunks]))

        still_p = VID_TMP / f"still_{site['code']}.png"
        canopy_p = VID_TMP / f"canopy_{site['code']}.png"
        shutil.copy2(fdir / f"frame_{int(total_frames * 0.62):04d}.png", still_p)
        shutil.copy2(fdir / "frame_0006.png", canopy_p)

        out_mp4 = OUT_STORE / f"{site['slug_base']}.mp4"
        out_gif = OUT_STORE / f"{site['slug_base']}.gif"
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps_mp4), "-i", str(fdir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast",
                "-movflags", "+faststart", str(out_mp4),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(out_mp4),
                "-vf", "fps=11,scale=760:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=160:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5",
                "-loop", "0", str(out_gif),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        shutil.copy2(out_mp4, OUT_REPO / out_mp4.name)
        shutil.copy2(out_gif, OUT_REPO / out_gif.name)
        shutil.rmtree(fdir, ignore_errors=True)
        print(
            f"Wrote {out_gif.name} ({out_gif.stat().st_size / (1024*1024):.2f} MB) & "
            f"{out_mp4.name} ({out_mp4.stat().st_size / (1024*1024):.2f} MB) | still={still_p}",
            flush=True,
        )


if __name__ == "__main__":
    main()
