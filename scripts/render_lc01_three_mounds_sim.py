#!/usr/bin/env python3
"""Render flat-plinth 0.33 m 3D Canopy-Strip & 360° Bare-Earth Orbit MP4 + GIF
specifically for LC-01 (the Outlying Three-Mound Terrace Group 520 m ENE of
Nuevo Jalisco on strip ChiapsCampeche_NFI_l0s473), with 3D pins on all 3 mounds.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import laspy
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.warp import transform as rio_transform
from rasterio.windows import from_bounds
from scipy import ndimage as ndi

sys.path.insert(0, "/agent/maya-lidar-discovery/scripts")
sys.path.insert(0, "/agent/maya-lidar-discovery/scripts/open_lidar_hunt_v2")
import raw_pointcloud_tools as rpt
import render_3d_video_simulations as r3d
import v2_scene_tools as vst

REPO_ROOT = Path("/agent/nuevo-jalisco-lacandon-lidar")
OUT_REPO_MEDIA = REPO_ROOT / "media"
OUT_STORE_MEDIA = Path("/cursor/stores/bc-adff6ec3-2baf-40bb-9bab-102a5f567f18/media/open-lidar-hunt-v3")
VID_TMP = Path("/agent/lidar/work/lacandon_deep_dive/video_sims_lc01")
VID_TMP.mkdir(parents=True, exist_ok=True)

SITE = {
    "code": "LC-01",
    "slug_base": "sim-lc01-three-mounds-terrace",
    "strip_id": "ChiapsCampeche_NFI_l0s473",
    # Centered directly on the middle mound of the 3-mound NW-SE row
    "lat": 16.72210,
    "lon": -91.16472,
    "half_m": 102.0,
    "z_exag": 2.55,
    "peak_label": "M1 (+3.5 m)",
    "title": "LC-01: Outlying Three-Mound Aligned Terrace Complex (520 m ENE of Nuevo Jalisco)",
    "region": "Lacandón Rainforest, Chiapas (16.72200° N, 91.16463° W) • Strip ChiapsCampeche_NFI_l0s473",
    "stats": "3 Aligned Closed-360° Mounds (+3.5 m, +3.3 m, +2.8 m) • 1,282 m² (<2°) Terrace • 38,258 .LAZ Ground Pts",
    "lit_status": "Unmapped Outlying Group Along ENE Lacanjá Trail (Tovalín 2004: 55) • 100% Virgin Rainforest (29–44 m Canopy)",
}


def prepare_lc01_scene(site: dict) -> dict:
    """Build 0.33 m flat-plinth scene directly from repo's cropped LC-01 rasters + .laz clouds."""
    crop_dir = REPO_ROOT / "lidar_data/lc01_three_mounds"
    half_m = site["half_m"]

    with rasterio.open(crop_dir / "DEM_0p33m.tif") as ds:
        xs, ys = rio_transform("EPSG:4326", ds.crs, [site["lon"]], [site["lat"]])
        cx, cy = float(xs[0]), float(ys[0])
        win = from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, ds.transform)
        dem = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        if ds.nodata is not None:
            dem = np.where(dem == ds.nodata, np.nan, dem)
        dem = np.where((np.abs(dem) > 1e6) | (dem < -100), np.nan, dem)
        res_m = float(abs(ds.res[0]))

    with rasterio.open(crop_dir / "G1_0p33m.tif") as gds:
        w_g1 = from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, gds.transform)
        g1_rgb = np.moveaxis(
            gds.read([1, 2, 3], window=w_g1, out_shape=(3, dem.shape[0], dem.shape[1]), boundless=True, fill_value=18),
            0,
            -1,
        )

    valid = np.isfinite(dem)
    base_z = float(np.nanpercentile(dem[valid], 12))
    dem_fill = np.where(valid, dem, base_z)
    dem_s = ndi.gaussian_filter(dem_fill, sigma=0.85)

    hs33 = rpt.multi_hillshade(dem, res_m=res_m)
    lrm33 = rpt.compute_lrm(dem, res_m=res_m, sigma_m=20.0)

    # Remove broad >38m hillside slope so the leveled terrace bench sits flat on the 3D plinth,
    # making the 3 discrete mounds & their intervening 2.5–3.5 m saddles unmistakably obvious!
    broad_trend = ndi.gaussian_filter(dem_s, sigma=38.0 / res_m)
    local_topo = np.clip((dem_s - broad_trend) + 0.9, 0.0, 7.5)

    lrm_s = ndi.gaussian_filter(np.nan_to_num(lrm33, nan=0.0), sigma=0.85)
    lrm_norm = np.clip((lrm_s + 0.25) / 3.6, 0.0, 1.0)

    z_struct_boost = np.clip(lrm_s, 0.0, 6.5) * 1.45
    z_surf_grid = 1.6 + 0.35 * local_topo + z_struct_boost

    dz_drow, dz_dcol = np.gradient(z_surf_grid * site["z_exag"], res_m, res_m)
    dz_dx = dz_dcol
    dz_dy = -dz_drow
    norm_len = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
    nx_grid = (-dz_dx / norm_len).astype(np.float32)
    ny_grid = (-dz_dy / norm_len).astype(np.float32)
    nz_grid = (1.0 / norm_len).astype(np.float32)

    albedo_grid = r3d.base_albedo_from_lrm(lrm_norm, g1_rgb)
    hs_f = np.clip(np.nan_to_num(hs33, nan=0.5), 0.0, 1.0)[..., None]
    hud_bare_arr = np.clip(albedo_grid * (0.35 + 0.80 * hs_f) * 255.0, 0, 255).astype(np.uint8)
    hud_bare_arr[~valid] = np.array([14, 18, 24], dtype=np.uint8)

    dlat = half_m / 110574.0
    dlon = half_m / (111320.0 * math.cos(math.radians(site["lat"])))
    sat_raw = vst.fetch_esri_bbox_rgb(
        site["lon"] - dlon, site["lat"] - dlat, site["lon"] + dlon, site["lat"] + dlat,
        out_w=512, out_h=512, zoom_level=16,
    )
    sat_img = Image.fromarray(sat_raw).resize((dem.shape[1], dem.shape[0]), Image.Resampling.BILINEAR)
    sat_rgb = np.asarray(sat_img, dtype=np.uint8)

    H, W = dem.shape
    step = max(1, int(round(min(H, W) / 360.0)))
    yy, xx = np.mgrid[0:H:step, 0:W:step]
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

    # Locate the 3 aligned mounds directly in local world coords relative to (cx, cy)!
    # Original LC-01 center was (16.721996, -91.164633); find exact local maxima around each of the 3 mounds
    mound_seeds_m = [
        ("NW Mound M1 (+3.5 m)", -39.5, +31.8),
        ("Center Mound M2 (+3.3 m)", +8.4, +5.1),
        ("SE Mound M3 (+2.8 m)", +31.1, -35.9),
    ]
    pins = []
    for lbl, sx_m, sy_m in mound_seeds_m:
        sc = int(round(((sx_m / half_m + 1.0) * 0.5) * (W - 1)))
        sr = int(round(((1.0 - (sy_m / half_m + 1.0) * 0.5)) * (H - 1)))
        rad = int(round(12.0 / res_m))
        r0, r1 = max(0, sr - rad), min(H, sr + rad)
        c0, c1 = max(0, sc - rad), min(W, sc + rad)
        sub = np.where(valid[r0:r1, c0:c1], lrm_s[r0:r1, c0:c1], -999.0)
        pr_rel, pc_rel = np.unravel_index(int(np.argmax(sub)), sub.shape)
        pr, pc = r0 + pr_rel, c0 + pc_rel
        px_w = float(((pc / max(W - 1, 1)) * 2.0 - 1.0) * half_m)
        py_w = float((1.0 - (pr / max(H - 1, 1)) * 2.0) * half_m)
        pz_w = float(z_surf_grid[pr, pc])
        pins.append((lbl, px_w, py_w, pz_w))

    # Load real 3D canopy returns from CANOPY_AND_GROUND.laz
    pc_all = laspy.read(crop_dir / "CANOPY_AND_GROUND.laz")
    m_box = (
        (pc_all.x >= cx - half_m) & (pc_all.x <= cx + half_m)
        & (pc_all.y >= cy - half_m) & (pc_all.y <= cy + half_m)
    )
    na_x = (np.array(pc_all.x[m_box], dtype=np.float64) - cx).astype(np.float32)
    na_y = (np.array(pc_all.y[m_box], dtype=np.float64) - cy).astype(np.float32)
    na_z_asl = np.array(pc_all.z[m_box], dtype=np.float32)

    gx = np.clip(((na_x / half_m + 1.0) * 0.5 * (W - 1)).astype(np.int32), 0, W - 1)
    gy = np.clip(((1.0 - (na_y / half_m + 1.0) * 0.5) * (H - 1)).astype(np.int32), 0, H - 1)
    agl = na_z_asl - dem_s[gy, gx]
    can_mask = (agl >= 2.2) & (agl <= 44.0) & valid[gy, gx]

    can_x = na_x[can_mask]
    can_y = na_y[can_mask]
    can_agl = agl[can_mask]
    can_z = z_surf_grid[gy[can_mask], gx[can_mask]].astype(np.float32) + can_agl * 0.48

    if len(can_x) > 80000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(can_x), size=80000, replace=False)
        can_x, can_y, can_z, can_agl = can_x[idx], can_y[idx], can_z[idx], can_agl[idx]

    c_gx = np.clip(((can_x / half_m + 1.0) * 0.5 * (W - 1)).astype(np.int32), 0, W - 1)
    c_gy = np.clip(((1.0 - (can_y / half_m + 1.0) * 0.5) * (H - 1)).astype(np.int32), 0, H - 1)
    sat_c = sat_rgb[c_gy, c_gx].astype(np.float32)
    h_norm = np.clip((can_agl - 2.0) / 32.0, 0.0, 1.0)[:, None]
    emerald = np.column_stack(
        [20.0 + 52.0 * h_norm[:, 0], 86.0 + 118.0 * h_norm[:, 0], 30.0 + 46.0 * h_norm[:, 0]]
    ).astype(np.float32)
    can_col = np.clip(0.40 * sat_c + 0.60 * emerald, 0, 255).astype(np.uint8)

    hud_bare = Image.fromarray(hud_bare_arr).resize((250, 250), Image.Resampling.BILINEAR)
    hud_sat = Image.fromarray(sat_rgb).resize((250, 250), Image.Resampling.BILINEAR)

    return {
        "site": site,
        "base_z": base_z,
        "peak": (pins[0][1], pins[0][2], pins[0][3]),
        "pins": pins,
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


def render_lc01_frame(scene: dict, frame_idx: int, total_frames: int, out_path: Path) -> None:
    r3d.render_frame(scene, frame_idx, total_frames, out_path)
    site = scene["site"]
    half_m = site["half_m"]
    z_exag = site["z_exag"]
    u = frame_idx / max(total_frames - 1, 1)
    az_deg = -35.0 + 330.0 * u
    pitch_deg = 29.0 + 4.5 * math.sin(u * math.pi * 2.0)
    az = math.radians(az_deg)
    pitch = math.radians(pitch_deg)
    ca, sa = math.cos(az), math.sin(az)
    cp, sp = math.cos(pitch), math.sin(pitch)

    if u < 0.14:
        sweep_pos = -1.50
    elif u < 0.44:
        sweep_pos = -1.45 + 2.95 * ((u - 0.14) / 0.30)
    else:
        sweep_pos = 2.00

    img = Image.open(out_path)
    draw = ImageDraw.Draw(img)
    _, _, f_small, f_mono = r3d.load_fonts()

    vp_x0, vp_y0, vp_w, vp_h = 20, 95, 950, 600
    cam_dist = half_m * 3.25
    fov = vp_w * 1.15
    cy_screen = vp_h * 0.56
    z_pivot = float(np.median(scene["surf_z"] * z_exag)) * 0.65

    # Draw 3D summit pins for Mound 2 and Mound 3 as well (Mound 1 is already drawn by r3d.render_frame!)
    pin_heights_m = [14.0, 10.5, 13.5]
    for idx_p, (lbl, pk_x, pk_y, pk_z_raw) in enumerate(scene["pins"]):
        if idx_p == 0:
            continue
        pk_diag = (pk_x - pk_y) / (1.4142 * half_m)
        if pk_diag <= sweep_pos:
            pk_z = pk_z_raw * z_exag
            prx = pk_x * ca - pk_y * sa
            pry = pk_x * sa + pk_y * ca
            prz_base = pk_z - z_pivot
            prz_top = (pk_z + pin_heights_m[idx_p] * z_exag) - z_pivot
            d_base = cam_dist + pry * cp - prz_base * sp
            sy_base = +pry * sp + prz_base * cp
            d_top = cam_dist + pry * cp - prz_top * sp
            sy_top = +pry * sp + prz_top * cp
            sx_b = int(vp_x0 + vp_w * 0.5 + (prx * fov) / d_base)
            sy_b = int(vp_y0 + cy_screen - (sy_base * fov) / d_base)
            sx_t = int(vp_x0 + vp_w * 0.5 + (prx * fov) / d_top)
            sy_t = int(vp_y0 + cy_screen - (sy_top * fov) / d_top)
            if (vp_x0 + 40 <= sx_b <= vp_x0 + vp_w - 40) and (vp_y0 + 50 <= sy_t <= vp_y0 + vp_h - 60):
                draw.line([sx_b, sy_b, sx_t, sy_t], fill=(56, 224, 255), width=2)
                draw.ellipse([sx_b - 4, sy_b - 4, sx_b + 4, sy_b + 4], fill=(0, 210, 255), outline=(255, 255, 255))
                tag = f"▲ {lbl}"
                tw = len(tag) * 7 + 14
                bx0 = max(vp_x0 + 14, min(sx_t - tw // 2, vp_x0 + vp_w - tw - 14))
                by0 = max(vp_y0 + 48, sy_t - 24)
                draw.rectangle([bx0, by0, bx0 + tw, by0 + 22], fill=(15, 23, 36), outline=(56, 224, 255), width=1)
                draw.text((bx0 + 7, by0 + 4), tag, fill=(165, 243, 252), font=f_small)

    # Bottom telemetry bar with exact WGS84 coordinates
    draw.rectangle([vp_x0 + 12, vp_y0 + vp_h - 42, vp_x0 + vp_w - 12, vp_y0 + vp_h - 10], fill=(12, 18, 26))
    telem = (
        f"ID: LC-01 (520 m ENE of #08)  |  WGS84: (16.72200, -91.16463)  |  "
        f"Az: {az_deg % 360:5.1f}°  |  Pitch: {pitch_deg:4.1f}°  |  Datum: {scene['base_z']:.0f} m ASL"
    )
    draw.text((vp_x0 + 18, vp_y0 + vp_h - 33), telem, fill=(210, 225, 245), font=f_mono)
    img.save(out_path, format="PNG", compress_level=1)


def _chunk_worker(args):
    scene, indices, fdir, total_frames = args
    for i in indices:
        render_lc01_frame(scene, i, total_frames, fdir / f"frame_{i:04d}.png")
    return len(indices)


def main() -> None:
    total_frames = 108
    fps_mp4 = 20
    scene = prepare_lc01_scene(SITE)
    fdir = VID_TMP / "frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True, exist_ok=True)

    n_workers = min(8, os.cpu_count() or 4)
    chunks = [list(range(w, total_frames, n_workers)) for w in range(n_workers)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_chunk_worker, [(scene, ch, fdir, total_frames) for ch in chunks]))

    shutil.copy2(fdir / "frame_0065.png", VID_TMP / "check_frame_0065.png")
    shutil.copy2(fdir / "frame_0088.png", VID_TMP / "check_frame_0088.png")

    out_mp4 = OUT_STORE_MEDIA / f"{SITE['slug_base']}.mp4"
    out_gif = OUT_STORE_MEDIA / f"{SITE['slug_base']}.gif"
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps_mp4), "-i", str(fdir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "fast",
            "-movflags", "+faststart", str(out_mp4),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(out_mp4),
            "-vf", "fps=12,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=180:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5",
            "-loop", "0", str(out_gif),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    shutil.copy2(out_mp4, OUT_REPO_MEDIA / out_mp4.name)
    shutil.copy2(out_gif, OUT_REPO_MEDIA / out_gif.name)
    shutil.rmtree(fdir, ignore_errors=True)
    print(
        f"Wrote {out_mp4.name} ({out_mp4.stat().st_size / 1e6:.2f} MB) & "
        f"{out_gif.name} ({out_gif.stat().st_size / 1e6:.2f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
