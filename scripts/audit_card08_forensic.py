#!/usr/bin/env python3
"""Forensic multi-scale raw .laz/.las + 0.33m G1/DEM audit of Candidate #08
(16.719521, -91.168757, strip ChiapsCampeche_NFI_l0s473) and spatial inventory
of all surrounding Lacandon G-LiHT strips and known archaeological registries.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy import ndimage as ndi
from scipy.spatial import Delaunay

sys.path.insert(0, "/agent/maya-lidar-discovery/scripts")
sys.path.insert(0, "/agent/maya-lidar-discovery/scripts/open_lidar_hunt_v2")
import raw_pointcloud_tools as rpt
import v2_scene_tools as vst

OUT_DIR = Path("/agent/lidar/work/lacandon_deep_dive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAT0 = 16.719521
LON0 = -91.168757
STRIP0 = "ChiapsCampeche_NFI_l0s473"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def inventory_surrounding_strips_and_sites() -> dict:
    ot_fp = json.loads(Path("/agent/lidar/work/carbon_raw_hunt/ot_456_footprints.json").read_text())
    nearby_strips = []
    for f in ot_fp:
        bb = f.get("bbox_wgs84")
        if not bb:
            continue
        clat = 0.5 * (bb[1] + bb[3])
        clon = 0.5 * (bb[0] + bb[2])
        # Minimum distance to bbox corners/center
        dkm = min(
            haversine_km(LAT0, LON0, clat, clon),
            haversine_km(LAT0, LON0, bb[1], bb[0]),
            haversine_km(LAT0, LON0, bb[3], bb[2]),
            haversine_km(LAT0, LON0, bb[1], bb[2]),
            haversine_km(LAT0, LON0, bb[3], bb[0]),
        )
        if dkm <= 45.0 or "ChiapsCampeche" in f["name"]:
            sid = rpt.strip_id_from_name(f["name"])
            nearby_strips.append(
                {
                    "strip_id": sid,
                    "name": f["name"],
                    "path": f["path"],
                    "area_km2": round(f["area_km2"], 3),
                    "bbox_wgs84": [round(x, 5) for x in bb],
                    "approx_dist_km": round(dkm, 2),
                    "has_0p33_dem": sid in rpt.DEM33_MAP,
                    "has_0p33_g1": sid in rpt.G1_33_MAP,
                    "has_laz": sid in rpt.LAZ_MAP,
                }
            )
    nearby_strips.sort(key=lambda x: x["approx_dist_km"])

    # Load EAAMS, INAH SIC, OSM
    eaams_gj = json.loads(Path("/agent/lidar/work/ce_deep/eaams/maya_sites.geojson").read_text())
    eaams_near = []
    for feat in eaams_gj["features"]:
        lon, lat = feat["geometry"]["coordinates"][:2]
        dkm = haversine_km(LAT0, LON0, lat, lon)
        if dkm <= 35.0:
            p = feat.get("properties") or {}
            eaams_near.append(
                {
                    "name": p.get("name") or p.get("NAME"),
                    "rank": p.get("rank"),
                    "state": p.get("state"),
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "dist_km": round(dkm, 3),
                    "reference": p.get("reference") or "",
                    "notes": p.get("notes") or "",
                }
            )
    eaams_near.sort(key=lambda x: x["dist_km"])

    sic_near = []
    sic_path = Path("/agent/lidar/work/gliht_west_mexico/registries/inah_sic_all_mexico.json")
    if sic_path.exists():
        for s in json.loads(sic_path.read_text()):
            lat = s.get("lat")
            lon = s.get("lon")
            if lat is None or lon is None:
                continue
            dkm = haversine_km(LAT0, LON0, float(lat), float(lon))
            if dkm <= 60.0:
                sic_near.append({**s, "dist_km": round(dkm, 3)})
    sic_near.sort(key=lambda x: x["dist_km"])

    osm_near = []
    osm_path = Path("/agent/lidar/work/gliht_west_mexico/registries/osm_mexico_archaeology.json")
    if osm_path.exists():
        for s in json.loads(osm_path.read_text()):
            lat = s.get("lat")
            lon = s.get("lon")
            if lat is None or lon is None:
                continue
            dkm = haversine_km(LAT0, LON0, float(lat), float(lon))
            if dkm <= 40.0:
                osm_near.append({**s, "dist_km": round(dkm, 3)})
    osm_near.sort(key=lambda x: x["dist_km"])

    return {
        "nearby_strips": nearby_strips,
        "eaams_within_35km": eaams_near,
        "sic_within_60km": sic_near,
        "osm_within_40km": osm_near,
    }


def audit_card08() -> dict:
    # Fetch 600m half-width (= 1200 x 1200 m) bundle around #08!
    print("Fetching 1200x1200m 0.33m bundle + full raw .laz/.las around #08...", flush=True)
    b600 = rpt.fetch_site_bundle(STRIP0, LAT0, LON0, half_m=600.0)
    np.savez_compressed(
        OUT_DIR / "card08_bundle_600m.npz",
        cx=b600["cx"],
        cy=b600["cy"],
        res_m=b600["res_m"],
        dem33=b600["dem33"],
        g1_33=b600["g1_33"],
        lrm33=b600["lrm33"],
        hs33=b600["hs33"],
        chm=b600["chm"] if b600["chm"] is not None else np.array([]),
        ot_x=b600["ot_pts"][0],
        ot_y=b600["ot_pts"][1],
        ot_z=b600["ot_pts"][2],
        ot_cls=b600["ot_pts"][3],
        na_x=b600["nasa_pts"][0],
        na_y=b600["nasa_pts"][1],
        na_z=b600["nasa_pts"][2],
        na_cls=b600["nasa_pts"][3],
    )

    cx, cy = b600["cx"], b600["cy"]
    res_m = b600["res_m"]
    dem = b600["dem33"]
    lrm = b600["lrm33"]
    g1 = b600["g1_33"]
    H, W = dem.shape
    ot_x, ot_y, ot_z, ot_cls = b600["ot_pts"]
    na_x, na_y, na_z, na_cls = b600["nasa_pts"]

    # Also check full strip dimensions of l0s473
    laz_path, las_path = rpt.ensure_pointclouds(STRIP0)
    full_laz = laspy.read(laz_path)
    full_x = np.array(full_laz.x)
    full_y = np.array(full_laz.y)
    full_cls = np.array(full_laz.classification)

    # Compute window statistics for 300x300m, 320x320m, 800x800m, 1000x1000m, 1200x1200m
    win_stats = {}
    yy_grid, xx_grid = np.mgrid[0:H, 0:W]
    wx_grid = (cx - 600.0) + (xx_grid + 0.5) * res_m
    wy_grid = (cy + 600.0) - (yy_grid + 0.5) * res_m

    for box_m in [300, 320, 800, 1000, 1200]:
        hm = box_m * 0.5
        in_pix = (np.abs(wx_grid - cx) <= hm) & (np.abs(wy_grid - cy) <= hm)
        valid_pix = in_pix & np.isfinite(dem)
        valid_area_m2 = float(valid_pix.sum() * res_m * res_m)

        m_ot = (np.abs(ot_x - cx) <= hm) & (np.abs(ot_y - cy) <= hm)
        m_na = (np.abs(na_x - cx) <= hm) & (np.abs(na_y - cy) <= hm)
        c2 = int((ot_cls[m_ot] == 2).sum())
        c8 = int((ot_cls[m_ot] == 8).sum())
        n_ot = int(m_ot.sum())
        n_na = int(m_na.sum())

        # Compute void distance map at 1m resolution inside valid swath
        g1m_n = box_m
        ox_rel = ot_x[m_ot] - (cx - hm)
        oy_rel = (cy + hm) - ot_y[m_ot]
        has_pt = np.zeros((g1m_n, g1m_n), dtype=bool)
        gi_x = np.clip(ox_rel.astype(int), 0, g1m_n - 1)
        gi_y = np.clip(oy_rel.astype(int), 0, g1m_n - 1)
        has_pt[gi_y, gi_x] = True
        void_dist_1m = ndi.distance_transform_edt(~has_pt)
        # Mask void_dist_1m to valid DEM swath
        dem_1m_sub = ndi.zoom(np.isfinite(dem[
            int((600 - hm) / res_m):int((600 + hm) / res_m),
            int((600 - hm) / res_m):int((600 + hm) / res_m),
        ]).astype(np.float32), g1m_n / max(1, int(2 * hm / res_m)), order=0) > 0.5
        vd_valid = void_dist_1m[dem_1m_sub[:g1m_n, :g1m_n]]

        win_stats[f"{box_m}x{box_m}m"] = {
            "box_area_m2": box_m * box_m,
            "valid_swath_area_m2": round(valid_area_m2, 1),
            "valid_swath_frac": round(valid_area_m2 / (box_m * box_m), 4),
            "laz_ground_pts_total": n_ot,
            "laz_class2_pts": c2,
            "laz_class8_pts": c8,
            "las_all_pts_total": n_na,
            "ground_density_valid_pts_m2": round(n_ot / max(valid_area_m2, 1.0), 4),
            "all_pulse_density_valid_pts_m2": round(n_na / max(valid_area_m2, 1.0), 3),
            "ground_penetration_pct": round(100.0 * n_ot / max(n_na, 1), 3),
            "void_dist_median_m": round(float(np.median(vd_valid)), 2) if vd_valid.size else None,
            "void_dist_p90_m": round(float(np.percentile(vd_valid, 90)), 2) if vd_valid.size else None,
            "void_dist_max_m": round(float(np.max(vd_valid)), 2) if vd_valid.size else None,
            "pct_area_gt_5m_from_ground_pt": round(100.0 * float((vd_valid > 5.0).mean()), 2) if vd_valid.size else None,
            "pct_area_gt_10m_from_ground_pt": round(100.0 * float((vd_valid > 10.0).mean()), 2) if vd_valid.size else None,
        }

    # Detailed feature-by-feature audit inside the 320x320m closeup around #08!
    hm = 160.0
    r0 = int(round((600.0 - hm) / res_m))
    r1 = int(round((600.0 + hm) / res_m))
    c0 = int(round((600.0 - hm) / res_m))
    c1 = int(round((600.0 + hm) / res_m))
    dem320 = dem[r0:r1, c0:c1]
    lrm320 = lrm[r0:r1, c0:c1]
    g1_320 = g1[r0:r1, c0:c1]
    h320, w320 = dem320.shape

    m_ot320 = (np.abs(ot_x - cx) <= hm) & (np.abs(ot_y - cy) <= hm)
    ox320 = ot_x[m_ot320]
    oy320 = ot_y[m_ot320]
    oz320 = ot_z[m_ot320]
    oc320 = ot_cls[m_ot320]

    # Map each ground point to (row, col) in dem320
    p_col = np.clip(((ox320 - (cx - hm)) / (2 * hm) * w320).astype(int), 0, w320 - 1)
    p_row = np.clip((((cy + hm) - oy320) / (2 * hm) * h320).astype(int), 0, h320 - 1)
    pt_count_grid = np.zeros((h320, w320), dtype=np.int32)
    np.add.at(pt_count_grid, (p_row, p_col), 1)
    dist_to_pt_m = ndi.distance_transform_edt(pt_count_grid == 0) * res_m

    # Label distinct positive LRM features (LRM >= 1.2m) inside 320x320m
    pos_mask = np.isfinite(lrm320) & (lrm320 >= 1.2)
    pos_mask = ndi.binary_opening(pos_mask, iterations=2)
    labeled, n_feat = ndi.label(pos_mask)
    features = []
    for lab in range(1, n_feat + 1):
        comp = labeled == lab
        area_m2 = float(comp.sum() * res_m * res_m)
        if area_m2 < 25.0:
            continue
        ys, xs = np.where(comp)
        f_pts = int(pt_count_grid[comp].sum())
        # Also count points within 2m collar of feature base
        dil = ndi.binary_dilation(comp, iterations=int(round(2.0 / res_m)))
        f_pts_collar = int(pt_count_grid[dil].sum())
        cy_f = (cy + hm) - (ys.mean() / h320) * (2 * hm)
        cx_f = (cx - hm) + (xs.mean() / w320) * (2 * hm)
        tr_inv = Transformer.from_crs(b600["crs"], "EPSG:4326", always_xy=True)
        flon, flat = tr_inv.transform(cx_f, cy_f)
        features.append(
            {
                "id": f"F{lab:02d}",
                "dx_m": round(float(cx_f - cx), 1),
                "dy_m": round(float(cy_f - cy), 1),
                "lat": round(float(flat), 6),
                "lon": round(float(flon), 6),
                "area_m2": round(area_m2, 1),
                "max_lrm_m": round(float(np.nanmax(lrm320[comp])), 2),
                "p90_lrm_m": round(float(np.nanpercentile(lrm320[comp], 90)), 2),
                "elev_max_m": round(float(np.nanmax(dem320[comp])), 2),
                "elev_min_m": round(float(np.nanmin(dem320[comp])), 2),
                "laz_pts_strictly_on_feature": f_pts,
                "laz_pts_with_2m_collar": f_pts_collar,
                "pts_per_m2_on_feature": round(f_pts / area_m2, 4),
                "median_dist_to_pt_m": round(float(np.median(dist_to_pt_m[comp])), 2),
                "max_dist_to_pt_m": round(float(np.max(dist_to_pt_m[comp])), 2),
            }
        )
    features.sort(key=lambda x: x["area_m2"] * x["max_lrm_m"], reverse=True)

    # Render diagnostic 6-panel forensic audit figure for our immediate visual inspection!
    fig, axes = plt.subplots(2, 3, figsize=(21, 13.5), dpi=150, facecolor="#0d1117")
    ext320 = [-hm, hm, -hm, hm]
    ext600 = [-600, 600, -600, 600]

    # 1. Raw 0.33m G1 (320x320m)
    axes[0, 0].imshow(g1_320, extent=ext320, origin="upper")
    axes[0, 0].set_title("1. Clean 0.33m G1 (320x320m around #08)", color="white", fontweight="bold")

    # 2. 0.33m G1 + EVERY raw .laz Ground Point + Delaunay Long-Edge (>15m) Overlay!
    axes[0, 1].imshow(g1_320, extent=ext320, origin="upper")
    pts_rel = np.column_stack([ox320 - cx, oy320 - cy])
    if len(pts_rel) >= 4:
        tri = Delaunay(pts_rel)
        # Draw long triangle edges (>18m) in cyan/blue to expose TIN facets!
        edges = set()
        for simp in tri.simplices:
            for a, b_i in [(simp[0], simp[1]), (simp[1], simp[2]), (simp[2], simp[0])]:
                e = (min(a, b_i), max(a, b_i))
                edges.add(e)
        segs_x, segs_y = [], []
        for a, b_i in edges:
            d = math.hypot(pts_rel[a, 0] - pts_rel[b_i, 0], pts_rel[a, 1] - pts_rel[b_i, 1])
            if d >= 18.0:
                segs_x.extend([pts_rel[a, 0], pts_rel[b_i, 0], np.nan])
                segs_y.extend([pts_rel[a, 1], pts_rel[b_i, 1], np.nan])
        axes[0, 1].plot(segs_x, segs_y, color="#00e5ff", lw=0.55, alpha=0.45, label="TIN edges >=18m")

    c2m = oc320 == 2
    c8m = oc320 == 8
    axes[0, 1].scatter(pts_rel[c2m, 0], pts_rel[c2m, 1], s=7, c="#ff3b30", alpha=0.9, linewidths=0.2, edgecolors="black", label=f"Class 2 Ground ({c2m.sum()})")
    axes[0, 1].scatter(pts_rel[c8m, 0], pts_rel[c8m, 1], s=11, c="#ffd60a", alpha=0.95, linewidths=0.3, edgecolors="black", label=f"Class 8 Key ({c8m.sum()})")
    for f in features[:10]:
        axes[0, 1].text(f["dx_m"], f["dy_m"], f"{f['id']}\n({f['laz_pts_strictly_on_feature']}p)", color="#4ade80", fontsize=7.5, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000aa", edgecolor="none"))
    axes[0, 1].legend(loc="lower right", fontsize=8, facecolor="#161b22", labelcolor="white")
    axes[0, 1].set_title(f"2. Raw .LAZ Ground Points (n={len(pts_rel)}) + Long TIN Edges (>=18m)", color="#ffd60a", fontweight="bold")

    # 3. 0.33m LRM + Ground Points colored by Z elevation
    im_lrm = axes[0, 2].imshow(lrm320, extent=ext320, origin="upper", cmap="inferno", vmin=-2.0, vmax=6.5)
    axes[0, 2].scatter(pts_rel[:, 0], pts_rel[:, 1], s=6, c="#00ffcc", alpha=0.85, linewidths=0)
    axes[0, 2].set_title("3. 0.33m LRM (-2 to +6.5m) + Cyan Ground Returns", color="white", fontweight="bold")

    # 4. Distance to nearest raw ground return (Void Map!)
    vd_plot = np.where(np.isfinite(dem320), dist_to_pt_m, np.nan)
    im_vd = axes[1, 0].imshow(vd_plot, extent=ext320, origin="upper", cmap="magma", vmin=0, vmax=22)
    plt.colorbar(im_vd, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Meters to nearest .laz ground point")
    axes[1, 0].contour(
        np.linspace(-hm, hm, w320),
        np.linspace(hm, -hm, h320),
        np.nan_to_num(lrm320, nan=0.0),
        levels=[1.5, 3.5],
        colors=["#4ade80", "#ffffff"],
        linewidths=0.9,
    )
    axes[1, 0].set_title("4. Canopy Void Distance Map + LRM Contours (1.5m, 3.5m)", color="white", fontweight="bold")

    # 5. Raw DEM Elevation + Contours + Ground Points colored by actual Z!
    vmin_z = float(np.nanpercentile(dem320, 2))
    vmax_z = float(np.nanpercentile(dem320, 98))
    axes[1, 1].imshow(dem320, extent=ext320, origin="upper", cmap="terrain", vmin=vmin_z, vmax=vmax_z)
    sc_z = axes[1, 1].scatter(pts_rel[:, 0], pts_rel[:, 1], c=oz320, cmap="terrain", vmin=vmin_z, vmax=vmax_z, s=12, edgecolors="black", linewidths=0.35)
    plt.colorbar(sc_z, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Elevation (m ASL)")
    axes[1, 1].set_title(f"5. DEM + Raw .LAZ Z-Values ({vmin_z:.0f}-{vmax_z:.0f} m ASL)", color="white", fontweight="bold")

    # 6. Wider 1.2 x 1.2 km 0.33m G1 Context
    axes[1, 2].imshow(g1, extent=ext600, origin="upper")
    axes[1, 2].plot([-hm, hm, hm, -hm, -hm], [-hm, -hm, hm, hm, -hm], color="#ffd60a", lw=1.5, ls="--")
    axes[1, 2].set_title("6. Wider 1.2 x 1.2 km 0.33m G1 Swath Context", color="white", fontweight="bold")

    for ax in axes.ravel():
        ax.tick_params(colors="#cbd5e1", labelsize=8)

    diag_png = OUT_DIR / "diag_card08_raw_points_overlay.png"
    fig.tight_layout()
    fig.savefig(diag_png, facecolor="#0d1117")
    plt.close(fig)

    return {
        "strip_full_laz_points": int(len(full_x)),
        "strip_utm_bbox": [
            round(float(full_x.min()), 1),
            round(float(full_y.min()), 1),
            round(float(full_x.max()), 1),
            round(float(full_y.max()), 1),
        ],
        "card08_utm": [round(cx, 2), round(cy, 2)],
        "window_stats": win_stats,
        "features_320x320m": features[:18],
        "diag_png": str(diag_png),
    }


def main() -> None:
    inv = inventory_surrounding_strips_and_sites()
    (OUT_DIR / "lacandon_inventory.json").write_text(json.dumps(inv, indent=2))
    print(f"Found {len(inv['nearby_strips'])} nearby/Campeche strips, {len(inv['eaams_within_35km'])} EAAMS sites within 35km.")

    res08 = audit_card08()
    (OUT_DIR / "card08_forensic_stats.json").write_text(json.dumps(res08, indent=2))
    print("Card #08 Window Stats:")
    print(json.dumps(res08["window_stats"], indent=2))
    print("\nTop Features in 320x320m around #08:")
    for f in res08["features_320x320m"][:12]:
        print(" ", f)


if __name__ == "__main__":
    main()
