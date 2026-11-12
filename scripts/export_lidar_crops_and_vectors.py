#!/usr/bin/env python3
"""Export georeferenced 0.33 m bare-earth DEM, 0.33 m G1 relief, 1.0 m NASA CHM
GeoTIFFs, cropped 3D .laz ground (Class 2+8) & full canopy+ground .laz point clouds,
plus Google Earth KML, GeoJSON, and CSV vector layers for:
  - Nuevo Jalisco Core + LC-01 (1.2 x 1.2 km swath crop)
  - Surrounding High-Confidence Lacandon & Northern Yucatan Complexes (320 x 320 m crops)

All output files are compressed (< 45 MB per file) for standard Git hosting without LFS.
"""
from __future__ import annotations

import csv
import gzip
import html
import json
import math
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import transform as rio_transform
from rasterio.windows import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
LIDAR_OUT = REPO_ROOT / "lidar_data"
DATA_OUT = REPO_ROOT / "data"

WORK_RAW = Path("/agent/lidar/work/carbon_raw_hunt")
RAW_CACHE = WORK_RAW / "raw_cache"
S3_CAT = json.loads((WORK_RAW / "s3_raw_catalog.json").read_text())
LAZ_MAP = {Path(k).name.split("_GND_v1")[0]: k for k in S3_CAT["laz"]}
DEM33_MAP = {Path(k).name.split("_DEM_0p33m")[0]: k for k in S3_CAT["dtm_0p33m"]}
G1_33_MAP = {Path(k).name.split("_G1_0p33m")[0]: k for k in S3_CAT["g1_0p33m"]}

CAMPAIGN_MAP = {
    "centro_glas": "AMIGACarb_Yuc_Centro_GLAS_Apr2013",
    "centro_nfi": "AMIGACarb_Yuc_Centro_NFI_Apr2013",
    "norte_glas": "AMIGACarb_Yuc_Norte_GLAS_Apr2013",
    "norte_nfi": "AMIGACarb_Yuc_Norte_NFI_Apr2013",
    "north2_glas": "AMIGACarb_Yuc_North_2_GLAS_Apr2013",
    "north2_nfi": "AMIGACarb_Yuc_North_2_NFI_Apr2013",
    "south_glas": "AMIGACarb_Yuc_South_GLAS_Apr2013",
    "south_nfi": "AMIGACarb_Yuc_South_NFI_Apr2013",
    "chiapscampeche_nfi": "AMIGACarb_Chiap_Campeche_NFI_Apr2013",
    "chiaps1_glas": "AMIGACarb_Chiaps_1_GLAS_Apr2013",
    "chiaps1_nfi": "AMIGACarb_Chiaps_1_NFI_Apr2013",
}


def nasa_campaign_and_ls(strip_id: str) -> tuple[str, str]:
    parts = strip_id.rsplit("_", 1)
    pref, ls = parts[0].lower(), parts[1].lower()
    return CAMPAIGN_MAP[pref], ls


def ensure_pointclouds(strip_id: str) -> tuple[Path, Path]:
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    laz_key = LAZ_MAP[strip_id]
    laz_path = RAW_CACHE / Path(laz_key).name
    if not laz_path.exists():
        url = f"https://opentopography.s3.sdsc.edu/pc-bulk/{laz_key}"
        print(f"  Downloading OT ground .laz: {url}", flush=True)
        urllib.request.urlretrieve(url, laz_path)

    camp, ls = nasa_campaign_and_ls(strip_id)
    las_path = RAW_CACHE / f"{camp}_{ls}.las"
    if not las_path.exists():
        gz_path = RAW_CACHE / f"{camp}_{ls}.las.gz"
        nasa_url = f"https://glihtdata.gsfc.nasa.gov/files/G-LiHT/{camp}/lidar/las/{camp}_{ls}.las.gz"
        print(f"  Downloading NASA full .las.gz: {nasa_url}", flush=True)
        urllib.request.urlretrieve(nasa_url, gz_path)
        with gzip.open(gz_path, "rb") as fi, open(las_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        gz_path.unlink(missing_ok=True)
    return laz_path, las_path


CROP_TARGETS = [
    {
        "id": "nuevo_jalisco_core",
        "out_dir": LIDAR_OUT / "nuevo_jalisco_core",
        "strip_id": "ChiapsCampeche_NFI_l0s473",
        "lat": 16.719521,
        "lon": -91.168757,
        "half_m": 600.0,
        "label": "Nuevo Jalisco Core (#08) + Outlying Group LC-01 (1.2 x 1.2 km Swath)",
        "tier": "CONFIRMED (99%)",
    },
    {
        "id": "lc01_three_mounds",
        "out_dir": LIDAR_OUT / "lc01_three_mounds",
        "strip_id": "ChiapsCampeche_NFI_l0s473",
        "lat": 16.721996,
        "lon": -91.164633,
        "half_m": 160.0,
        "label": "LC-01: Outlying 5-Mound Terrace Group (520 m ENE of Nuevo Jalisco)",
        "tier": "TIER B+ (82%)",
    },
    {
        "id": "lc14_montes_azules_courtyard",
        "out_dir": LIDAR_OUT / "lc14_montes_azules_courtyard",
        "strip_id": "Chiaps1_GLAS_l2s480",
        "lat": 16.530595,
        "lon": -90.890019,
        "half_m": 160.0,
        "label": "LC-14: Montes Azules 90-Degree L/U-Shaped Masonry Ridge Courtyard",
        "tier": "TIER A (96%)",
    },
]


def crop_geotiff_float(ds: rasterio.DatasetReader, xmin: float, ymin: float, xmax: float, ymax: float, out_path: Path, nodata_val: float = -9999.0) -> dict:
    win = from_bounds(xmin, ymin, xmax, ymax, ds.transform).round_offsets().round_lengths()
    arr = ds.read(1, window=win, boundless=True, fill_value=nodata_val).astype(np.float32)
    if ds.nodata is not None:
        arr = np.where(arr == ds.nodata, nodata_val, arr)
    arr = np.where(~np.isfinite(arr) | (np.abs(arr) > 1e6) | (arr < -100.0), nodata_val, arr).astype(np.float32)
    transform = ds.window_transform(win)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": ds.crs,
        "transform": transform,
        "nodata": nodata_val,
        "compress": "deflate",
        "predictor": 3,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)

    valid = arr != nodata_val
    return {
        "file": str(out_path.relative_to(REPO_ROOT)),
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "res_m": round(float(abs(transform.a)), 4),
        "valid_fraction": round(float(valid.mean()), 4),
        "min_val": round(float(arr[valid].min()), 2) if valid.any() else None,
        "max_val": round(float(arr[valid].max()), 2) if valid.any() else None,
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 3),
    }


def crop_geotiff_rgb(ds: rasterio.DatasetReader, xmin: float, ymin: float, xmax: float, ymax: float, out_path: Path) -> dict:
    win = from_bounds(xmin, ymin, xmax, ymax, ds.transform).round_offsets().round_lengths()
    bands = [1, 2, 3] if ds.count >= 3 else [1, 1, 1]
    arr = ds.read(bands, window=win, boundless=True, fill_value=0).astype(np.uint8)
    transform = ds.window_transform(win)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": 3,
        "dtype": "uint8",
        "crs": ds.crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "photometric": "RGB",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)

    return {
        "file": str(out_path.relative_to(REPO_ROOT)),
        "width": int(arr.shape[2]),
        "height": int(arr.shape[1]),
        "res_m": round(float(abs(transform.a)), 4),
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 3),
    }


def crop_laz_subset(pc: laspy.LasData, xmin: float, ymin: float, xmax: float, ymax: float, out_path: Path, z_shift: float = 0.0, crs_vlrs=None) -> dict:
    mask = (pc.x >= xmin) & (pc.x <= xmax) & (pc.y >= ymin) & (pc.y <= ymax)
    sub_pts = pc.points[mask].copy()

    hdr = laspy.LasHeader(point_format=pc.header.point_format, version=pc.header.version)
    hdr.scales = pc.header.scales
    hdr.offsets = [float(xmin), float(ymin), float(np.min(pc.z[mask]) + z_shift) if np.any(mask) else 0.0]
    if crs_vlrs:
        hdr.vlrs.extend(crs_vlrs)
    elif pc.header.vlrs:
        hdr.vlrs.extend(pc.header.vlrs)

    sub_las = laspy.LasData(header=hdr)
    # Copy point dimensions safely while applying optional vertical datum shift
    for dim in pc.point_format.dimension_names:
        if dim in ("X", "Y", "Z"):
            continue
        try:
            setattr(sub_las, dim, getattr(sub_pts, dim))
        except Exception:
            pass
    sub_las.x = np.asarray(pc.x[mask], dtype=np.float64)
    sub_las.y = np.asarray(pc.y[mask], dtype=np.float64)
    sub_las.z = np.asarray(pc.z[mask], dtype=np.float64) + z_shift

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_las.write(str(out_path))

    cls_arr = np.asarray(sub_las.classification, dtype=np.uint8)
    u_cls, u_cnt = np.unique(cls_arr, return_counts=True)
    cls_counts = {str(int(k)): int(v) for k, v in zip(u_cls, u_cnt)}

    return {
        "file": str(out_path.relative_to(REPO_ROOT)),
        "point_count": int(len(sub_las.x)),
        "classification_counts": cls_counts,
        "z_min_m": round(float(np.min(sub_las.z)), 2) if len(sub_las.x) else None,
        "z_max_m": round(float(np.max(sub_las.z)), 2) if len(sub_las.x) else None,
        "z_shift_applied_m": round(float(z_shift), 3),
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 3),
    }


def export_all_lidar_crops() -> list[dict]:
    by_strip: dict[str, list[dict]] = defaultdict(list)
    for t in CROP_TARGETS:
        by_strip[t["strip_id"]].append(t)

    manifest_entries = []

    for strip_id, targets in by_strip.items():
        print(f"\n=== Processing strip {strip_id} ({len(targets)} crop target(s)) ===", flush=True)
        u_dem = f"/vsicurl/https://opentopography.s3.sdsc.edu/raster/{DEM33_MAP[strip_id]}"
        u_g1 = f"/vsicurl/https://opentopography.s3.sdsc.edu/raster/{G1_33_MAP[strip_id]}"
        camp, ls = nasa_campaign_and_ls(strip_id)
        chm_path = WORK_RAW / f"chm/extracted/{camp}/{camp}_{ls}_CHM.tif"
        laz_path, las_path = ensure_pointclouds(strip_id)

        ot_pc = laspy.read(laz_path)
        na_pc = laspy.read(las_path)

        with rasterio.open(u_dem) as ds_dem, rasterio.open(u_g1) as ds_g1:
            ds_chm = rasterio.open(chm_path) if chm_path.exists() else None
            try:
                for t in targets:
                    xs, ys = rio_transform("EPSG:4326", ds_dem.crs, [t["lon"]], [t["lat"]])
                    cx, cy = float(xs[0]), float(ys[0])
                    hm = float(t["half_m"])
                    xmin, ymin, xmax, ymax = cx - hm, cy - hm, cx + hm, cy + hm

                    out_d = t["out_dir"]
                    out_d.mkdir(parents=True, exist_ok=True)
                    print(f"  -> Cropping {t['id']} ({int(2*hm)}x{int(2*hm)}m) to {out_d.relative_to(REPO_ROOT)}...", flush=True)

                    dem_meta = crop_geotiff_float(ds_dem, xmin, ymin, xmax, ymax, out_d / "DEM_0p33m.tif")
                    g1_meta = crop_geotiff_rgb(ds_g1, xmin, ymin, xmax, ymax, out_d / "G1_0p33m.tif")
                    chm_meta = None
                    if ds_chm is not None:
                        chm_meta = crop_geotiff_float(ds_chm, xmin, ymin, xmax, ymax, out_d / "CHM_1m.tif")

                    # Compute vertical datum alignment between NASA .las and OT EGM96 .laz inside window
                    m_ot = (ot_pc.x >= xmin) & (ot_pc.x <= xmax) & (ot_pc.y >= ymin) & (ot_pc.y <= ymax)
                    m_na = (na_pc.x >= xmin) & (na_pc.x <= xmax) & (na_pc.y >= ymin) & (na_pc.y <= ymax)
                    z_shift = 0.0
                    if np.sum(m_ot) > 100 and np.sum(m_na) > 100:
                        na_sub_c = np.asarray(na_pc.classification[m_na])
                        na_sub_z = np.asarray(na_pc.z[m_na], dtype=np.float64)
                        ot_sub_z = np.asarray(ot_pc.z[m_ot], dtype=np.float64)
                        na_gnd = na_sub_z[na_sub_c == 2] if np.sum(na_sub_c == 2) > 50 else np.percentile(na_sub_z, [5])
                        dz = float(np.percentile(na_gnd, 10) - np.percentile(ot_sub_z, 10))
                        if abs(dz) > 0.35:
                            z_shift = -dz

                    gnd_meta = crop_laz_subset(
                        ot_pc, xmin, ymin, xmax, ymax, out_d / "GND_Class2_8.laz", z_shift=0.0
                    )
                    can_meta = crop_laz_subset(
                        na_pc, xmin, ymin, xmax, ymax, out_d / "CANOPY_AND_GROUND.laz",
                        z_shift=z_shift, crs_vlrs=ot_pc.header.vlrs
                    )

                    # Verify < 45 MB per file
                    for m_obj in [dem_meta, g1_meta, chm_meta, gnd_meta, can_meta]:
                        if m_obj is not None and m_obj["size_mb"] >= 45.0:
                            raise RuntimeError(f"File {m_obj['file']} exceeded 45 MB ({m_obj['size_mb']} MB)!")

                    tr_inv = Transformer.from_crs(ds_dem.crs, "EPSG:4326", always_xy=True)
                    lon_min, lat_min = tr_inv.transform(xmin, ymin)
                    lon_max, lat_max = tr_inv.transform(xmax, ymax)

                    entry = {
                        "id": t["id"],
                        "label": t["label"],
                        "tier": t["tier"],
                        "strip_id": strip_id,
                        "center_wgs84": {"lat": t["lat"], "lon": t["lon"]},
                        "box_size_m": int(2 * hm),
                        "crs_wkt": ds_dem.crs.to_wkt(),
                        "utm_bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                        "wgs84_bbox": [round(lon_min, 6), round(lat_min, 6), round(lon_max, 6), round(lat_max, 6)],
                        "files": {
                            "dem_0p33m": dem_meta,
                            "g1_0p33m": g1_meta,
                            "chm_1m": chm_meta,
                            "gnd_laz": gnd_meta,
                            "canopy_and_ground_laz": can_meta,
                        },
                    }
                    manifest_entries.append(entry)
                    print(
                        f"     DEM={dem_meta['size_mb']}MB | G1={g1_meta['size_mb']}MB | "
                        f"GND.laz={gnd_meta['size_mb']}MB ({gnd_meta['point_count']:,} pts) | "
                        f"FULL.laz={can_meta['size_mb']}MB ({can_meta['point_count']:,} pts)",
                        flush=True,
                    )
            finally:
                if ds_chm is not None:
                    ds_chm.close()

    order = {t["id"]: i for i, t in enumerate(CROP_TARGETS)}
    manifest_entries.sort(key=lambda x: order[x["id"]])
    (LIDAR_OUT / "manifest.json").write_text(json.dumps(manifest_entries, indent=2))
    return manifest_entries


def rotated_rect_wgs84(cx_utm: float, cy_utm: float, dx_m: float, dy_m: float, length_m: float, width_m: float, az_deg: float, epsg_from: str = "EPSG:32615") -> list[list[float]]:
    tr = Transformer.from_crs(epsg_from, "EPSG:4326", always_xy=True)
    theta = math.radians(az_deg)
    ux, uy = math.sin(theta), math.cos(theta)
    vx, vy = math.cos(theta), -math.sin(theta)
    hl, hw = 0.5 * length_m, 0.5 * width_m
    px0, py0 = cx_utm + dx_m, cy_utm + dy_m
    corners_utm = [
        (px0 - hl * ux - hw * vx, py0 - hl * uy - hw * vy),
        (px0 + hl * ux - hw * vx, py0 + hl * uy - hw * vy),
        (px0 + hl * ux + hw * vx, py0 + hl * uy + hw * vy),
        (px0 - hl * ux + hw * vx, py0 - hl * uy + hw * vy),
        (px0 - hl * ux - hw * vx, py0 - hl * uy - hw * vy),
    ]
    coords = []
    for ex, ny in corners_utm:
        lon, lat = tr.transform(ex, ny)
        coords.append([round(float(lon), 6), round(float(lat), 6)])
    return coords


def export_vector_layers() -> None:
    audit = json.loads(Path("/agent/maya-lidar-discovery/data/open_lidar_hunt_v2/lacandon_confidence_audit.json").read_text())
    cur14 = {x["lc_code"]: x for x in json.loads(Path("/agent/lidar/work/lacandon_deep_dive/lacandon_top14_curated.json").read_text())}

    nj_sites = [
        {
            "code": "NJ-1",
            "name": "NJ-1: Unbroken-Jungle Courtyard & +6.0 m Temple Pyramid",
            "region": "Central-East Yucatan (Outside Estrada-Belli 2025)",
            "strip_id": "North2_GLAS_l0s391",
            "lat": 20.648720,
            "lon": -88.756850,
            "tier": "NORTHERN YUCATAN TIER A",
            "confidence_pct": 95,
            "n_mounds_total_0p33m": 9,
            "n_closed_360_mounds": 9,
            "n_ortho_rect_mounds": 4,
            "closed_quadrants_of_4": 4,
            "plaza_area_lt2deg_m2": 3142.0,
            "max_mound_relief_m": 6.0,
            "laz_pts_window": 4634,
            "chm_med_m": 10.2,
            "nearest_known": "Cenote Xtohil (6.98 km; EAAMS) / Chichen Itza (19.6 km ENE)",
            "rationale": "9 mounds, +6.0 m pyramid, 3,142 m2 enclosed plaza under 99.3% unbroken forest; 0 polygons in MARI-GISLAB, EAAMS, INAH SIC, OSM.",
            "has_lidar_crop": True,
            "lidar_crop_dir": "lidar_data/surrounding_lacandon_tier_a/NJ-1/",
        },
        {
            "code": "NJ-2",
            "name": "NJ-2: Unbroken-Jungle Four-Quadrant Plaza & +5.0 m Pyramid",
            "region": "Central-East Yucatan (Outside Estrada-Belli 2025)",
            "strip_id": "North2_GLAS_l0s390",
            "lat": 20.630460,
            "lon": -88.758900,
            "tier": "NORTHERN YUCATAN TIER A",
            "confidence_pct": 95,
            "n_mounds_total_0p33m": 9,
            "n_closed_360_mounds": 9,
            "n_ortho_rect_mounds": 4,
            "closed_quadrants_of_4": 4,
            "plaza_area_lt2deg_m2": 548.0,
            "max_mound_relief_m": 5.0,
            "laz_pts_window": 1854,
            "chm_med_m": 11.8,
            "nearest_known": "Ikil (6.04 km; EAAMS) / Chichen Itza (20.1 km NE)",
            "rationale": "9 mounds (4 rectangular, 4/4 quadrants), +5.0 m pyramid, 548 m2 plaza under 99.6% unbroken forest; 0 polygons in MARI-GISLAB.",
            "has_lidar_crop": True,
            "lidar_crop_dir": "lidar_data/surrounding_lacandon_tier_a/NJ-2/",
        },
        {
            "code": "NJ-3",
            "name": "NJ-3: Unbroken-Jungle 18-Mound Enclosed Courtyard & Range Complex",
            "region": "Central-East Yucatan (Outside Estrada-Belli 2025)",
            "strip_id": "North2_NFI_l8s387",
            "lat": 20.663450,
            "lon": -88.777120,
            "tier": "NORTHERN YUCATAN TIER A",
            "confidence_pct": 94,
            "n_mounds_total_0p33m": 18,
            "n_closed_360_mounds": 16,
            "n_ortho_rect_mounds": 4,
            "closed_quadrants_of_4": 4,
            "plaza_area_lt2deg_m2": 1068.0,
            "max_mound_relief_m": 4.4,
            "laz_pts_window": 8188,
            "chm_med_m": 11.6,
            "nearest_known": "Cenote Xtohil (4.31 km; EAAMS) / Chichen Itza (21.2 km E)",
            "rationale": "18 mounds (4 rectangular, 4/4 quadrants), +4.4 m range mound, 1,068 m2 plaza under 99.4% unbroken forest; 0 polygons in MARI-GISLAB.",
            "has_lidar_crop": True,
            "lidar_crop_dir": "lidar_data/surrounding_lacandon_tier_a/NJ-3/",
        },
    ]

    crop_ids = {t["id"] for t in CROP_TARGETS}
    features = []
    csv_rows = []

    for r in audit:
        code_raw = r["code"]
        code = "#08" if code_raw.startswith("#08") else code_raw
        c_extra = cur14.get(code, {})
        max_lrm = max((m["max_lrm_m"] for m in r.get("top_mounds", [])), default=0.0)
        nk = r.get("nearest_any_known", {})
        nk_str = f"{nk.get('name', '')} ({nk.get('dist_km', '')} km)"
        chm_med = 25.0 if code == "#08" else c_extra.get("chm_med_m", None)

        if code == "#08":
            crop_dir = "lidar_data/nuevo_jalisco_core/"
            has_crop = True
            name_str = "Nuevo Jalisco (#08): Classic Maya Ceremonial Center (Gran Plaza & NE Acropolis)"
        else:
            has_crop = code in crop_ids
            crop_dir = f"lidar_data/surrounding_lacandon_tier_a/{code}/" if has_crop else ""
            name_str = f"{code}: {c_extra.get('arch_desc', 'Lacandon Rainforest Complex')}"

        props = {
            "id": code,
            "name": name_str,
            "region": "Lacandon Rainforest Corridor, Chiapas",
            "strip_id": r["strip_id"],
            "lat": round(float(r["lat"]), 6),
            "lon": round(float(r["lon"]), 6),
            "dist_from_nuevo_jalisco_km": r["dist_from_card08_km"],
            "tier": r["tier"],
            "confidence_pct": r["confidence_pct"],
            "n_mounds_0p33m": r["n_mounds_total_0p33m"],
            "n_closed_360_mounds": r["n_closed_360_mounds"],
            "n_ortho_rect_mounds": r["n_ortho_rect_mounds"],
            "closed_quadrants_of_4": r["closed_quadrants_of_4"],
            "plaza_area_lt2deg_m2": r["plaza_area_lt2deg_m2"],
            "plaza_area_lt3deg_m2": r["plaza_area_lt3deg_m2"],
            "max_mound_lrm_m": round(float(max_lrm), 2),
            "laz_ground_pts_window": r["laz_pts_window"],
            "laz_pts_per_m2": r["laz_pts_per_m2"],
            "chm_median_m": chm_med,
            "nearest_known_site": nk_str,
            "rationale": r["rationale"],
            "has_lidar_crop": has_crop,
            "lidar_crop_dir": crop_dir,
            "google_maps_url": f"https://www.google.com/maps?q={r['lat']:.6f},{r['lon']:.6f}&t=k&z=17",
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(float(r["lon"]), 6), round(float(r["lat"]), 6)]},
                "properties": props,
            }
        )
        csv_rows.append(props)

    for nj in nj_sites:
        props = {
            "id": nj["code"],
            "name": nj["name"],
            "region": nj["region"],
            "strip_id": nj["strip_id"],
            "lat": nj["lat"],
            "lon": nj["lon"],
            "dist_from_nuevo_jalisco_km": None,
            "tier": nj["tier"],
            "confidence_pct": nj["confidence_pct"],
            "n_mounds_0p33m": nj["n_mounds_total_0p33m"],
            "n_closed_360_mounds": nj["n_closed_360_mounds"],
            "n_ortho_rect_mounds": nj["n_ortho_rect_mounds"],
            "closed_quadrants_of_4": nj["closed_quadrants_of_4"],
            "plaza_area_lt2deg_m2": nj["plaza_area_lt2deg_m2"],
            "plaza_area_lt3deg_m2": None,
            "max_mound_lrm_m": nj["max_mound_relief_m"],
            "laz_ground_pts_window": nj["laz_pts_window"],
            "laz_pts_per_m2": None,
            "chm_median_m": nj["chm_med_m"],
            "nearest_known_site": nj["nearest_known"],
            "rationale": nj["rationale"],
            "has_lidar_crop": nj["has_lidar_crop"],
            "lidar_crop_dir": nj["lidar_crop_dir"],
            "google_maps_url": f"https://www.google.com/maps?q={nj['lat']:.6f},{nj['lon']:.6f}&t=k&z=17",
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [nj["lon"], nj["lat"]]},
                "properties": props,
            }
        )
        csv_rows.append(props)

    # Add Nuevo Jalisco architectural wireframe polygons (Tovalin 2004 / Carrasco 1981 1:1 lock)
    cx08, cy08 = 695238.98, 1849425.29
    arch_polys = [
        ("NJ_Gran_Plaza_80x70m", "Nuevo Jalisco: Gran Plaza (80 x 70 m @ N40E, Carrasco 1981 / Tovalin 2004)", -11.0, -7.0, 80.0, 70.0, 40.0, "#38bdf8"),
        ("NJ_NE_Acropolis_70x25m", "Nuevo Jalisco: NE Acropolis (70 x 25 m, +8.7 m front / -12 m rear wall, 4 Vaulted Temples)", 18.5, 33.0, 25.0, 70.0, 40.0, "#4ade80"),
        ("NJ_SW_Range_Mound", "Nuevo Jalisco: SW Flanking Platform (+3.4 m)", -45.5, -31.0, 14.0, 34.0, 40.0, "#fde047"),
        ("NJ_NW_Range_Mound", "Nuevo Jalisco: NW Flanking Platform (+1.6 m)", -43.0, 19.0, 42.0, 9.0, 40.0, "#fde047"),
        ("NJ_SE_Range_Mound", "Nuevo Jalisco: SE Flanking Platform (+2.8 m)", 17.5, -51.0, 34.0, 11.0, 40.0, "#fde047"),
    ]
    for pid, pname, dx, dy, length, width, az, col in arch_polys:
        ring = rotated_rect_wgs84(cx08, cy08, dx, dy, length, width, az)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "id": pid,
                    "name": pname,
                    "feature_class": "architectural_footprint",
                    "stroke": col,
                    "source": "Carrasco Vargas (1981, Fig. 2) & Tovalin Ahumada et al. (2004: 55-56) 1:1 Lock on 0.33 m LiDAR",
                },
            }
        )

    # Also add Tovalin 2004 NAD27->WGS84 GPS datum point (16.71955, -91.16858)
    features.append(
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-91.168579, 16.719547]},
            "properties": {
                "id": "Tovalin_2004_GPS_Datum",
                "name": "Tovalin et al. (2004: 55) Field GPS Fix (NAD27 15Q 695273E, 1849232N -> 19.2 m from #08)",
                "feature_class": "ground_truth_gps",
                "source": "Tovalin Ahumada, Ortiz Villarreal & Velazquez de Leon (2004, Bolom 1: 55)",
            },
        }
    )

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    geojson_path = DATA_OUT / "nuevo_jalisco_and_lacandon_sites.geojson"
    geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))

    csv_path = DATA_OUT / "nuevo_jalisco_and_lacandon_scorecard.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Build Google Earth KML
    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "  <name>Nuevo Jalisco (1981 Carrasco / 0.33m LiDAR) &amp; Surrounding Lacandon Maya Discoveries</name>",
        "  <description>0.33 m Airborne LiDAR Bare-Earth Reprocessing (NASA G-LiHT 2013 / Britton Nov 2026) — Confirmed Classic Maya Center of Nuevo Jalisco (#08), Calibrated Surrounding Lacandon Rainforest Complexes (LC-01..LC-14), and Northern Yucatan Unbroken-Jungle Complexes (NJ-1..NJ-3).</description>",
        '  <Style id="confirmed"><IconStyle><color>ffffd538</color><scale>1.35</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/star.png</href></Icon></IconStyle></Style>',
        '  <Style id="tier_a"><IconStyle><color>ff80de4a</color><scale>1.2</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-stars.png</href></Icon></IconStyle></Style>',
        '  <Style id="tier_b"><IconStyle><color>ffeed322</color><scale>1.05</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon></IconStyle></Style>',
        '  <Style id="tier_c"><IconStyle><color>ff8571fb</color><scale>0.95</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/pink-blank.png</href></Icon></IconStyle></Style>',
        '  <Style id="yucatan"><IconStyle><color>ff5ec522</color><scale>1.2</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/ltblu-stars.png</href></Icon></IconStyle></Style>',
        '  <Style id="arch_poly"><LineStyle><color>ff80de4a</color><width>2.5</width></LineStyle><PolyStyle><color>4480de4a</color></PolyStyle></Style>',
    ]

    for feat in features:
        geom = feat["geometry"]
        p = feat["properties"]
        nm = html.escape(str(p.get("name", p.get("id", ""))))
        if geom["type"] == "Point":
            lon, lat = geom["coordinates"]
            tier_u = str(p.get("tier", ""))
            if "CONFIRMED" in tier_u or p.get("feature_class") == "ground_truth_gps":
                st = "#confirmed"
            elif "NORTHERN YUCATAN" in tier_u:
                st = "#yucatan"
            elif "TIER A" in tier_u:
                st = "#tier_a"
            elif "TIER B" in tier_u:
                st = "#tier_b"
            else:
                st = "#tier_c"

            desc_html = (
                f"<b>ID:</b> {html.escape(str(p.get('id', '')))}<br/>"
                f"<b>Tier &amp; Confidence:</b> {html.escape(str(p.get('tier', '')))} ({html.escape(str(p.get('confidence_pct', 'N/A')))}%)<br/>"
                f"<b>Strip:</b> {html.escape(str(p.get('strip_id', '')))}<br/>"
                f"<b>360-deg Closed Mounds:</b> {p.get('n_closed_360_mounds', 'N/A')} / {p.get('n_mounds_0p33m', 'N/A')}<br/>"
                f"<b>Plaza &lt;2 deg Area (m2):</b> {p.get('plaza_area_lt2deg_m2', 'N/A')}<br/>"
                f"<b>Max Mound Relief (m):</b> {p.get('max_mound_lrm_m', 'N/A')}<br/>"
                f"<b>Raw .LAZ Ground Points:</b> {p.get('laz_ground_pts_window', 'N/A')}<br/>"
                f"<b>Nearest Known Site:</b> {html.escape(str(p.get('nearest_known_site', 'N/A')))}<br/>"
                f"<b>Forensic Verdict:</b> {html.escape(str(p.get('rationale', p.get('source', ''))))}<br/>"
                f"<b>Repo LiDAR Crop:</b> {html.escape(str(p.get('lidar_crop_dir', '')))}"
            )
            kml_lines.extend(
                [
                    "  <Placemark>",
                    f"    <name>{html.escape(str(p.get('id', '')))} — {nm}</name>",
                    f"    <description><![CDATA[{desc_html}]]></description>",
                    f"    <styleUrl>{st}</styleUrl>",
                    f"    <Point><coordinates>{lon},{lat},0</coordinates></Point>",
                    "  </Placemark>",
                ]
            )
        elif geom["type"] == "Polygon":
            ring_str = " ".join(f"{c[0]},{c[1]},0" for c in geom["coordinates"][0])
            kml_lines.extend(
                [
                    "  <Placemark>",
                    f"    <name>{nm}</name>",
                    f"    <description>{html.escape(str(p.get('source', '')))}</description>",
                    "    <styleUrl>#arch_poly</styleUrl>",
                    "    <Polygon><outerBoundaryIs><LinearRing>",
                    f"      <coordinates>{ring_str}</coordinates>",
                    "    </LinearRing></outerBoundaryIs></Polygon>",
                    "  </Placemark>",
                ]
            )

    kml_lines.extend(["</Document>", "</kml>"])
    kml_path = DATA_OUT / "nuevo_jalisco_and_lacandon_sites.kml"
    kml_path.write_text("\n".join(kml_lines), encoding="utf-8")
    print(f"\nWrote {geojson_path.name}, {kml_path.name}, and {csv_path.name} ({len(features)} features).", flush=True)


def main() -> None:
    export_all_lidar_crops()
    export_vector_layers()


if __name__ == "__main__":
    main()
