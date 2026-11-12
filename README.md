# Nuevo Jalisco & Outlying Three-Mound Terrace (`LC-01`) — 0.33 m Airborne LiDAR Rediscovery

[![License: MIT / CC BY 4.0](https://img.shields.io/badge/License-MIT%20%2F%20CC%20BY%204.0-green.svg)](LICENSE)
[![OpenTopography Source](https://img.shields.io/badge/OpenTopography-OTLAS.112026.32615.1-blue)](https://doi.org/10.5069/G9Z899N1)

High-resolution **`0.33 m` bare-earth airborne LiDAR** and **raw 3D `.laz`/`.las` point-cloud** characterization of the remote Late Classic Maya ceremonial center of **Nuevo Jalisco** (`16.719521° N, 91.168757° W`, Ocosingo, Chiapas — strip `ChiapsCampeche_NFI_l0s473`) and an unmapped **outlying three-mound terrace complex (`LC-01`, `16.721996° N, 91.164633° W`)** located **`520 m` east-northeast** under unbroken `25–38 m` Lacandón rainforest canopy.

![1981 Carrasco Jungle Plan Rotated 42° CW vs 0.33 m Bare-Earth LiDAR](media/carrasco-1981-plan-rotated-vs-0p33m-lidar.png)

---

## 1. Overview & Why This Was Resolved Now

- **Digitally "Missing" & Unvisited Since `2001`:** Only two brief on-foot INAH reconnaissance visits have ever been recorded at *Nuevo Jalisco* — **Ramón Carrasco Vargas in `1976`** (published **1981**) and **Alejandro Tovalín Ahumada's team in `2001`** (published **2004**, **25 years ago**; tourism permit denied, zero trails, `2.5-hour` bushwhack across 15 branches of the Río El Cedro, zero excavations). Because Carrasco's 1976 pre-GPS coordinate plotted **`3.98 km` off** and Tovalín's 2001 `NAD27` UTM fix sat in a 1,000-copy regional print journal (*Bolom* No. 1), *Nuevo Jalisco* was completely absent from `EAAMS` (`5,223` sites), `INAH SIC`, Tulane `MARI-GISLAB`, `OpenStreetMap`, and `Wikidata`.
- **The 2016 LiDAR Bottleneck (`Golden et al. 2016: 306`):** NASA Goddard's environmental **G-LiHT** imager flew strip `l0s473` over *Nuevo Jalisco* in April 2013. Evaluating that pass in *Journal of Archaeological Science: Reports* (**9**: 293–308), **Charles Golden, Andrew Scherer, Whittaker Schroder et al. (2016: 306)** noted verbatim that single-pass point density under high canopy and dense undergrowth made it difficult *"to fully resolve the major architecture at Nuevo Jalisco"* and prevented *"identify[ing] smaller structures on the landscape surrounding Nuevo Jalisco in the lidar data due to lack of sufficient ground points."*
- **The Nov 4, 2026 `0.33 m` Reprocessing (`Britton 2026`):** Re-classifying NASA's raw 2013 laser pulses from scratch with `TerraScan`'s adaptive TIN ground routine (`OTLAS.112026.32615.1`, [DOI: 10.5069/G9Z899N1](https://doi.org/10.5069/G9Z899N1)) snaps **both** *Nuevo Jalisco*'s ceremonial core and its outlying **`LC-01` three-mound terrace (`520 m` ENE)** into razor-sharp 3D relief.

![Golden et al. (2016: 305–306) Verbatim Excerpts on Nuevo Jalisco](media/golden-2016-nuevo-jalisco-excerpt.png)

---

## 2. Nuevo Jalisco Ceremonial Core (`16.719521° N, 91.168757° W`) — 1:1 Match to 1981 Jungle Drawings & 2004 Dimensions

![Nuevo Jalisco 1981 Sketches & 2004 INAH Dimensions vs 0.33 m Bare-Earth LiDAR](media/nuevo-jalisco-2004-literature-vs-lidar.jpg)

| Metric / Feature | 1976/1981 Carrasco Drawings & 2004 Tovalín Field Report | Nov 2026 `0.33 m` LiDAR + Raw `.LAZ` Ground Points |
| --- | --- | --- |
| **Coordinates** | Tovalín 2001 `NAD27` UTM `15Q 695,273 E, 1,849,232 N` $\rightarrow$ `16.71955° N, 91.16858° W` | **`16.71952° N, 91.16876° W`** (**`19.2 m`** from Tovalín's GPS pin at foot of Acropolis) |
| **Gran Plaza** | `80 × 70 m`, long axis oriented **`40°` E of N**, framed by 4 perimeter platforms (`1–3 m` tall) | **`~80 × 70 m`** (`4,614 m²` `< 2.0°` level floor), dominant azimuth **`39.0°–40.0°`**, SW mound **`+3.37 m`** (`192` `.laz` pts) |
| **NE Acropolis** | `70 × 25 m` basal platform, **`+7 m`** above plaza on SW front, **`-12 m`** rear NE retaining-wall drop | **`~70 × 25 m`** (`2,535 m²`), **`+8.69 m` LRM** front rise, **`-11.5 m`** rear NE drop, **`1,425` raw `.laz` ground pts** (`0.56 pts/m²`, zero voids) |
| **Summit Temples** | **4 vaulted buildings (`Edificios 1–4`)** with stepped *Ik'* doorways, roofcombs & stone column-altars | **4 discrete summit crown peaks** matching Carrasco's `1:100` plan (`Fig. 2`) building-for-building |
| **Dating** | **Late Classic (`c. AD 630–800`)** architecture; *Nuevo Jalisco Panels 1 & 2* dated **`16 Feb AD 717` – `14 May AD 732`** (*Sak Tz'i'* king `K'ab' Chan Te'`) | Hidden beneath **`99.0%` unbroken Lacandón rainforest** (`25 m` median, `35–38 m` emergent trees; zero roads or clearings) |

![Nuevo Jalisco Forensic Raw .LAZ Point-Cloud & Cross-Section Audit](media/nuevo-jalisco-forensic-deep-dive.jpg)

![3D Canopy-Strip Orbit of Nuevo Jalisco Acropolis & Gran Plaza](media/sim-nuevo-jalisco-acropolis.gif)

- **Full-Resolution 3D Canopy-Strip Orbit Video (`MP4`):** [`media/sim-nuevo-jalisco-acropolis.mp4`](media/sim-nuevo-jalisco-acropolis.mp4)

### Original Archival Drawings from Carrasco Vargas (1981, *Estudios de Cultura Maya* XIII)
- **Ceremonial Center Plan (`Fig. 2`, p. 345):** [`media/carrasco-1981-fig2-nuevo-jalisco-plan.png`](media/carrasco-1981-fig2-nuevo-jalisco-plan.png)
- **Isometric Jungle Perspective Sketch (`Fig. 3`, p. 346):** [`media/carrasco-1981-fig3-nuevo-jalisco-perspective-sketch.png`](media/carrasco-1981-fig3-nuevo-jalisco-perspective-sketch.png)

---

## 3. Outlying Three-Mound Terrace Group `LC-01` (`16.721996° N, 91.164633° W` — `520 m` ENE)

Visible in the upper-right of the `1.2 × 1.2 km` `l0s473` swath (`+437 m` East, `+278 m` North of *Nuevo Jalisco*'s Gran Plaza), **`LC-01`** sits under `100%` closed `29 m` rainforest right along the east-northeast approach bearing from *Lacanjá-Chansayab* where Tovalín et al. (2004: 55) noted passing *"varias plataformas pequeñas y de escasa altura, dispersas a lo largo del camino."*

![LC-01 Forensic Deep-Dive: 0.33 m Contours & Longitudinal/Transverse .LAZ Slices](media/lc01-three-mounds-forensic-deep-dive.jpg)

![3D Canopy-Strip Orbit of LC-01 Outlying Three-Mound Terrace](media/sim-lc01-three-mounds-terrace.gif)

- **Full-Resolution 3D Canopy-Strip Orbit Video (`MP4`):** [`media/sim-lc01-three-mounds-terrace.mp4`](media/sim-lc01-three-mounds-terrace.mp4)
- **Raw `.LAZ` Ground Coverage:** **`38,258` classified ground returns** (`28,905` Class 2 + `9,353` Class 8; `841,549` total `.las` canopy+ground returns) across the `320 × 320 m` window (`0%` mound voids `> 6 m`).
- **Longitudinal (`L → L'`) & Transverse (`T → T'`) Slices:** Rather than a continuous limestone ridge, `LC-01` resolves into **three prominent aligned mounds (`+3.47 m`, `+3.32 m`, and `+2.84 m`)** separated by `2.5–3.5 m` saddles — each passing **`360°` 8-azimuth DEM closure** — flanking a **`1,282 m²` (`< 2.0°`) leveled terrace bench** (`82%` confidence outlying Maya platform group; unmapped in prior surveys).

---

## 4. Repository Structure & Raw LiDAR Files (`QGIS` / `CloudCompare` / `Google Earth`)

All cropped `0.33 m` GeoTIFF rasters and compressed 3D `.laz` laser point clouds are stored directly in [`lidar_data/`](lidar_data/) (no Git LFS required):

```text
nuevo-jalisco-lacandon-lidar/
├── lidar_data/
│   ├── manifest.json                                   # Exact UTM/WGS84 bounds, CRS & point counts
│   ├── nuevo_jalisco_core/                             # 1.2 × 1.2 km swath covering BOTH #08 & LC-01!
│   │   ├── DEM_0p33m.tif                               # 0.33 m bare-earth DEM (UTM 15N + EGM96)
│   │   ├── G1_0p33m.tif                                # 0.33 m Britton G1 architectural relief GeoTIFF
│   │   ├── CHM_1m.tif                                  # 1.0 m NASA Canopy Height Model (up to 44.6 m)
│   │   ├── GND_Class2_8.laz                            # 204,808 bare-earth ground laser returns
│   │   └── CANOPY_AND_GROUND.laz                       # 3,495,846 full 3D canopy + ground returns
│   └── lc01_three_mounds/                              # 320 × 320 m tight crop over LC-01 (520 m ENE)
│       ├── DEM_0p33m.tif, G1_0p33m.tif, CHM_1m.tif
│       ├── GND_Class2_8.laz                            # 38,258 ground laser returns
│       └── CANOPY_AND_GROUND.laz                       # 841,549 full 3D canopy + ground returns
├── data/
│   ├── nuevo_jalisco_sites.kml                         # Google Earth pins (#08 Core, 2001 GPS, LC-01)
│   ├── nuevo_jalisco_sites.geojson                     # QGIS / ArcGIS vector points & metadata
│   ├── nuevo_jalisco_summary.csv                       # Clean summary metrics table
│   ├── nuevo_jalisco_core_forensic_stats.json          # Point-by-feature .laz counts on #08
│   └── lc01_three_mounds_forensic_audit.json           # 360° closure & terrace metrics (#08 & LC-01)
├── media/                                              # Side-by-side 1981/2004/2016 figures, zooms & 3D GIF/MP4
├── papers/                                             # Primary Spanish & English reference PDFs (1981–2020)
├── docs/
│   ├── NUEVO_JALISCO_HISTORY_AND_DATING.md             # Full 1950s–2026 visit history & AD 717–732 epigraphy
│   └── OUTREACH_GUIDE_GOLDEN_SCHERER.md                # Verbatim Golden 2016 citations & email template
└── scripts/                                            # Reproducible Python .laz audits & 3D renderers
```

---

## 5. Credits & Primary References

- **Airborne LiDAR:** Cook, B. D. et al. (2013), *NASA G-LiHT AMIGA-Carb Campaign* (Public Domain); **0.33 m Reprocessing:** Britton, B. (Nov 4, 2026), *G-LiHT Yucatan West 2013: Reprocessed Ground Points, DEMs and Visualizations*, OpenTopography ([10.5069/G9Z899N1](https://doi.org/10.5069/G9Z899N1), CC BY 4.0).
- **Archaeological Literature:** Carrasco Vargas, R. (1981), *Estudios de Cultura Maya* XIII: 339–348; Tovalín Ahumada, A., V. M. Ortiz Villarreal & A. Velázquez de León Collins (2004), *Bolom* 1: 55–89; Tovalín Ahumada, A. (2020), *BAR International Series* 2985: 228–254; Bíró, P. (2005), *Mesoweb*; Golden, C., T. Murtha, B. Cook, D. S. Shaffer, W. Schroder, E. J. Hermitt, O. Alcover Firpi & A. K. Scherer (2016), *JAS: Reports* 9: 293–308 ([10.1016/j.jasrep.2016.07.029](https://doi.org/10.1016/j.jasrep.2016.07.029)).
