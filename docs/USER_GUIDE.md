# User Guide

[English](USER_GUIDE.md) | [한국어](USER_GUIDE.ko.md)

**Version:** 0.1.0

## 1. Purpose and Safety

Pedicle Screw Simulator is a research and education desktop application for reviewing CT data, segmenting vertebrae, planning pedicle screws, and inspecting screw-aligned MPR images.

> This software is not a certified medical device. Do not use it as the sole basis for diagnosis, surgery, navigation, or patient care. A qualified clinician must independently verify every result.

Do not publish clinical DICOM files or identifying screenshots. See [Local DICOM Data](../data/README.md).

## 2. Requirements

- Python 3.12
- macOS, Linux, or Windows for the provided launcher scripts
- Display area of at least 1280×760
- Sufficient RAM for the CT volume
- Optional CUDA-capable NVIDIA GPU for faster TotalSegmentator inference

Standalone builds are provided for Apple Silicon macOS and Windows x64. Intel macOS is not supported.

## 3. Install and Launch

```bash
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
./scripts/run_app.sh
```

On Windows, use the PowerShell launcher instead:

```powershell
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

Optional TotalSegmentator installation:

```bash
./scripts/run_app.sh --with-totalseg
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --with-totalseg
```

Maintenance commands:

```bash
./scripts/run_app.sh --check
./scripts/run_app.sh --test
./scripts/run_app.sh --setup-only
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --check
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --test
```

### Standalone macOS and Windows packages

Standalone packages do not require a separate Python installation. Download the archive produced by the GitHub **Build desktop packages** workflow, extract the complete folder, and run the `.app` or `.exe`. Keep the Windows `_internal` folder beside the executable.

The standalone package includes TotalSegmentator, PyTorch, and nnU-Net. The first automatic segmentation downloads the open `total` task model weights and later runs reuse the local cache. See [Desktop Build Guide](BUILDING_DESKTOP.md).

## 4. Workspace

The Planning workspace contains:

- **Axial, Sagittal, Coronal MPR:** synchronized CT sections with crosshairs, segmentation, measurements, and screw overlays
- **3D viewport:** CT volume, vertebral meshes, screws, and optional MPR planes
- **Workflow panel:** study, segmentation, planning, selected screw, and validation controls
- **Tools palette:** Select, Add Screw, Distance, and Angle

Use **Planning** for a large 3D view and compact MPR panels. Use **MPR Focus** for a larger 2×2 review layout.

## 5. Standard Planning Workflow

### 5.1 Open a CT Study

1. Select **Open DICOM Folder** or press `Ctrl+O`.
2. Select a directory containing a DICOM series.
3. If multiple series are present, choose the intended CT series.
4. Confirm the anatomy and orientation in all MPR views and 3D.

### 5.2 Run Segmentation

1. Select **Run Auto Segmentation**.
2. The application prioritizes GPU execution and retries on CPU when necessary.
3. Wait for vertebral labels and meshes to appear.
4. Review segmentation boundaries before planning.

In standalone builds TotalSegmentator is already included. If its first model download or inference fails, the application produces a threshold fallback and reports the reason. Never assume fallback output is clinically accurate. Source installations can add the AI runtime with `--with-totalseg`.

### 5.3 Select Vertebral Levels

Use the **Visible / Plan Levels** checkboxes.

- Checked levels are visible in 3D.
- **Plan Screws** uses the same checked levels.
- **Isolate Vertebrae** masks non-vertebral anatomy in MPR.
- **Restore Full Volume** returns to the original CT.

### 5.4 Generate Proposals

Select **Plan Screws**. Proposals appear immediately in the screw list and remain editable.

Current defaults:

- Lengths are proposed in 5 mm increments, normally up to 50 mm.
- A 55 mm screw is allowed only when the safe in-bone corridor is at least 60 mm.
- Preferred diameter is 6.5 mm for S1 and L3–L5, 6.0 mm for L1–L2, and 5.5 mm for T1–T12.
- Automatic diameter is limited to 7.0 mm; manual adjustment is limited to 7.5 mm.

These values are workflow presets, not universal clinical recommendations.

### 5.5 Planning Parameters

The collapsible **Planning parameters** group below **Planning** exposes the settings the automatic planner uses to size and place screws. Each field is validated and saved immediately (Qt `QSettings`), so a value survives an application restart.

| Field | Default | Meaning |
|---|---:|---|
| Pedicle fill | 0.80 | Screw diameter as a fraction of the narrowest measured pedicle (isthmus) width |
| Wall clearance | 1.0 mm | Minimum distance kept between the screw and the cortical wall |
| Anterior margin | 4.0 mm | Safety margin kept behind the anterior vertebral body cortex |
| Max convergence | 35° | Largest medial convergence angle the planner may use |
| HU threshold | 123 HU | Trajectory HU below which the loosening-risk warning is flagged |

Select **Reset Defaults** to restore these five built-in values and save them immediately. The lateral-divergence limit (−5°, the most lateral angle the planner may still choose) is fixed in this version and is not exposed in the panel.

### 5.6 Review in Screw MPR

1. Select a screw from the list, MPR, or 3D.
2. Select **Screw MPR**.
3. Review Oblique Axial, Oblique Sagittal, and Cross-section images.
4. Move **Position** from entry to tip to inspect the corridor.
5. Select **Std MPR** to restore standard planes.

Screw MPR shows the selected screw. Standard MPR shows screws intersecting the current slice.

### 5.7 Screw Metrics and Grading

The **Selected Screw** panel reports:

- **Convergence:** the signed axial angle toward the midline. Positive is medial (tip toward the midline); negative is lateral.
- **Craniocaudal:** the signed elevation of the trajectory above the axial plane. Positive is cranial. For a converging screw this differs slightly from a sagittal-projection angle.
- **Safety (grade):** a Gertzbein-Robbins grade computed from the cylinder-surface distance to the vertebra boundary on the TotalSegmentator mask. HU plays no role in the grade; the reported mean and minimum HU along the trajectory are informational only. The grade reads `N/A` until a TotalSegmentator mask exists for the screw's vertebra.

Automatic sizing keeps the diameter at or below 80% of the measured pedicle isthmus width with at least 1 mm of cortical clearance on each side, keeps the tip at least 4 mm behind the anterior cortex, and selects lengths from the 25-55 mm catalogue in 5 mm steps. These values are workflow presets, not universal clinical recommendations.

Loaded volumes are reoriented to LPS (identity direction) before display. An oblique acquisition is resampled onto an identity-direction grid, and the info panel notes "(oblique volume resampled)" when this occurs.

### 5.8 Screw Quality Metrics

Once a screw is graded against a segmentation, the **Selected Screw** panel (Body HU, Wall margin, Facet, Heary rows) and the CSV/JSON export report a bundle of literature-based bone-quality and safety measurements:

- **Trajectory HU (mean / min):** the mean and minimum Hounsfield Unit sampled along the screw's cylindrical trajectory.
- **Pedicle HU:** mean HU restricted to trajectory samples within 10 mm of the pedicle isthmus centre. Auto-planned screws only — a manually placed screw has no isthmus centre to sample around.
- **Vertebral body HU:** mean HU of an 8×8×6 mm ellipsoidal region of interest at the vertebral body centre, intersected with that vertebra's segmentation label. Auto-planned screws only, for the same reason.
- **Trajectory/body HU ratio:** trajectory mean HU divided by vertebral body HU.
- **Minimum cortical wall distance ("Wall margin"):** the closest approach, in mm, between the screw and the cortical wall.
- **Heary breach direction:** the anatomical direction of the worst cortical breach — medial, lateral, anterior, posterior, superior, or inferior (Heary 2004). Reported as "mediolateral" for a medial/lateral breach on a manually placed screw with no known side, and "none" when there is no breach.
- **Facet-joint violation grade (0-3):** an approximation of the Babu (2012) grading, measured from the proximal third of the screw to the cephalad vertebra's segmentation label — 0 no contact, 1 abuts the facet within 1 mm, 2 enters it by less than 1 mm, 3 penetrates it by 1 mm or more.

The panel and export also warn when a measurement crosses a literature threshold:

| Metric | Threshold | Warning | Reference |
|---|---|---|---|
| Trajectory HU | below 123 HU | Loosening risk | Yamamoto 2025; Dhar 2026 |
| Vertebral body HU | below 132 HU | Osteoporosis | Sankar 2026 |
| Vertebral body HU | below 141 HU (and not already osteoporotic) | Low bone density | Sankar 2026 |
| Trajectory/body HU ratio | below 1.0 | Loosening risk | Yang 2026 |
| Facet-joint violation grade | 2 or higher | Facet violation | Babu 2012 |

These bone-quality warnings are generated only for automatically planned screws; a manually placed screw still receives the full metric bundle but not the bone-quality warnings. Re-grading a plan — after running segmentation again or on plan load, whenever a segmentation is already available — regenerates the breach-distance and cortical-clearance warnings for every screw.

## 6. Edit Screws

### 6.1 Direct Editing

1. Double-click the large head to move the entry point only.
2. Double-click the pointed tip or distal shaft to move the tip only.
3. Double-click the middle shaft to move entry and tip together.
4. Move the pointer without holding a button.
5. Double-click again to finish.
6. Press `Esc` to leave movement mode while retaining the last valid position.

During MPR editing the CT image remains fixed and the screw moves. After editing, Screw MPR realigns to the updated trajectory.

### 6.2 Diameter

The diameter field accepts shorthand:

| Input | Result |
|---:|---:|
| `65` | 6.5 mm |
| `55` | 5.5 mm |
| `60` | 6.0 mm |
| `7` | 7.0 mm |

Values are normalized to 0.5 mm increments between 4.0 and 7.5 mm.

### 6.3 Delete

Select a screw and use **Delete Screw** or the `Delete` key.

## 7. Manual Tools

### Add Screw

1. Select **Add Screw**.
2. Click the entry point in an MPR view.
3. Click the target point in the same MPR view.
4. Review and adjust the new screw.

### Distance and Angle

- **Distance:** click two points in one MPR view.
- **Angle:** click three points; the second point is the vertex.

Measurements belong to the cut where they were created. They hide on another cut and reappear when the original cut is restored.

- Select **Show Cut** to return to the source cut.
- Click a measurement to select it and display yellow handles.
- Drag a handle to correct one point.
- Select **Edit** to re-measure the item.
- Select **Delete** or press `Delete` to remove it.

## 8. Display Controls

### MPR

| Control | Action |
|---|---|
| Mouse wheel | Change standard MPR slice |
| Ctrl/Cmd + wheel | Zoom |
| `Pan` then left-drag | Move image and overlays |
| `− / + / Fit` | Zoom out, zoom in, or fit |
| Right-drag | Adjust window/level |

### 3D

| Control | Action |
|---|---|
| Left-drag | Rotate |
| Mouse wheel | Zoom |
| `Pan` then drag or Shift-drag | Move model |
| Double-click anatomy | Focus on a point |
| `Reset View` | Restore sagittal startup orientation |
| `Vertebra Transparency` | Reveal or obscure internal screws |
| `Planes On / Off` | Show or hide MPR planes |

## 9. Save and Export

- Save and load planning data in JSON format.
- Export supported planning tables as CSV.
- Export supported bone surfaces as STL.

Plan files use schema version 3, which adds a `metrics` field per screw carrying the bone-quality and safety measurements described in section 5.8, Screw Quality Metrics (trajectory/pedicle/body HU, HU ratio, minimum wall distance, Heary breach direction, facet violation grade); schema version 2 added `mean_hu`, `min_hu`, `warnings`, and `source` for each screw. Plan files saved by earlier versions still load; a plan loaded while a segmentation is already available is re-graded immediately, which fills in the schema v3 metrics. CSV export includes the renamed `convergence_angle_deg` and `craniocaudal_angle_deg` columns, `mean_hu`, `min_hu`, `source`, `warnings`, and the schema v3 metric columns `trajectory_mean_hu`, `pedicle_mean_hu`, `body_mean_hu`, `hu_ratio`, `min_wall_mm`, `heary_direction`, and `facet_grade`.

Planning files, screenshots, and meshes may still be identifiable derivatives. Review them before sharing.

## 10. Troubleshooting

### CT Appears Too Small

Use `+`, enable `Pan` and drag, or select `Fit`. Use **Fit MPR** to reset all MPR views.

### TotalSegmentator Is Slow

- Use a CUDA GPU when available.
- Close memory-intensive applications.
- Allow the first run to finish downloading model weights; subsequent runs reuse them.
- Remember that lower-resolution segmentation may reduce boundary accuracy.

### A Planned Screw Is Missing

The planner may skip a side when it cannot find an allowed contained trajectory. Check segmentation and place or edit the screw manually.

### Startup Problem

```bash
./scripts/run_app.sh --check
tail -100 logs/app.log
```

When reporting a problem, include the platform, Python version, steps to reproduce, and a redacted log excerpt. Do not attach clinical data.

## 11. Version, Credits, and Citation

Open **Help → About Pedicle Screw Simulator** to view the installed version, creator, affiliation, email, website, source repository, MIT License, and research-use notice.

- Creator: Sang-Min Park, MD, Ph.D.
- Organization: Seoul National University Bundang Hospital
- Academic affiliation: Seoul National University College of Medicine
- Website: [https://sangmin.me](https://sangmin.me)
- Citation: see [`CITATION.cff`](../CITATION.cff)

## 12. Validating Plans

`scripts/validate_plans.py` compares a predicted plan (automatic planner output, a trainee plan, ...) against a reference plan (an expert plan, ground truth) using the deviation measures reported in the pedicle-screw planning literature: entry- and tip-point mean absolute deviation (MAD), the 3D angle between screw axes, convergence and craniocaudal angle deltas, diameter and length agreement (Bland-Altman bias and limits of agreement), pedicle-centre offset, and screw-volume Dice overlap.

Run it from the repository root, outside the desktop app, with two saved plan JSON files:

```bash
python scripts/validate_plans.py --pred pred_plan.json --ref ref_plan.json --out report
```

This prints the cohort summary (`key: value` lines) to the console and writes:

- `report.csv` — one row per matched screw, with `level, side, head_mad_mm, tip_mad_mm, axis_angle_deg, convergence_delta_deg, craniocaudal_delta_deg, diameter_delta_mm, length_delta_mm, pedicle_center_offset_mm, dice`.
- `report.json` — the cohort summary (matched/unmatched counts, MAD and axis-angle mean/SD, diameter and length bias with 95% limits of agreement, mean Dice).

Screws are matched between the two plans by vertebra level and side; use `--voxel-mm` to change the rasterisation grid used for the Dice computation (default `0.5` mm; coarser grids run faster but slightly underestimate overlap for very long or thin screws).

**Interpreting the numbers.** There is no universal pass/fail threshold — read the summary against the spread reported for human raters. In the inter-rater agreement literature (Scherer 2022), independent expert planners on the same cases differ by a mean of about 4.9 mm at the entry point and 4.4 mm at the tip, with a mean axis-angle difference of about 5.3°. A predicted plan that falls within roughly this range of a reference plan is consistent with inter-observer variability; deviations well beyond it warrant closer review of the planner output or the reference plan itself.
