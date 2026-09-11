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
| Wall clearance | 0.0 mm | Minimum distance kept between the screw and the cortical wall |
| Anterior margin | 4.0 mm | Safety margin kept behind the anterior vertebral body cortex |
| Max convergence | 35° | Largest medial convergence angle the planner may use |
| HU threshold | 123 HU | Trajectory HU below which the loosening-risk warning is flagged |
| Narrow pedicle (mm) | 5.0 mm | Pedicle width below which the smallest catalogue screw is planned and the level is marked narrow |
| Lateral breach cap (mm) | 2.0 mm | Lateral (in-out-in) breach a narrow pedicle may accept; the medial wall is never breached |
| Parallel to upper endplate | On | Aims the trajectory along the upper endplate instead of horizontally |
| Endplate band | 10° | How far from the endplate direction the optimizer may angle the screw while the option above is on |
| Construct alignment | 0.30 | Weight given to lining screw heads up for the rod and agreeing on convergence across levels (see 5.6) |

Select **Reset Defaults** to restore these ten built-in values, along with the planner mode, trajectory family, and the Safety and Density objective weights described below, and save them all immediately. The lateral-divergence limit (−5°, the most lateral angle the planner may still choose) is fixed in this version and is not exposed in the panel.

### 5.6 Trajectory Optimizer

The **Planner** combo in Planning Parameters selects between two back-ends for **Plan Screws**:

- **Optimizer** (default) — enumerates a dense grid of straight candidate trajectories per pedicle (entry offset × convergence angle × craniocaudal angle × catalogue length), grades every candidate in a single pass, discards infeasible ones, and ranks the rest by a weighted sum of five normalised objectives (each 0–1):
  - **Safety** — minimum cortical wall clearance, saturating at 3 mm.
  - **Density** — mean trajectory HU, normalised over 100–600 HU.
  - **Length** — screw length as a fraction of the longest catalogue length.
  - **Endplate** — how parallel the trajectory is to the upper endplate, within a 15° tolerance.
  - **Centering** — how close the trajectory passes to the pedicle isthmus centre, relative to the isthmus half-width.
- **Legacy** — the original greedy entry/target search used before the optimizer was added.

Default weights (shown as a percentage of the nominal weight, 0–300%, in the panel): Safety 100% (1.0), Density 50% (0.5), Length 20% (0.2), Endplate 30% (0.3), Centering 30% (0.3). A sixth weight, **Construct alignment** (default 30% / 0.3, internally the `rod` weight), does not affect single-screw scoring; it only governs how much a multi-screw plan may trade an individual screw's score to line up the screw heads on the same side and to bring converging levels' angles into agreement (see below). Only the Safety, Density, and Construct alignment weights are exposed as sliders in the panel; Length, Endplate, and Centering stay at their defaults in this version.

A candidate is feasible only when it keeps at least the configured wall clearance, keeps its convergence angle within the configured range, and keeps the configured anterior margin over its distal 4 mm segment. For a normal-width pedicle it must also have zero cortical breach; for a narrow pedicle -- narrower than the "Narrow pedicle (mm)" threshold, 5.0 mm by default -- the optimizer instead places the smallest catalogue screw (4.0 mm, drawn red in the Selected Screw panel), keeps the medial, canal-side wall intact, and accepts a lateral, in-out-in breach up to the "Lateral breach cap (mm)" limit (2.0 mm by default); both spin boxes sit in Planning Parameters next to Wall clearance. An entry point that would have to be seated more than 6 mm inside the posterior cortex to fit the screw's cross-section is also rejected as unreachable, since a real drill cannot pass through that much bone to reach the corridor. If no diameter within two catalogue steps below the pedicle's recommended diameter admits a feasible trajectory, the pedicle is planned by the legacy method instead and the screw's warnings include "Optimizer found no feasible trajectory; legacy planner used." For a narrow pedicle specifically, that legacy fallback never drops the side either: it still places whichever entry the lateral-shift search finds least medial, and if a breach into the canal remains even after that search, the screw carries a "Medial breach _x_.x mm — canal side" warning rather than being silently accepted or discarded. In testing, Optimizer-mode screws never score a worse Gertzbein grade or meaningfully less wall clearance than the same case planned in Legacy mode.

When more than one screw is planned on the same side, the optimizer re-ranks each pedicle's top candidates so the screw heads line up along a common line and neighbouring levels' convergence angles agree — a proxy for how much the rod has to be bent and twisted. It may trade away at most 10% of a screw's own best score to reduce this misalignment, weighted by the Construct alignment slider. After planning, the status bar and the auto-screw status line report the result as "Construct: rod fit *x* mm (L), *y* mm (R) · convergence spread *a*° (L), *b*° (R)," and each screw's metrics carry `score`, `score_components`, `rod_misalignment_mm`, and `convergence_deviation_deg`. Legacy planning places each screw on its own, so there is nothing to harmonise: the status line instead reads "Construct alignment needs Optimizer mode."

Runtime is approximately 2–3 seconds per pedicle on a typical CT; multi-level cases take proportionally longer because pedicles are planned one at a time. There is currently no per-level progress indicator during planning — only the three coarse status messages "Analyzing vertebral pedicles...", "Analyzed N vertebrae, M with pedicle data. Planning screws...", and "Planned N screw trajectories."

### 5.7 Cortical Bone Trajectory (CBT) Mode

The **Trajectory** combo in Planning Parameters selects the trajectory family **Plan Screws** aims for:

- **Traditional** (default) — a convergent pedicle screw following the pedicle axis, planned by either back-end above.
- **Cortical bone trajectory** — a short, narrow screw that starts at the pars/lamina junction, just inferior and medial to the pedicle isthmus, and runs cranially and laterally into the vertebral body: the mirror image of a traditional screw. It gains its pull-out strength from cortical bone contact along the way rather than from filling the pedicle, which is the technique's advantage in osteoporotic bone. Selecting CBT replaces the trajectory search entirely — neither the optimizer's candidate grid nor the legacy planner's medial search is used — regardless of the Planner mode selected.

Starting angles follow the CBT literature: cranial angle ≈25° and lateral angle ≈12°, each swept ±5° in 2.5° steps to find the best-scoring direction. The implant catalogue is 5.0/5.5/6.0 mm diameters and 30/35/40 mm lengths — narrower and shorter than the traditional catalogue. Candidates are ranked by safety and density in equal proportion (cortical purchase is the point of the technique), and must have zero breach and the configured wall clearance to be feasible; ties favor the longer, then the wider, screw.

Every planned CBT screw carries a fixed warning reminding the reviewer to rule out the technique's contraindications on the CT, since the planner has no way to detect them automatically: "CBT consensus contraindications: spondylolisthesis grade >= 3, pars defect, absent lamina/isthmus, rotational deformity > 2° (Zhang 2024)." A side that has no feasible CBT trajectory is dropped rather than silently substituted with a traditional trajectory, since the two techniques place their heads in different locations and mixing them would break the construct.

CBT screws are graded through the exact same finalization path as traditional screws — the same mask-based Gertzbein-Robbins breach and wall-clearance logic, HU statistics, bone-quality warnings, and facet/Heary classification described in sections 5.9 and 5.10 — so a CBT screw's grade and metrics are directly comparable to a traditional one in the Selected Screw panel and in exports. Starting angles are from Zeng et al. (2024, *Orthopaedic Surgery*, CT-based CBT trajectory morphometry); the contraindication note is from Zhang et al. (2024, *Asian Spine Journal*, Delphi consensus on CBT indications).

### 5.8 Review in Screw MPR

1. Select a screw from the list, MPR, or 3D.
2. Select **Screw MPR**.
3. Review Oblique Axial, Oblique Sagittal, and Cross-section images.
4. Move **Position** from entry to tip to inspect the corridor.
5. Select **Std MPR** to restore standard planes.

Screw MPR shows the selected screw. Standard MPR shows screws intersecting the current slice.

### 5.9 Screw Metrics and Grading

The **Selected Screw** panel reports:

- **Convergence:** the signed axial angle toward the midline. Positive is medial (tip toward the midline); negative is lateral.
- **Craniocaudal:** the signed elevation of the trajectory above the axial plane. Positive is cranial. For a converging screw this differs slightly from a sagittal-projection angle.
- **Endplate:** the signed angle of the trajectory relative to the upper endplate; positive is tip-cranial, 0 is parallel. The "Parallel to upper endplate" planning option aims directly for 0° within the configured endplate band.
- **Alignment:** how this screw sits in the multi-screw construct — its rod-line offset (`rod _x_ mm`) and how far its convergence angle differs from its neighbours (`conv ±_y_°`). Populated only when the construct was harmonised by the Optimizer (see 5.6); Legacy-mode screws leave this row blank.
- **Safety (grade):** a Gertzbein-Robbins grade computed from the cylinder-surface distance to the vertebra boundary on the TotalSegmentator mask. HU plays no role in the grade; the reported mean and minimum HU along the trajectory are informational only. The grade reads `N/A` until a TotalSegmentator mask exists for the screw's vertebra.
- **Pedicle:** the measured pedicle isthmus width. It is shown in red when the level fell below the "Narrow pedicle" threshold, meaning the planner used the smallest catalogue diameter and capped the lateral (in-out-in) breach to protect the medial wall. The same red marking, with the width followed by `?` and the row reading "width not trusted", means the analyser rejected the measurement itself (it fell outside the plausible band for that level, in either direction). The narrow policy is applied either way, but only a trusted width is quoted as a finding.

Automatic sizing keeps the diameter at or below 80% of the measured pedicle isthmus width, keeps the tip at least 4 mm behind the anterior cortex, and selects lengths from the 25-55 mm catalogue in 5 mm steps. Wall clearance now defaults to 0 mm; raise it in Planning parameters if you want a buffer beyond the zero-breach requirement described above. A clearance of exactly 1.0 mm saved by an older build is reset to 0 mm once, with a note in the status bar, because that value was the old default rather than a choice; any other saved value (0.5 mm, 1.5 mm) is respected and stays visible in that spin box, as is a 1.0 mm you set yourself afterwards. The "Parallel to upper endplate" checkbox and its tolerance spin box (the endplate band, 10° by default) aim the trajectory at the upper endplate instead of horizontally; both live in Planning parameters alongside the sizing fields. These values are workflow presets, not universal clinical recommendations.

Loaded volumes are reoriented to LPS (identity direction) before display. An oblique acquisition is resampled onto an identity-direction grid, and the info panel notes "(oblique volume resampled)" when this occurs.

### 5.10 Screw Quality Metrics

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

### 5.11 Optional Pedicle Subregion Model

Segmentation → Advanced offers **Use pedicle subregion model**, off by default. When enabled, the app looks for a locally installed nnU-Net model that segments a vertebra into pedicle/corpus/lamina/spinous/transverse/articular subregions ([MICN-Lab/Spine_Subregions](https://github.com/MICN-Lab/Spine_Subregions); Da Mutten et al., *J Imaging Inform Med* 2026), located either through the **Model directory** field or the `PSS_SUBREGION_MODEL_DIR` environment variable. Both must point at an nnU-Net results folder containing `dataset.json` and `fold_*/checkpoint_final.pth`.

This is a source-install feature: it requires a non-frozen Python environment with `nnunetv2` installed, and its memory needs are similar to TotalSegmentator's `3d_fullres` configuration (a GPU is recommended). The standalone macOS and Windows packages do not support it — see [Desktop Build Guide](BUILDING_DESKTOP.md). Note: the Spine_Subregions release currently published upstream uses an nnU-Net v1-style folder layout, not the nnU-Net v2 layout this app expects, so it will not be recognized until it is re-exported — see the Desktop Build Guide for details.

When the model runs successfully, each side's pedicle isthmus is measured from its label first, falling back to the coronal cross-section search only for a side the label does not resolve. The segmentation status line reports "· pedicle model used", "· pedicle model ran but found no pedicle voxels" when the model ran but its output contains no pedicle label at all, or "· pedicle model unavailable: <reason>" when it could not run — a stage-2 failure never blocks the TotalSegmentator result underneath it.

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

Plan files use schema version 3, which adds a `metrics` field per screw carrying the bone-quality and safety measurements described in section 5.10, Screw Quality Metrics (trajectory/pedicle/body HU, HU ratio, minimum wall distance, Heary breach direction, facet violation grade); schema version 2 added `mean_hu`, `min_hu`, `warnings`, and `source` for each screw. Plan files saved by earlier versions still load; a plan loaded while a segmentation is already available is re-graded immediately, which fills in the schema v3 metrics. CSV export includes the renamed `convergence_angle_deg` and `craniocaudal_angle_deg` columns, `mean_hu`, `min_hu`, `source`, `warnings`, and the schema v3 metric columns `trajectory_mean_hu`, `pedicle_mean_hu`, `body_mean_hu`, `hu_ratio`, `min_wall_mm`, `heary_direction`, and `facet_grade`.

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
```

Log location depends on how the application is running:

- **Source build (running from `python main.py` or `run_app.sh`)**: `logs/app.log` inside the project directory.
- **Packaged build on macOS**: `~/Library/Logs/PedicleScrewSimulator/app.log`.
- **Packaged build on Windows**: `%LOCALAPPDATA%\PedicleScrewSimulator\logs\app.log`.

Inspect the last lines of the relevant file, for example:

```bash
tail -100 logs/app.log
```

```powershell
Get-Content "$env:LOCALAPPDATA\PedicleScrewSimulator\logs\app.log" -Tail 100
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
