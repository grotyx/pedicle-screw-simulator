# User Guide

[English](USER_GUIDE.md) | [한국어](USER_GUIDE.ko.md)

**Version:** 0.2.2

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

- **Workflow bar:** a bar above the MPR/3D views with four steps, ① Study, ② Segment, ③ Plan, and ④ Review, that navigate the right-hand panel (see below) — it no longer runs any action itself
- **Axial, Sagittal, Coronal MPR:** synchronized CT sections with crosshairs, segmentation, measurements, and screw overlays
- **3D viewport:** CT volume, vertebral meshes, screws, and optional MPR planes — while **Screw MPR** is active these are the screw-aligned planes instead of the standard axial/sagittal/coronal ones (see 5.8); its header carries a **Vertebrae / Full CT** toggle (see 5.2)
- **Right-hand step panel:** one page per workflow-bar step — Study, Segment, Plan, Review — described page by page below
- **Tools palette:** Select, Add Screw, Distance, and Angle

### Workflow bar

The bar above the MPR/3D views reads "① Study → ② Segment → ③ Plan → ④ Review." Clicking a step shows that step's page in the right-hand panel; every step is always clickable, since every page is reachable whatever the study's state — the bar only navigates, it no longer runs Open DICOM, Run Auto Segmentation, or Plan Screws itself (those buttons live on their own pages, described below).

A finished step shows a check mark ("✓ Study"). The next thing to do is highlighted as the primary action so the bar always points at what to do next, and a small marker under the currently shown step's button tracks which page is on screen right now (independent of which step is "next"). Hovering a step shows a tooltip describing what to do there, or what still blocks it.

| Step | Counts as done when | Tooltip |
|---|---|---|
| ① Study | A DICOM study is loaded | "Study loaded — open another one here", or "Open a DICOM series folder" |
| ② Segment | A TotalSegmentator mask exists (a threshold fallback does not count) | "Run TotalSegmentator on the loaded study", or "Open a DICOM series first" |
| ③ Plan | An automatically planned screw exists | "Plan screws for the selected vertebral levels", "Select vertebral levels in the Plan step", or "Run segmentation first" |
| ④ Review | Never shown as done — it is the destination, not a step to finish | "Review the screws level by level", or "Plan or place screws first" |

A manually placed screw does not count toward ③; deleting the automatic screws reopens it. A later step never shows done while an earlier one is not. Loading a new study resets the whole bar and switches the panel to the Segment page; a TotalSegmentator run that finds vertebra labels switches it to the Plan page; selecting a screw (from the list, an MPR view, or 3D) or turning on Screw MPR switches it to the Review page — only turning Screw MPR *on* does that: once it is on, the panel stays on whatever page you switch to, and turning it off again does not itself switch pages.

### The right-hand panel, page by page

- **Study:** the **Open DICOM…** button, the **Study** info section, then the collapsed **Window/Level** (window/level sliders and Bone/Soft Tissue presets) and **3D Rendering** (transfer-function preset and opacity) sections.
- **Segment:** the **Segmentation** group (**Run Auto Segmentation**, **Refine boundaries against CT**, the status line, and the isolate/restore button — see 5.2) and a hint about the automatic isolation and the 3D header toggle.
- **Plan:** the **Planning** group, holding **Levels to plan (also shown in 3D)** at the top (the level checkboxes and their **All**/**Clear** buttons, formerly in the Study tab's Segmentation section — see 5.3), then the mode combo, a review notice, **Plan Screws**, status, and **Clear All Screws** — followed by the collapsed **Planning parameters** and **Manual Screw Defaults** sections.
- **Review:** built around the per-level screw list, which takes most of the page (see 5.9/5.10 for its layout, and 6 for editing); **Measurements** are collapsed at the bottom (see 7).

### 3D header: Vertebrae / Full CT

The 3D viewport's header carries a two-state **Vertebrae / Full CT** toggle next to its title. It always reflects what is actually on screen — including after a failed automatic isolation, which snaps it back to Full CT — and it is the only thing that now decides whether the CT volume is visible in 3D (see 5.2).

### Layouts

Use **Planning** for a large 3D view and compact MPR panels. Use **MPR Focus** for a larger 2×2 review layout.

### Orientation markers

Each MPR pane labels its four edges with the anatomical direction you are looking toward: **A**nterior, **P**osterior, patient **L**eft, patient **R**ight, **S**uperior, **I**nferior. The axial view opens in radiological convention — anterior at the top, patient left on the right of the screen — so a left pedicle appears on the right of the image. A letter followed by an apostrophe means the axis is oblique to the screen and the label is the nearest direction rather than an exact one.

Always confirm the markers against the patient's known laterality before planning. They are derived from the DICOM direction cosines, which a mislabelled or hand-edited series can misstate.

### Maximise a single pane

Double-click a pane's header, or press `Ctrl+M`, to expand the pane you are working in to the full viewing area. The same gesture restores the previous layout. `Ctrl+M` follows the pane your pointer last worked in, whichever tool is active.

### Themes

The **Theme** selector in the **View** menu offers one light theme (Soft Light) and two dark ones (Graphite Blue, the default, and Graphite Mint). The choice is remembered between sessions.

## 5. Standard Planning Workflow

The workflow bar's four pages (see section 4) roughly follow this section's order: Study is 5.1, Segment is 5.2, and Plan covers 5.3 (levels), 5.4 (proposals), and 5.5 (parameters). Reviewing and editing the resulting proposals follow in 5.8 onward, in the Review page (5.9/5.10), and in section 6.

### 5.1 Open a CT Study

1. On the **Study** page, select **Open DICOM…** (the toolbar button and the File menu's **Open DICOM Folder**, `Ctrl+O`, still work too).
2. Select a directory containing a DICOM series.
3. If multiple series are present, choose the intended CT series.
4. Confirm the anatomy and orientation in all MPR views and 3D.

### 5.2 Run Segmentation

1. On the **Segment** page, select **Run Auto Segmentation**.
2. The application prioritizes GPU execution and retries on CPU when necessary.
3. Wait for vertebral labels and meshes to appear.
4. Review segmentation boundaries before planning.

In standalone builds TotalSegmentator is already included. If its first model download or inference fails, the application produces a threshold fallback and reports the reason. Never assume fallback output is clinically accurate. Source installations can add the AI runtime with `--with-totalseg`.

#### Automatic isolation and the 3D toggle

A successful TotalSegmentator run (one that actually detects vertebra labels) isolates the vertebrae automatically — no button press needed — and switches the panel to the **Plan** step. A threshold fallback never isolates automatically, since it carries no vertebra labels to isolate on; the isolate/restore button on the Segment page stays disabled for it, matching the 3D header toggle.

The status line names the detected range, for example "Segmentation ready · 7 vertebrae detected (T12–S1) · Refined (CT-guided)", followed by the pedicle-model suffix when that optional stage ran (5.11).

Switch between the isolated and full views either with the **Vertebrae / Full CT** toggle on the 3D viewport's header, or with the same button on the Segment page (it reads **Isolate Vertebrae** or **Restore Full Volume** depending on the current state); both controls always agree. The toggle now owns whether the CT volume itself is visible in 3D — checking or unchecking a level in **Levels to plan** (5.3) changes which vertebrae are shown, but no longer hides or shows the CT underneath them in Full CT mode.

#### Refine boundaries against CT

TotalSegmentator infers at about 1.5 mm and its result is upsampled to the CT grid, which on a sub-millimetre study leaves visible stair-steps on every surface. Those steps are not anatomy, and they distort both the 3D rendering and the pedicle isthmus width measured from the mask.

Leave **Refine boundaries against CT** enabled (the default) to re-decide the boundary band against the CT itself: each label is anti-aliased, voxels within 1.5 mm of a surface are re-assigned by bone density, interior holes are filled in 3D, and only the largest connected component of each vertebra is kept. It adds roughly one to two seconds on a 512 × 512 × 292 study and can be cancelled.

The status line reports which mask is in use — `Refined (CT-guided)`, `Refined (anti-alias only)` (the CT-guided step could not run, but the anti-alias pass still smoothed the label), or `Raw mask` — and names any label the refinement declined to change. A refinement that would have altered a label's volume by more than 30 % is rejected and the raw label is kept, on the assumption that such a change is a failure rather than a correction.

Turn the option off when you want the model output exactly as produced, for example when comparing against another tool's mask.

**Re-running segmentation discards the pedicle analyses behind the current plan.** Screw grades are recomputed against the new mask immediately; pedicle width, the narrow verdict, and the endplate angle keep the values recorded when the plan was made — they come from an analysis of the mask that was just replaced, so they are no longer re-derived when a screw is edited, until you generate proposals again.

### 5.3 Select Vertebral Levels

Use the **Levels to plan** checkboxes on the **Plan** page (moved there from the Study tab's Segmentation section, since level selection is a planning decision).

- Checked levels are visible in 3D — but, since the 3D header toggle now owns CT visibility (5.2), unchecking a level no longer hides the CT itself in Full CT mode.
- **Plan Screws** uses the same checked levels.
- **All** / **Clear** check or uncheck every detected level at once.

### 5.4 Generate Proposals

On the **Plan** page, select **Plan Screws**. Proposals appear immediately in the screw list and remain editable. **Clear All Screws**, just below it, removes every screw at once.

Current defaults:

- Length is the longest catalogue length (25–55 mm in 5 mm steps) whose tip still keeps the configured Anterior margin of bone ahead of it, measured along the screw's own axis — see below.
- Preferred diameter is 6.5 mm for S1 and L3–L5, 6.0 mm for L1–L2, and 5.5 mm for T1–T12.
- Automatic diameter is limited to 7.0 mm; manual adjustment is limited to 7.5 mm.

These values are workflow presets, not universal clinical recommendations.

#### Entry point and screw length

Every candidate's head is carried back along the screw's own axis to the last bone its centreline meets — the dorsal cortex where a drill would actually start. On a traditional trajectory this lands the head on the posterior surface at the lateral facet region, the classic junction of the transverse process and the superior articular process. The planner finds this point geometrically on the segmentation mask; **it does not detect the facet or transverse process as anatomical landmarks — always confirm the entry on the MPR before relying on it.**

A candidate is discarded as unreachable when the head still has the same vertebra's bone within 15 mm behind it along the screw axis (for example, under a lamina with an air pocket in between): a real drill would have to pass through that bone first. This is the dorsal-approach rule, and it is the only reachability test the **Optimizer** applies to a candidate head position.

Once the head is seated, length is the longest catalogue length whose tip still keeps the configured Anterior margin (4.0 mm by default) of bone ahead of it, measured from the tip along the screw's own axis to the anterior vertebral-body cortex — the margin the way a surgeon states it, not clearance measured all around the distal cylinder. Only the longest feasible length on each trajectory is kept, because scoring a shorter screw on the same trajectory would let the density objective reward it for staying inside dense pedicle bone; the planner's score, including the Length weight (5.6), then chooses between trajectories, not between lengths on one trajectory.

The **Legacy** planner and the automatic per-side legacy fallback (5.6) move the head and tip only after their own search has chosen and validated a trajectory (entry, target and diameter). They then use the first of three options that breaches no more than that validated screw. The first is the head re-seated on the dorsal cortex along the screw's own axis, considered only if it passes the same 15 mm dorsal-approach test, with the tip extended to the longest catalogue length that still keeps the Anterior margin ahead of it. The second is the original head with the tip extended the same way. The third is the validated screw unchanged. "No more" means less medial breach than the unchanged screw, or the same medial breach and no more breach in total. The unchanged screw always qualifies, so a legacy screw is never made less safe to make it longer or to move its head. **Cortical bone trajectory (CBT)** mode is unaffected: it keeps its own entry landmark (5.7) and selects candidates with its own full-length feasibility test.

On the project's sample study, with the refined mask and default settings, this moved screw count from 12 to 13 (S1-left is now planned), legacy fallbacks from 1 to 0, grades from A 10 / B 2 to A 13, the L5-right medial breach from 1.41 mm to 0 everywhere, and head burial from 5–21 mm to 0–0.2 mm. Lengths (left / right) became L1 35/35, L2 50/45, L3 35/40, L4 40/35, L5 45/35, T12 35/25 mm, where most sides had been 25 mm before (L1, L2, and L4 on the left among them).

### 5.5 Planning Parameters

The collapsed **Planning parameters** section on the **Plan** page exposes the settings the automatic planner uses to size and place screws (the collapsed **Manual Screw Defaults** section next to it holds the length/diameter defaults for manually added screws). Each field is validated and saved immediately (Qt `QSettings`), so a value survives an application restart.

| Field | Default | Meaning |
|---|---:|---|
| Pedicle fill | 0.80 | Screw diameter as a fraction of the narrowest measured pedicle (isthmus) width |
| Wall clearance | 0.0 mm | Minimum distance kept between the screw and the cortical wall |
| Anterior margin | 4.0 mm | Bone kept ahead of the tip, measured along the screw's axis to the anterior vertebral body cortex |
| Max convergence | 35° | Largest medial convergence angle the planner may use |
| HU threshold | 123 HU | Trajectory HU below which the loosening-risk warning is flagged |
| Narrow pedicle (mm) | 5.0 mm | Pedicle width below which the smallest catalogue screw is planned and the level is marked narrow |
| Lateral breach cap (mm) | 2.0 mm | Lateral (in-out-in) breach a narrow pedicle may accept; the medial wall is never breached |
| Parallel to upper endplate | On | Aims the trajectory along the upper endplate instead of horizontally |
| Endplate band | 10° | How far from the endplate direction the optimizer may angle the screw while the option above is on |
| Construct alignment | 0.30 | Weight given to lining screw heads up for the rod and agreeing on convergence across levels (see 5.6) |

Select **Reset Defaults** to restore these ten built-in values, along with the planner mode, trajectory family, and the Safety and Density objective weights described below, and save them all immediately. The lateral-divergence limit (−5°, the most lateral angle the planner may still choose) is fixed in this version and is not exposed in the panel.

#### Endplate reference for a rough fit

"Parallel to upper endplate" aims each level along that level's own fitted upper-endplate plane — but on a real study, some levels' superior surface is not a clean plane to begin with: a compression fracture or a Schmorl node breaks it into a shape the plane fit cannot follow, and a screw aimed at that broken fit would be tilted by the irregularity rather than by the true endplate. The planner tells the two cases apart by the fit's own residual (RMSE):

- **RMSE ≤ 1.5 mm:** the fit is trusted and used directly — `endplate_reference` = `own`, no warning.
- **1.5–3.0 mm:** still the level's own fit — `own` — but flagged with the rough-fit warning ("Upper endplate fit is rough (RMSE _x_.x mm) — check the sagittal view"), since a single lumbar CT voxel is about 1 mm and 1.5 mm is roughly where the fit stops being distinguishable from the mask's own stair-steps.
- **Above 3.0 mm, or no fit at all:** the level's own fit is no longer trusted to aim with. The threshold sits about two to three lumbar CT slices deep — well past the stair-step noise the 1.5 mm warning is about, and about the depth a real compression fracture or Schmorl node reaches. The planner instead aims along the inverse-distance-weighted average of the nearest well-fitted ("trusted") levels within two levels above and below — `endplate_reference` = `neighbours` — with a warning naming which level(s) it borrowed from: "Endplate reference: own upper-endplate fit too rough (RMSE _x_.x mm); aimed along T12 and L3 — check the sagittal view" for a rough own fit, or "Endplate reference: no upper-endplate fit; aimed along T12 and L3 — check the sagittal view" when there was no own fit to be rough about at all. With no trusted neighbour within reach, the trajectory falls back to horizontal — `endplate_reference` = `none` — with its own warning ("...and no well-fitted neighbour; used horizontal sagittal trajectory", or "Upper endplate unavailable; used horizontal sagittal trajectory" when there was no fit to be rough about at all).

S1 and the sacrum are excluded from this borrowing in both directions: the lumbosacral angle differs from L5's own by 15–30°, so S1 never lends its normal to L5 and never borrows one back, whatever its own fit looks like.

Whichever reference was actually used, the planner reports the endplate angle against it — never against a rejected own fit — and the screw's later re-grade (5.9, ScrewTool) measures the same angle against the same reference, so the two never disagree. The reference is recorded per screw as `endplate_reference` (`own` / `neighbours` / `none`), shown in the Review page's Details section (5.9) and exported as the last CSV column (section 9).

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

A candidate is feasible only when it keeps at least the configured wall clearance, keeps its convergence angle within the configured range, and keeps the configured Anterior margin of bone ahead of its tip, measured along the screw's own axis to the anterior vertebral-body cortex — the distal shaft, like the rest of the shaft, otherwise only needs to stay contained and keep the configured Wall clearance (see 5.4, "Entry point and screw length"). For a normal-width pedicle it must also have zero cortical breach; for a narrow pedicle -- narrower than the "Narrow pedicle (mm)" threshold, 5.0 mm by default -- the optimizer instead places the smallest catalogue screw (4.0 mm, drawn red on the Review page), keeps the medial, canal-side wall intact, and accepts a lateral, in-out-in breach up to the "Lateral breach cap (mm)" limit (2.0 mm by default); both spin boxes sit in Planning Parameters next to Wall clearance. The head itself is seated on the dorsal cortex along the screw's own axis (see 5.4) and rejected as unreachable when the same vertebra's bone still lies within 15 mm behind it along that axis, since a real drill would have to pass through that bone first. Only the longest feasible length on each trajectory is ranked, so the Length weight above compares trajectories against each other rather than lengths on the same trajectory. If no diameter within two catalogue steps below the pedicle's recommended diameter admits a feasible trajectory, the pedicle is planned by the legacy method instead and the screw's warnings include "Optimizer found no feasible trajectory; legacy planner used." For a narrow pedicle specifically, that legacy fallback never drops the side either: it still places whichever entry the lateral-shift search finds least medial, and if a breach into the canal remains even after that search, the screw carries a "Medial breach _x_.x mm — canal side" warning rather than being silently accepted or discarded. In testing, Optimizer-mode screws never score a worse Gertzbein grade or meaningfully less wall clearance than the same case planned in Legacy mode.

When more than one screw is planned on the same side, the optimizer re-ranks each pedicle's top candidates so the screw heads line up along a common line and neighbouring levels' convergence angles agree — a proxy for how much the rod has to be bent and twisted. It may trade away at most 10% of a screw's own best score to reduce this misalignment, weighted by the Construct alignment slider. After planning, the status bar and the auto-screw status line report the result as "Construct: rod fit *x* mm (L), *y* mm (R) · convergence spread *a*° (L), *b*° (R)," and each screw's metrics carry `score`, `score_components`, `rod_misalignment_mm`, and `convergence_deviation_deg`. Legacy planning places each screw on its own, so there is nothing to harmonise: the status line instead reads "Construct alignment needs Optimizer mode."

Runtime is approximately 2–3 seconds per pedicle on a typical CT; multi-level cases take proportionally longer because pedicles are planned one at a time. There is currently no per-level progress indicator during planning — only the three coarse status messages "Analyzing vertebral pedicles...", "Analyzed N vertebrae, M with pedicle data. Planning screws...", and "Planned N screw trajectories."

### 5.7 Cortical Bone Trajectory (CBT) Mode

The **Trajectory** combo in Planning Parameters selects the trajectory family **Plan Screws** aims for:

- **Traditional** (default) — a convergent pedicle screw following the pedicle axis, planned by either back-end above.
- **Cortical bone trajectory** — a short, narrow screw that starts at the pars/lamina junction, just inferior and medial to the pedicle isthmus, and runs cranially and laterally into the vertebral body: the mirror image of a traditional screw. It gains its pull-out strength from cortical bone contact along the way rather than from filling the pedicle, which is the technique's advantage in osteoporotic bone. Selecting CBT replaces the trajectory search entirely — neither the optimizer's candidate grid nor the legacy planner's medial search is used — regardless of the Planner mode selected.

Starting angles follow the CBT literature: cranial angle ≈25° and lateral angle ≈12°, each swept ±5° in 2.5° steps to find the best-scoring direction. The implant catalogue is 5.0/5.5/6.0 mm diameters and 30/35/40 mm lengths — narrower and shorter than the traditional catalogue. Candidates are ranked by safety and density in equal proportion (cortical purchase is the point of the technique), and must have zero breach and the configured wall clearance to be feasible; ties favor the longer, then the wider, screw.

Every planned CBT screw carries a fixed warning reminding the reviewer to rule out the technique's contraindications on the CT, since the planner has no way to detect them automatically: "CBT consensus contraindications: spondylolisthesis grade >= 3, pars defect, absent lamina/isthmus, rotational deformity > 2° (Zhang 2024)." A side that has no feasible CBT trajectory is dropped rather than silently substituted with a traditional trajectory, since the two techniques place their heads in different locations and mixing them would break the construct.

CBT screws are graded through the exact same finalization path as traditional screws — the same mask-based Gertzbein-Robbins breach and wall-clearance logic, HU statistics, bone-quality warnings, and facet/Heary classification described in sections 5.9 and 5.10 — so a CBT screw's grade and metrics are directly comparable to a traditional one on the Review page and in exports. Starting angles are from Zeng et al. (2024, *Orthopaedic Surgery*, CT-based CBT trajectory morphometry); the contraindication note is from Zhang et al. (2024, *Asian Spine Journal*, Delphi consensus on CBT indications).

### 5.8 Review in Screw MPR

1. Select a screw from the list, MPR, or 3D.
2. Select **Screw MPR**.
3. Review Oblique Axial, Oblique Sagittal, and Cross-section images.
4. Move **Position** from entry to tip to inspect the corridor.
5. Select **Std MPR** to restore standard planes.

Screw MPR shows the selected screw. Standard MPR shows screws intersecting the current slice.

#### Moving the screw-aligned planes

The oblique planes are not frozen. You can look around the screw without leaving screw-aligned review:

| Control | Action |
|---|---|
| Mouse wheel | Slide the plane along the screw axis (the same travel as **Position**) |
| `Shift` + wheel | Rotate the plane around the screw axis |
| Middle-drag | Offset the plane sideways, keeping the screw direction |
| `Ctrl/Cmd` + wheel | Zoom |
| Right-drag | Adjust window/level |

Each pane's readout shows the current rotation and offset, so a view you have moved is never mistaken for the canonical one. Rotation is useful for checking the medial wall along the whole corridor rather than only in the plane the screw happens to define; a sideways offset lets you compare the trajectory against the pedicle wall a few millimetres to one side of it.

Selecting a different screw, or returning to **Std MPR** and back, resets rotation and offset to zero.

#### Screw MPR in 3D

While Screw MPR is active, the 3D view replaces the standard axial/sagittal/coronal plane indicators — which no pane shows any more in this mode — with the three screw-aligned planes, each drawn as an 80 mm square centred on the screw and coloured like the header of the pane showing it: Oblique Axial in the axial pane's colour, Oblique Sagittal in the sagittal pane's, Cross-section in the coronal pane's.

The cross-section itself is shown as a real textured CT slice in 3D, not just a coloured indicator plane: it follows **Position**, rotation, and the plane offsets exactly like the Cross-section pane, and it samples the CT at the Study page's Window/Level sliders (8). A right-drag window/level change made inside an MPR pane changes only that pane, not the sliders, so it is not carried over to the 3D slice — use the sliders when you want the two to match. This holds in both **Vertebrae** and **Full CT** mode (see 4).

Only the vertebra the selected screw is graded against — the same one its Gertzbein-Robbins grade and Endplate angle are measured on — is opened at that plane; every other level in the scene stays completely whole, in both 3D modes. The slice itself shows the cut vertebra's own section at full opacity, with the surrounding anatomy faded, so the section you are studying is unmistakable against its context. Screws themselves are never cut. A vertebra mesh rebuilt while Screw MPR is open — for example after re-running segmentation — is cut at the same plane, and moving **Position** from entry to tip (or the mouse wheel over the Cross-section pane, 1 mm per notch) moves the cut with it. When the selected screw's vertebra cannot be resolved from the segmentation, the slice still shows but nothing is cut.

**Planes On / Off** (section 8) shows or hides whichever set of planes is current, standard or screw-aligned; it never affects the textured slice itself. **Cut View**, in the 3D pane's top-left corner while Screw MPR is active, points the camera straight down the screw at the cross-section from the entry side — useful for judging the trajectory's position within the cut face at a glance. It only reframes the camera on request; it never moves the view on its own. **Std MPR** removes the cut and the slice, and restores the standard planes.

### 5.9 Screw Metrics and Grading

#### The Review page layout

The **Review** page is built around the per-level screw list (formerly a small table squeezed under a fixed "Planning Cockpit" summary), which now takes most of the page. Above the list, the key numbers for the selected screw are shown large: level and side, diameter (editable in place), length, and a coloured Gertzbein-Robbins grade chip. Just below that sits a single collapsed line, "⚠ _N_ warnings" (or "✓ No warnings"), that expands on click to the full warning text — the same text every warning list in the app already used, just collapsed by default so it does not compete with the numbers above it. The action buttons — ‹ › previous/next navigation, **Screw MPR** / **Std MPR**, **Edit** (a menu, see section 6), and, alone on a second row, **Delete Screw** — sit in a fixed two-row block right under the list. While Screw MPR is active, a **Position** slider, **Rotation** spin box, and **Reset view** button appear in two rows under the action buttons, Position on the first and Rotation with Reset view on the second (5.8).

The list itself carries a **⚠** column at its right in place of the old **Source** column: a plain count of that screw's warnings (blank at zero), so which screws deserve a second look is visible at a glance without opening each row — auto/manual provenance moved to the collapsed Details section below (it stays in the CSV export, section 9, where it already lived), since a screw's warning count matters more for a quick scan than how it was placed.

Below the action row, the remaining per-screw metrics (Convergence, Craniocaudal, Endplate, Alignment, Trajectory HU, Source, Body HU, Wall margin, Facet, Heary, Trajectory, Pedicle) sit in a collapsed **Details** section, described in full below — the Gertzbein-Robbins grade itself is not repeated there, since it is already the chip shown large above the list. **Measurements** (section 7) are collapsed at the bottom of the same page.

The **Review** page reports:

- **Convergence:** the signed axial angle toward the midline. Positive is medial (tip toward the midline); negative is lateral.
- **Craniocaudal:** the signed elevation of the trajectory above the axial plane. Positive is cranial. For a converging screw this differs slightly from a sagittal-projection angle.
- **Endplate:** the signed angle of the trajectory relative to the reference it was measured against; positive is tip-cranial, 0 is parallel. The "Parallel to upper endplate" planning option aims directly for 0° within the configured endplate band. The reference is normally the level's own upper-endplate fit, but for a rough or missing fit it is instead the nearest well-fitted neighbouring level(s) — see "Endplate reference for a rough fit" in 5.5. When the reference is `own` the row reads a plain "+2.3°"; when it is `neighbours` it reads "+2.3° vs T12, L3" and the tooltip names the borrowed level(s); when it is `none` there is no reference to measure against, so the row reads "--".
- **Alignment:** how this screw sits in the multi-screw construct — its rod-line offset (`rod _x_ mm`) and how far its convergence angle differs from its neighbours (`conv ±_y_°`). Both describe the whole side rather than the one screw, so dragging any head re-measures them for every screw on that rod, and deleting one refits the line over those that remain. Populated only when the construct was harmonised by the Optimizer (see 5.6); Legacy-mode screws and screws you place by hand leave this row blank.
- **Safety (grade):** a Gertzbein-Robbins grade computed from the cylinder-surface distance to the vertebra boundary on the TotalSegmentator mask. HU plays no role in the grade; the reported mean and minimum HU along the trajectory are informational only. The grade reads `N/A` until a TotalSegmentator mask exists for the screw's vertebra.
- **Pedicle:** the measured pedicle isthmus width. It is shown in red when the level fell below the "Narrow pedicle" threshold, meaning the planner used the smallest catalogue diameter and capped the lateral (in-out-in) breach to protect the medial wall. The same red marking, with the width followed by `?` and the row reading "width not trusted", means the analyser rejected the measurement itself (it fell outside the plausible band for that level, in either direction). The narrow policy is applied either way, but only a trusted width is quoted as a finding.

**Entry cortex.** Grading starts 3 mm past the point where the screw's axis first enters the vertebra, not at the head. A head seated on the dorsal cortex (5.4) straddles the surface it enters, so grading it from the head would read every screw as a breach of up to its own radius before it ever reaches the pedicle — on the project's sample study, 11 of the 12 re-seated screws graded B or C for nothing but the entry cortex. Gertzbein-Robbins grades the pedicle wall, not the entry cortex, so this 3 mm entry zone is excluded from the check. It is measured from where the axis enters bone rather than from the head itself: a head placed proud of the bone also skips its own stretch in air before the axis enters the vertebra, and a head buried in bone gets only its first 3 mm treated leniently — far short of the isthmus, where a medial breach actually matters. A screw whose axis never enters the vertebra is graded from its head, so it still reads as the breach it is. The same 3 mm is excluded from every figure taken from grading — the grade itself, breach distance and direction split, the Wall margin, and the mean/min HU in the Trajectory HU row (CSV `mean_hu` / `min_hu`). This applies to every graded screw: automatic proposals from the Optimizer and Legacy planners and the displayed grade of CBT screws, manually placed screws, screws you drag or otherwise edit, and screws re-graded after segmentation or on plan load — so dragging a planned screw never re-grades it by a stricter rule than the one it was planned under. One consequence: a plan saved by an earlier build may show better grades when it is re-graded now — on load while a segmentation is available, or after running segmentation again — because the entry cortex is no longer graded (see also section 9).

Automatic sizing keeps the diameter at or below 80% of the measured pedicle isthmus width, keeps the tip at least 4 mm behind the anterior cortex measured along the screw's own axis, and selects lengths from the 25-55 mm catalogue in 5 mm steps. Wall clearance now defaults to 0 mm; raise it in Planning parameters if you want a buffer beyond the zero-breach requirement described above. A clearance of exactly 1.0 mm saved by an older build is reset to 0 mm once, with a note in the status bar, because that value was the old default rather than a choice; any other saved value (0.5 mm, 1.5 mm) is respected and stays visible in that spin box, as is a 1.0 mm you set yourself afterwards. The "Parallel to upper endplate" checkbox and its tolerance spin box (the endplate band, 10° by default) aim the trajectory at the upper endplate instead of horizontally; both live in Planning parameters alongside the sizing fields. These values are workflow presets, not universal clinical recommendations.

Loaded volumes are reoriented to LPS (identity direction) before display. An oblique acquisition is resampled onto an identity-direction grid, and the info panel notes "(oblique volume resampled)" when this occurs.

### 5.10 Screw Quality Metrics

Once a screw is graded against a segmentation, the Review page's Details section (Body HU, Wall margin, Facet, Heary rows) and the CSV/JSON export report a bundle of literature-based bone-quality and safety measurements:

- **Trajectory HU (mean / min), exported as `trajectory_mean_hu` (CSV and JSON) and `trajectory_min_hu` (JSON `metrics` only):** the mean and minimum HU (Hounsfield units) sampled along the screw's whole cylindrical trajectory, entry zone included. This is a separate figure from the **Trajectory HU** row in Details (CSV `mean_hu` / `min_hu`, see 5.9), which excludes the 3 mm entry zone.
- **Pedicle HU:** mean HU restricted to trajectory samples within 10 mm of the pedicle isthmus centre. Auto-planned screws only — a manually placed screw has no isthmus centre to sample around.
- **Vertebral body HU:** mean HU of an 8×8×6 mm ellipsoidal region of interest at the vertebral body centre, intersected with that vertebra's segmentation label. Auto-planned screws only, for the same reason.
- **Trajectory/body HU ratio:** trajectory mean HU divided by vertebral body HU.
- **Minimum cortical wall distance ("Wall margin"):** the closest approach, in mm, between the screw and the cortical wall (past the 3 mm entry zone, see 5.9).
- **Heary breach direction:** the anatomical direction of the worst cortical breach — medial, lateral, anterior, posterior, superior, or inferior (Heary 2004). Reported as "mediolateral" for a medial/lateral breach on a manually placed screw with no known side, and "none" when there is no breach.
- **Facet-joint violation grade (0-3):** an approximation of the Babu (2012) grading, measured from the proximal third of the screw to the cephalad vertebra's segmentation label — 0 no contact, 1 abuts the facet within 1 mm, 2 enters it by less than 1 mm, 3 penetrates it by 1 mm or more.

Details and the export also warn when a measurement crosses a literature threshold:

| Metric | Threshold | Warning | Reference |
|---|---|---|---|
| Trajectory HU (`trajectory_mean_hu`) | below 123 HU | Loosening risk | Yamamoto 2025; Dhar 2026 |
| Vertebral body HU | below 132 HU | Osteoporosis | Sankar 2026 |
| Vertebral body HU | below 141 HU (and not already osteoporotic) | Low bone density | Sankar 2026 |
| Trajectory/body HU ratio | below 1.0 | Loosening risk | Yang 2026 |
| Facet-joint violation grade | 2 or higher | Facet violation | Babu 2012 |

These bone-quality warnings are generated only for automatically planned screws; a manually placed screw still receives the full metric bundle but not the bone-quality warnings. Re-grading a plan — after running segmentation again or on plan load, whenever a segmentation is already available — regenerates the breach-distance and cortical-clearance warnings for every screw.

### 5.11 Optional Pedicle Subregion Model

The **Segment** page's **Advanced options** toggle shows and hides **Use pedicle subregion model**, off by default. When enabled, the app looks for a locally installed nnU-Net model that segments a vertebra into pedicle/corpus/lamina/spinous/transverse/articular subregions ([MICN-Lab/Spine_Subregions](https://github.com/MICN-Lab/Spine_Subregions); Da Mutten et al., *J Imaging Inform Med* 2026), located either through the **Model directory** field or the `PSS_SUBREGION_MODEL_DIR` environment variable. Both must point at an nnU-Net results folder containing `dataset.json` and `fold_*/checkpoint_final.pth`.

This is a source-install feature: it requires a non-frozen Python environment with `nnunetv2` installed, and its memory needs are similar to TotalSegmentator's `3d_fullres` configuration (a GPU is recommended). The standalone macOS and Windows packages do not support it — see [Desktop Build Guide](BUILDING_DESKTOP.md). Note: the Spine_Subregions release currently published upstream uses an nnU-Net v1-style folder layout, not the nnU-Net v2 layout this app expects, so it will not be recognized until it is re-exported — see the Desktop Build Guide for details.

When the model runs successfully, each side's pedicle isthmus is measured from its label first, falling back to the coronal cross-section search only for a side the label does not resolve. The segmentation status line reports "· pedicle model used", "· pedicle model ran but found no pedicle voxels" when the model ran but its output contains no pedicle label at all, or "· pedicle model unavailable: <reason>" when it could not run — a stage-2 failure never blocks the TotalSegmentator result underneath it.

## 6. Edit Screws

### 6.1 The Edit Menu

The Review page's **Edit** button opens a menu with four entries: **Move entry point**, **Move tip point**, **Move whole screw**, and **Cancel edit**. Choosing one of the first three starts that edit mode on the selected screw exactly as double-clicking the head, tip, or shaft would (6.2); **Cancel edit** ends edit mode; the screw keeps the position it was last moved to — it does not undo the move, matching `Esc` in direct editing (6.2). **Delete Screw** and the ‹ › previous/next buttons sit in the same two-row action block, with **Delete Screw** alone on the second row (5.9).

### 6.2 Direct Editing

Double-clicking still works exactly as before, alongside the Edit menu:

1. Double-click the large head to move the entry point only.
2. Double-click the pointed tip or distal shaft to move the tip only.
3. Double-click the middle shaft to move entry and tip together.
4. Move the pointer without holding a button.
5. Double-click again to finish.
6. Press `Esc` to leave movement mode while retaining the last valid position.

During MPR editing the CT image remains fixed and the screw moves. After editing, Screw MPR realigns to the updated trajectory.

### 6.3 Diameter

The diameter field accepts shorthand:

| Input | Result |
|---:|---:|
| `65` | 6.5 mm |
| `55` | 5.5 mm |
| `60` | 6.0 mm |
| `7` | 7.0 mm |

Values are normalized to 0.5 mm increments between 4.0 and 7.5 mm.

### 6.4 Delete

Select a screw and use **Delete Screw** or the `Delete` key.

## 7. Manual Tools

Add Screw, Distance, and Angle are the Tools-palette gestures; **Measurements** (the mode combo, list, and Show Cut / Edit / Delete buttons) live in a collapsed section at the bottom of the **Review** page (5.9).

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

**Window/Level** (window/level sliders, Bone and Soft Tissue presets) and **3D Rendering** (transfer-function preset, opacity — formerly named **Validation**) are both collapsed sections on the **Study** page (see section 4).

### MPR

| Control | Action |
|---|---|
| Mouse wheel | Change standard MPR slice |
| Ctrl/Cmd + wheel | Zoom |
| `Pan` then left-drag | Move image and overlays |
| `− / + / Fit` | Zoom out, zoom in, or fit |
| Right-drag | Adjust window/level |
| Double-click header, or `Ctrl+M` | Maximise this pane; repeat to restore |

In Screw MPR the wheel and middle-drag move the screw-aligned planes instead; see 5.8.

### 3D

| Control | Action |
|---|---|
| Left-drag | Rotate |
| Mouse wheel | Zoom |
| `Pan` then drag or Shift-drag | Move model |
| Double-click anatomy | Focus on a point |
| `Reset View` | Restore sagittal startup orientation |
| `Vertebra Transparency` | Reveal or obscure internal screws |
| `Planes On / Off` | Show or hide the MPR planes — the screw-aligned ones while Screw MPR is active |
| Double-click header, or `Ctrl+M` | Maximise the 3D pane; repeat to restore |

See 5.8 for how Screw MPR changes what these planes show and cut.

Volume rendering uses the GPU on Windows and Linux and the CPU ray caster on macOS, where the OpenGL-to-Metal translation layer stalls during 3D texture upload. The choice is automatic and is recorded in the log.

## 9. Save and Export

- Save and load planning data in JSON format.
- Export supported planning tables as CSV.
- Export supported bone surfaces as STL.

Plan files use schema version 3, which adds a `metrics` field per screw carrying the bone-quality and safety measurements described in section 5.10, Screw Quality Metrics (trajectory/pedicle/body HU, HU ratio, minimum wall distance, Heary breach direction, facet violation grade); schema version 2 added `mean_hu`, `min_hu`, `warnings`, and `source` for each screw. Plan files saved by earlier versions still load; a plan loaded while a segmentation is already available is re-graded immediately, which fills in the schema v3 metrics. A plan saved by an earlier build may show better grades once re-graded this way, because the entry cortex is no longer graded (see 5.9) — nothing in the plan file itself changes. CSV export includes the renamed `convergence_angle_deg` and `craniocaudal_angle_deg` columns, `mean_hu`, `min_hu`, `source`, `warnings`, the schema v3 metric columns `trajectory_mean_hu`, `pedicle_mean_hu`, `body_mean_hu`, `hu_ratio`, `min_wall_mm`, `heary_direction`, and `facet_grade`, and — as the new last column — `endplate_reference` (`own` / `neighbours` / `none`, or blank when the screw never had an endplate reference recorded — for example a manually placed screw on a level with no pedicle analysis (not planned and not within two levels of a planned level, or placed after segmentation was re-run, which discards the analyses), or a screw from a plan saved by an earlier version), naming which reference the `endplate_angle_deg` column was measured against (see 5.5, "Endplate reference for a rough fit"). A planned screw keeps its planner-recorded value through later re-grades.

Planning files, screenshots, and meshes may still be identifiable derivatives. Review them before sharing.

## 10. Troubleshooting

### CT Appears Too Small

Use `+`, enable `Pan` and drag, or select `Fit`. Use **Fit MPR** to reset all MPR views.

### TotalSegmentator Is Slow

- Use a CUDA GPU when available.
- Close memory-intensive applications.
- Allow the first run to finish downloading model weights; subsequent runs reuse them.
- Remember that lower-resolution segmentation may reduce boundary accuracy.

### 3D View Is Slow or Stutters

Volume rendering falls back to CPU ray casting when no usable GPU context is available — over Remote Desktop, inside a virtual machine, or with software OpenGL. The application detects this and re-downsamples the volume automatically, so the view stays responsive at lower detail; the log records `render mode=cpu-raycast` when it happens. For full-detail rendering, run on the machine directly with a working GPU driver. Lowering `Vertebra Transparency` or turning `Planes Off` also reduces the load.

### A Planned Screw Is Missing

The planner may skip a side when it cannot find an allowed contained trajectory. Check segmentation and place or edit the screw manually.

### Screws Are Not Parallel to a Fractured Endplate

If a level's superior surface is broken by a compression fracture or a Schmorl node, the planner may aim that level's screw along a neighbouring level's endplate instead of its own — see "Endplate reference for a rough fit" (5.5). Expand the Review page's "⚠ _N_ warnings" line and look for an "Endplate reference: …" warning, or "Upper endplate unavailable; used horizontal sagittal trajectory". The Endplate row in the Review page's Details section does not spell out `own`/`neighbours`/`none` in words: `own` reads a plain angle ("+2.3°"); `neighbours` reads the angle with the borrowed level(s) named ("+2.3° vs T12, L3", also in the tooltip); `none` reads "--" because there is no reference left to measure against. The literal `own` / `neighbours` / `none` value is exported as the CSV `endplate_reference` column (9). This is expected behaviour for a level whose own fit is too rough to trust, not a bug — confirm the trajectory on the sagittal view either way.

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
- Release history: see [`CHANGELOG.md`](../CHANGELOG.md)

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
