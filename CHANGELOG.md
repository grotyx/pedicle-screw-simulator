# Changelog

All notable changes to Pedicle Screw Simulator are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The repository-root `VERSION` file is the single source of truth for the
current version; every other place the version appears is derived from it.

> **Research and education use only.** Nothing in this changelog implies
> clinical validation. Every segmentation, screw proposal, dimension, breach
> grade, and warning requires independent review by a qualified clinician.

## [Unreleased]

### Added

- The Heary breach direction now travels with its secondary axis:
  `heary_secondary` in JSON `metrics`, a new `heary_secondary` CSV column
  after `endplate_reference`, and a "primary + secondary" Heary row on the
  Review page, so a superomedial breach keeps its medial component.
- The Review screw list supports multi-select batch delete with
  Ctrl+Z undo, a level/side/grade text filter, and Delete / [ / ]
  shortcuts. Undo never restores screws across a study or plan load, and
  hidden rows are excluded from delete and navigation.
- One shared cancellable job dialog serves DICOM loading (now
  cancellable), segmentation, planning, and STL export, with a one-line
  log tail; the planning cancel note names the side being finished.
  The Study step shows a spinner while DICOM loads.

### Changed

- The legacy planner requires grade A (zero breach) on a normal-width
  pedicle, like the optimizer and CBT: a side that only grades B is
  skipped rather than placed. The narrow-pedicle path is unchanged.
- The planner and the screw tool size from the analyser's conservative
  lower bound whenever it disagrees with the headline width by more than
  1.0 mm, and an edited screw is judged by the same rule.
- The vertebral-body HU region scales with the level's segmented volume
  (down to half the default for small levels); an ROI under 100 voxels
  reports unmeasured instead of a noisy mean.
- The facet check reads the nearest segmented level above by centroid
  height instead of `label + 1`, so a missing middle level no longer
  skips the true superior neighbour.
- The Study info section shows acquisition geometry only; patient
  identifiers from the DICOM headers are never read into the application.
- The optimizer rejects colliding screws between adjacent levels
  (tip/shaft clearance under radii sum + 1 mm) and boosts the density
  weight in osteoporotic bone (body HU below 132).
- The right-hand panel is built by per-step builders; the workflow bar
  derives from session state with a running spinner; the status bar
  carries a mode chip (tool, armed edit, Screw MPR identity, isolation).
- Screw drags coalesce to 30 Hz with release replay; viewers share one
  render path and reuse one cell picker each.
- A colliding construct warns and reports the pair: the lower-score side
  falls back to the legacy planner, and a side with no collision-free
  option is recorded in the dropped-sides note.
- A screw dragged to another level drops the old level's body/pedicle HU
  and ratio instead of mixing them, and CBT records its zero entry zone.

### Fixed

- The pedicle analyser rejects a mask whose direction is not LPS
  identity instead of silently mirroring left/right and anterior/posterior;
  planning reports the misalignment and asks for a re-run of segmentation.
- Pedicle width measurement uses the physical x/y spacing separately, so
  anisotropic grids no longer mis-scale the width.
- A threshold fallback reports `success=False` (still `method=
  "threshold_fallback"` for the UI), so an empty plan is not misread as
  success.
- Segmentation geometry warnings share the grader's tolerances (relative
  spacing, origin within a fraction of a voxel), so a NIfTI float32
  origin round-trip no longer warns spuriously.
- Implant catalogues from stored settings are sorted and validated
  ascending, so a custom order cannot silently mis-size screws.
- The per-voxel scalar presence check in mesh building delegates to the
  numpy path; the slow Python VTK loop survives only as a no-numpy
  fallback.
- A screw inside the cephalad label counts as entered even on float
  distance slack; stored planner settings reject NaN/inf.
- The dorsal approach check sweeps sparsely out to 40 mm, so a far
  lamina pocket behind a head no longer reads as reachable.

## [0.2.3] - 2026-09-13

### Added

- `AGENTS.md` states the patient-data, git and verification rules for every
  coding agent working in the repository; `CLAUDE.md` imports it.

### Fixed

- Re-running segmentation while Screw MPR was open could leave the 3D cut on
  the vertebra resolved from the old segmentation (or on none) until Screw MPR
  was left and re-entered. The label is now resolved again whenever the
  segmentation changes.
- In the same situation, when the cut cannot be rebuilt at once (for example
  while a one-click edit is armed), the 3D slice no longer keeps the old
  mask's shading; it is drawn fully opaque until the next update.
- Loading a plan now selects its first screw, as automatic planning does, and
  the Review header shows the screw count ("3 screws") instead of "No screws"
  whenever screws exist but none is selected.
- Isolate Vertebrae and the 3D **Vertebrae / Full CT** toggle stay disabled
  when a TotalSegmentator run finds no vertebra. Isolating on such a mask
  blanked every view while the status reported success.

## [0.2.2] - 2026-09-12

A step-based right-hand panel that follows the workflow bar, a local 3D cut
that opens only the vertebra a Screw MPR screw is graded against instead of
clipping the whole scene, and a planner fix that stops aiming a screw along
an untrustworthy upper-endplate fit (a compression fracture or Schmorl node)
by borrowing the nearest well-fitted neighbouring level instead.

### Added

- A four-page step panel (Study / Segment / Plan / Review) replaces the
  pinned "Planning Cockpit" summary above a three-tab widget. The Review
  page is built around the per-level screw list, with the selected screw's
  key numbers (level/side, diameter, length, grade) shown large above it,
  action buttons under the list, a collapsed "⚠ N warnings" line, and the
  remaining metrics collapsed into a Details section.
- The screw plan table's last column is now a per-row warning count (⚠),
  replacing Source (see Changed).
- The 3D viewport header carries a **Vertebrae / Full CT** toggle that always
  reflects, and now owns, whether the CT volume is visible in 3D. The 3D
  segmentation overlay stays under the "Show 3D" checkbox: isolating hides
  it, and returning to Full CT shows it again only if that box is ticked.
- Vertebrae are isolated automatically once a TotalSegmentator run detects
  vertebra labels (never for a threshold fallback), and the panel switches to
  the Plan step.
- Screw MPR's 3D cross-section is a real textured CT slice, following
  Position, rotation, and the plane offsets exactly like the pane, rendered
  at the Study page's Window/Level settings; only the vertebra the selected screw is
  graded against is cut open at that plane, shown opaque against a faded
  surrounding, while every other level -- in both Vertebrae and Full CT mode
  -- stays whole. A new **Cut View** button points the camera down the screw
  at the cross-section from the entry side.
- `endplate_reference` (`own` / `neighbours` / `none`) records which
  upper-endplate fit a screw's trajectory and angle were actually measured
  against; shown in the Review page's Details section and exported as the
  last CSV column.

### Changed

- The workflow bar now only navigates the four step pages; it no longer runs
  Open DICOM, Run Auto Segmentation, or Plan Screws itself.
- The vertebral-level checkboxes ("Visible / Plan Levels") moved from the
  Study tab's Segmentation section to the Plan page ("Levels to plan"), and
  no longer hide or show the CT volume itself in Full CT mode -- that is now
  the 3D header toggle's job.
- The Tools tab's "Validation" section moved to the Study page and was
  renamed "3D Rendering".
- Isolate Vertebrae and the 3D **Vertebrae / Full CT** toggle are disabled
  for a threshold-fallback mask, which carries no vertebra labels to isolate
  on; previously Isolate Vertebrae was enabled after any finished
  segmentation, fallback included.
- The Source column moved from the screw plan table to the collapsed Details
  section; it stays in the CSV export, where it already was.
- Measurements moved from the Tools tab to a collapsed section at the bottom
  of the Review page, after the screw list, action buttons, Screw MPR
  controls, and Details.

### Fixed

- Screw MPR no longer hides the 3D model: it used to clip the whole volume
  and the whole vertebral mesh at one infinite plane, which for an oblique
  screw removed everything cranial to the cut and, at every level, stripped
  the posterior elements of every vertebra, not just the screw's own.
- A level whose own upper-endplate fit is too rough to trust (RMSE above
  3.0 mm, or no fit at all -- an L1/L2-type compression fracture or Schmorl
  node) no longer tilts its screw to follow that broken fit. The planner
  instead aims along the inverse-distance-weighted average of the nearest
  well-fitted levels within two levels above and below (S1 and the sacrum
  neither lend nor borrow), or falls back to a horizontal trajectory with no
  such neighbour -- and reports the endplate angle, in both the planner and
  ScrewTool's re-grade, against the reference actually used.
- The pedicle subregion options had no way to open from the UI (already true
  in 0.2.1); they now open from the **Advanced options** toggle on the
  Segment step.

## [0.2.1] - 2026-09-12

A workflow bar for the three main planning steps, a 3D view of Screw MPR, and
a planner fix that seats screw heads on the dorsal cortex, runs each screw to
the anterior margin, and grades from where the axis enters bone rather than
from the head.

### Added

- A workflow bar above the MPR/3D views reading
  "① Open DICOM → ② Segment → ③ Plan Screws". Each step runs the existing
  action it names. A finished step shows a check mark; the next step is
  highlighted even while disabled, and a disabled step's tooltip says what
  unlocks it. Step 2 needs a real TotalSegmentator mask to count as done
  (a threshold fallback does not); step 3 needs automatically planned
  screws, and deleting them reopens it. Loading a new study resets the bar.
- Screw MPR now has a 3D counterpart: the 3D view shows the three
  screw-aligned planes, each an 80 mm square coloured like the pane header
  that displays it, in place of the standard axial/sagittal/coronal
  indicators. The CT volume and vertebra meshes are cut at the cross-section,
  keeping the tip side, and the cut follows Position; screws are never cut.

### Changed

- **Screw heads are now seated on the dorsal cortex, along the screw's own
  axis** -- the last bone the centreline meets, where a drill would start --
  instead of where the whole cross-section first fitted. A head is rejected
  as unreachable when the same vertebra's bone still lies within 15 mm
  behind it along that axis; this replaces the old 6 mm posterior-cortex
  burial bound.
- **Grading now starts 3 mm past where the screw's axis enters bone**
  (`ENTRY_ZONE_MM`), not at the head, because a head seated on the cortex
  straddled the surface it entered and read as a breach of its own radius.
  This applies to every graded screw: automatic proposals (Optimizer,
  Legacy, and the displayed grade of CBT screws), manual and edited screws,
  and re-grading on segmentation or plan load.
- **Screw length is the longest catalogue length that keeps the Anterior
  margin (4 mm by default) ahead of the tip, measured along the screw's own
  axis** to the anterior vertebral-body cortex, rather than clearance
  required all around the distal cylinder; the distal shaft now only needs
  containment and the configured Wall clearance, like the rest of the shaft.
  Only the longest feasible length per trajectory is ranked, so the Length
  weight compares trajectories rather than lengths on one trajectory.
- The Legacy planner and the automatic per-side legacy fallback validate a
  trajectory first, then try, in order: the head re-seated on the dorsal
  cortex (only if it passes the same 15 mm dorsal-approach test) with the
  tip run out to the Anterior margin; the original head with the tip run
  out the same way; the validated screw unchanged. The first option that
  breaches no more than the validated screw, medially first and then in
  total, is used, so a legacy screw is never made less safe to make it
  longer or to move its head. Cortical bone trajectory (CBT) selection is
  unchanged.
- `Planes On / Off` now applies to whichever plane set is currently shown,
  standard or screw-aligned.
- A plan re-graded on load or after re-running segmentation may show better
  grades than when it was saved by an earlier build, because the entry
  cortex is no longer graded -- nothing in the plan file itself changes.

### Fixed

- The 0.2.0 README and user guides said the app offers four themes ("one
  dark and three light"). There are three, unchanged since 0.1.0: Soft Light,
  Graphite Blue and Graphite Mint; 0.2.0 only changed the default to Graphite
  Blue. The 0.2.0 entry below has been corrected to match.
- Planned screws were short with buried heads: measured along each screw's
  own axis on the sample study, heads sat 5-21 mm inside the lamina and tips
  stopped 5-20 mm short of the anterior cortex. On the sample study (refined
  mask, default settings):

  |                  | before           | after                    |
  |------------------|------------------|--------------------------|
  | screws           | 12               | 13 (S1-left now planned) |
  | legacy fallbacks | 1                | 0                        |
  | grades           | A 10 / B 2       | A 13                     |
  | medial breach    | L5-right 1.41 mm | 0 everywhere             |
  | head burial      | 5-21 mm          | 0-0.2 mm                 |

  Lengths (left / right) became L1 35/35, L2 50/45, L3 35/40, L4 40/35,
  L5 45/35, T12 35/25 mm, where most sides had been 25 mm before (L1, L2
  and L4 on the left among them).
- While Screw MPR was active, the 3D view kept drawing the standard
  axial/sagittal/coronal planes and left the volume uncut.

## [0.2.0] - 2026-09-11

The first release after the public baseline. It adds automatic trajectory
optimisation, a clinical policy for narrow pedicles, cortical bone
trajectories, CT-guided mask refinement, and a substantially reworked
interface.

### Added

**Planning**

- Multi-objective trajectory optimiser, now the **default** planning mode. It
  generates candidate trajectories over convergence, craniocaudal angle, entry
  position, and length, then scores them on safety, bone density, length,
  endplate agreement, and pedicle centring. The previous rule-based planner
  remains available as **Legacy** mode and as an automatic per-side fallback.
- Construct-level harmonisation: entry points on each side are fitted to a rod
  line and neighbouring convergence angles are pulled together, without letting
  any screw fall below 90 % of its own best safety score. Reported as
  `Construct: rod fit … · convergence spread …` in the status bar and as the
  inspector's **Alignment** row.
- **Narrow-pedicle policy.** A pedicle is never a reason to skip a side. Below
  the configurable narrow threshold (5.0 mm by default) the level takes the
  smallest catalogue implant, is drawn in red, and is planned with the medial
  (canal-side) wall protected: medial breach 0, lateral breach capped, entry
  walked laterally until both hold. This is the in-out-in technique made
  explicit rather than left to the surgeon to discover.
- **Parallel to upper endplate** planning option (on by default) with a
  configurable tolerance band, and a signed endplate angle reported per screw.
- **Cortical bone trajectory (CBT)** planning mode with consensus default
  diameters and lengths, its own entry landmark at the inferomedial isthmus
  corner, and a contraindication note on every CBT screw.
- Editable planning parameters in the UI: pedicle fill ratio, wall clearance,
  anterior margin, maximum convergence, trajectory HU threshold, narrow-pedicle
  threshold, narrow lateral-breach cap, endplate tolerance, planner mode,
  trajectory type, and the three optimiser weights.
- Gertzbein–Robbins grading from cropped Euclidean distance maps, split by
  direction into medial, lateral, and craniocaudal breach, plus the remaining
  medial wall thickness.
- Bone-quality and breach metrics on every screw: trajectory/pedicle/vertebral
  body HU with literature thresholds, trajectory-to-body HU ratio, Heary breach
  direction, and facet violation grade.
- `scripts/validate_plans.py` and plan comparison metrics (mean absolute
  deviation, axis angle, cylinder Dice, Bland–Altman) for comparing two plans.
- `scripts/check_narrow_policy.py`, a Qt-free acceptance CLI that runs the
  analyser and planner against a real study and fails on a narrow-pedicle
  policy violation.

**Segmentation and measurement**

- **CT-guided mask refinement** after TotalSegmentator: per-label anti-aliasing,
  a boundary band re-decided against the CT, 3D hole filling, and largest-
  component selection. This removes the stair-stepping caused by upsampling a
  1.5 mm inference mask onto a sub-millimetre CT grid. Cancellable, with a
  volume-change guard that rejects a refinement that moved too much.
- Optional locally installed pedicle subregion nnU-Net stage that feeds the
  analyser label-based isthmus boundaries (source installs only).
- Robust pedicle width measurement: centroid continuity tracking across slices,
  sliver area and width floors, a neighbourhood-median width checked against an
  inscribed-diameter estimate, a per-level plausibility band with an axial
  second opinion, and a tilt guard on the fitted pedicle axis.
- Upper endplate plane fitting with a reported RMSE and a warning when the fit
  is too rough to trust.
- Cancel button that terminates the TotalSegmentator subprocess.

**Interface**

- Axial view opens anterior-up in radiological convention, with **A / P / L / R**
  orientation letters on every MPR pane.
- **Screw MPR panes are now interactive**: rotate the oblique planes around the
  screw axis, offset them along and across the screw, and scroll or drag
  without leaving screw-aligned review. Rotation and offset are shown in the
  pane readout.
- **Screw Review table** with per-screw level, side, pedicle width, diameter,
  length, grade chip, and source, replacing the plain list.
- Planning cockpit above the control tabs: diameter, length, convergence,
  craniocaudal angle, endplate angle, construct alignment, grade, trajectory
  HU, body HU, wall margin, facet, Heary direction, trajectory type, and
  pedicle verdict for the selected screw.
- Tabbed control panel (Study / Planning / Tools) and tool icons.
- Pane maximise by header double-click or Ctrl+M, which follows the pane the
  pointer is working in.
- JSON plan schema v3 carrying screw metadata, signed angles, and the full
  metric bundle; CSV export gains pedicle width, narrow-pedicle and
  width-uncertain flags, directional breaches, and the endplate angle.

**Project**

- macOS and Windows standalone desktop packaging, with TotalSegmentator bundled.
- Windows PowerShell launcher (`scripts/run_app.ps1`).
- Continuous integration running ruff and pytest on every push and pull request.
- MIT license, academic attribution, `CITATION.cff`, and third-party notices.
- This changelog.

### Changed

- **Default planner mode is now Optimizer.** A persisted `legacy` mode from an
  earlier build is migrated once, with a note in the status bar; a Legacy mode
  chosen after that migration is respected.
- **Default wall clearance is now 0 mm.** At 1 mm the optimiser could not
  satisfy the containment rule on most sides of a real study and silently fell
  back to the legacy planner. A persisted clearance of exactly 1.0 mm (the old
  default, rather than a choice) is reset once, with a note in the status bar;
  any other persisted value is left alone.
- A width the analyser cannot stand behind is reported as **not trusted**
  rather than as narrow, in the plan table, the cockpit, the warnings, and the
  CSV export. The narrow policy still applies to it; only the wording changed,
  because the plausibility band rejects implausibly *wide* measurements too.
- GPU-capable volume rendering (`vtkSmartVolumeMapper`) on every platform except
  macOS, where the OpenGL-to-Metal translation layer stalls on 3D texture
  upload and the CPU ray caster stays. A smart mapper that turns out to be ray
  casting on the CPU re-downsamples rather than grinding at full resolution.
- Screw diameter is never left unset: a pedicle too small for the smallest
  catalogue implant still receives one, marked, instead of leaving the side bare.
- The default theme is now the dark Graphite Blue; it was the light Soft
  Light.  The same three themes are offered, and a theme you already chose
  is kept.
- Plan files assert that they carry no patient identifiers.
- Ruff import sorting and lint rules apply to the whole tree; file-wide ignores
  were retired.

### Fixed

- **Only a fraction of levels were planned.** The coronal isthmus search kept
  the component nearest the midline, so a 1.6 mm stair-step sliver could
  displace the real 7.4 mm pedicle and the level was then dropped as too narrow.
  On the sample study the measured L3 widths went from 3.1 / 4.7 mm to
  8.6 / 8.8 mm.
- **The optimiser fell back to the legacy planner on most sides.** Candidate
  generation swept convergence as an offset from the pedicle axis while scoring
  filtered the absolute angle, so levels with a strongly converging axis had
  every candidate rejected. On the sample study this moved the grades from
  A:2 / B:10 with ten legacy fallbacks to A:10 / B:2 with one.
- A pedicle the analyser never found was read back as a 0.0 mm measurement,
  painting a manually placed screw red and warning that a 4.0 mm implant filled
  100 % of a pedicle that was never measured.
- Re-running segmentation on the same study re-graded every screw against the
  new mask while re-deriving its pedicle width, narrow verdict, and endplate
  angle from analyses computed on the old one.
- The axial width re-check accepted the nearest non-body component with no
  distance or side guard, so a transverse-process fragment could silently
  replace a real isthmus measurement.
- Construct alignment metrics were measured once at plan time and never
  refreshed, so the inspector's Alignment row described a trajectory the user
  had already dragged away from.
- CBT screws never carried an upper-endplate normal, so their endplate angle
  appeared only after the first drag.
- A one-time settings migration could mark itself done while the value it
  existed to retire survived, if any other planner key in the store was
  unreadable.
- Ctrl+M maximised the last-maximised pane rather than the pane in use.
- Orientation letters and the Screw MPR rotation pivot used Qt logical pixels
  instead of device pixels on HiDPI displays.
- A 3D pane restored from maximise could stay blank until the next interaction.
- Oblique volume resampling assigned axes in a way that was not always a true
  permutation.
- Numerous grading, drag-frame, and plan-lifecycle fixes: grades and metrics no
  longer blink or go stale during a drag that passes outside the mask, a
  replaced planning run is stopped, and per-level analyses are dropped when the
  study they describe is replaced.

### Migration notes

- Two one-time settings migrations run on first launch and announce themselves
  in the status bar: planner mode `legacy` → `optimizer`, and wall clearance
  `1.0 mm` → `0 mm`. Both are recorded so they never run twice, and anything
  you choose afterwards is respected.
- Plans saved by 0.1.0 load unchanged. Plans saved by 0.2.0 use schema v3 and
  carry metrics that earlier versions ignore.
- The CSV export gained columns. They are appended, never reordered, so a
  reader that keys on the header row keeps working.

## [0.1.0] - 2026-07-12

Initial public baseline.

- Multi-series DICOM CT loading with LPS reorientation and oblique resampling.
- Synchronized axial, sagittal, and coronal MPR with pan, zoom, and
  window/level.
- VTK volume rendering and per-vertebra 3D meshes.
- Optional GPU-first TotalSegmentator integration with CPU retry.
- Multi-level vertebra selection and rule-based automatic screw proposals.
- Standard and screw-aligned oblique MPR review.
- Direct entry, tip, and whole-screw editing in MPR and 3D.
- Manual screw placement, distance measurement, and angle measurement.
- JSON plan save/load, CSV and STL export.
- Three UI themes.

[0.2.3]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/grotyx/pedicle-screw-simulator/releases/tag/v0.1.0
