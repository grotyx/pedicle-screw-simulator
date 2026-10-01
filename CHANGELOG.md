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

## [0.2.5] - 2026-10-02

Review tools on top of 0.2.4's safer planning: mark each screw as
reviewed, read warnings sorted by kind, show the rod line, and export a
one-file planning report. A script compares the old and new planning
defaults on your own study.

### Added

- Review page: a "✓ Reviewed" button marks the selected screw as checked.
  The mark shows in the Construct map and the screw table, is saved in the
  plan file, and is cleared by any change to the screw's position, size or
  grade; the Review step shows done once every screw is reviewed.
- Review warnings are sorted into Safety, Image and Info: the table's ⚠
  column shows a chip coloured by the worst category, and the tooltip and
  the Review warning line group the lines by category ("⚠ 1 safety ·
  2 image"). The categories sort lines for reading; they are not a risk
  rating.
- A "Rod Line" toggle in the 3D view (off by default) draws each side's
  straight line through the screw heads, the line the Alignment figure is
  measured against, in 3D and as a dashed projection on the sagittal and
  coronal views. It is not a rod-bending plan.
- File > Export Report... writes one self-contained HTML planning report
  (planner settings, per-screw table, construct map, pane screenshots)
  under a research-use banner and without patient identifiers.
- scripts/compare_planning_defaults.py compares, on one study, the planning
  result of the previous behaviour (grade-B breaches accepted, narrow sides
  always placed) with the current defaults, level by level.

### Changed

- The screw table's pedicle-width column header reads "Ped." (its tooltip
  is unchanged) so Level stays readable next to the reviewed marks.

### Fixed

- The Screw MPR 3D slice uses vtkImageBinaryThreshold when the installed
  VTK has it, so newer VTK no longer warns about a deprecated class.

## [0.2.4] - 2026-10-01

Safer automatic planning and a new workstation layout. By default automatic
planning now places no screw with a cortical breach and never one that
breaches into the canal; a narrow pedicle that cannot hold a contained
screw is left unplanned with a reason unless you opt in; and re-running
Plan Screws asks before it replaces earlier automatic screws. The views
are rearranged around one large main view with a vertical step rail, a
tool dock and a level-by-level construct map.

### Added

- A Construct map at the top of the Review page summarizes the plan level by
  level, cranial to caudal, as Right | vertebra | Left grade chips with
  diameter x length. Screws without a level are listed under Manual, screws
  without a known side in the centre column. Clicking a chip selects that
  screw.
- Loading a study warns (without blocking) when slice spacing is not
  uniform, which usually means missing slices, or when the series has a
  gantry tilt; either can distort lengths along the scan axis.
- The user guide lists every keyboard shortcut.

### Changed

- Automatic planning accepts no cortical breach by default (Gertzbein grade
  A). The new Planning Parameters setting "Accepted breach" ("A only — no
  breach", "Up to B (< 2 mm)", "Up to C (< 4 mm)") replaces "Lateral breach
  cap", which used to let a narrow pedicle breach laterally by up to 2 mm;
  a previously saved cap is ignored. A narrow pedicle may only breach
  laterally, within the accepted grade. A medial (canal-side) breach is
  never accepted on any pedicle; before, a normal-width screw from the
  legacy fallback could be kept with a grade-B breach into the canal.
- A narrow pedicle whose best trajectory is not contained (a breach beyond
  the accepted grade, or any medial or craniocaudal breach) is left
  unplanned, and the skip reason names the figure that failed. Before, the
  legacy fallback placed its smallest screw anyway. The new option "Place
  narrow screws even if not contained" (off by default) restores that
  placement, and such a screw carries a warning naming the option.
- Running Plan Screws again asks before replacing the automatic screws on
  the levels being planned, including ones you adjusted. A new screw
  replaces the automatic screw on the same level and side, and only after
  the new plan succeeds, so the rod-fit figures no longer mix two runs.
  Automatic screws on a level or side the new plan skips, hand-placed
  screws and screws on other levels are kept. Before, a second run added a
  second set of screws.
- CBT holds its anterior margin along the screw like the optimizer and keeps
  the longest feasible length per direction (zero breach and wall clearance
  still required), so it no longer drops or over-shortens screws the
  optimizer's rule accepts. The chosen CBT screw can differ from before.
- The planner's density objective (optimizer and CBT) averages HU only over
  screw-cylinder voxels inside the planned vertebra, matching the per-screw
  HU figures. Soft tissue, fat or CSF outside the bone no longer lowers a
  candidate's density, and a candidate with no sample in the vertebra scores
  the worst density. The chosen screw can differ from before.
- Implant catalogues are sorted on load and the automatic diameter is always
  a catalogue size.
- The four workflow steps run down a vertical step rail at the left edge
  instead of a bar above the views. Each right-hand page opens with a header
  ("Step N of 4", title, one-line hint).
- The Planning layout shows one large main view (3D by default) with the
  other three views as thumbnails underneath. The enlarge button (tooltip
  "Show in the main view") on a thumbnail swaps it into the main view;
  header double-click still maximises. Turning Screw MPR on makes the axial
  view the main view. MPR Focus (2x2) is unchanged.
- The tool buttons (Select, Add Screw, Distance, Angle, Screw MPR, Fit MPR,
  Fit 3D) moved from the top toolbar into a tool dock on its own row below
  the main view. The 3D zoom in/out and fit commands are also in the View
  menu.
- The top toolbar keeps Open DICOM and adds a study chip (voxel spacing
  only, for example "CT · 0.39 × 0.39 × 1.00 mm") and a RESEARCH USE ONLY
  badge.
- The tool dock and the floating controls over the views (MPR zoom buttons,
  3D overlays) share one flat, theme-driven style in all three themes.
- Plan Save, Load and Export dialogs open in the last plan folder (or
  Documents), never the application folder, and root-level plan exports are
  git-ignored. Plans store only a SHA-256 digest of the series UID; older
  plans with the raw UID still load and are not re-saved with it.
- Measurements taken in Screw MPR are marked "(screw view)" and keep that
  mark in saved plans. They can no longer be edited or dragged, because
  their oblique plane is not recorded; delete and re-measure them. They no
  longer jump to a standard slice they do not lie on.

### Fixed

- A pedicle measured by the axial fallback reported up to three times its
  width: the width was taken from the isthmus slice and its two neighbours
  pooled together. It is now measured on the isthmus slice alone.
- A screw whose tip ran through the anterior cortex had its Heary breach
  direction reported as "none". The direction is now read from the nearest
  point of the vertebra rather than from the screw's own centreline, so tip
  breaches read "anterior".
- Trajectory and pedicle HU (Details, CSV mean_hu/min_hu and
  trajectory_mean_hu) are sampled only inside the screw's own vertebra
  label, so soft tissue, fat and the spinal canal no longer lower the mean
  or set the minimum. A screw with no samples inside the vertebra shows N/A
  instead of a false loosening-risk warning. The vertebral-body HU region
  no longer copies the whole CT and mask for every screw.
- The endplate band uses the true 3-D angle to the endplate, so a coronally
  tilted endplate is judged the same on both sides.
- Optimizer warnings for a level without an endplate reference no longer
  claim a horizontal trajectory.
- The legacy planner skips a side whose posterior surface is cut off instead
  of entering at the anterior cortex.
- Typing a diameter with a decimal point ("7.5", "5,5") entered the wrong
  value (5.0). Typed values are now taken as written; the digit-only
  shorthand ("65" → 6.5 mm) still works.
- DICOM load and scan errors no longer write the folder path (often named
  after the patient) or any exception text that may contain it to app.log
  or the error dialog. The log records only the error type, with code
  locations at debug level, and the dialog shows a fixed message. Loading
  no longer re-reads every slice to check its geometry.
- An empty SliceThickness or KVP tag no longer discards all study metadata
  ("Loaded 0 slices").
- Unsigned oblique series no longer turn the -1000 background into bone
  values after reorientation.
- The views no longer stay frozen and black when a step of a study load
  fails.
- Opening a new study left the previous study's screw projections drawn on
  the axial, sagittal and coronal views, and kept its vertebral-only volume.
- Plan loading rejects NaN/Infinity values, malformed measurement entries
  and plans from a newer version with a clear error. Saving a plan is
  atomic, so a crash cannot truncate the existing file, and no longer fails
  when a measurement has no screw-view entry. CSV export neutralises text
  cells that start with =, +, - or @.
- Finishing or cancelling a segmentation run could crash the app ("QThread:
  Destroyed while thread is still running"). Worker threads are now kept
  until they stop, and planning threads are freed promptly.
- Cancel now stops TotalSegmentator's and the subregion model's worker
  processes too, not just the parent process, without freezing the window
  while they are stopped. Quitting the app no longer leaves
  TotalSegmentator running.
- Earlier segmentation runs' temporary CT copies are deleted after a
  successful run, and startup survives another instance cleaning up at the
  same time.
- A failure reading a GPU segmentation result is reported instead of
  silently rerunning the whole segmentation on the CPU.
- MPR reference lines now follow the current slice positions after
  scrolling or clicking in any view, after loading a study and on entering
  or leaving Screw MPR, and each line uses the colour of the plane it marks
  (all three views had them swapped).
- The Angle tool no longer gets stuck when two of its points coincide, and
  changing the measure mode while editing a measurement keeps its plane.
- Starting a screw drag in one view releases any other view's drag lock, so
  one screw cannot be moved by two views at once.
- The Side column of the screw table shows L or R, so it no longer
  truncates to "L…" at the default panel width.
- Stepping the segmentation label no longer freezes the window: changes are
  coalesced and recently built surfaces are cached. Stepping the 3D label
  through an empty label no longer re-shows a hidden overlay.
- 3D and MPR panes repaint after the window is minimised and restored.
- The Screw MPR 2D mask overlay no longer clips on oblique planes.
- CI and the run scripts no longer pass a second -q to pytest (the summary
  line was hidden), and CI now runs git diff --check. validate_plans --out
  no longer truncates dotted names, and run_app.sh stops with a clear
  message when python3 is older than 3.12. The CITATION release date now
  follows the changelog (a test checks it), and CONTRIBUTING names the project and its checks
  correctly.
- The re-plan confirmation shows its question in the dialog body, so it is
  visible on macOS, which hides message-box titles.

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

[Unreleased]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/grotyx/pedicle-screw-simulator/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/grotyx/pedicle-screw-simulator/releases/tag/v0.1.0
