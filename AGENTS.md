# Agent instructions

Rules for any coding agent working in this repository (Claude Code, Codex,
OpenCode, or others). `CONTRIBUTING.md` still applies; this file adds what an
agent needs to know that a human contributor would pick up in review.

Pedicle Screw Simulator is research and education software, not a certified
medical device. Never describe an automatic result as clinically validated.

## Patient data (hard rules)

- Never open, list, read, copy, or reference anything under `data/` except
  `data/README.md`. It can hold real clinical CT. This includes reading it
  "just to check" and pointing a script or test at it.
- Tests use synthetic phantoms built in code only. Never add DICOM, NIfTI,
  segmentation masks, screenshots, logs, plans, or meshes from a real study.
- Never print, log, or echo patient identifiers (names, IDs, birth dates,
  study dates, accession numbers, institution names), including in commit
  messages and agent reports.

## Git

- Do not commit unless your task says you are the reviewer. Implementers leave
  their changes in the working tree for review.
- Stage files by explicit path. Never `git add -A`, `git add -u`, or
  `git add .`.
- Never run bare `git stash` or `git stash pop`: the stash is shared by every
  worktree and every concurrent agent. Set work aside with a WIP commit.
- Never push, tag, rewrite history, or delete branches unless the task says
  so.
- Commit subjects follow `type(scope): summary` (`feat`, `fix`, `docs`,
  `refactor`, `test`, `chore`), in the imperative, with a body that says why.

## Making a change

1. Write or update a regression test first and see it fail.
2. Keep one problem per change; do not refactor unrelated code.
3. Match the surrounding code: naming, comment density, idiom.
4. Keep DICOM LPS geometry intact through every coordinate transform, and
   never let automatic planning fabricate a screw when it fails.
5. Update the documentation when controls, workflows, requirements, or
   outputs change. English and Korean files are pairs and change together:
   `README.md` / `README.ko.md`, `docs/USER_GUIDE.md` /
   `docs/USER_GUIDE.ko.md`, `CHANGELOG.md` / `CHANGELOG.ko.md`, and the other
   `*.ko.md` files.
6. Record user-visible changes under `## [Unreleased]` in both changelogs
   (Keep a Changelog: Added / Changed / Fixed).
7. `VERSION` is the single source of truth for the version. Change it only in
   a release task; `tests/test_version.py` checks every file derived from it.

## Checks

Run these before reporting a change as done, and report the real output:

```bash
python -m pytest tests -p no:cacheprovider
python -m ruff check src tests scripts main.py
git diff --check
```

- `pyproject.toml` already passes `-q` to pytest. Adding another `-q` hides
  the pass/fail summary line.
- On Windows, run anything that opens a VTK/OpenGL window (the app, GUI
  drives) from PowerShell; Git Bash cannot create a GL context. Tests run
  offscreen from either shell.
- If a check fails, say so with the output. Do not weaken or skip a test to
  make it pass.

## Roles in a multi-agent run

- **Planner:** reads the code and writes a concrete plan (files, functions,
  tests, acceptance checks). Does not edit files.
- **Implementer:** follows the plan in its own worktree, runs the checks, and
  reports what changed and the check results. Does not commit.
- **Reviewer:** reads the diff against the plan, runs the checks itself, and
  either sends back specific fixes or stages by path and commits.
