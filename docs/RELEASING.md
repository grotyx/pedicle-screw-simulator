# Releasing RadWright Screw to Users

[English](RELEASING.md) | [한국어](RELEASING.ko.md)

How a version of RadWright Screw (internal name Pedicle Screw Simulator) goes from
`main` to people's computers, and what is still missing before a wider release.
Build details are in [Desktop Build Guide](BUILDING_DESKTOP.md).

## What users get

| Platform | File | How it installs | Updates |
|:--|:--|:--|:--|
| Windows x64 + NVIDIA GPU | `PedicleScrewSimulator-<ver>-Windows-x64-Setup.exe` + `…-Setup-1.bin` | Per user, `%LOCALAPPDATA%\Programs\RadWright Screw`, no administrator prompt | Automatic, changed layers only |
| Windows x64 (portable) | `PedicleScrewSimulator-<ver>-Windows-x64.zip` | Extract anywhere | None (download again) |
| macOS Apple Silicon | `PedicleScrewSimulator-<ver>-macOS-arm64.zip` | Extract, right-click > Open | None |

Requirements to tell users: Windows 10/11 64-bit, an NVIDIA GPU with a current driver
(segmentation runs on the GPU), about 6.5 GB free disk for the app (the Spine Only models
are included), and an internet connection for updates. Only the Total task or Fast quality
downloads extra models (up to 1.4 GB) on first use.

## Release steps

1. **Prepare `main`.** Move `[Unreleased]` into a new version section in both
   changelogs; bump `VERSION`, `CITATION.cff`, README / USER_GUIDE / BUILDING_DESKTOP
   pairs and `THIRD_PARTY_NOTICES.md` (`tests/test_version.py` checks them). Run the
   checks in `AGENTS.md`.
2. **Tag and build.** `git tag -a vX.Y.Z -m "RadWright Screw X.Y.Z"` and push the tag.
   The **Build desktop packages** workflow builds Windows (ZIP, installer, update layers)
   and macOS, runs `--self-check`, and uploads the artifacts. A local Windows build
   (`scripts\build_desktop.py`) produces the same files.
3. **Smoke-test as a normal user.** On a Windows account *without* administrator rights:
   run the installer, start the app from the Start menu, load a sample CT, run
   segmentation once, then uninstall. (RadWright Spine shipped a build that every user
   could not start because the developer PC always ran as administrator.)
4. **Publish to R2.** Everything users download is hosted on R2, because GitHub limits
   each file to 2 GB and the installer `.bin` is about 2.8 GB. Put the macOS ZIP in
   `release\` next to the Windows files, then from the Windows build machine run
   `python scripts\publish_update.py release\update --downloads <the .exe, .bin and
   ZIPs> --dry-run` (the exact command is in the build guide). Check that only the
   expected layers are new, then run it without `--dry-run`. The installer and ZIPs land
   in `radwright-screw/downloads/<version>/`; installed apps see the new version on
   their next start.
5. **Create the GitHub Release** (`gh release create vX.Y.Z --verify-tag --notes-file
   <notes>`) with the release notes and the R2 download links, no attached files.
6. **Announce.** Release notes in plain language: what changed for the user, that the
   software is for research and education only, and any action needed.

## Rolling back

Copy `radwright-screw/versions/<previous>/components.json` over
`radwright-screw/latest/components.json` in R2. Apps that have not updated yet stay on
the previous version. Apps that already updated keep the new version; ship a fixed
version with a higher number.

## Before a wider release (not done yet)

- **Download page.** RadWright Spine has a site with the latest version, download links,
  requirements and SmartScreen instructions. RadWright Screw has none yet; it would link
  to `radwright-screw/downloads/<version>/` on R2.
- **User manual.** `docs/USER_GUIDE*.md` is a developer-facing reference. A user manual
  (installation, first plan, reading the grades, limits) is to be written and reviewed
  by the owner before release.
- **Terms of use.** RadWright Spine asks users to accept a EULA in the installer and on
  first start. RadWright Screw shows a research-use banner but has no EULA; decide on
  the text (research and education only, not a medical device, the user verifies every
  result) before handing it to people outside the lab.
- **Code signing.** The installer and app are unsigned, so Windows SmartScreen warns on
  first run (**More info** > **Run anyway**). A code-signing certificate removes the
  warning.
- **Signed update manifest.** The app trusts `latest/components.json` because it comes over
  HTTPS from our own domain, and checks each layer's SHA-256 against it. Anyone who gets the
  R2 write keys could therefore push code to every installed copy. Signing the manifest with
  an offline Ed25519 key (public key built into the app) and Authenticode-signing the binaries
  closes that; until then keep `~/.spinemetrics/r2.env` on the build PC only.
- **First R2 upload.** Nothing has been published under `radwright-screw/` yet; until
  then the update check finds nothing and stays silent.
- **Logo.** Done: `packaging/icons/` (RadWright family style, an axial vertebra with two
  long pedicle screws). It is in the executable, the installer, the macOS app and the window;
  the wordmarks are ready for a download page and the manual.
