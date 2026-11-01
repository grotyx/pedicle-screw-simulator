"""
Layered self-update for the Windows installer build.

The PyInstaller folder is split into three layers that change at different
rates: ``cuda`` (the CUDA DLLs in ``_internal/torch/lib``, several GB, changes
only with torch), ``runtime`` (every other third-party package under
``_internal``) and ``app`` (the executable, which carries our code, plus the
loose files). Each layer is identified by a hash of its file list and file
hashes, so a release only uploads, and a client only downloads, the layers
whose content actually changed.

The server holds ``latest/components.json`` and immutable
``layers/<name>-<id>.zip`` archives. The installed copy keeps its own
``components.json`` beside the executable; comparing the two gives the plan.
Files are swapped by a PowerShell script after the app exits, because the
running executable cannot replace itself.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src import __version__

logger = logging.getLogger(__name__)

MANIFEST_NAME = "components.json"
PRODUCT = "RadWright Screw"
#: Where releases are published; ``PSS_UPDATE_URL`` overrides it (testing).
DEFAULT_UPDATE_URL = "https://dl.spinemetrics.sangmin.me/radwright-screw"
#: Inno Setup's uninstall key for this AppId (see packaging/PedicleScrewSimulator.iss).
UNINSTALL_KEY = (
    r"HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall"
    r"\{7FC11677-0A42-427F-A57C-60FED2B6B1FD}_is1"
)
CUDA_PREFIX = "_internal/torch/lib/"
_CHUNK = 1 << 20
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+[A-Za-z0-9.-]*")
_ARCHIVE_RE = re.compile(r"layers/[a-z]+-[0-9a-f]{16}\.zip")
_PART_RE = re.compile(r"[A-Za-z0-9 ._+()@-]+")


class UpdateError(RuntimeError):
    """An update could not be checked, downloaded or staged."""


@dataclass
class UpdatePlan:
    version: str
    layers: List[str]
    download_bytes: int


def update_url() -> str:
    return os.environ.get("PSS_UPDATE_URL", DEFAULT_UPDATE_URL).rstrip("/")


def layer_of(rel: str) -> str:
    """Layer for a POSIX path relative to the application folder."""
    if rel.startswith(CUDA_PREFIX):
        return "cuda"
    parts = rel.split("/")
    if parts[0] == "_internal" and len(parts) > 2:
        return "runtime"
    return "app"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def build_layers(dist: Path, version: str, out: Path) -> Dict:
    """Write one ZIP per layer under ``out/layers`` and the manifest.

    The manifest is written to ``out`` and into ``dist`` itself, so the
    installer ships it and the installed copy knows which layers it has.
    """
    dist, out = Path(dist), Path(out)
    files: Dict[str, List[str]] = {"cuda": [], "runtime": [], "app": []}
    for path in sorted(dist.rglob("*")):
        rel = path.relative_to(dist).as_posix()
        if path.is_file() and rel != MANIFEST_NAME:
            files[layer_of(rel)].append(rel)

    (out / "layers").mkdir(parents=True, exist_ok=True)
    layers = {}
    for name, rels in files.items():
        listing = "".join(f"{rel}\0{_sha256(dist / rel)}\n" for rel in rels)
        layer_id = hashlib.sha256(listing.encode("utf-8")).hexdigest()[:16]
        archive = f"layers/{name}-{layer_id}.zip"
        target = out / archive
        if not target.exists():
            # Stored, not deflated: the big DLLs barely compress and the layer
            # is fetched once per change.
            with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                for rel in rels:
                    zf.write(dist / rel, rel)
        layers[name] = {
            "id": layer_id,
            "archive": archive,
            "size": target.stat().st_size,
            "sha256": _sha256(target),
            "files": rels,
        }
    manifest = {"product": PRODUCT, "version": version, "layers": layers}
    text = json.dumps(manifest, indent=1)
    (out / MANIFEST_NAME).write_text(text, encoding="utf-8")
    (dist / MANIFEST_NAME).write_text(text, encoding="utf-8")
    return manifest


def _safe_rel(rel) -> bool:
    """A relative POSIX path that cannot leave the application folder."""
    if not isinstance(rel, str) or not rel:
        return False
    parts = rel.split("/")
    return all(part not in ("", ".", "..") and _PART_RE.fullmatch(part) for part in parts)


def validate_manifest(manifest) -> Dict:
    """Reject a manifest whose version, archive names or file paths are unsafe.

    The server's manifest drives file writes, deletes and the apply script,
    so nothing from it may point outside the install or staging folders.
    """
    try:
        if not _VERSION_RE.fullmatch(manifest["version"]):
            raise UpdateError(f"Update manifest has an invalid version: {manifest['version']!r}")
        for name, layer in manifest["layers"].items():
            if name not in ("cuda", "runtime", "app"):
                raise UpdateError(f"Update manifest has an unknown layer: {name!r}")
            if not _ARCHIVE_RE.fullmatch(layer["archive"]):
                raise UpdateError(f"Update manifest has an invalid archive: {layer['archive']!r}")
            bad = [rel for rel in layer["files"] if not _safe_rel(rel)]
            if bad:
                raise UpdateError(f"Update manifest has an unsafe path: {bad[0]!r}")
            int(layer["size"]), str(layer["sha256"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise UpdateError(f"Update manifest is malformed: {exc}") from exc
    return manifest


def _version_key(version: str):
    match = re.match(r"(\d+)\.(\d+)\.(\d+)(.*)", version.strip())
    if not match:
        return (0, 0, 0, 0, version)
    major, minor, patch, rest = match.groups()
    # A pre-release ("rc1", "-beta") sorts before the release itself.
    return (int(major), int(minor), int(patch), 0 if rest else 1, rest)


def is_newer(candidate: str, current: str) -> bool:
    return _version_key(candidate) > _version_key(current)


def plan_update(local: Dict, remote: Dict) -> Optional[UpdatePlan]:
    """Layers to download, or None when the server has nothing newer."""
    if not is_newer(remote.get("version", ""), local.get("version", "")):
        return None
    changed = [
        name
        for name, layer in remote["layers"].items()
        if local.get("layers", {}).get(name, {}).get("id") != layer["id"]
    ]
    size = sum(remote["layers"][name]["size"] for name in changed)
    return UpdatePlan(remote["version"], changed, size)


def load_local_manifest(install_dir: Path) -> Optional[Dict]:
    path = Path(install_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _open(url: str, timeout: float):
    # Cloudflare in front of R2 answers 403 to urllib's default "Python-urllib" agent.
    request = urllib.request.Request(url, headers={"User-Agent": f"RadWright-Screw/{__version__}"})
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_manifest(base_url: str, timeout: float = 15.0) -> Dict:
    url = f"{base_url}/latest/{MANIFEST_NAME}"
    try:
        with _open(url, timeout) as response:
            manifest = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise UpdateError(f"Could not read the update manifest: {exc}") from exc
    return validate_manifest(manifest)


def download_layer(
    base_url: str,
    layer: Dict,
    dest: Path,
    progress: Optional[Callable[[int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Download one layer archive and verify its checksum.

    An archive already downloaded with the right checksum is reused, so an
    interrupted update resumes per layer.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(layer["archive"]).name
    if target.is_file() and _sha256(target) == layer["sha256"]:
        if progress:
            progress(layer["size"])
        return target
    partial = target.with_suffix(".part")
    digest = hashlib.sha256()
    try:
        with _open(f"{base_url}/{layer['archive']}", 60) as response, \
                open(partial, "wb") as handle:
            for block in iter(lambda: response.read(_CHUNK), b""):
                if cancelled and cancelled():
                    raise UpdateError("Update download cancelled")
                handle.write(block)
                digest.update(block)
                if progress:
                    progress(len(block))
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"Could not download {layer['archive']}: {exc}") from exc
    except UpdateError:
        partial.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != layer["sha256"]:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"{layer['archive']} failed its checksum")
    partial.replace(target)
    return target


def stage_update(plan: UpdatePlan, local: Dict, remote: Dict, archives: List[Path], staging: Path) -> Path:
    """Extract the downloaded layers and write the plan the apply script follows."""
    staging = Path(staging)
    files_dir = staging / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True)
    copy: List[str] = []
    remove: List[str] = []
    validate_manifest(remote)
    for name, archive in zip(plan.layers, archives, strict=True):
        new_files = remote["layers"][name]["files"]
        with zipfile.ZipFile(archive) as zf:
            if sorted(zf.namelist()) != sorted(new_files):
                raise UpdateError(f"{Path(archive).name} does not match its file list")
            zf.extractall(files_dir)
        copy += new_files
        old_files = local.get("layers", {}).get(name, {}).get("files", [])
        remove += sorted(set(old_files) - set(new_files))
    (files_dir / MANIFEST_NAME).write_text(json.dumps(remote, indent=1), encoding="utf-8")
    copy.append(MANIFEST_NAME)
    (staging / "apply.json").write_text(
        json.dumps({"version": plan.version, "files": copy, "remove": remove}), encoding="utf-8"
    )
    return staging


_APPLY_SCRIPT = r"""$ErrorActionPreference = 'Stop'
$staging = '__STAGING__'
$install = '__INSTALL__'
$exe = '__EXE__'
$log = Join-Path $staging 'apply.log'
try { Wait-Process -Id __PID__ -Timeout 300 -ErrorAction SilentlyContinue } catch {}
Start-Sleep -Seconds 1
$plan = Get-Content -LiteralPath (Join-Path $staging 'apply.json') -Raw | ConvertFrom-Json
$root = [System.IO.Path]::GetFullPath($install).TrimEnd('\') + '\'
foreach ($rel in @($plan.remove) + @($plan.files)) {
    if (-not [System.IO.Path]::GetFullPath((Join-Path $install $rel)).StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        "failed: unsafe path $rel" | Out-File -LiteralPath $log -Encoding utf8
        Start-Process -FilePath $exe
        exit 1
    }
}
$backup = Join-Path $staging 'backup'
$moved = New-Object System.Collections.ArrayList
$copied = New-Object System.Collections.ArrayList
try {
    foreach ($rel in @($plan.remove) + @($plan.files)) {
        $dst = Join-Path $install $rel
        if (Test-Path -LiteralPath $dst -PathType Leaf) {
            $bak = Join-Path $backup $rel
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $bak) | Out-Null
            Move-Item -LiteralPath $dst -Destination $bak -Force
            [void]$moved.Add($rel)
        }
    }
    foreach ($rel in $plan.files) {
        $dst = Join-Path $install $rel
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Copy-Item -LiteralPath (Join-Path (Join-Path $staging 'files') $rel) -Destination $dst -Force
        [void]$copied.Add($rel)
    }
    if (Test-Path -LiteralPath '__KEY__') {
        Set-ItemProperty -LiteralPath '__KEY__' -Name DisplayVersion -Value $plan.version
        Set-ItemProperty -LiteralPath '__KEY__' -Name DisplayName -Value ('__PRODUCT__ ' + $plan.version)
    }
    "applied $($plan.version)" | Out-File -LiteralPath $log -Encoding utf8
    Remove-Item -LiteralPath (Join-Path $staging 'files') -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction SilentlyContinue
} catch {
    "failed: $_" | Out-File -LiteralPath $log -Encoding utf8
    foreach ($rel in $copied) { Remove-Item -LiteralPath (Join-Path $install $rel) -Force -ErrorAction SilentlyContinue }
    foreach ($rel in $moved) { Move-Item -LiteralPath (Join-Path $backup $rel) -Destination (Join-Path $install $rel) -Force }
}
Start-Process -FilePath $exe
"""


def write_apply_script(staging: Path, install_dir: Path, pid: int, exe: Path) -> Path:
    """Write the PowerShell script that swaps the files once ``pid`` exits.

    Every file it overwrites or removes is moved to a backup first; on any
    error the copied files are removed and the backup is moved back, so a
    failed update leaves the previous version runnable.
    """
    def quote(value) -> str:
        # PowerShell also ends a single-quoted string at the typographic quotes U+2018-U+201B.
        text = str(value)
        for mark in ("'", "\u2018", "\u2019", "\u201a", "\u201b"):
            text = text.replace(mark, mark * 2)
        return text

    script = (
        _APPLY_SCRIPT.replace("__STAGING__", quote(staging))
        .replace("__INSTALL__", quote(install_dir))
        .replace("__EXE__", quote(exe))
        .replace("__PID__", str(int(pid)))
        .replace("__KEY__", UNINSTALL_KEY)
        .replace("__PRODUCT__", PRODUCT)
    )
    path = Path(staging) / "apply_update.ps1"
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 with BOM so Windows PowerShell 5 reads non-ASCII paths correctly.
    path.write_text(script, encoding="utf-8-sig")
    return path
