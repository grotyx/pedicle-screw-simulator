#!/usr/bin/env python3
"""Upload a Windows build's update layers to Cloudflare R2.

Usage:
    python scripts/publish_update.py release/update [--bucket spinemetrics-dl]
        [--prefix radwright-screw] [--downloads FILE ...] [--dry-run]

Layers are content-addressed (``layers/<name>-<id>.zip``), so a layer already
on the server is skipped and an unchanged CUDA runtime is never re-uploaded.
``latest/components.json`` is written last, so clients only see a release
once every layer it names is in place. A copy goes to
``versions/<version>/components.json`` for rollback.

``--downloads`` also uploads the files people download (installer ``.exe`` +
``.bin``, portable ZIP, macOS ZIP) to ``downloads/<version>/<file name>``.
GitHub Releases cap each file at 2 GB and the installer ``.bin`` is larger.

Credentials: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_ACCOUNT_ID from the
environment, else ``~/.spinemetrics/r2.env`` (KEY=VALUE lines, never commit).
Needs ``boto3`` (``pip install boto3``); the app itself does not.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

KEYS = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ACCOUNT_ID")


def _credentials() -> dict:
    env = {key: os.environ.get(key, "") for key in KEYS}
    path = Path.home() / ".spinemetrics" / "r2.env"
    if not all(env.values()) and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in KEYS and not env[key.strip()]:
                env[key.strip()] = value.strip().strip("'\"")
    missing = [key for key in KEYS if not env[key]]
    if missing:
        raise SystemExit(f"Missing R2 credentials: {', '.join(missing)}")
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("update_dir", type=Path)
    parser.add_argument("--bucket", default="spinemetrics-dl")
    parser.add_argument("--prefix", default="radwright-screw")
    parser.add_argument("--downloads", type=Path, nargs="+", default=[],
                        help="Installer, .bin and ZIP files to put under downloads/<version>/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.update_dir / "components.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    uploads = [(args.update_dir / layer["archive"], f"{args.prefix}/{layer['archive']}")
               for layer in manifest["layers"].values()]
    downloads = [(path, f"{args.prefix}/downloads/{manifest['version']}/{path.name}")
                 for path in args.downloads]
    version_key = f"{args.prefix}/versions/{manifest['version']}/components.json"
    latest_key = f"{args.prefix}/latest/components.json"

    if args.dry_run:
        for source, key in uploads:
            print(f"layer  {key}  ({source.stat().st_size / 2**20:.0f} MB)")
        for source, key in downloads:
            print(f"download  {key}  ({source.stat().st_size / 2**20:.0f} MB)")
        print(f"manifest  {version_key}\nmanifest  {latest_key}")
        return

    import boto3
    from botocore.exceptions import ClientError

    creds = _credentials()
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{creds['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=creds["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    for source, key in uploads:
        try:
            client.head_object(Bucket=args.bucket, Key=key)
            print(f"present  {key}")
            continue
        except ClientError:
            pass
        print(f"upload   {key}  ({source.stat().st_size / 2**20:.0f} MB)")
        client.upload_file(str(source), args.bucket, key,
                           ExtraArgs={"ContentType": "application/zip"})
    # Downloads go up before the manifest so a new version is never announced
    # while its installer is still missing.
    for source, key in downloads:
        print(f"upload   {key}  ({source.stat().st_size / 2**20:.0f} MB)")
        client.upload_file(str(source), args.bucket, key,
                           ExtraArgs={"ContentType": "application/octet-stream"})
    for key in (version_key, latest_key):
        client.upload_file(
            str(manifest_path), args.bucket, key,
            ExtraArgs={"ContentType": "application/json", "CacheControl": "no-cache"},
        )
        print(f"manifest {key}")


if __name__ == "__main__":
    main()
