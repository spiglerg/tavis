#!/usr/bin/env python3
"""Download TAVIS robot and task assets from Hugging Face.

Assets (robot USDs, task USDs, meshes, textures) live on the Hugging Face
dataset repository ``tavis-benchmark/tavis-assets`` rather than in this
git repository, because the anonymous review mirror (anonymous.4open.science)
does not proxy GitHub Releases or LFS, and because asset revisions decouple
naturally from code releases.

Usage
-----
Explicit::

    python -m tavis.download_assets            # download if missing
    python -m tavis.download_assets --force    # re-download
    python -m tavis.download_assets --check    # report status, no I/O

Implicit: ``tavis.make_env.make_tavis_env`` calls :func:`download_assets`
on the first run, so any eval or teleoperation entry point fetches assets
transparently the first time. Training does not construct a simulation
environment and therefore does not need the assets to be present.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._version import __assets_version__


HF_REPO_ID = "tavis-benchmark/tavis-assets"
HF_REPO_TYPE = "dataset"

# Pin to a tag on the asset repo once one is published; fall back to ``main``
# until then. The version literal is kept in sync with the code release in
# ``tavis/_version.py``.
HF_REVISION = "main"

PACKAGE_DIR = Path(__file__).parent.absolute()
ASSETS_DIR = PACKAGE_DIR / "assets"

# Files that must exist after a successful download. Used as a fast presence
# check (avoids re-hitting the network on every env construction) and as a
# post-download integrity check.
REQUIRED_FILES = (
    "robots/GR1/GR1T2_robotiq85.usd",
    "robots/reachy2/reachy2.usd",
    "tasks/packing_table.usd",
)


def check_assets(verbose: bool = False) -> bool:
    """Return True iff every required asset file is present under :data:`ASSETS_DIR`."""
    missing = [f for f in REQUIRED_FILES if not (ASSETS_DIR / f).exists()]
    if missing and verbose:
        print(f"Missing required assets under {ASSETS_DIR}:")
        for f in missing:
            print(f"  - {f}")
    return not missing


def download_assets(force: bool = False) -> bool:
    """Fetch the TAVIS asset bundle from Hugging Face into :data:`ASSETS_DIR`.

    Idempotent: a no-op if all required files are already present, unless
    ``force=True``. Returns True on success, False on failure.
    """
    if not force and check_assets():
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required to download assets. Install with:")
        print("    uv pip install huggingface_hub   # or pip install huggingface_hub")
        return False

    print(
        f"Downloading TAVIS assets (revision={HF_REVISION}) from "
        f"https://huggingface.co/datasets/{HF_REPO_ID} ..."
    )
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
            local_dir=str(ASSETS_DIR),
            local_dir_use_symlinks=False,
        )
    except Exception as e:  # noqa: BLE001 — surface any download failure clearly
        print(f"\nFailed to download assets: {e}")
        print("\nManual download:")
        print(f"  1. Open https://huggingface.co/datasets/{HF_REPO_ID}/tree/{HF_REVISION}")
        print(f"  2. Place the directory contents under: {ASSETS_DIR}")
        return False

    if not check_assets(verbose=True):
        print("\nDownload completed but required files are still missing.")
        return False

    print(f"Assets installed under {ASSETS_DIR}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download TAVIS robot and task assets from Hugging Face")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Re-download even if assets are already present")
    parser.add_argument("--check", "-c", action="store_true",
                        help="Report whether required assets are present, without downloading")
    args = parser.parse_args()

    if args.check:
        ok = check_assets(verbose=True)
        if ok:
            print(f"All required assets present under {ASSETS_DIR}.")
        sys.exit(0 if ok else 1)

    sys.exit(0 if download_assets(force=args.force) else 1)


if __name__ == "__main__":
    main()
