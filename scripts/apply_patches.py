"""
scripts/apply_patches.py
─────────────────────────
Copies the extended sglang_patch/ modules over the corresponding files in the
installed SGLang package.  Run once before starting the server.

Uses importlib to find the real install location — no hardcoded paths.
"""
from __future__ import annotations

import importlib
import logging
import pathlib
import shutil
import sys

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def find_sglang_root() -> pathlib.Path:
    try:
        import sglang
        return pathlib.Path(sglang.__file__).parent
    except ImportError:
        logger.error("SGLang is not installed.  Install it first: pip install sglang[all]")
        sys.exit(1)


PATCH_MAP = {
    # source (our file)                                   : destination (in sglang package)
    "sglang_patch/managers/router/radix_cache.py":        "srt/managers/controller/radix_cache.py",
    "sglang_patch/managers/router/model_runner.py":       "srt/managers/controller/model_runner.py",
    "sglang_patch/managers/router/scheduler.py": "srt/managers/controller/schedule_heuristic.py",
    "sglang_patch/server/server_args.py":                 "srt/server_args.py",
}


def apply_patches(dry_run: bool = False) -> None:
    sglang_root = find_sglang_root()
    logger.info("SGLang root: %s", sglang_root)

    for rel_src, rel_dst in PATCH_MAP.items():
        src = PROJECT_ROOT / rel_src
        dst = sglang_root / rel_dst

        if not src.exists():
            logger.error("Patch source not found: %s", src)
            sys.exit(1)

        if not dst.parent.exists():
            logger.error(
                "Destination directory does not exist: %s\n"
                "This patch map entry may be wrong for your SGLang version.",
                dst.parent,
            )
            sys.exit(1)

        if dry_run:
            logger.info("[dry-run] would copy %s → %s", src, dst)
        else:
            # Back up original
            backup = dst.with_suffix(".py.orig")
            if dst.exists() and not backup.exists():
                shutil.copy2(dst, backup)
                logger.info("Backed up %s → %s", dst.name, backup.name)

            shutil.copy2(src, dst)
            logger.info("Applied patch: %s → %s", rel_src, rel_dst)

    if not dry_run:
        logger.info("All patches applied.  Original files backed up as *.py.orig")


def restore_originals() -> None:
    """Restore backed-up originals (undo patches)."""
    sglang_root = find_sglang_root()
    for _, rel_dst in PATCH_MAP.items():
        dst = sglang_root / rel_dst
        backup = dst.with_suffix(".py.orig")
        if backup.exists():
            shutil.copy2(backup, dst)
            logger.info("Restored %s", dst.name)
        else:
            logger.warning("No backup found for %s", dst.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--restore", action="store_true", help="Restore original SGLang files")
    args = p.parse_args()

    if args.restore:
        restore_originals()
    else:
        apply_patches(dry_run=args.dry_run)
