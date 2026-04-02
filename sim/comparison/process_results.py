"""Process saved simulation results and compute metrics.

Usage (from repo root):
    uv run python -m sim.comparison.process_results

Reads ``results/<tag>/<tag>_results.npz`` + scalars for each tag,
computes metrics, and saves ``<tag>_metrics.json``.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from sim.comparison.metrics import compute_all_metrics, save_metrics

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_tag_results(tag: str, tag_dir: pathlib.Path) -> dict | None:
    """Load saved results for a tag from .npz + .json files."""
    npz_path = tag_dir / f"{tag}_results.npz"
    scalars_path = tag_dir / f"{tag}_scalars.json"

    if not npz_path.exists():
        print(f"  [SKIP] {tag}: {npz_path} not found")
        return None

    with np.load(npz_path) as data:
        results = {k: data[k] for k in data.files}

    if scalars_path.exists():
        with open(scalars_path) as f:
            scalars = json.load(f)
        results.update(scalars)

    return results


def discover_tags() -> list[tuple[str, pathlib.Path]]:
    """Discover tags from subdirectories in results/."""
    tags = []
    if not RESULTS_DIR.exists():
        return tags
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if subdir.is_dir():
            npz_files = list(subdir.glob("*_results.npz"))
            for npz in npz_files:
                tag = npz.stem.replace("_results", "")
                tags.append((tag, subdir))
    return tags


def main() -> None:
    """Process results for all discovered tags."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tags = discover_tags()
    if not tags:
        print(f"No results found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"Discovered {len(tags)} result(s): {', '.join(t for t, _ in tags)}")

    for tag, tag_dir in tags:
        print(f"\nProcessing {tag}...")
        results = load_tag_results(tag, tag_dir)
        if results is None:
            continue

        metrics = compute_all_metrics(results, tag)
        save_metrics(metrics, tag_dir / f"{tag}_metrics.json")


if __name__ == "__main__":
    main()
