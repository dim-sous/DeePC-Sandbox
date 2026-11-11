"""``uv run lab`` -- start the local page and open it.

Runs from the project root, because the out-dir and the ``deepc`` command are relative to it:
the Lab launches the same commands a person types, and they are typed there. Under WSL the
browser may not open by itself: pass ``--no-open`` and open the printed address.
"""

from __future__ import annotations

import argparse
import os

from terminal import DEFAULT_OUT_DIR

from .app import serve

DEFAULT_PORT = 8765


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lab",
        description="The Lab: one local page over a piece -- its runs on the left, ticked and "
                    "compared column by column; their panels on the right as live charts, one "
                    "row per shown run; a form over its settings in a drawer; the terminal's "
                    "own output along the bottom. It launches `deepc run` and `deepc draw` for "
                    "you and computes nothing itself.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"port on 127.0.0.1 (default: {DEFAULT_PORT})")
    p.add_argument("--no-open", action="store_true",
                   help="do not open a browser; just print the address")
    p.add_argument("--root", default=None,
                   help="project root the commands run in (default: the current directory)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"where `deepc run` writes records, relative to the root "
                        f"(default: {DEFAULT_OUT_DIR})")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = os.path.abspath(args.root or os.getcwd())
    print(f"Lab on http://127.0.0.1:{args.port}   (root {root}, records under {args.out_dir}/, "
          f"ctrl-c to stop)")
    serve(args.port, root, args.out_dir, show=not args.no_open)     # until ctrl-c
    print("\nstopped")
    return 0
