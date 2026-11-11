"""``deepc``: run a piece and save its record. BESS's ``bess run`` (``devkit/terminal``); its
``draw`` lands with the figure layer.

    deepc run <piece> [config.json] [--set SETTING=VALUE ...] [--sweep SETTING=V1,V2,... ...]
              [--name NAME] [--out-dir DIR]

``run`` builds the piece's Config (the file's values, else the defaults), applies each ``--set``,
runs once per ``--sweep`` combination, and saves one record per run at
``<out-dir>/<piece>/<name>[-setting-value].json``, printing the path and the numbers. An unknown
setting, a value outside its bounds or one set behind a closed gate is refused by the Config, in
its words.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from harness import NAMES, get_component
from record.config import from_dict, parse_override, with_overrides

DEFAULT_OUT_DIR = "results"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepc", description=__doc__.split("\n\n")[0].strip())
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one piece under one config; save the record")
    r.add_argument("piece", choices=NAMES, help="a registered piece")
    r.add_argument("config", nargs="?", default=None,
                   help="a config JSON; absent means the Config's defaults")
    r.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SETTING=VALUE", help="override one setting (repeatable)")
    r.add_argument("--sweep", dest="sweeps", action="append", default=[],
                   metavar="SETTING=V1,V2,...",
                   help="run once per value (repeatable: once per combination); each record "
                        "is named <name>-<setting>-<value>")
    r.add_argument("--name", default=None,
                   help="the record's name; default: the config file's stem, else 'default'")
    r.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    r.set_defaults(func=run_command)
    return p


def run_command(args: argparse.Namespace) -> int:
    component = get_component(args.piece)
    if args.config:
        with open(args.config) as fh:
            cfg = from_dict(component.Config, json.load(fh), where=args.config)
    else:
        cfg = component.Config()
    pairs = [parse_override(item) for item in args.overrides]
    name = args.name or (Path(args.config).stem if args.config else "default")

    axes, swept = [], []
    for item in args.sweeps:
        field, values = parse_override(item)
        if not values:
            raise SystemExit(f"--sweep expects SETTING=V1,V2,..., got {item!r}")
        if field in swept or any(f == field for f, _ in pairs):
            raise SystemExit(f"{field!r} is given twice (--sweep/--set); a setting is swept "
                             f"once, and not also set")
        texts = [v.strip() for v in values.split(",")]
        if len(set(texts)) != len(texts):
            raise SystemExit(f"--sweep {item!r} repeats a value; each run would overwrite the last")
        parted = [v for v in texts if "/" in v or os.sep in v]
        if parted:
            raise SystemExit(f"--sweep {field}: {parted[0]!r} holds a path separator, and each "
                             f"run is named <name>-{field}-<value>; run each with --set")
        swept.append(field)
        axes.append([(field, v) for v in texts])

    for combo in (itertools.product(*axes) if axes else [()]):
        # --set pairs are applied WITH the sweep value, so a gate a sweep opens is open when
        # the setting behind it is checked.
        run_cfg = with_overrides(cfg, pairs + list(combo))
        run_name = "-".join([name] + [f"{f}-{v}" for f, v in combo])
        rec = dataclasses.replace(component.run(run_cfg), name=run_name)
        path = rec.save(os.path.join(args.out_dir, args.piece, f"{run_name}.json"))
        print(path)
        for key, value in rec.numbers.items():
            print(f"  {key} = {value:.6g}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f"deepc: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
