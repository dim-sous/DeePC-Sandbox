"""``deepc``: run a piece, draw its records. BESS's ``bess`` (``devkit/terminal``).

    deepc run  <piece> [config.json] [--set SETTING=VALUE ...] [--sweep SETTING=V1,V2,... ...]
              [--name NAME] [--out-dir DIR]
    deepc draw <record.json> [<record.json> ...] [--panel NAME ...] [--number NAME --vs SETTING]
              [--runs HOW] [--out-dir DIR]

``run`` builds the piece's Config (the file's values, else the defaults), applies each
``--set``, runs once per ``--sweep`` combination, and saves one record per run at
``<out-dir>/<piece>/<name>[-setting-value].json``, printing the path and the numbers. An
unknown setting, a value outside its bounds or one set behind a closed gate is refused by the
Config, in its words. ``draw`` writes the panels the records declare (every one, or each
``--panel``) as PNGs with JSON sidecars, over every record given, the runs shared as ``--runs``
says; with ``--number`` it draws that summary number against the setting ``--vs`` instead. Default
``<out-dir>``: a folder beside the first record named after the records drawn (:func:`folder_of`).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from harness import NAMES, get_component
from record import read_header
from record.config import from_dict, parse_override, with_overrides

DEFAULT_OUT_DIR = "results"

#: How ``deepc draw`` shares several runs: on a panel, one plot, a row each, or pooled by a
#: statistic (``figure.chart.RUNS``); on a ``--number`` draw, each run a point or the mean / RMS
#: of a colour's runs at one value. The Lab's card offers the same.
RUNS_CHOICES = ("one", "rows", "mean", "rms", "each")


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

    d = sub.add_parser("draw", help="draw the panels records declare, or a number against a "
                                     "setting")
    d.add_argument("records", nargs="+", metavar="record.json")
    d.add_argument("--panel", dest="panels", action="append", default=None, metavar="NAME",
                   help="a panel to draw (repeatable); default: every panel the first declares")
    d.add_argument("--number", default=None, metavar="NAME",
                   help="draw this summary number against the setting --vs instead")
    d.add_argument("--vs", default=None, metavar="SETTING",
                   help="with --number: the setting on the bottom axis")
    d.add_argument("--runs", default=None, choices=RUNS_CHOICES,
                   help="several runs on one plot (one), a row each (rows), or pooled (mean, "
                        "rms); on a --number draw, each run a point (each) or pooled")
    d.add_argument("--out-dir", default=None,
                   help="default: beside the first record, in a folder named after the records")
    d.set_defaults(func=draw_command)
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


#: The longest default folder name a draw writes, in bytes: a filesystem refuses a name past
#: 255, and a sweep's runs are named ``<name>-<setting>-<value>``.
_FOLDER_BYTES = 120


def folder_of(names: Sequence[str]) -> str:
    """The folder a draw of the runs ``names`` writes into by default: their names joined by
    ``+``; past :data:`_FOLDER_BYTES`, the first name, how many more and a digest of them all
    (``<first>+<n>-more-<8 hex>``), so two different sets of runs never share a folder."""
    joined = "+".join(names)
    if len(joined.encode()) <= _FOLDER_BYTES:
        return joined
    digest = hashlib.sha1(joined.encode()).hexdigest()[:8]
    return f"{names[0][:_FOLDER_BYTES // 2]}+{len(names) - 1}-more-{digest}"


def draw_command(args: argparse.Namespace) -> int:
    from figure import draw, draw_sweep          # matplotlib only when drawing

    pieces = {read_header(path)["component"] for path in args.records}
    if len(pieces) > 1:
        raise ValueError(f"the records are of different pieces {sorted(pieces)}; one draw "
                         f"compares runs of one piece")
    component = get_component(pieces.pop())
    pairs = [(Path(path).stem, component.Record.load(path)) for path in args.records]
    shaping = {"runs": args.runs} if args.runs else {}
    first = Path(args.records[0])
    out_dir = Path(args.out_dir) if args.out_dir else first.parent / folder_of(
        [name for name, _ in pairs])
    if args.number:
        if args.panels:
            raise ValueError("--number draws one summary number; --panel names a chart -- give one")
        if args.vs is None:
            raise ValueError("--number needs --vs SETTING: the setting on the bottom axis")
        paths = draw_sweep(pairs, out_dir, args.number, vs=args.vs, **shaping)
    else:
        if args.vs is not None:
            raise ValueError("--vs shapes a --number draw")
        paths = draw(pairs, out_dir, panels=args.panels, **shaping)
    for path in paths:
        print(path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f"deepc: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
