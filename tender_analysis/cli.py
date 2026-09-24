"""Command-line reduction.

    python -m tender_analysis.cli list   <sif-dir>
    python -m tender_analysis.cli reduce <sif-dir> --measurement LABEL --out DIR
                                  [--central-pix P] [--n 7] [--no-i0]
                                  [--thresholds a [b [c [d]]]]
    python -m tender_analysis.cli batch  <sif-dir> --out DIR [--max-workers N]

``reduce`` runs ONE measurement (a label from ``list``, i.e.
``Measurement.label()``) and writes the chemcat CSV: ``write_xas_csv`` for RIXS
(HERFD), ``write_xes_csv`` for XES, to ``DIR/<label>.csv``. ``batch`` indexes
the directory and runs every measurement through ``BeamtimeIndex.run_all``
(text outputs + ``run_manifest.json`` under ``DIR/analysis/``); it exits
non-zero when any measurement failed.
"""

from __future__ import annotations

import argparse
import os
import sys

__all__ = ["main", "build_parser"]


def _thresholds(values):
    if values is None:
        return None
    if not 1 <= len(values) <= 4:
        raise SystemExit("--thresholds takes 1 to 4 values ([bcg_cutoff] low [xray] hi)")
    return values


def _overrides(args) -> dict:
    kw = {"n": args.n, "i0_corr": not args.no_i0}
    if args.central_pix is not None:
        kw["central_pix"] = args.central_pix
    th = _thresholds(args.thresholds)
    if th is not None:
        kw["threshold"] = th
    return kw


def _cmd_list(args) -> int:
    from .dataset import index_beamtime

    idx = index_beamtime(args.directory)
    for m in idx:
        print(f"{m.label()}\t{m.kind}\t{len(m.data_paths)} files\t{len(m.dark_paths)} dark")
    for reason, paths in idx.skipped.items():
        print(f"# skipped ({reason}): {len(paths)}", file=sys.stderr)
    return 0


def _cmd_reduce(args) -> int:
    from .dataset import index_beamtime
    from .export import write_xas_csv, write_xes_csv

    idx = index_beamtime(args.directory)
    matches = [m for m in idx if m.label() == args.measurement]
    if not matches:
        labels = ", ".join(m.label() for m in idx) or "(none)"
        print(f"no measurement {args.measurement!r} in {args.directory}; "
              f"available: {labels}", file=sys.stderr)
        return 2
    m = matches[0]
    result = m.run(**_overrides(args))
    path = os.path.join(args.out, m.label() + ".csv")
    if m.kind == "RIXS":
        write_xas_csv(result, path, source="tender_analysis cli reduce (OnePotRIXS.herfd)")
    else:
        write_xes_csv(result, path, source="tender_analysis cli reduce (OnePot.run)")
    print(path)
    return 0


def _cmd_batch(args) -> int:
    from .dataset import index_beamtime

    idx = index_beamtime(args.directory)
    runs = idx.run_all(save_root=args.out, max_workers=args.max_workers,
                       verbose=not args.quiet, **_overrides(args))
    failed = [r for r in runs if not r.ok]
    for r in failed:
        print(f"FAILED {r.measurement.label()}: {r.error!r}", file=sys.stderr)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tender_analysis.cli",
                                description="Tender SIF XES/RIXS reduction.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("directory", help="directory of .sif files (one sample)")
        sp.add_argument("--central-pix", type=int, default=None,
                        help="HERFD band centre pixel (default: gaussian auto-fit)")
        sp.add_argument("--n", type=int, default=7, help="HERFD band width in pixels")
        sp.add_argument("--no-i0", action="store_true", help="skip I0 normalization")
        sp.add_argument("--thresholds", type=float, nargs="+", default=None,
                        metavar="ADU", help="[bcg_cutoff] low [xray] hi, as Thresholds")

    sp = sub.add_parser("list", help="list the measurements in a directory")
    sp.add_argument("directory")
    sp.set_defaults(func=_cmd_list)

    sp = sub.add_parser("reduce", help="reduce one measurement to a chemcat CSV")
    common(sp)
    sp.add_argument("--measurement", required=True, help="label from `list`")
    sp.add_argument("--out", required=True, help="output directory")
    sp.set_defaults(func=_cmd_reduce)

    sp = sub.add_parser("batch", help="run every measurement (index_beamtime + run_all)")
    common(sp)
    sp.add_argument("--out", required=True, help="save_root for run_all")
    sp.add_argument("--max-workers", type=int, default=None,
                    help="process-pool size (default: sequential)")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=_cmd_batch)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # guard required: batch uses a spawn process pool
    sys.exit(main())
