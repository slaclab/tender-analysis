"""Header-only compatibility scan of a directory of ``.sif`` files.

:func:`index_beamtime` answers "how do these files group into measurements".
It does not answer "is this file actually readable", because it never opens
one: :func:`~tender_analysis.files.find_sif_files` selects by glob and
:func:`~tender_analysis.dataset.parse_sif_name` reads only the name. A
truncated file, a ``.sif`` that is not an Andor file, or one whose comment
block carries no ``mono`` all survive indexing and fail later, inside the
analysis, as an exception nobody can attribute to a file.

This module adds the missing validation and reports it as data instead of
raising: every input file comes back either inside a measurement or in
``rejected`` with a reason. It reads at most the first 8 kB of each file --
no frame decode -- so it is fast enough to run inline in a web request, which
is what the chemcat portal's Processing tab does before offering to run
anything.

    report = scan_directory("/data/15/BL6-2A/2026-02_Smith")
    report.measurements   # [{label, kind, sample, n_files, energy_min, ...}]
    report.rejected       # [{file, reason}]
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

from .dataset import index_beamtime, parse_sif_name

__all__ = ["ScanReport", "scan_directory", "REJECT_REASONS", "SIF_MAGIC"]

# Every Andor Multi-Channel file opens with this ASCII line. sif_parser itself
# checks for it, but only after the file is committed to being decoded.
SIF_MAGIC = b"Andor Technology Multi-Channel File"

# Bytes of header read per file: the same window
# :attr:`tender_analysis.sif_io.SifFile.comment` parses, so "scan says the mono
# is there" and "the pipeline can find the mono" cannot disagree.
_HEADER_BYTES = 8192

_MONO_RE = re.compile(rb"mono\s*=\s*(-?\d+\.?\d*)")

#: Every reason a file can be left out of a measurement, and what it means.
REJECT_REASONS = {
    "not_sif": "filename does not end in .sif",
    "bad_header": "not an Andor Multi-Channel file (truncated or wrong format)",
    "no_mono": "no parseable 'mono = ' incident energy in the file header",
    "unreadable": "the file could not be opened",
    "dark": "dark frame, paired into a measurement as background rather than data",
    "aux": "alignment / calibration / test file, excluded from the standard workflow",
    "echem": "operando echem naming, excluded from the standard workflow",
    "elastic": "elastic-scattering scan, used for pixel->energy calibration",
    "energy_out_of_range": ("incident energy outside the tender range "
                            "(1900-4300 eV) or absent from the name"),
    "unparsed": "the filename could not be parsed",
}


@dataclass
class ScanReport:
    """What a directory holds, and what in it cannot be analyzed.

    ``measurements`` mirrors :class:`~tender_analysis.dataset.Measurement` as
    plain JSON-able dicts (the portal serializes this straight into a
    response). ``rejected`` accounts for every other ``.sif`` in the directory,
    so the two lists together cover the input set with nothing silently
    dropped.
    """

    directory: str
    measurements: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    n_files: int = 0

    @property
    def n_measurements(self) -> int:
        return len(self.measurements)

    def as_dict(self) -> dict:
        return {"directory": self.directory, "n_files": self.n_files,
                "measurements": self.measurements, "rejected": self.rejected,
                "reasons": REJECT_REASONS}


def _validate_header(path: str) -> str | None:
    """``None`` if the file is a readable Andor sif with a mono value, else a
    rejection reason. Reads the header window only."""
    if not path.lower().endswith(".sif"):
        return "not_sif"
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEADER_BYTES)
    except OSError:
        return "unreadable"
    if not head.startswith(SIF_MAGIC):
        return "bad_header"
    if not _MONO_RE.search(head):
        return "no_mono"
    return None


def _energy_range(paths: list[str]) -> tuple[float | None, float | None]:
    """(min, max) incident energy over a measurement's files, from their names."""
    energies = []
    for p in paths:
        rec = parse_sif_name(p)
        if rec is not None and rec.energy is not None:
            energies.append(rec.energy)
    return (min(energies), max(energies)) if energies else (None, None)


def scan_directory(directory: str, *, recursive: bool = False,
                   skip_echem: bool = True, skip_aux: bool = True) -> ScanReport:
    """Group and validate every ``.sif`` under ``directory``.

    Grouping is :func:`index_beamtime`'s, unchanged -- this wraps it rather
    than reimplementing it, so the measurements offered by a portal and the
    measurements a notebook runs are the same objects. Validation is the part
    that is new: a file that fails the header check is reported as
    ``rejected`` and is EXCLUDED from the measurement it would otherwise have
    joined, so "Process All" never dispatches a job that is going to die on a
    truncated frame stack.
    """
    directory = str(directory)
    if glob.has_magic(directory):
        paths = glob.glob(directory, recursive=recursive)
    elif recursive:
        paths = glob.glob(os.path.join(directory, "**", "*.sif"), recursive=True)
    else:
        paths = glob.glob(os.path.join(directory, "*.sif"))
    paths = sorted(paths)

    rejected: list[dict] = []
    bad: set[str] = set()
    for p in paths:
        reason = _validate_header(p)
        if reason is not None:
            rejected.append({"file": os.path.basename(p), "path": p, "reason": reason})
            bad.add(p)

    index = index_beamtime(directory, recursive=recursive,
                           skip_echem=skip_echem, skip_aux=skip_aux)

    # Everything index_beamtime itself set aside, reported with the same
    # vocabulary as the header failures so a caller has ONE list to render.
    for reason, skipped_paths in (index.skipped or {}).items():
        for p in skipped_paths:
            if p in bad:
                continue
            rejected.append({"file": os.path.basename(p), "path": p,
                             "reason": reason if reason in REJECT_REASONS
                             else "energy_out_of_range"})
            bad.add(p)

    measurements: list[dict] = []
    for m in index.measurements:
        data_paths = [p for p in m.data_paths if p not in bad]
        dark_paths = [p for p in m.dark_paths if p not in bad]
        if not data_paths:
            # Every data file rejected: the group is not runnable. Its files are
            # already in `rejected` with their own reasons.
            continue
        e_min, e_max = _energy_range(data_paths)
        measurements.append({
            "label": m.label(),
            "kind": m.kind,
            "sample": m.sample,
            "emission_line": m.emission_line,
            "incident_energy": m.incident_energy,
            "series_index": m.series_index,
            "n_files": len(data_paths),
            "n_dark": len(dark_paths),
            "energy_min": e_min,
            "energy_max": e_max,
            "files": [os.path.basename(p) for p in data_paths],
            "dark_files": [os.path.basename(p) for p in dark_paths],
        })

    # Darks are paired into measurements as background, never analyzed on their
    # own -- account for them explicitly rather than leaving them unmentioned.
    paired_darks = {f for m in measurements for f in m["dark_files"]}
    for p in paths:
        if p in bad:
            continue
        base = os.path.basename(p)
        if base in paired_darks:
            rejected.append({"file": base, "path": p, "reason": "dark"})

    measurements.sort(key=lambda m: (m["kind"], m["sample"], m["label"]))
    rejected.sort(key=lambda r: (r["reason"], r["file"]))
    return ScanReport(directory=directory, measurements=measurements,
                      rejected=rejected, n_files=len(paths))
