"""Pure, testable helpers for :mod:`tomo_cropper`.

This module deliberately has no GUI or MRC dependencies.  Keeping filename,
coordinate, and history handling here lets those rules be tested without a
display or large tomography files.

A tomogram may carry several crop plans (one per region of interest).  Each
plan has an integer ``crop_index`` that both names its output files and keys
its row in the history CSV:

    index 1, 2, 3 ...  ->  <name>_crop1.mrc, <name>_crop2.mrc, ...
    index 0 (legacy)   ->  <name>_crop.mrc

Index 0 exists only so that crops written before multi-plan support keep their
original unnumbered filenames and stay recognised by the history.  New plans
are always numbered from 1.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence


OUT_TAG = "_crop"
CSV_NAME = "tomo_crop_history.csv"
CSV_FIELDS = [
    "prefix",
    "folder",
    "view_file",
    "view_bin",
    "crop_index",
    "cx0",
    "cx1",
    "cy0",
    "cy1",
    "cz0",
    "cz1",
    "timestamp",
]

# Matches any family member: <prefix>_[ODD_|EVN_]Vol[_b<N>].mrc
# Bin may be an integer or decimal; absence of _b means bin 1.
MEMBER_RE = re.compile(
    r"^(?P<prefix>.+?)_(?:(?P<kind>ODD|EVN)_)?Vol"
    r"(?:_b(?P<bin>\d+(?:\.\d+)?))?\.mrc$"
)

# Matches a crop output of any family member: ..._crop.mrc or ..._crop<N>.mrc
CROP_RE = re.compile(
    r"^(?P<prefix>.+?)_(?:(?P<kind>ODD|EVN)_)?Vol"
    r"(?:_b(?P<bin>\d+(?:\.\d+)?))?"
    r"_crop(?P<index>\d*)\.mrc$"
)

# Any filename ending in _crop.mrc / _crop<N>.mrc is an output, never a source.
OUT_SUFFIX_RE = re.compile(r"_crop\d*\.mrc$")


def parse_member(name: str) -> tuple[str, str, float] | None:
    """Return ``(prefix, kind, bin)`` for a family filename.

    Crop outputs (numbered or not) and filenames with a zero bin factor are
    rejected.  ``kind`` is an empty string for a main volume and ``bin``
    defaults to ``1.0``.
    """
    if OUT_SUFFIX_RE.search(name):
        return None
    match = MEMBER_RE.match(name)
    if not match:
        return None
    bin_text = match.group("bin")
    bin_factor = float(bin_text) if bin_text else 1.0
    if not math.isfinite(bin_factor) or bin_factor <= 0:
        return None
    return match.group("prefix"), (match.group("kind") or ""), bin_factor


def parse_crop(name: str) -> tuple[str, str, float, int] | None:
    """Return ``(prefix, kind, bin, crop_index)`` for a crop output filename.

    An unnumbered legacy ``_crop.mrc`` reports index ``0``.
    """
    match = CROP_RE.match(name)
    if not match:
        return None
    bin_text = match.group("bin")
    bin_factor = float(bin_text) if bin_text else 1.0
    if not math.isfinite(bin_factor) or bin_factor <= 0:
        return None
    index_text = match.group("index")
    return (
        match.group("prefix"),
        (match.group("kind") or ""),
        bin_factor,
        int(index_text) if index_text else 0,
    )


def fmt_bin(bin_factor: float) -> str:
    """Format a bin factor without an unnecessary decimal suffix."""
    value = float(bin_factor)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def crop_suffix(crop_index: int) -> str:
    """Filename suffix for a crop plan: ``_crop`` at 0, ``_crop<N>`` above."""
    index = int(crop_index)
    if index < 0:
        raise ValueError("crop index must not be negative")
    return OUT_TAG if index == 0 else f"{OUT_TAG}{index}"


def crop_path(source_path: str, crop_index: int = 0) -> str:
    """Output path for cropping ``source_path`` under a given plan index."""
    if not source_path.lower().endswith(".mrc"):
        raise ValueError("a source volume must be a .mrc file")
    return source_path[:-4] + crop_suffix(crop_index) + ".mrc"


def family_files(folder: str, prefix: str) -> list[tuple[str, float]]:
    """Find all uncropped family members, ordered by bin then filename."""
    found: list[tuple[str, float]] = []
    pattern = os.path.join(folder, glob.escape(prefix) + "*.mrc")
    for path in glob.glob(pattern):
        parsed = parse_member(os.path.basename(path))
        if parsed and parsed[0] == prefix:
            found.append((path, parsed[2]))
    return sorted(found, key=lambda item: (item[1], item[0]))


def existing_crop_indices(folder: str, prefix: str) -> set[int]:
    """Crop indices already present on disk for ``prefix`` in ``folder``."""
    indices: set[int] = set()
    pattern = os.path.join(folder, glob.escape(prefix) + "*_crop*.mrc")
    for path in glob.glob(pattern):
        parsed = parse_crop(os.path.basename(path))
        if parsed and parsed[0] == prefix:
            indices.add(parsed[3])
    return indices


def next_crop_index(used: Iterable[int]) -> int:
    """Lowest unused plan index at or above 1."""
    taken = {int(value) for value in used}
    index = 1
    while index in taken:
        index += 1
    return index


def normalize_box(
    coordinates: Iterable[int], shape_zyx: Sequence[int]
) -> tuple[int, int, int, int, int, int] | None:
    """Order and clamp an XYZ box to a ZYX volume shape.

    Coordinates use NumPy slicing semantics: lower bounds are inclusive and
    upper bounds are exclusive.  ``None`` is returned for an empty box.
    """
    values = tuple(int(value) for value in coordinates)
    if len(values) != 6 or len(shape_zyx) != 3:
        raise ValueError("a crop box needs six coordinates and a 3-D shape")
    x0, x1 = sorted(values[0:2])
    y0, y1 = sorted(values[2:4])
    z0, z1 = sorted(values[4:6])
    nz, ny, nx = (int(value) for value in shape_zyx)
    x0, x1 = max(0, min(x0, nx)), max(0, min(x1, nx))
    y0, y1 = max(0, min(y0, ny)), max(0, min(y1, ny))
    z0, z1 = max(0, min(z0, nz)), max(0, min(z1, nz))
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return None
    return x0, x1, y0, y1, z0, z1


def canonical_box(
    view_box: Iterable[int], view_bin: float
) -> tuple[int, int, int, int, int, int]:
    """Scale a view-space box into canonical bin-1 coordinates."""
    bin_factor = float(view_bin)
    if not math.isfinite(bin_factor) or bin_factor <= 0:
        raise ValueError("bin factor must be a positive finite number")
    values = tuple(int(round(value * bin_factor)) for value in view_box)
    if len(values) != 6:
        raise ValueError("a crop box needs six coordinates")
    return values


def project_box(
    canonical: Iterable[int], target_bin: float, shape_zyx: Sequence[int]
) -> tuple[int, int, int, int, int, int]:
    """Project a canonical box into a target volume and clamp it.

    Raises ``ValueError`` if the target bin is invalid or clamping produces an
    empty crop.
    """
    bin_factor = float(target_bin)
    if not math.isfinite(bin_factor) or bin_factor <= 0:
        raise ValueError("bin factor must be a positive finite number")
    values = tuple(int(round(value / bin_factor)) for value in canonical)
    if len(values) != 6:
        raise ValueError("a crop box needs six coordinates")
    result = normalize_box(values, shape_zyx)
    if result is None:
        raise ValueError("empty crop after clamping to this file's extent")
    return result


def load_history(path: str) -> dict[str, list[dict[str, object]]]:
    """Load valid history rows grouped by prefix.

    Each prefix maps to its crop plans ordered by ``crop_index``.  The latest
    row for a given ``(prefix, crop_index)`` wins.  Rows written before
    multi-plan support carry no ``crop_index`` column and load as index ``0``,
    which keeps their unnumbered ``_crop.mrc`` outputs addressable.
    """
    plans: dict[tuple[str, int], dict[str, object]] = {}
    if not os.path.isfile(path):
        return {}

    with open(path, newline="", encoding="utf-8-sig") as handle:
        for source_row in csv.DictReader(handle):
            try:
                prefix = (source_row.get("prefix") or "").strip()
                if not prefix:
                    continue
                row: dict[str, object] = dict(source_row)
                for key in ("cx0", "cx1", "cy0", "cy1", "cz0", "cz1"):
                    coordinate = float(source_row[key])
                    if not math.isfinite(coordinate):
                        raise ValueError(f"{key} must be finite")
                    row[key] = int(round(coordinate))
                view_bin = float(source_row["view_bin"])
                if not math.isfinite(view_bin) or view_bin <= 0:
                    continue
                row["view_bin"] = view_bin
                index_text = str(source_row.get("crop_index", "") or "0").strip()
                crop_index = int(float(index_text))
                if crop_index < 0:
                    continue
                row["crop_index"] = crop_index
            except (KeyError, OverflowError, TypeError, ValueError):
                continue
            plans[(prefix, crop_index)] = row

    history: dict[str, list[dict[str, object]]] = {}
    for (prefix, _index), row in plans.items():
        history.setdefault(prefix, []).append(row)
    for rows in history.values():
        rows.sort(key=lambda item: int(item["crop_index"]))
    return history


def upsert_history_rows(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    """Insert or replace several history rows in one atomic rewrite.

    Rows are keyed by ``(prefix, crop_index)``; a row with no ``crop_index``
    is treated as the legacy index ``0``.
    """
    incoming = [dict(row) for row in rows]
    if not incoming:
        return
    for row in incoming:
        if not str(row.get("prefix", "")).strip():
            raise ValueError("history row requires a prefix")
        index = int(float(row.get("crop_index", 0) or 0))
        if index < 0:
            raise ValueError("crop index must not be negative")
        row["crop_index"] = index

    existing = load_history(path)
    merged: dict[tuple[str, int], dict[str, object]] = {}
    for prefix, plan_rows in existing.items():
        for plan_row in plan_rows:
            merged[(prefix, int(plan_row["crop_index"]))] = plan_row
    for row in incoming:
        merged[(str(row["prefix"]).strip(), int(row["crop_index"]))] = row

    ordered = sorted(merged.items(), key=lambda item: (item[0][0], item[0][1]))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=folder,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for _key, history_row in ordered:
                writer.writerow(
                    {key: history_row.get(key, "") for key in CSV_FIELDS}
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def upsert_history(path: str, row: Mapping[str, object]) -> None:
    """Insert or replace one history row using an atomic same-folder rewrite."""
    upsert_history_rows(path, [row])
