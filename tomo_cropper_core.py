"""Pure, testable helpers for :mod:`tomo_cropper`.

This module deliberately has no GUI or MRC dependencies.  Keeping filename,
coordinate, and history handling here lets those rules be tested without a
display or large tomography files.
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


def parse_member(name: str) -> tuple[str, str, float] | None:
    """Return ``(prefix, kind, bin)`` for a family filename.

    Crop outputs and filenames with a zero bin factor are rejected.  ``kind``
    is an empty string for a main volume and ``bin`` defaults to ``1.0``.
    """
    if name.endswith(OUT_TAG + ".mrc"):
        return None
    match = MEMBER_RE.match(name)
    if not match:
        return None
    bin_text = match.group("bin")
    bin_factor = float(bin_text) if bin_text else 1.0
    if not math.isfinite(bin_factor) or bin_factor <= 0:
        return None
    return match.group("prefix"), (match.group("kind") or ""), bin_factor


def fmt_bin(bin_factor: float) -> str:
    """Format a bin factor without an unnecessary decimal suffix."""
    value = float(bin_factor)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def family_files(folder: str, prefix: str) -> list[tuple[str, float]]:
    """Find all uncropped family members, ordered by bin then filename."""
    found: list[tuple[str, float]] = []
    pattern = os.path.join(folder, glob.escape(prefix) + "*.mrc")
    for path in glob.glob(pattern):
        parsed = parse_member(os.path.basename(path))
        if parsed and parsed[0] == prefix:
            found.append((path, parsed[2]))
    return sorted(found, key=lambda item: (item[1], item[0]))


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


def load_history(path: str) -> dict[str, dict[str, object]]:
    """Load valid history rows, with the latest row for each prefix winning."""
    history: dict[str, dict[str, object]] = {}
    if not os.path.isfile(path):
        return history

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
            except (KeyError, OverflowError, TypeError, ValueError):
                continue
            history[prefix] = row
    return history


def upsert_history(path: str, row: Mapping[str, object]) -> None:
    """Insert or replace a history row using an atomic same-folder rewrite."""
    prefix = str(row.get("prefix", "")).strip()
    if not prefix:
        raise ValueError("history row requires a prefix")

    history = load_history(path)
    history[prefix] = dict(row)
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
            for history_row in history.values():
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
