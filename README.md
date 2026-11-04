# tomo_cropper

`tomo_cropper` lets you draw one or more crop boxes on a binned MRC tomogram
and apply the same physical regions to every available binning and half-map in
that tomogram family. Crop coordinates are saved so newly generated family
members can receive the same crops later.

Several regions of interest in one tomogram — for example several VLPs — are
queued as numbered **crop plans** and written together in a single pass.

Using a bin 4 (or more heavily binned) volume as the view keeps the GUI quicker
and uses less memory while lower-bin and original tomograms are cropped with
the corresponding coordinates.

<img width="1482" height="1087" alt="image" src="https://github.com/user-attachments/assets/4085f214-8f90-4e9e-8afc-c892ffdf37ca" />

![Crop confirmation](https://github.com/user-attachments/assets/4b058ae6-44e8-4b73-b453-a290f4671722)

## Features

- Draw X/Y bounds on a slice and enter the Z range numerically.
- Queue several crop plans per tomogram and write them all in one pass.
- Apply each physical crop to integer and non-integer binnings.
- Crop main, `ODD`, and `EVN` volumes together when they are present.
- Save canonical bin-1 coordinates in `tomo_crop_history.csv`.
- Find newly added, uncropped family members from a saved history CSV.
- Write and validate a temporary MRC before replacing an existing crop.
- Keep source volumes by default; optional deletion requires confirmation.

## Requirements and installation

- Python 3.10 or newer
- Tk support for Python

Tk is normally included with Python on Windows and macOS. On Debian/Ubuntu,
install it with your system package manager if needed:

```bash
sudo apt install python3-tk
```

Create a virtual environment and install the project:

```bash
python -m venv .venv
python -m pip install -e .
```

`numpy`, `mrcfile`, and `matplotlib` are installed from `pyproject.toml`.

## Launch

After installation:

```bash
tomo-cropper
```

You can also launch directly from the repository:

```bash
python tomo_cropper.py
```

## Expected filenames

Only `.mrc` volumes that match this family pattern are discovered:

```text
<prefix>_Vol.mrc
<prefix>_Vol_b<N>.mrc
<prefix>_ODD_Vol[_b<N>].mrc
<prefix>_EVN_Vol[_b<N>].mrc
```

`N` may be an integer or decimal, such as `2`, `4`, or `1.5`. With a prefix of
`Grid9_TS_01.mrc`, one family could contain:

```text
Grid9_TS_01.mrc_Vol.mrc
Grid9_TS_01.mrc_ODD_Vol_b1.5.mrc
Grid9_TS_01.mrc_EVN_Vol_b1.5.mrc
Grid9_TS_01.mrc_Vol_b4.mrc
```

Crop outputs use the suffix `_crop<N>.mrc`, where `N` is the crop plan number:

```text
Grid9_TS_01.mrc_Vol_b4_crop1.mrc
Grid9_TS_01.mrc_Vol_b4_crop2.mrc
```

Unnumbered `_crop.mrc` files written before multi-plan support are read as plan
number 0 and are left in place; new plans are numbered from 1. Crop outputs,
numbered or not, are never treated as source family members.

## Crop plans

Each region you want from a tomogram is a numbered crop plan:

1. Drag an X/Y box on the slice and set the Z range.
2. Press **+ New crop plan**. The box joins the plan list and gets the next
   free number.
3. Repeat for every region in this tomogram.
4. Press **Crop all plans**.

Selecting a plan in the list loads it back into the X/Y/Z entries; **Update
selected** overwrites that plan with the current entries, **Delete selected**
drops it from the queue, and **Clear all plans** empties the queue. Queued
plans are drawn on the slice in blue with their numbers, the selected plan in
amber, and the live entry box in red; a dashed outline means the current Z
slice lies outside that plan's Z range.

Plan numbers never reuse an index that already exists on disk or in the history
for that prefix, so returning to a tomogram later adds `_crop3`, `_crop4`, ...
rather than overwriting earlier work. Cropping with an empty queue falls back
to the single box in the entries.

## Coordinate model

The crop box is stored in bin-1 (unbinned) voxel coordinates. A box drawn on a
view binned by `V` becomes `canonical = view box * V`. For a target volume
binned by `T`, the crop becomes `canonical / T`, rounded and clamped to the
target MRC's real header dimensions. This also handles non-integer bins and
volumes whose dimensions are not exact multiples.

## Crop history and source deletion

Every successful **Crop all plans** operation records one row per plan in
`tomo_crop_history.csv` beside the data: prefix, folder, view file, view bin,
crop index, canonical box, and timestamp. Rows are keyed by prefix and crop
index, so plans accumulate rather than replace one another. Reopening a view
file restores that prefix's saved plans into the plan list. **Crop uncropped
(CSV)** scans those records for family members that do not yet have the
corresponding `_crop<N>.mrc` file.

Source deletion is disabled when the application starts. If you enable it, the
application asks for confirmation after the crop and only offers source files
whose output was written and validated **for every queued plan** — a source
still needed by a later plan is never offered. If history saving fails, source
files are kept.

## Tests

The core filename, coordinate, and CSV rules use only the Python standard
library and can be tested without a GUI or sample MRC data:

```bash
python -m unittest discover -s tests -v
```

## License

MIT. See [LICENSE](LICENSE).
