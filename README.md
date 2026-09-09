# tomo_cropper

`tomo_cropper` lets you draw one crop box on a binned MRC tomogram and apply
the same physical region to every available binning and half-map in that
tomogram family. Crop coordinates are saved so newly generated family members
can receive the same crop later.

Using a bin 4 (or more heavily binned) volume as the view keeps the GUI quicker
and uses less memory while lower-bin and original tomograms are cropped with
the corresponding coordinates.

![Main crop view](https://github.com/user-attachments/assets/a5b64a80-9e66-4919-82f6-d7b8aeb008e3)

![Crop confirmation](https://github.com/user-attachments/assets/4b058ae6-44e8-4b73-b453-a290f4671722)

## Features

- Draw X/Y bounds on a slice and enter the Z range numerically.
- Apply one physical crop to integer and non-integer binnings.
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

Crop outputs use the suffix `_crop.mrc` and are not treated as source family
members.

## Coordinate model

The crop box is stored in bin-1 (unbinned) voxel coordinates. A box drawn on a
view binned by `V` becomes `canonical = view box * V`. For a target volume
binned by `T`, the crop becomes `canonical / T`, rounded and clamped to the
target MRC's real header dimensions. This also handles non-integer bins and
volumes whose dimensions are not exact multiples.

## Crop history and source deletion

Every successful **Apply crop** operation records the prefix, folder, view
file, view bin, canonical box, and timestamp in `tomo_crop_history.csv` beside
the data. **Crop uncropped (CSV)** scans those records for family members that
do not yet have a corresponding `_crop.mrc` file.

Source deletion is disabled when the application starts. If you enable it, the
application asks for confirmation after each crop and only offers source files
whose output was written and validated. If history saving fails during a
normal Apply operation, source files are kept.

## Tests

The core filename, coordinate, and CSV rules use only the Python standard
library and can be tested without a GUI or sample MRC data:

```bash
python -m unittest discover -s tests -v
```

## License

MIT. See [LICENSE](LICENSE).
