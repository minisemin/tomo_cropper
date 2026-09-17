#!/usr/bin/env python3
"""
tomo_cropper.py -- draw ONE crop box (X/Y by dragging, Z by entries) on a view
volume and write the SAME physical region for every binning variant of that
tomogram that exists on disk.

Open a view file (usually the bin3 volume):

    <prefix>_Vol_b3.mrc          (prefix e.g. Grid9_TS_01.mrc)

On "Apply crop" the box is written for every family member of that prefix found
in the folder, at whatever binnings exist -- for example:

    bin1  : <prefix>_Vol.mrc      _ODD_Vol.mrc      _EVN_Vol.mrc
    bin1.5: <prefix>_Vol_b1.5.mrc _ODD_Vol_b1.5.mrc _EVN_Vol_b1.5.mrc
    bin2  : <prefix>_Vol_b2.mrc   ...
    bin3  : <prefix>_Vol_b3.mrc   ...
    bin4  : <prefix>_Vol_b4.mrc   ...

CANONICAL COORDINATES  (handles any binning, integer or not)
------------------------------------------------------------
The crop box is stored in bin1 (unbinned) voxel coordinates. Drawing on a view
binned by V gives canonical = box * V; cropping a file binned by T uses
canonical / T (rounded, then clamped to that file's real header extent). So
non-integer bins like 1.5 and volumes whose sizes are not exact multiples all
crop consistently -- edge voxels the binning never represented are simply
dropped.

CROP HISTORY  (tomo_crop_history.csv, in the data folder)
---------------------------------------------------------
Every Apply records the prefix, folder, view file, view bin, canonical box and a
timestamp. Re-open any file with the same prefix and its previous crop box is
propagated into the entries automatically. "Crop uncropped (CSV)" scans a
history CSV, finds every binning variant of each recorded prefix that has no
crop yet, lists them for confirmation, and crops them all with the saved box.

Output: <name>_crop.mrc, voxel size preserved.
Requires:  pip install mrcfile numpy matplotlib
"""

import os
import logging
import datetime
import uuid
import numpy as np
import mrcfile
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from tomo_cropper_core import (
    CSV_NAME,
    OUT_TAG,
    canonical_box as to_canonical_box,
    family_files,
    fmt_bin,
    load_history,
    normalize_box,
    parse_member,
    project_box,
    upsert_history,
)

log = logging.getLogger("tomo_cropper")


class Cropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomogram cropper -- draw on a view volume, crop all binnings")
        self._view_mrc = None    # owns the active memory map
        self.vol = None          # view volume (numpy, z,y,x)
        self.path = None         # view file path
        self.prefix = None       # <prefix>
        self.viewbin = 1.0       # bin factor of the view (float)
        self.binlabel = "b1"     # display label
        self.z = 0
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # ---- top bar: open + info -------------------------------------
        bar = tk.Frame(root); bar.pack(fill="x", padx=6, pady=4)
        tk.Button(bar, text="Open view (_Vol[_b*].mrc)",
                  command=self.open_file).pack(side="left")
        self.info = tk.Label(bar, text="no file", anchor="w")
        self.info.pack(side="left", padx=10)

        # ---- Z navigation slider --------------------------------------
        zbar = tk.Frame(root); zbar.pack(fill="x", padx=6)
        tk.Label(zbar, text="view Z-slice:").pack(side="left")
        self.zslider = tk.Scale(zbar, from_=0, to=0, orient="horizontal",
                                command=self.on_z, length=300)
        self.zslider.pack(side="left", fill="x", expand=True)

        # ---- numeric X / Y / Z crop ranges (view voxels) --------------
        grid = tk.Frame(root); grid.pack(fill="x", padx=6, pady=2)
        self.ex0, self.ex1 = self._range_row(grid, 0, "crop X from", "to")
        self.ey0, self.ey1 = self._range_row(grid, 1, "crop Y from", "to")
        self.ez0, self.ez1 = self._range_row(grid, 2, "crop Z from", "to")
        tk.Label(grid, text="(view voxels; drag on image to set X/Y, "
                            "or type any value)").grid(row=0, column=4,
                                                       rowspan=3, padx=10, sticky="w")
        for e in (self.ex0, self.ex1, self.ey0, self.ey1):
            e.bind("<Return>", lambda _e: self.draw())
            e.bind("<FocusOut>", lambda _e: self.draw())

        # ---- options + actions ----------------------------------------
        act = tk.Frame(root); act.pack(fill="x", padx=6, pady=4)
        self.delorig = tk.IntVar(value=0)
        tk.Checkbutton(act, text="delete originals after successful crop (asks again)",
                       variable=self.delorig).pack(side="left", padx=4)
        tk.Button(act, text="Reset to full", command=self.reset_full).pack(side="left", padx=4)
        tk.Button(act, text="Apply crop", command=self.apply,
                  bg="#2e7d32", fg="white").pack(side="left", padx=4)
        tk.Button(act, text="Crop uncropped (CSV)", command=self.crop_from_csv,
                  bg="#1565c0", fg="white").pack(side="left", padx=4)
        self.status = tk.Label(act, text="", anchor="w")
        self.status.pack(side="left", padx=10)

        # ---- image canvas ---------------------------------------------
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self.rs = RectangleSelector(self.ax, self.on_box, useblit=True, button=[1],
                                    interactive=True, minspanx=3, minspany=3,
                                    spancoords="pixels")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _range_row(parent, r, lbl0, lbl1):
        tk.Label(parent, text=lbl0).grid(row=r, column=0, sticky="e")
        a = tk.Entry(parent, width=7); a.grid(row=r, column=1, padx=2)
        tk.Label(parent, text=lbl1).grid(row=r, column=2)
        b = tk.Entry(parent, width=7); b.grid(row=r, column=3, padx=2)
        return a, b

    @staticmethod
    def _set(entry, val):
        entry.delete(0, "end"); entry.insert(0, str(val))

    def _csv_path(self, folder=None):
        if folder is None:
            folder = os.path.dirname(self.path) if self.path else "."
        return os.path.join(folder, CSV_NAME)

    def _close_view_mapping(self):
        """Release the active view before replacing or deleting its file."""
        handle = getattr(self, "_view_mrc", None)
        self.vol = None
        self._view_mrc = None
        if handle is not None:
            handle.close()

    def _map_view(self, path):
        """Map a 3-D view and keep its file handle alive for the GUI lifetime."""
        candidate = None
        try:
            candidate = mrcfile.mmap(path, mode="r", permissive=True)
            if candidate.data is None or candidate.data.ndim != 3:
                raise ValueError("the selected MRC file is not a 3-D volume")
        except Exception:
            if candidate is not None:
                candidate.close()
            raise

        # Only release the current view after the replacement is known-good.
        self._close_view_mapping()
        self._view_mrc = candidate
        self.vol = candidate.data

    def close(self):
        """Release mapped files before closing the Tk window."""
        self._close_view_mapping()
        self.root.destroy()

    def current_box(self):
        """(x0,x1,y0,y1,z0,z1) in VIEW voxels from the entries, clamped and
        ordered low->high. None if invalid/empty."""
        if self.vol is None:
            return None
        try:
            values = (
                int(self.ex0.get()), int(self.ex1.get()),
                int(self.ey0.get()), int(self.ey1.get()),
                int(self.ez0.get()), int(self.ez1.get()),
            )
        except ValueError:
            return None
        return normalize_box(values, self.vol.shape)

    def canonical_box(self):
        """View box scaled up to bin1 (unbinned) voxels. None if no valid box."""
        box = self.current_box()
        if box is None:
            return None
        return to_canonical_box(box, self.viewbin)

    # ------------------------------------------------------------------ events
    def open_file(self):
        p = filedialog.askopenfilename(
            title="Select a view file  <prefix>_Vol[_b<N>].mrc",
            filetypes=[("MRC", "*.mrc"), ("all", "*.*")])
        if not p:
            return
        parsed = parse_member(os.path.basename(p))
        if not parsed:
            messagebox.showwarning(
                "Wrong file",
                "Open a main volume named <prefix>_Vol.mrc or <prefix>_Vol_b<N>.mrc\n"
                "(e.g. Grid9_TS_01.mrc_Vol_b3.mrc).")
            return
        prefix, kind, binf = parsed
        if kind:
            messagebox.showwarning(
                "Open the main volume",
                "Open the main _Vol file, not the _ODD_ / _EVN_ half-maps.\n"
                "The half-maps are cropped automatically alongside it.")
            return
        try:
            self._map_view(p)
        except Exception as ex:
            log.error("could not open view %s: %s", p, ex)
            messagebox.showerror("Could not open MRC", f"{p}\n\n{ex}")
            return
        self.path = p
        self.prefix = prefix
        self.viewbin = binf
        self.binlabel = "b" + fmt_bin(binf)
        nz, ny, nx = self.vol.shape
        log.info("opened view %s  (x,y,z = %d,%d,%d)  prefix=%s  view bin=%s",
                 os.path.basename(p), nx, ny, nz, self.prefix, fmt_bin(binf))
        self.info.config(
            text=f"{os.path.basename(p)}   x,y,z = {nx},{ny},{nz}   "
                 f"prefix: {self.prefix}   view bin: {self.binlabel}")
        self.zslider.config(from_=0, to=nz - 1)
        self.z = nz // 2
        self.zslider.set(self.z)
        # propagate a previous crop for this prefix, if any
        if not self._propagate_from_history():
            self.reset_full()

    def _propagate_from_history(self):
        try:
            row = load_history(self._csv_path()).get(self.prefix)
        except Exception as ex:
            log.error("could not read history %s: %s", self._csv_path(), ex)
            self.status.config(text="could not read crop history; using full volume")
            return False
        if not row:
            return False
        V = self.viewbin
        nz, ny, nx = self.vol.shape
        x0 = int(round(row["cx0"] / V)); x1 = int(round(row["cx1"] / V))
        y0 = int(round(row["cy0"] / V)); y1 = int(round(row["cy1"] / V))
        z0 = int(round(row["cz0"] / V)); z1 = int(round(row["cz1"] / V))
        self._set(self.ex0, max(0, min(x0, nx))); self._set(self.ex1, max(0, min(x1, nx)))
        self._set(self.ey0, max(0, min(y0, ny))); self._set(self.ey1, max(0, min(y1, ny)))
        self._set(self.ez0, max(0, min(z0, nz))); self._set(self.ez1, max(0, min(z1, nz)))
        log.info("propagated previous crop for %s (canonical x[%d:%d] y[%d:%d] z[%d:%d])",
                 self.prefix, row["cx0"], row["cx1"], row["cy0"], row["cy1"],
                 row["cz0"], row["cz1"])
        if self.current_box() is None:
            log.warning("saved crop for %s is empty in this view", self.prefix)
            return False
        self.status.config(text="loaded previous crop box for this prefix")
        self.draw()
        return True

    def reset_full(self):
        if self.vol is None:
            return
        nz, ny, nx = self.vol.shape
        self._set(self.ex0, 0); self._set(self.ex1, nx)
        self._set(self.ey0, 0); self._set(self.ey1, ny)
        self._set(self.ez0, 0); self._set(self.ez1, nz)
        self.draw()

    def clear_selection(self):
        """Drop the crop box so the next file starts clean."""
        for e in (self.ex0, self.ex1, self.ey0, self.ey1, self.ez0, self.ez1):
            e.delete(0, "end")
        try:
            self.rs.set_visible(False)
            self.rs.set_active(True)
        except Exception:
            pass
        self.status.config(text="")
        self.draw()

    def on_z(self, val):
        if self.vol is not None:
            self.z = int(val)
            self.draw()

    def on_box(self, e_press, e_release):
        if e_press.xdata is None or e_release.xdata is None:
            return
        x0, x1 = sorted((e_press.xdata, e_release.xdata))
        y0, y1 = sorted((e_press.ydata, e_release.ydata))
        nz, ny, nx = self.vol.shape
        self._set(self.ex0, int(round(max(0, x0))))
        self._set(self.ex1, int(round(min(nx, x1))))
        self._set(self.ey0, int(round(max(0, y0))))
        self._set(self.ey1, int(round(min(ny, y1))))
        log.debug("box drag set  x[%s:%s] y[%s:%s]",
                  self.ex0.get(), self.ex1.get(), self.ey0.get(), self.ey1.get())
        self.draw()

    def draw(self):
        if self.vol is None:
            return
        sl = self.vol[self.z]
        self.ax.clear()
        finite = sl[np.isfinite(sl)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [2, 98])
            if vmax <= vmin:
                vmax = vmin + 1
        else:
            vmin, vmax = 0, 1
        self.ax.imshow(sl, cmap="gray", origin="lower",
                       vmin=vmin, vmax=vmax, aspect="equal")
        box = self.current_box()
        if box:
            x0, x1, y0, y1, z0, z1 = box
            inz = z0 <= self.z < z1
            self.ax.add_patch(matplotlib.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, ec="red", fc="none",
                lw=1.5, ls="-" if inz else "--"))
            self.status.config(
                text=f"box (view): x {x0}-{x1}  y {y0}-{y1}  z {z0}-{z1}"
                     f"   -> {x1-x0} x {y1-y0} x {z1-z0}")
        self.ax.set_title(f"Z={self.z}   (solid box = inside Z range, dashed = outside)")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ crop core
    def _crop_one(self, path, canon):
        """Crop one family file using a canonical (bin1) box. The file's own bin
        is read from its name; canonical/bin gives its voxel range, clamped to
        the real header extent."""
        parsed = parse_member(os.path.basename(path))
        if parsed is None:
            raise ValueError("filename is not a recognized tomogram family member")
        T = parsed[2]
        with mrcfile.mmap(path, mode="r", permissive=True) as m:
            if m.data is None or m.data.ndim != 3:
                raise ValueError("source MRC is not a 3-D volume")
            nz_t, ny_t, nx_t = m.data.shape
            xs, xe, ys, ye, zs, ze = project_box(canon, T, m.data.shape)
            log.info("  crop %s  bin=%s  src(x,y,z)=%d,%d,%d  "
                     "-> x[%d:%d] y[%d:%d] z[%d:%d]",
                     os.path.basename(path), fmt_bin(T), nx_t, ny_t, nz_t,
                     xs, xe, ys, ye, zs, ze)
            sub = np.array(m.data[zs:ze, ys:ye, xs:xe])
            vsize = m.voxel_size.copy()
        out = path[:-4] + OUT_TAG + ".mrc"
        temporary = f"{out}.tmp-{uuid.uuid4().hex}"
        try:
            with mrcfile.new(temporary, overwrite=False) as o:
                o.set_data(sub)
                o.voxel_size = vsize
                o.update_header_from_data()
                o.update_header_stats()
            # Validate the completed temporary file before it can replace a
            # previous crop. os.replace keeps replacement atomic on the same
            # filesystem.
            with mrcfile.mmap(temporary, mode="r", permissive=False) as check:
                if check.data is None or check.data.shape != sub.shape:
                    raise ValueError("written crop failed shape validation")
            os.replace(temporary, out)
            temporary = None
        finally:
            if temporary and os.path.exists(temporary):
                os.remove(temporary)
        log.info("  wrote %s  (x,y,z = %d,%d,%d)",
                 os.path.basename(out), sub.shape[2], sub.shape[1], sub.shape[0])
        return out, sub.shape, T

    def _crop_files(self, items):
        """Crop items and return ``(written, failed, safe_to_delete)``."""
        written, failed, to_delete = [], [], []
        for path, canon in items:
            try:
                out, shp, T = self._crop_one(path, canon)
                written.append(f"{os.path.basename(out)}  "
                               f"{shp[2]}x{shp[1]}x{shp[0]}  "
                               f"(bin {fmt_bin(T)})")
                if (os.path.isfile(out) and os.path.getsize(out) > 0
                        and os.path.abspath(out) != os.path.abspath(path)):
                    to_delete.append(path)
            except Exception as ex:
                log.error("  FAILED %s: %s", os.path.basename(path), ex)
                failed.append(f"{os.path.basename(path)}: {ex}")
        return written, failed, to_delete

    def _delete_originals(self, to_delete):
        """Confirm once, delete verified-cropped originals, invalidate the view
        if it was among them. Honors the delete-originals checkbox."""
        deleted = []
        if not to_delete:
            return deleted
        if not self.delorig.get():
            log.info("delete-originals off; keeping %d original(s)", len(to_delete))
            return deleted
        names = "\n  ".join(os.path.basename(f) for f in to_delete)
        log.info("delete requested for %d original(s); awaiting confirmation",
                 len(to_delete))
        if messagebox.askyesno(
                "Delete uncropped originals?",
                f"Cropping succeeded for {len(to_delete)} file(s).\n\n"
                f"Delete these UNCROPPED originals now?\n"
                f"This is permanent and cannot be undone:\n\n  {names}"):
            view_path = self.path
            view_key = (
                os.path.normcase(os.path.abspath(view_path))
                if view_path
                else None
            )
            view_is_target = bool(
                view_key
                and any(
                    os.path.normcase(os.path.abspath(f)) == view_key
                    for f in to_delete
                )
            )
            if view_is_target:
                # Windows cannot remove a file while its mmap is open.
                self._close_view_mapping()
            deleted_paths = []
            for f in to_delete:
                try:
                    os.remove(f)
                    log.info("  deleted %s", os.path.basename(f))
                    deleted.append(os.path.basename(f))
                    deleted_paths.append(f)
                except Exception as ex:
                    log.error("  DELETE FAILED %s: %s", os.path.basename(f), ex)
            if view_is_target and any(
                    os.path.normcase(os.path.abspath(f)) == view_key
                    for f in deleted_paths):
                self.path = None
                self.prefix = None
                self.info.config(text="view file was deleted - open a new file")
            elif view_is_target:
                self.info.config(text="view was closed - reopen the file to continue")
        else:
            log.info("deletion cancelled by user; originals kept")
        return deleted

    # ------------------------------------------------------------------ actions
    def apply(self):
        if self.vol is None:
            messagebox.showwarning("Nothing to do", "Open a view file first.")
            return
        canon = self.canonical_box()
        if canon is None:
            messagebox.showwarning(
                "Invalid box",
                "Set a valid X/Y/Z range (drag a box and/or type integers; "
                "each range must span at least 1 voxel).")
            return

        folder = os.path.dirname(self.path)
        fam = family_files(folder, self.prefix)
        log.info("apply crop  prefix=%s  canonical(bin1) x[%d:%d] y[%d:%d] z[%d:%d]"
                 "  family=%d file(s)",
                 self.prefix, canon[0], canon[1], canon[2], canon[3],
                 canon[4], canon[5], len(fam))
        if not fam:
            messagebox.showwarning("No family files",
                                   f"No _Vol files found for prefix {self.prefix}.")
            return

        written, failed, to_delete = self._crop_files(
            [(path, canon) for path, _ in fam]
        )

        # record history BEFORE any deletion (so the box survives even if the
        # view file is removed)
        row = {
            "prefix": self.prefix, "folder": folder,
            "view_file": os.path.basename(self.path), "view_bin": fmt_bin(self.viewbin),
            "cx0": canon[0], "cx1": canon[1], "cy0": canon[2],
            "cy1": canon[3], "cz0": canon[4], "cz1": canon[5],
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        history_saved = False
        if written:
            try:
                upsert_history(self._csv_path(folder), row)
                history_saved = True
                log.info("history updated: %s", self._csv_path(folder))
            except Exception as ex:
                log.error("could not write history: %s", ex)
                failed.append(f"history was not saved: {ex}")

        # Keep originals if history persistence failed.  A valid history row is
        # part of the recovery path when source volumes are removed.
        deleted = self._delete_originals(to_delete if history_saved else [])

        log.info("done: %d written, %d failed, %d deleted",
                 len(written), len(failed), len(deleted))
        msg = ("Wrote:\n  " + "\n  ".join(written)) if written else "Wrote nothing."
        if failed:
            msg += "\n\nFailed / kept original:\n  " + "\n  ".join(failed)
        if deleted:
            msg += "\n\nDeleted originals:\n  " + "\n  ".join(deleted)
        if history_saved:
            msg += f"\n\nHistory: {CSV_NAME}"
        self.status.config(
            text=f"done: {len(written)} written, {len(failed)} failed, "
                 f"{len(deleted)} deleted"
        )
        if written:
            self.clear_selection()
        dialog = messagebox.showwarning if failed else messagebox.showinfo
        dialog("Crop complete", msg)

    def crop_from_csv(self):
        """Catch-up mode: scan a history CSV, find every binning variant of each
        recorded prefix that still has no crop, confirm, and crop them all."""
        default = self._csv_path() if self.path else os.getcwd()
        p = filedialog.askopenfilename(
            title="Select crop history CSV",
            initialdir=os.path.dirname(default),
            initialfile=CSV_NAME,
            filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        if not p:
            return
        try:
            hist = load_history(p)
        except Exception as ex:
            log.error("could not read history %s: %s", p, ex)
            messagebox.showerror("Could not read history", f"{p}\n\n{ex}")
            return
        if not hist:
            messagebox.showinfo("Empty", "No usable rows found in that CSV.")
            return

        # build the to-do list: uncropped family files per prefix
        todo, groups, unavailable = [], {}, []
        for prefix, row in hist.items():
            folder = str(row.get("folder", "") or "").strip()
            if not folder or not os.path.isdir(folder):
                log.warning("history folder missing for %s: %s", prefix, folder)
                unavailable.append((prefix, folder or "(empty path)"))
                continue
            canon = (row["cx0"], row["cx1"], row["cy0"],
                     row["cy1"], row["cz0"], row["cz1"])
            for fp, b in family_files(folder, prefix):
                if not os.path.isfile(fp[:-4] + OUT_TAG + ".mrc"):
                    todo.append((fp, canon))
                    groups.setdefault(prefix, []).append((os.path.basename(fp), b))

        unavailable_count = len(unavailable)
        unavailable_label = "entry" if unavailable_count == 1 else "entries"
        if not todo:
            if unavailable:
                skipped = "\n  ".join(
                    f"{prefix}: {folder}" for prefix, folder in unavailable
                )
                messagebox.showwarning(
                    "Folders unavailable",
                    "No uncropped files could be queued.\n\n"
                    f"Skipped {unavailable_count} CSV {unavailable_label} because "
                    "their "
                    f"folders are missing or unavailable:\n\n  {skipped}",
                )
            else:
                messagebox.showinfo(
                    "Nothing to do",
                    "Every family file listed in the CSV already has a crop.",
                )
            log.info("crop_from_csv: nothing uncropped")
            return

        # "pop up" the files that will be cropped, grouped by prefix
        lines = []
        for prefix in sorted(groups):
            lines.append(prefix + ":")
            for name, b in sorted(groups[prefix], key=lambda t: t[1]):
                lines.append(f"    {name}   (bin {fmt_bin(b)})")
        listing = "\n".join(lines)
        log.info("crop_from_csv: %d uncropped file(s) across %d prefix(es)",
                 len(todo), len(groups))
        unavailable_note = ""
        if unavailable:
            unavailable_note = (
                f"\n\nSkipped {unavailable_count} CSV {unavailable_label} whose folders "
                "are missing or unavailable."
            )
        if not messagebox.askyesno(
                "Crop uncropped files?",
                f"Found {len(todo)} uncropped file(s) in {len(groups)} prefix(es):\n\n"
                f"{listing}{unavailable_note}\n\n"
                "Crop all available files with their saved boxes?"):
            log.info("crop_from_csv: cancelled by user")
            return

        written, failed, to_delete = self._crop_files(todo)
        deleted = self._delete_originals(to_delete)
        log.info("crop_from_csv done: %d written, %d failed, %d deleted",
                 len(written), len(failed), len(deleted))
        msg = ("Wrote:\n  " + "\n  ".join(written)) if written else "Wrote nothing."
        if failed:
            msg += "\n\nFailed / kept original:\n  " + "\n  ".join(failed)
        if deleted:
            msg += "\n\nDeleted originals:\n  " + "\n  ".join(deleted)
        if unavailable:
            msg += (
                f"\n\nSkipped {unavailable_count} CSV {unavailable_label}: "
                "folder missing or unavailable."
            )
        self.status.config(
            text=f"CSV crop: {len(written)} written, {len(failed)} failed, "
                 f"{len(deleted)} deleted, {unavailable_count} unavailable"
        )
        dialog = (
            messagebox.showwarning
            if failed or unavailable
            else messagebox.showinfo
        )
        dialog("Crop complete", msg)


def main():
    """Launch the desktop application."""
    logging.basicConfig(
        level=logging.INFO,          # set to logging.DEBUG for box-drag traces
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")
    log.info("tomo_cropper started")
    root = tk.Tk()
    Cropper(root)
    root.mainloop()
    log.info("tomo_cropper closed")


if __name__ == "__main__":
    main()
