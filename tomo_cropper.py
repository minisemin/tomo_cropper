#!/usr/bin/env python3
"""
tomo_cropper.py -- draw one or more crop boxes (X/Y by dragging, Z by entries)
on a view volume and write the SAME physical regions for every binning variant
of that tomogram that exists on disk.

Open a view file (usually the bin3 volume):

    <prefix>_Vol_b3.mrc          (prefix e.g. Grid9_TS_01.mrc)

Each box you keep becomes a numbered CROP PLAN.  On "Crop all plans" every plan
is written for every family member of that prefix found in the folder, at
whatever binnings exist -- for example:

    bin1  : <prefix>_Vol.mrc      _ODD_Vol.mrc      _EVN_Vol.mrc
    bin1.5: <prefix>_Vol_b1.5.mrc _ODD_Vol_b1.5.mrc _EVN_Vol_b1.5.mrc
    bin2  : <prefix>_Vol_b2.mrc   ...
    bin3  : <prefix>_Vol_b3.mrc   ...
    bin4  : <prefix>_Vol_b4.mrc   ...

CROP PLANS  (several VLPs per tomogram, one session)
----------------------------------------------------
Drag a box, set Z, press "+ New crop plan" -- the box is added to the plan list
and numbered.  Repeat for every region you want from this tomogram.  Selecting
a plan loads it back into the entries; "Update selected" overwrites it with the
current entries.  "Crop all plans" then writes every plan for every binning in
one pass.  Outputs are numbered by plan:

    <name>_crop1.mrc, <name>_crop2.mrc, ...

Plan numbers never reuse an index already present on disk or in the history for
that prefix, so a second session adds _crop3, _crop4 ... rather than
overwriting earlier work.  Unnumbered <name>_crop.mrc files written before
multi-plan support are recognised as plan index 0 and are left alone.

CANONICAL COORDINATES  (handles any binning, integer or not)
------------------------------------------------------------
Each crop box is stored in bin1 (unbinned) voxel coordinates. Drawing on a view
binned by V gives canonical = box * V; cropping a file binned by T uses
canonical / T (rounded, then clamped to that file's real header extent). So
non-integer bins like 1.5 and volumes whose sizes are not exact multiples all
crop consistently -- edge voxels the binning never represented are simply
dropped.

CROP HISTORY  (tomo_crop_history.csv, in the data folder)
---------------------------------------------------------
Every crop records the prefix, folder, view file, view bin, plan index,
canonical box and a timestamp -- one row per plan. Re-open any file with the
same prefix and its previous plans are restored into the plan list
automatically. "Crop uncropped (CSV)" scans a history CSV, finds every binning
variant of each recorded prefix/plan that has no crop yet, lists them for
confirmation, and crops them all with the saved boxes.

Output: <name>_crop<N>.mrc, voxel size preserved.
Requires:  pip install mrcfile numpy matplotlib
"""

import os
import logging
import datetime
import uuid
from collections import Counter

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
    canonical_box as to_canonical_box,
    crop_path,
    crop_suffix,
    existing_crop_indices,
    family_files,
    fmt_bin,
    load_history,
    next_crop_index,
    normalize_box,
    parse_member,
    project_box,
    upsert_history_rows,
)

log = logging.getLogger("tomo_cropper")

PLAN_COLOR = "#00b0ff"
SELECTED_COLOR = "#ffb300"


class Cropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomogram cropper -- several crop plans, all binnings")
        self._view_mrc = None    # owns the active memory map
        self.vol = None          # view volume (numpy, z,y,x)
        self.path = None         # view file path
        self.prefix = None       # <prefix>
        self.viewbin = 1.0       # bin factor of the view (float)
        self.binlabel = "b1"     # display label
        self.z = 0
        self.plans = []          # [{"index": int, "canon": tuple, "view": tuple}]
        self._used_indices = set()
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

        # ---- crop plans -----------------------------------------------
        plan_frame = tk.LabelFrame(root, text="Crop plans for this tomogram")
        plan_frame.pack(fill="x", padx=6, pady=4)
        listbox_frame = tk.Frame(plan_frame)
        listbox_frame.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self.plan_list = tk.Listbox(listbox_frame, height=5,
                                    exportselection=False,
                                    font=("TkFixedFont", 9))
        scroll = tk.Scrollbar(listbox_frame, orient="vertical",
                              command=self.plan_list.yview)
        self.plan_list.configure(yscrollcommand=scroll.set)
        self.plan_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        self.plan_list.bind("<<ListboxSelect>>", self.on_plan_select)

        plan_buttons = tk.Frame(plan_frame)
        plan_buttons.pack(side="left", fill="y", padx=4, pady=4)
        tk.Button(plan_buttons, text="+ New crop plan", width=16,
                  command=self.add_plan, bg="#37474f", fg="white"
                  ).pack(fill="x", pady=1)
        tk.Button(plan_buttons, text="Update selected", width=16,
                  command=self.update_plan).pack(fill="x", pady=1)
        tk.Button(plan_buttons, text="Delete selected", width=16,
                  command=self.delete_plan).pack(fill="x", pady=1)
        tk.Button(plan_buttons, text="Clear all plans", width=16,
                  command=self.clear_plans).pack(fill="x", pady=1)

        # ---- options + actions ----------------------------------------
        act = tk.Frame(root); act.pack(fill="x", padx=6, pady=4)
        self.delorig = tk.IntVar(value=0)
        tk.Checkbutton(act, text="delete originals after successful crop (asks again)",
                       variable=self.delorig).pack(side="left", padx=4)
        tk.Button(act, text="Reset to full", command=self.reset_full).pack(side="left", padx=4)
        tk.Button(act, text="Crop all plans", command=self.apply,
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

    def _progress(self, text):
        """Show a line in the status bar and let Tk repaint mid-crop."""
        status = getattr(self, "status", None)
        if status is None:
            return
        try:
            status.config(text=text)
            root = getattr(self, "root", None)
            if root is not None:
                root.update_idletasks()
        except Exception:
            pass

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

    def _view_box(self, canon):
        """Project a canonical (bin1) box back into view voxels, or None."""
        if self.vol is None:
            return None
        V = self.viewbin
        values = tuple(int(round(value / V)) for value in canon)
        return normalize_box(values, self.vol.shape)

    # ------------------------------------------------------------------ plans
    def _allocate_index(self):
        """Lowest plan number not already used on disk, in history, or queued."""
        index = next_crop_index(self._used_indices)
        self._used_indices.add(index)
        return index

    def _refresh_used_indices(self):
        """Collect plan numbers already taken for the open prefix."""
        used = set()
        if self.path and self.prefix:
            folder = os.path.dirname(self.path)
            try:
                used |= existing_crop_indices(folder, self.prefix)
            except Exception as ex:
                log.warning("could not scan existing crops: %s", ex)
            try:
                for row in load_history(self._csv_path(folder)).get(self.prefix, []):
                    used.add(int(row["crop_index"]))
            except Exception as ex:
                log.warning("could not read history for plan numbering: %s", ex)
        used |= {int(plan["index"]) for plan in self.plans}
        self._used_indices = used

    @staticmethod
    def _plan_label(plan):
        x0, x1, y0, y1, z0, z1 = plan["view"]
        return (f"{crop_suffix(plan['index']):>7}  "
                f"x {x0:>5}-{x1:<5} y {y0:>5}-{y1:<5} z {z0:>4}-{z1:<4}"
                f"  [{x1-x0} x {y1-y0} x {z1-z0}]")

    def _refresh_plans(self, select=None):
        self.plan_list.delete(0, "end")
        for plan in self.plans:
            self.plan_list.insert("end", self._plan_label(plan))
        if select is not None and 0 <= select < len(self.plans):
            self.plan_list.selection_clear(0, "end")
            self.plan_list.selection_set(select)
            self.plan_list.see(select)
        self.draw()

    def _selected_plan(self):
        selection = self.plan_list.curselection()
        return selection[0] if selection else None

    def add_plan(self):
        """Keep the current X/Y/Z box as a new numbered crop plan."""
        if self.vol is None:
            messagebox.showwarning("Nothing to do", "Open a view file first.")
            return
        view = self.current_box()
        canon = self.canonical_box()
        if view is None or canon is None:
            messagebox.showwarning(
                "Invalid box",
                "Set a valid X/Y/Z range (drag a box and/or type integers; "
                "each range must span at least 1 voxel).")
            return
        index = self._allocate_index()
        self.plans.append({"index": index, "canon": canon, "view": view})
        log.info("added plan %s  view x[%d:%d] y[%d:%d] z[%d:%d]",
                 crop_suffix(index), *view)
        self._refresh_plans(select=len(self.plans) - 1)
        self.status.config(
            text=f"{len(self.plans)} plan(s) queued; next output "
                 f"suffix {crop_suffix(next_crop_index(self._used_indices))}")

    def update_plan(self):
        """Overwrite the selected plan with the current entries."""
        position = self._selected_plan()
        if position is None:
            messagebox.showinfo("No plan selected",
                                "Select a plan in the list first.")
            return
        view = self.current_box()
        canon = self.canonical_box()
        if view is None or canon is None:
            messagebox.showwarning("Invalid box", "Set a valid X/Y/Z range first.")
            return
        self.plans[position]["view"] = view
        self.plans[position]["canon"] = canon
        log.info("updated plan %s  view x[%d:%d] y[%d:%d] z[%d:%d]",
                 crop_suffix(self.plans[position]["index"]), *view)
        self._refresh_plans(select=position)

    def delete_plan(self):
        position = self._selected_plan()
        if position is None:
            messagebox.showinfo("No plan selected",
                                "Select a plan in the list first.")
            return
        plan = self.plans.pop(position)
        log.info("removed plan %s from the queue", crop_suffix(plan["index"]))
        self._refresh_plans(select=min(position, len(self.plans) - 1))

    def clear_plans(self):
        if self.plans and not messagebox.askyesno(
                "Clear plans?",
                f"Discard all {len(self.plans)} queued crop plan(s)?\n"
                "Already written crop files are not affected."):
            return
        self.plans = []
        self._refresh_used_indices()
        self._refresh_plans()
        self.status.config(text="plan queue cleared")

    def on_plan_select(self, _event=None):
        """Load the selected plan back into the X/Y/Z entries."""
        position = self._selected_plan()
        if position is None:
            return
        x0, x1, y0, y1, z0, z1 = self.plans[position]["view"]
        self._set(self.ex0, x0); self._set(self.ex1, x1)
        self._set(self.ey0, y0); self._set(self.ey1, y1)
        self._set(self.ez0, z0); self._set(self.ez1, z1)
        self.draw()

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
        self.plans = []
        nz, ny, nx = self.vol.shape
        log.info("opened view %s  (x,y,z = %d,%d,%d)  prefix=%s  view bin=%s",
                 os.path.basename(p), nx, ny, nz, self.prefix, fmt_bin(binf))
        self.info.config(
            text=f"{os.path.basename(p)}   x,y,z = {nx},{ny},{nz}   "
                 f"prefix: {self.prefix}   view bin: {self.binlabel}")
        self.zslider.config(from_=0, to=nz - 1)
        self.z = nz // 2
        self.zslider.set(self.z)
        # restore previous plans for this prefix, if any
        restored = self._propagate_from_history()
        self._refresh_used_indices()
        if not restored:
            self.reset_full()
        self._refresh_plans(select=0 if self.plans else None)

    def _propagate_from_history(self):
        """Rebuild the plan list for this prefix from the history CSV."""
        try:
            rows = load_history(self._csv_path()).get(self.prefix, [])
        except Exception as ex:
            log.error("could not read history %s: %s", self._csv_path(), ex)
            self.status.config(text="could not read crop history; using full volume")
            return False
        if not rows:
            return False
        restored, dropped = [], 0
        for row in rows:
            canon = (row["cx0"], row["cx1"], row["cy0"],
                     row["cy1"], row["cz0"], row["cz1"])
            view = self._view_box(canon)
            if view is None:
                dropped += 1
                log.warning("saved plan %s for %s is empty in this view",
                            crop_suffix(int(row["crop_index"])), self.prefix)
                continue
            restored.append({"index": int(row["crop_index"]),
                             "canon": canon, "view": view})
        if not restored:
            return False
        self.plans = restored
        x0, x1, y0, y1, z0, z1 = restored[0]["view"]
        self._set(self.ex0, x0); self._set(self.ex1, x1)
        self._set(self.ey0, y0); self._set(self.ey1, y1)
        self._set(self.ez0, z0); self._set(self.ez1, z1)
        log.info("restored %d saved plan(s) for %s", len(restored), self.prefix)
        note = f"restored {len(restored)} saved crop plan(s) for this prefix"
        if dropped:
            note += f" ({dropped} empty in this view, skipped)"
        self.status.config(text=note)
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
        """Drop the entries and the plan queue so the next file starts clean."""
        for e in (self.ex0, self.ex1, self.ey0, self.ey1, self.ez0, self.ez1):
            e.delete(0, "end")
        self.plans = []
        try:
            self.plan_list.delete(0, "end")
        except Exception:
            pass
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

        # queued plans
        selected = self._selected_plan()
        for position, plan in enumerate(self.plans):
            x0, x1, y0, y1, z0, z1 = plan["view"]
            inz = z0 <= self.z < z1
            color = SELECTED_COLOR if position == selected else PLAN_COLOR
            self.ax.add_patch(matplotlib.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, ec=color, fc="none",
                lw=1.5, ls="-" if inz else "--"))
            self.ax.text(x0 + 2, y0 + 2, crop_suffix(plan["index"]).lstrip("_"),
                         color=color, fontsize=8, va="bottom", ha="left")

        # the live box in the entries
        box = self.current_box()
        if box:
            x0, x1, y0, y1, z0, z1 = box
            inz = z0 <= self.z < z1
            self.ax.add_patch(matplotlib.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, ec="red", fc="none",
                lw=1.5, ls="-" if inz else "--"))
            self.status.config(
                text=f"box (view): x {x0}-{x1}  y {y0}-{y1}  z {z0}-{z1}"
                     f"   -> {x1-x0} x {y1-y0} x {z1-z0}"
                     f"   |  {len(self.plans)} plan(s) queued")
        self.ax.set_title(f"Z={self.z}   (red = current box, blue = queued plans; "
                          f"solid = inside Z range)")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ crop core
    def _crop_one(self, path, canon, crop_index=0):
        """Crop one family file using a canonical (bin1) box. The file's own bin
        is read from its name; canonical/bin gives its voxel range, clamped to
        the real header extent. ``crop_index`` selects the output suffix."""
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
                     "-> x[%d:%d] y[%d:%d] z[%d:%d]  as %s",
                     os.path.basename(path), fmt_bin(T), nx_t, ny_t, nz_t,
                     xs, xe, ys, ye, zs, ze, crop_suffix(crop_index))
            sub = np.array(m.data[zs:ze, ys:ye, xs:xe])
            vsize = m.voxel_size.copy()
        out = crop_path(path, crop_index)
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

    def _crop_files(self, items, label=""):
        """Crop ``(path, canonical_box, crop_index)`` items.

        Returns ``(written, failed, cropped_sources)``.  ``cropped_sources``
        has one entry per verified output, so a source that belongs to several
        plans appears several times -- the caller counts them before offering
        anything for deletion.
        """
        written, failed, cropped = [], [], []
        total = len(items)
        for number, item in enumerate(items, 1):
            path, canon, crop_index = item if len(item) == 3 else (*item, 0)
            self._progress(f"{label}cropping {number}/{total}: "
                           f"{os.path.basename(path)} -> {crop_suffix(crop_index)}")
            try:
                out, shp, T = self._crop_one(path, canon, crop_index)
                written.append(f"{os.path.basename(out)}  "
                               f"{shp[2]}x{shp[1]}x{shp[0]}  "
                               f"(bin {fmt_bin(T)})")
                if (os.path.isfile(out) and os.path.getsize(out) > 0
                        and os.path.abspath(out) != os.path.abspath(path)):
                    cropped.append(path)
            except Exception as ex:
                log.error("  FAILED %s (%s): %s",
                          os.path.basename(path), crop_suffix(crop_index), ex)
                failed.append(f"{os.path.basename(path)} "
                              f"[{crop_suffix(crop_index)}]: {ex}")
        return written, failed, cropped

    @staticmethod
    def _fully_cropped(items, cropped_sources):
        """Sources whose every queued plan produced a verified output."""
        needed = Counter(item[0] for item in items)
        done = Counter(cropped_sources)
        return [path for path in needed if done.get(path, 0) == needed[path]]

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
        """Write every queued crop plan for every binning of this prefix."""
        if self.vol is None:
            messagebox.showwarning("Nothing to do", "Open a view file first.")
            return

        plans = list(getattr(self, "plans", []) or [])
        if not plans:
            # nothing queued: treat the live entries as a single plan
            canon = self.canonical_box()
            if canon is None:
                messagebox.showwarning(
                    "Nothing to crop",
                    "Queue at least one crop plan (\"+ New crop plan\"), or set a "
                    "valid X/Y/Z range to crop a single region.")
                return
            plans = [{"index": self._allocate_index(),
                      "canon": canon, "view": self.current_box()}]

        folder = os.path.dirname(self.path)
        fam = family_files(folder, self.prefix)
        if not fam:
            messagebox.showwarning("No family files",
                                   f"No _Vol files found for prefix {self.prefix}.")
            return

        log.info("apply: prefix=%s  %d plan(s) x %d family file(s)",
                 self.prefix, len(plans), len(fam))
        for plan in plans:
            log.info("  plan %s canonical(bin1) x[%d:%d] y[%d:%d] z[%d:%d]",
                     crop_suffix(plan["index"]), *plan["canon"])

        if len(plans) > 1:
            listing = "\n  ".join(
                f"{crop_suffix(plan['index'])}: "
                f"x {plan['view'][0]}-{plan['view'][1]}  "
                f"y {plan['view'][2]}-{plan['view'][3]}  "
                f"z {plan['view'][4]}-{plan['view'][5]}"
                for plan in plans)
            if not messagebox.askyesno(
                    "Crop all plans?",
                    f"{len(plans)} crop plan(s) x {len(fam)} family file(s) "
                    f"= {len(plans) * len(fam)} output file(s).\n\n"
                    f"Plans (view voxels):\n  {listing}\n\n"
                    "Write them all now?"):
                log.info("apply cancelled by user")
                return

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        all_written, all_failed, all_cropped, plan_rows = [], [], [], []
        all_items = []
        for number, plan in enumerate(plans, 1):
            items = [(path, plan["canon"], plan["index"]) for path, _ in fam]
            all_items.extend(items)
            label = (f"plan {number}/{len(plans)} ({crop_suffix(plan['index'])}) - "
                     if len(plans) > 1 else "")
            written, failed, cropped = self._crop_files(items, label=label)
            all_written.extend(written)
            all_failed.extend(failed)
            all_cropped.extend(cropped)
            if written:
                plan_rows.append({
                    "prefix": self.prefix, "folder": folder,
                    "view_file": os.path.basename(self.path),
                    "view_bin": fmt_bin(self.viewbin),
                    "crop_index": plan["index"],
                    "cx0": plan["canon"][0], "cx1": plan["canon"][1],
                    "cy0": plan["canon"][2], "cy1": plan["canon"][3],
                    "cz0": plan["canon"][4], "cz1": plan["canon"][5],
                    "timestamp": timestamp,
                })

        # record history BEFORE any deletion (so the boxes survive even if the
        # view file is removed)
        history_saved = False
        if plan_rows:
            try:
                upsert_history_rows(self._csv_path(folder), plan_rows)
                history_saved = True
                log.info("history updated (%d row(s)): %s",
                         len(plan_rows), self._csv_path(folder))
            except Exception as ex:
                log.error("could not write history: %s", ex)
                all_failed.append(f"history was not saved: {ex}")

        # Keep originals if history persistence failed, and only offer a source
        # whose every plan was written -- otherwise a later plan loses its input.
        to_delete = self._fully_cropped(all_items, all_cropped)
        deleted = self._delete_originals(to_delete if history_saved else [])

        log.info("done: %d written, %d failed, %d deleted",
                 len(all_written), len(all_failed), len(deleted))
        msg = (f"Wrote {len(all_written)} file(s) from {len(plans)} plan(s):\n  "
               + "\n  ".join(all_written)) if all_written else "Wrote nothing."
        if all_failed:
            msg += "\n\nFailed / kept original:\n  " + "\n  ".join(all_failed)
        if deleted:
            msg += "\n\nDeleted originals:\n  " + "\n  ".join(deleted)
        if history_saved:
            msg += f"\n\nHistory: {CSV_NAME}"
        self.status.config(
            text=f"done: {len(all_written)} written, {len(all_failed)} failed, "
                 f"{len(deleted)} deleted"
        )
        if all_written:
            self.clear_selection()
        dialog = messagebox.showwarning if all_failed else messagebox.showinfo
        dialog("Crop complete", msg)

    def crop_from_csv(self):
        """Catch-up mode: scan a history CSV, find every binning variant of each
        recorded prefix/plan that still has no crop, confirm, and crop them."""
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

        # build the to-do list: uncropped family files per prefix and plan
        todo, groups, unavailable = [], {}, []
        for prefix, rows in hist.items():
            for row in rows if isinstance(rows, list) else [rows]:
                crop_index = int(row.get("crop_index", 0) or 0)
                folder = str(row.get("folder", "") or "").strip()
                if not folder or not os.path.isdir(folder):
                    log.warning("history folder missing for %s %s: %s",
                                prefix, crop_suffix(crop_index), folder)
                    unavailable.append((f"{prefix} [{crop_suffix(crop_index)}]",
                                        folder or "(empty path)"))
                    continue
                canon = (row["cx0"], row["cx1"], row["cy0"],
                         row["cy1"], row["cz0"], row["cz1"])
                for fp, b in family_files(folder, prefix):
                    out = crop_path(fp, crop_index)
                    if not os.path.isfile(out):
                        todo.append((fp, canon, crop_index))
                        groups.setdefault(prefix, []).append(
                            (os.path.basename(out), b, crop_index))

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

        # "pop up" the files that will be written, grouped by prefix
        lines = []
        for prefix in sorted(groups):
            lines.append(prefix + ":")
            for name, b, index in sorted(groups[prefix], key=lambda t: (t[2], t[1])):
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

        written, failed, cropped = self._crop_files(todo, label="CSV catch-up - ")
        deleted = self._delete_originals(self._fully_cropped(todo, cropped))
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
