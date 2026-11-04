import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HAS_MRC_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("matplotlib", "mrcfile", "numpy", "tkinter")
)


@unittest.skipUnless(HAS_MRC_DEPENDENCIES, "MRC/GUI dependencies are not installed")
class MrcCropTests(unittest.TestCase):
    def test_view_uses_a_live_memory_map_and_preserves_it_on_invalid_open(self):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import mrcfile
        import numpy as np

        from tomo_cropper import Cropper

        with tempfile.TemporaryDirectory() as folder:
            valid = Path(folder, "valid_Vol_b4.mrc")
            invalid = Path(folder, "invalid_Vol_b4.mrc")
            with mrcfile.new(valid) as handle:
                handle.set_data(np.zeros((2, 3, 4), dtype=np.float32))
            with mrcfile.new(invalid) as handle:
                handle.set_data(np.zeros((3, 4), dtype=np.float32))

            cropper = Cropper.__new__(Cropper)
            cropper._view_mrc = None
            cropper.vol = None
            try:
                cropper._map_view(str(valid))
                active_handle = cropper._view_mrc

                self.assertIs(cropper.vol, active_handle.data)
                self.assertIsInstance(cropper.vol, np.memmap)
                with self.assertRaisesRegex(ValueError, "3-D"):
                    cropper._map_view(str(invalid))
                self.assertIs(cropper._view_mrc, active_handle)
                self.assertIs(cropper.vol, active_handle.data)
            finally:
                cropper._close_view_mapping()

        self.assertIsNone(cropper.vol)
        self.assertIsNone(cropper._view_mrc)

    def test_crop_writes_expected_data_shape_and_voxel_size(self):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import mrcfile
        import numpy as np

        from tomo_cropper import Cropper

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "sample_Vol_b2.mrc")
            data = np.arange(6 * 8 * 10, dtype=np.float32).reshape(6, 8, 10)
            with mrcfile.new(source) as handle:
                handle.set_data(data)
                handle.voxel_size = (2.5, 3.5, 4.5)

            cropper = Cropper.__new__(Cropper)
            output, shape, bin_factor = cropper._crop_one(
                str(source),
                (2, 8, 2, 6, 2, 10),
            )

            with mrcfile.open(output, permissive=False) as handle:
                actual = handle.data.copy()
                voxel_size = (
                    float(handle.voxel_size.x),
                    float(handle.voxel_size.y),
                    float(handle.voxel_size.z),
                )

            temporary_files = list(Path(folder).glob("*.tmp-*"))

        np.testing.assert_array_equal(actual, data[1:5, 1:3, 1:4])
        self.assertEqual(shape, (4, 2, 3))
        self.assertEqual(bin_factor, 2.0)
        self.assertEqual(voxel_size, (2.5, 3.5, 4.5))
        self.assertEqual(temporary_files, [])

    def test_failed_replace_preserves_an_existing_crop(self):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import mrcfile
        import numpy as np

        from tomo_cropper import Cropper

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "sample_Vol_b2.mrc")
            output = Path(folder, "sample_Vol_b2_crop.mrc")
            source_data = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
            previous_crop = np.full((1, 1, 1), 99, dtype=np.float32)
            with mrcfile.new(source) as handle:
                handle.set_data(source_data)
            with mrcfile.new(output) as handle:
                handle.set_data(previous_crop)

            cropper = Cropper.__new__(Cropper)
            with mock.patch(
                "tomo_cropper.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    cropper._crop_one(str(source), (0, 4, 0, 4, 0, 4))

            with mrcfile.open(output, permissive=False) as handle:
                actual = handle.data.copy()
            temporary_files = list(Path(folder).glob("*.tmp-*"))

        np.testing.assert_array_equal(actual, previous_crop)
        self.assertEqual(temporary_files, [])

    def test_history_failure_prevents_original_deletion(self):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        from tomo_cropper import Cropper

        cropper = Cropper.__new__(Cropper)
        cropper.vol = object()
        cropper.path = os.path.abspath("sample_Vol_b4.mrc")
        cropper.prefix = "sample"
        cropper.viewbin = 4.0
        cropper.plans = []
        cropper._used_indices = set()
        cropper.status = mock.Mock()
        cropper.canonical_box = mock.Mock(return_value=(0, 4, 0, 4, 0, 4))
        cropper.current_box = mock.Mock(return_value=(0, 1, 0, 1, 0, 1))
        cropper._crop_files = mock.Mock(
            return_value=(["sample_Vol_b4_crop1.mrc"], [], [cropper.path])
        )
        cropper._delete_originals = mock.Mock(return_value=[])
        cropper.clear_selection = mock.Mock()

        with (
            mock.patch(
                "tomo_cropper.family_files",
                return_value=[(cropper.path, 4.0)],
            ),
            mock.patch(
                "tomo_cropper.upsert_history_rows",
                side_effect=OSError("simulated history failure"),
            ),
            mock.patch("tomo_cropper.messagebox.showwarning") as warning,
        ):
            cropper.apply()

        cropper._delete_originals.assert_called_once_with([])
        warning.assert_called_once()
        self.assertIn("history was not saved", warning.call_args.args[1])

    def test_deleting_the_active_view_closes_its_memory_map(self):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import mrcfile
        import numpy as np

        from tomo_cropper import Cropper

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "sample_Vol_b4.mrc")
            with mrcfile.new(source) as handle:
                handle.set_data(np.zeros((2, 2, 2), dtype=np.float32))

            cropper = Cropper.__new__(Cropper)
            cropper._view_mrc = None
            cropper.vol = None
            cropper._map_view(str(source))
            cropper.path = str(source)
            cropper.prefix = "sample"
            cropper.delorig = mock.Mock()
            cropper.delorig.get.return_value = 1
            cropper.info = mock.Mock()

            with mock.patch(
                "tomo_cropper.messagebox.askyesno",
                return_value=True,
            ):
                deleted = cropper._delete_originals([str(source)])

            self.assertFalse(source.exists())

        self.assertEqual(deleted, ["sample_Vol_b4.mrc"])
        self.assertIsNone(cropper._view_mrc)
        self.assertIsNone(cropper.vol)
        self.assertIsNone(cropper.path)

    def test_two_plans_write_numbered_crops_for_every_family_member(self):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import mrcfile
        import numpy as np

        from tomo_cropper import Cropper
        from tomo_cropper_core import CSV_NAME, load_history

        with tempfile.TemporaryDirectory() as folder:
            data = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
            for name in ("sample_Vol_b2.mrc", "sample_ODD_Vol_b2.mrc"):
                with mrcfile.new(Path(folder, name)) as handle:
                    handle.set_data(data)

            view = Path(folder, "sample_Vol_b2.mrc")
            cropper = Cropper.__new__(Cropper)
            cropper.vol = data
            cropper.path = str(view)
            cropper.prefix = "sample"
            cropper.viewbin = 2.0
            cropper.root = None
            cropper.status = mock.Mock()
            cropper.clear_selection = mock.Mock()
            cropper.delorig = mock.Mock()
            cropper.delorig.get.return_value = 0
            # two regions, in canonical bin-1 voxels (view bin 2 -> x2)
            cropper.plans = [
                {"index": 1, "canon": (0, 8, 0, 8, 0, 8),
                 "view": (0, 4, 0, 4, 0, 4)},
                {"index": 2, "canon": (16, 24, 16, 24, 8, 16),
                 "view": (8, 12, 8, 12, 4, 8)},
            ]
            cropper._used_indices = {1, 2}

            with (
                mock.patch("tomo_cropper.messagebox.askyesno", return_value=True),
                mock.patch("tomo_cropper.messagebox.showinfo") as info,
                mock.patch("tomo_cropper.messagebox.showwarning") as warning,
            ):
                cropper.apply()

            warning.assert_not_called()
            info.assert_called_once()

            written = sorted(p.name for p in Path(folder).glob("*_crop*.mrc"))
            self.assertEqual(
                written,
                [
                    "sample_ODD_Vol_b2_crop1.mrc",
                    "sample_ODD_Vol_b2_crop2.mrc",
                    "sample_Vol_b2_crop1.mrc",
                    "sample_Vol_b2_crop2.mrc",
                ],
            )

            with mrcfile.open(Path(folder, "sample_Vol_b2_crop2.mrc")) as handle:
                second = handle.data.copy()
            np.testing.assert_array_equal(second, data[4:8, 8:12, 8:12])

            plans = load_history(str(Path(folder, CSV_NAME)))["sample"]

            # originals are untouched while delete-originals is off
            self.assertTrue(view.exists())

        self.assertEqual([row["crop_index"] for row in plans], [1, 2])
        self.assertEqual([row["cx0"] for row in plans], [0, 16])

    def test_missing_history_folder_is_reported_as_unavailable(self):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        from tomo_cropper import Cropper

        with tempfile.TemporaryDirectory() as folder:
            missing = str(Path(folder, "moved-data"))
            history = {
                "sample": [
                    {
                        "folder": missing,
                        "crop_index": 1,
                        "cx0": 0,
                        "cx1": 4,
                        "cy0": 0,
                        "cy1": 4,
                        "cz0": 0,
                        "cz1": 4,
                    }
                ]
            }
            cropper = Cropper.__new__(Cropper)
            cropper.path = None

            with (
                mock.patch(
                    "tomo_cropper.filedialog.askopenfilename",
                    return_value=str(Path(folder, "history.csv")),
                ),
                mock.patch("tomo_cropper.load_history", return_value=history),
                mock.patch("tomo_cropper.messagebox.showwarning") as warning,
                mock.patch("tomo_cropper.messagebox.showinfo") as info,
            ):
                cropper.crop_from_csv()

        warning.assert_called_once()
        self.assertIn("missing or unavailable", warning.call_args.args[1])
        info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
