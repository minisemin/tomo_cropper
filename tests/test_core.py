import csv
import tempfile
import unittest
from pathlib import Path

from tomo_cropper_core import (
    CSV_FIELDS,
    canonical_box,
    family_files,
    fmt_bin,
    load_history,
    normalize_box,
    parse_member,
    project_box,
    upsert_history,
)


def history_row(prefix, offset=0):
    return {
        "prefix": prefix,
        "folder": "/data",
        "view_file": f"{prefix}_Vol_b4.mrc",
        "view_bin": "4",
        "cx0": offset,
        "cx1": offset + 40,
        "cy0": 0,
        "cy1": 80,
        "cz0": 0,
        "cz1": 120,
        "timestamp": "2026-09-08T12:00:00",
    }


class FilenameTests(unittest.TestCase):
    def test_parse_main_and_half_maps(self):
        self.assertEqual(
            parse_member("Grid9_TS_01.mrc_Vol_b1.5.mrc"),
            ("Grid9_TS_01.mrc", "", 1.5),
        )
        self.assertEqual(
            parse_member("Grid9_TS_01.mrc_ODD_Vol.mrc"),
            ("Grid9_TS_01.mrc", "ODD", 1.0),
        )
        self.assertEqual(
            parse_member("Grid9_TS_01.mrc_EVN_Vol_b4.mrc"),
            ("Grid9_TS_01.mrc", "EVN", 4.0),
        )

    def test_rejects_outputs_and_invalid_bins(self):
        self.assertIsNone(parse_member("Grid9_TS_01.mrc_Vol_b4_crop.mrc"))
        self.assertIsNone(parse_member("Grid9_TS_01.mrc_Vol_b0.mrc"))
        self.assertIsNone(parse_member("notes.txt"))

    def test_bin_formatting(self):
        self.assertEqual(fmt_bin(4.0), "4")
        self.assertEqual(fmt_bin(1.5), "1.5")

    def test_family_discovery_is_exact_and_sorted(self):
        with tempfile.TemporaryDirectory() as folder:
            names = [
                "sample_Vol_b4.mrc",
                "sample_ODD_Vol.mrc",
                "sample_Vol_b1.5.mrc",
                "sample_Vol_b4_crop.mrc",
                "sample-other_Vol_b2.mrc",
                "sample_notes.mrc",
            ]
            for name in names:
                Path(folder, name).touch()

            result = family_files(folder, "sample")

        self.assertEqual(
            [(Path(path).name, bin_factor) for path, bin_factor in result],
            [
                ("sample_ODD_Vol.mrc", 1.0),
                ("sample_Vol_b1.5.mrc", 1.5),
                ("sample_Vol_b4.mrc", 4.0),
            ],
        )


class CoordinateTests(unittest.TestCase):
    def test_normalize_orders_and_clamps(self):
        self.assertEqual(
            normalize_box((12, -2, 9, 2, 11, 1), (10, 8, 6)),
            (0, 6, 2, 8, 1, 10),
        )

    def test_normalize_rejects_empty_box(self):
        self.assertIsNone(normalize_box((2, 2, 0, 5, 0, 5), (5, 5, 5)))

    def test_canonical_and_target_projection(self):
        canonical = canonical_box((2, 8, 4, 10, 6, 12), 1.5)
        self.assertEqual(canonical, (3, 12, 6, 15, 9, 18))
        self.assertEqual(
            project_box(canonical, 1.5, (20, 20, 20)),
            (2, 8, 4, 10, 6, 12),
        )

    def test_projection_clamps_to_real_header_shape(self):
        self.assertEqual(
            project_box((3, 12, 6, 21, 0, 30), 1.5, (15, 12, 7)),
            (2, 7, 4, 12, 0, 15),
        )

    def test_projection_rejects_invalid_bin(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            project_box((0, 1, 0, 1, 0, 1), 0, (1, 1, 1))


class HistoryTests(unittest.TestCase):
    def test_upsert_round_trip_and_preserves_other_prefixes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder, "history.csv"))
            upsert_history(path, history_row("first"))
            upsert_history(path, history_row("second", offset=10))
            upsert_history(path, history_row("first", offset=20))

            history = load_history(path)
            leftovers = list(Path(folder).glob(".history.csv.*.tmp"))

        self.assertEqual(set(history), {"first", "second"})
        self.assertEqual(history["first"]["cx0"], 20)
        self.assertEqual(history["first"]["view_bin"], 4.0)
        self.assertEqual(leftovers, [])

    def test_malformed_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "history.csv")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerow(history_row("valid"))
                invalid = history_row("invalid")
                invalid["cx0"] = "not-a-number"
                writer.writerow(invalid)

            history = load_history(str(path))

        self.assertEqual(set(history), {"valid"})

    def test_non_finite_coordinate_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "history.csv")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerow(history_row("valid"))
                for prefix, value in (("infinite", "inf"), ("nan", "nan")):
                    invalid = history_row(prefix)
                    invalid["cx0"] = value
                    writer.writerow(invalid)

            history = load_history(str(path))

        self.assertEqual(set(history), {"valid"})

    def test_upsert_requires_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder, "history.csv"))
            with self.assertRaisesRegex(ValueError, "prefix"):
                upsert_history(path, history_row(""))


if __name__ == "__main__":
    unittest.main()
