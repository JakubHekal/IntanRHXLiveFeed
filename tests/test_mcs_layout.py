import unittest

import numpy as np

from leech.device.mcs_mea.layout import (
    channel_names,
    electrode_positions,
    mea_layout,
    mea_name,
)


MEA60_LABELS = [
    "47", "48", "46", "45", "38", "37", "28", "36", "27", "17",
    "26", "16", "35", "25", "Ref", "14", "24", "34", "13", "23",
    "12", "22", "33", "21", "32", "31", "44", "43", "41", "42",
    "52", "51", "53", "54", "61", "62", "71", "63", "72", "82",
    "73", "83", "64", "74", "84", "85", "75", "65", "86", "76",
    "87", "77", "66", "78", "67", "68", "55", "56", "58", "57",
]

MEA120_LABELS = [
    "F7", "F8", "F12", "F11", "F10", "F9", "E12", "E11", "E10", "E9",
    "D12", "D11", "D10", "D9", "C11", "C10", "B10", "E8", "C9", "B9",
    "A9", "D8", "C8", "B8", "A8", "D7", "C7", "B7", "A7", "E7",
    "F6", "E6", "A6", "B6", "C6", "D6", "A5", "B5", "C5", "D5",
    "A4", "B4", "C4", "D4", "B3", "C3", "C2", "E5", "D3", "D2",
    "D1", "E4", "E3", "E2", "E1", "F4", "F3", "F2", "F1", "F5",
    "G6", "G5", "G1", "G2", "G3", "G4", "H1", "H2", "H3", "H4",
    "J1", "J2", "J3", "J4", "K2", "K3", "L3", "H5", "K4", "L4",
    "M4", "J5", "K5", "L5", "M5", "J6", "K6", "L6", "M6", "H6",
    "G7", "H7", "M7", "L7", "K7", "J7", "M8", "L8", "K8", "J8",
    "M9", "L9", "K9", "J9", "L10", "K10", "K11", "H8", "J10", "J11",
    "J12", "H9", "H10", "H11", "H12", "G9", "G10", "G11", "G12", "G8",
]


class McsLayoutTest(unittest.TestCase):
    def test_60_channel_channel_order_and_positions(self):
        self.assertEqual(channel_names(60), MEA60_LABELS)
        xs, ys = electrode_positions(60)
        self.assertEqual(len(set(zip(xs, ys))), 60)
        for corner in ((0.0, 0.0), (7.0, 0.0), (0.0, -7.0), (7.0, -7.0)):
            self.assertNotIn(corner, set(zip(xs, ys)))
        ref_index = MEA60_LABELS.index("Ref")
        self.assertEqual((xs[ref_index], ys[ref_index]), (0.0, -4.0))

    def test_120_channel_channel_order_and_positions(self):
        self.assertEqual(channel_names(120), MEA120_LABELS)
        xs, ys = electrode_positions(120)
        self.assertEqual(len(set(zip(xs, ys))), 120)
        row_counts = [int(np.sum(ys == -row)) for row in range(12)]
        self.assertEqual(row_counts, [6, 8, 10, 12, 12, 12, 12, 12, 12, 10, 8, 6])
        self.assertEqual((xs[0], ys[0]), (6.0, -5.0))
        self.assertEqual((xs[-1], ys[-1]), (7.0, -6.0))

    def test_verified_product_names(self):
        self.assertEqual(mea_name(60), "60MEA200/30iR")
        self.assertEqual(mea_layout(60), "60MEA200_30_iR")
        self.assertEqual(mea_name(120), "120MEA200/30iR")
        self.assertEqual(mea_layout(120), "120MEA200_30_iR")


if __name__ == "__main__":
    unittest.main()
