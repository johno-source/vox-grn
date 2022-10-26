# The moving average filter is something I expected to have in a python library. 
# Develop it using TDD
import unittest
import numpy as np
from vad_utils import MovingAverageFilter

class TestMovingAverage(unittest.TestCase):
    def testnullcase(self):
        with self.assertRaises(ValueError):
            dut = MovingAverageFilter(0)

    def testallones(self):
        dut = MovingAverageFilter(10)
        for ans in np.arange(0.1, 1.0, 0.1):
            self.assertAlmostEqual(ans, dut.filt(1))

        self.assertAlmostEqual(1.0, dut.filt(1))
        self.assertAlmostEqual(1.0, dut.filt(1))
        self.assertAlmostEqual(1.0, dut.filt(1))

    def testalltrue(self):
        dut = MovingAverageFilter(10)
        for ans in np.arange(0.1, 1.0, 0.1):
            self.assertAlmostEqual(ans, dut.filt(True))

        self.assertAlmostEqual(1.0, dut.filt(True))
        self.assertAlmostEqual(1.0, dut.filt(True))
        self.assertAlmostEqual(1.0, dut.filt(True))

    def testmixed(self):
        dut = MovingAverageFilter(4)
        self.assertEqual(0.25, dut.filt(1))
        self.assertEqual(0.5, dut.filt(1))
        self.assertEqual(0.75, dut.filt(1))
        self.assertEqual(0.75, dut.filt(0))
        self.assertEqual(0.5, dut.filt(0))
