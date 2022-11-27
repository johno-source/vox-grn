

# define our unittests
import unittest
from vad_utils import *
from pydub import AudioSegment
import pickle as pkl

class TestVADFilter(unittest.TestCase):
    def testnovoice(self):
        dut = VADFilter()
        for _ in range(100):
            self.assertEqual(dut.filt(0), False)

    def testTurnOnOff(self):
        dut = VADFilter()
        for _ in range(8):
            self.assertEqual(dut.filt(1), False)
        
        self.assertEqual(dut.filt(1), True)
        for _ in range(8):
            self.assertEqual(dut.filt(0), True)
        self.assertEqual(dut.filt(0), False)
        

class TestBreakIntoSegments(unittest.TestCase):
    def testZeroSegments(self):
        self.assertEqual([], divide_into_segments([], 6.0))

    def testOneSegment(self):
        self.assertEqual([VoiceSegment(0, convert_seconds_to_frames(6.0))], divide_into_segments([VoiceSegment(0, convert_seconds_to_frames(6.0))], 6.0))

    def testTruncate(self):
        self.assertEqual([VoiceSegment(0, convert_seconds_to_frames(6.0))], divide_into_segments([VoiceSegment(0, convert_seconds_to_frames(8.0))], 6.0))

    def testMerge(self):
        self.assertEqual([VoiceSegment(0, convert_seconds_to_frames(6.0))], divide_into_segments([
            VoiceSegment(0, convert_seconds_to_frames(4.0)),
            VoiceSegment(convert_seconds_to_frames(4.9), convert_seconds_to_frames(6.0))], 6.0))

    def testMultiMerge(self):
        self.assertEqual([VoiceSegment(0, convert_seconds_to_frames(6.0))], divide_into_segments([
            VoiceSegment(0, convert_seconds_to_frames(3.0)),
            VoiceSegment(convert_seconds_to_frames(3.9), convert_seconds_to_frames(5.0)),
            VoiceSegment(convert_seconds_to_frames(5.5), convert_seconds_to_frames(7.0))
            ], 6.0))

    def testTwoSegments(self):
        self.assertEqual([VoiceSegment(0, convert_seconds_to_frames(6.0)), VoiceSegment(convert_seconds_to_frames(6.5), convert_seconds_to_frames(12.5))], 
            divide_into_segments([
                VoiceSegment(0, convert_seconds_to_frames(6.2)),
                VoiceSegment(convert_seconds_to_frames(6.5), convert_seconds_to_frames(13.0))
            ], 6.0))

class TestPickledSegments(unittest.TestCase):

    def testPickleAndUnpickle(self):
        dut = [VoiceSegment(0, convert_seconds_to_frames(6.0)), VoiceSegment(convert_seconds_to_frames(6.5), convert_seconds_to_frames(12.5))]
        with open('test.pkl', 'wb') as pklFile:
            pkl.dump(dut, pklFile)
        with open('test.pkl', 'rb') as pklFile:
            self.assertEqual(dut, pkl.load(pklFile))

class Test_audio_to_raw_voice_segments(unittest.TestCase):
    """This tests the voice segment generation rather than the VAD, which is called by these functions."""

    def testNoVoice(self):
        self.assertEqual([], audio_to_raw_voice_segments(AudioSegment.silent(duration=10.0)))

    def testOnRealVoice(self):
        test_seg = AudioSegment.from_file('/media/programs/Programs/03/03410/A03410/From_CM/C03410B-01.wav', format='wav')
        segs = audio_to_raw_voice_segments(test_seg)
        self.assertGreaterEqual(len(segs), 20)
        prior_stop = 0.0
        for seg in segs:
            self.assertGreater(seg.start, prior_stop)
            self.assertGreater(seg.stop, seg.start)
            prior_stop = seg.stop
