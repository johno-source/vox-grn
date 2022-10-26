

# define our unittests
import unittest
from vad_utils import *
from pydub import AudioSegment

class TestConvertFloatToPCM(unittest.TestCase):

    def testConvertToPCM(self):
        wav = [1.0, 0.5, 0.0, -0.5, -1.0]
        self.assertEqual(convert_float_to_pcm([]), b'')
        self.assertEqual(convert_float_to_pcm(wav), b'\xff\x7f\x00\x40\x00\x00\x00\xc0\x00\x80')

    def testConvertToPCMOutOfRange(self):
        wav = [10.0, 0.5, 0.0, -0.5, -100.0]
        self.assertEqual(convert_float_to_pcm([]), b'')
        self.assertEqual(convert_float_to_pcm(wav), b'\xff\x7f\x00\x40\x00\x00\x00\xc0\x00\x80')

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

class TestAudioSegmentToVAD(unittest.TestCase):
    def testAudioSegmentToBytes(self):
        test_audio_segment = sound = AudioSegment(
            data=b'\xff\x7f\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00\xc0\x00\x80\x00\x00',
            sample_width=4,
            frame_rate=44100,
            channels=1
        )
