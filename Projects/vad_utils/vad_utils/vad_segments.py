"""
    VADSegments is used to condition wav files into a format needed to run webrtcvad and to convert the output of webrtcvad into AudioSegments that describe candidates for
    file creation for our own HF data set.
"""
import numpy as np
from pydub import AudioSegment


# Define constants
SAMPLING_RATE = 16000
FRAME_SIZE_MS = 30
SAMPLES_PER_FRAME = (SAMPLING_RATE * FRAME_SIZE_MS) / 1000

def convert_seconds_to_frames(secs):
    return int(SAMPLING_RATE * secs / SAMPLES_PER_FRAME)

def convert_frames_to_seconds(frames):
    return SAMPLES_PER_FRAME * frames / SAMPLING_RATE

def convert_frames_to_ms(frames):
    return FRAME_SIZE_MS * frames


MERGE_TOLERANCE = convert_seconds_to_frames(1.0)

# Input data conditioning. 
def convert_float_to_pcm(sig):
    """Input is a float array range -1.0 to 1.0. Output is byte array, big endian"""
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")

    i = np.iinfo('int16')
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    out_int16 = (sig * abs_max + offset).clip(i.min, i.max).astype('<i2')
    return out_int16.tobytes()


# Now the problem is that vad works on small frames. This gives the granularity for determining where voice starts and ends.
# These classes let us break the audio into suitable frames
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from floating point wav audio data.
    Takes the desired frame duration in milliseconds, the wav data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    # The 2 in this is because the PCM data is assumed to be 16 bit and the data in bytes.
    audio = convert_float_to_pcm(audio)
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def generate_frames_from_audio_segments(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from pydub AudioSegments.
    Takes the desired frame duration in milliseconds, the AudioSegment, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    assert isinstance(audio, AudioSegment)
    
    # convert to mono 16 bit PWM data
    if audio.channels != 1:
        audio = audio.set_channels(1)

    if audio.sample_width != 2:
        audio = audio.set_sample_width(2)

    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
        
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    a = audio.raw_data
    while offset + n < len(a):
        yield Frame(a[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


# webrtcvad output conditioning
class VADFilter:
    """The writers of webrtcvad use this algorithm on the output of webrtcvad. It is effectively a moving average filter with hysteresis."""
    def __init__(self):
        from vad_utils import MovingAverageFilter
        self.__maf = MovingAverageFilter(10)
        self.__isVoice = False

    def filt(self, newval):
        """Filter the output of webrtcvad. It is a moving average filter with hysteresis.

        Parameters
        ----------
            isVoice : bool
                true indicates that the corresponding frame is voice
            
        Returns
        -------
            isVoice : bool
                a filtered version of the input
        """
        isVoice = self.__maf.filt(newval)
        if not self.__isVoice:
            if isVoice >= 0.9:
                self.__isVoice = True
        elif isVoice <= 0.1:
            self.__isVoice = False
        return self.__isVoice

class VoiceSegment:
    """Defines the start and end of a segment of audio. Internally the values are kept in frames. These can be converted to seconds or samples as required."""
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f'Start: {convert_frames_to_seconds(self.start)} Stop: {convert_frames_to_seconds(self.stop)}'

def form_segments(vadout):
    """Convert the output of the VAD into AudioSegments. 

        Parameters
        ----------
            vadout : iterable(bool)
                true indicates that the corresponding frame is voice
            
        Returns
        -------
            segments : list[AudioSegment]
                a list of the segments where voiced audio was detected
    """
    in_segment = False
    start = 0
    segments = []
    for frame, v in enumerate(vadout):
        if v and not in_segment:
            in_segment = True
            start = frame
        elif not v and in_segment:
            in_segment = False
            segments.append(VoiceSegment(start, frame))

    return segments

def __take_segment_from_list(segs, start, frames, merge_tolerance):
    """Used to organise frames into segments. Calls recursively."""
    if len(segs) > 0:
        seg_frames = segs[0].stop - start
        if seg_frames >= frames:
            created_segment = [VoiceSegment(start, start+frames)]
            if len(segs) > 1:
                return created_segment + __take_segment_from_list(segs[1:], segs[1].start, frames, merge_tolerance)
            return created_segment
        elif len(segs) > 1:
            if (segs[0].stop + merge_tolerance) >= segs[1].start:
                return __take_segment_from_list(segs[1:], start, frames, merge_tolerance)
            return __take_segment_from_list(segs[1:], segs[1].start, frames, merge_tolerance)
    return []


def divide_into_segments(segs, seconds, merge_tolerance=MERGE_TOLERANCE):
    """Convert the raw AudioSegments into fixed time length AudioSegments using the following rules:
            1. Always start a new segment at the start of a raw segment. Partially used raw segments are discarded.
            2. Merge raw segments provided the start time of the subsequent segment leaves less that a merge_tolerance gap. The gap is not removed.

        Parameters
        ----------
            segs : iterable(AudioSegment)
                An arbitrary list of sequenced AudioSegments. It is assumed that the segments do not overlap.

            seconds : float
                The number of seconds desired for each output segment.

            merge_tolerance : float
                The amount of non voiced audio permitted in a segment. Default is 1 second.
            
        Returns
        -------
            segments : list[AudioSegment]
                a list of the segments each seconds long. An empty list means no suitable segments were found.
    """
    return __take_segment_from_list(segs, segs[0].start, convert_seconds_to_frames(seconds), convert_seconds_to_frames(merge_tolerance)) if len(segs) > 0 else []
