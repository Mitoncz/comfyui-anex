"""Reference audio as the node sends it: a ComfyUI AUDIO written as WAV,
and the limits checked before anything is uploaded. Needs numpy only."""
import io
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anex_comfy import audio  # noqa: E402


def an_audio(seconds, rate=44100, channels=2, batch=True):
    frames = int(seconds * rate)
    t = np.arange(frames) / rate
    wave_ = np.stack([np.sin(2 * np.pi * 220 * (c + 1) * t) * 0.5
                      for c in range(channels)]).astype(np.float32)
    return {"waveform": wave_[None] if batch else wave_, "sample_rate": rate}


def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        return (w.getnchannels(), w.getsampwidth(), w.getframerate(),
                w.getnframes())


def test_wav_keeps_rate_channels_and_length():
    data = audio.wav_bytes(an_audio(3.0))
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert read_wav(data) == (2, 2, 44100, 132300)


def test_wav_of_an_unbatched_mono_waveform():
    data = audio.wav_bytes(an_audio(2.5, rate=16000, channels=1, batch=False))
    assert read_wav(data) == (1, 2, 16000, 40000)


def test_more_than_two_channels_mixed_to_mono():
    assert read_wav(audio.wav_bytes(an_audio(2.0, channels=6)))[0] == 1


def test_samples_clipped_to_full_scale():
    loud = {"waveform": np.full((1, 1, 32000), 3.0, np.float32),
            "sample_rate": 16000}
    with wave.open(io.BytesIO(audio.wav_bytes(loud))) as w:
        pcm = np.frombuffer(w.readframes(10), "<i2")
    assert (pcm == 32767).all()


def test_a_torch_like_waveform_is_read():
    class Tensor:
        def __init__(self, a):
            self.a = a

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self.a
    clip = an_audio(2.0)
    clip["waveform"] = Tensor(clip["waveform"])
    assert audio.seconds(clip) == pytest.approx(2.0)


def test_limits_checked_before_upload():
    assert audio.check([an_audio(2.0), an_audio(13.0)]) == [
        pytest.approx(2.0), pytest.approx(13.0)]
    with pytest.raises(ValueError, match="Audio 1 is 1.5 s"):
        audio.check([an_audio(1.5)])
    with pytest.raises(ValueError, match="Audio 2 is 15.5 s"):
        audio.check([an_audio(3), an_audio(15.5)])
    with pytest.raises(ValueError, match="total 16.0 s"):
        audio.check([an_audio(8), an_audio(8)])
    with pytest.raises(ValueError, match="at most 3"):
        audio.check([an_audio(2)] * 4)
    with pytest.raises(ValueError, match="not a ComfyUI AUDIO"):
        audio.check([{"samples": []}])
