"""A ComfyUI AUDIO as the WAV file a reference audio is sent as.

anex.sh takes reference audio as WAV or MP3; the node writes every AUDIO it
is given as 16-bit PCM WAV, whatever it was loaded from, so no other format
ever leaves ComfyUI. Kept free of ComfyUI imports so it can be tested alone.
"""
from __future__ import annotations

import io
import wave

import numpy as np

# anex.sh's limits on reference audio (MiniMax's): up to 3 files, each 2 to
# 15 seconds and 15 MB, and 15 seconds in total.
MAX_REFERENCE_AUDIOS = 3
REFERENCE_AUDIO_SECONDS = (2, 15)
MAX_TOTAL_AUDIO_SECONDS = 15
MAX_AUDIO_BYTES = 15 * 1024 * 1024
# More channels than this are mixed down to mono: the model reads speech and
# sound, not surround.
MAX_CHANNELS = 2


def _samples(audio) -> tuple[np.ndarray, int]:
    """(samples as a (channels, frames) float array, sample rate) of one
    AUDIO: a dict with `waveform` (batch, channels, frames) and
    `sample_rate`. Only the first item of a batch is taken."""
    try:
        waveform, rate = audio["waveform"], int(audio["sample_rate"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("an audio input is not a ComfyUI AUDIO") from None
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().float().numpy()
    data = np.asarray(waveform, dtype=np.float32)
    if data.ndim == 3:
        data = data[0]
    elif data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or rate <= 0:
        raise ValueError("an audio input has no samples")
    return data, rate


def seconds(audio) -> float:
    """How long one AUDIO lasts."""
    data, rate = _samples(audio)
    return data.shape[1] / rate


def wav_bytes(audio) -> bytes:
    """One AUDIO as 16-bit PCM WAV bytes, at its own sample rate, mono or
    stereo."""
    data, rate = _samples(audio)
    if data.shape[0] > MAX_CHANNELS:
        data = data.mean(axis=0, keepdims=True)
    pcm = (np.clip(data, -1.0, 1.0) * 32767).round().astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(pcm.shape[0])
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.T.tobytes())
    return buf.getvalue()


def check(audios: list) -> list[float]:
    """Refuse audio anex.sh would refuse, before anything is uploaded or
    billed; returns each one's seconds. Audio n is the n-th connected."""
    if len(audios) > MAX_REFERENCE_AUDIOS:
        raise ValueError(f"{len(audios)} reference audios; at most "
                         f"{MAX_REFERENCE_AUDIOS}")
    low, high = REFERENCE_AUDIO_SECONDS
    lengths = [seconds(a) for a in audios]
    for n, length in enumerate(lengths, 1):
        if not low <= length <= high:
            raise ValueError(f"Audio {n} is {length:.1f} s; each reference "
                             f"audio must be {low} to {high} s")
    total = sum(lengths)
    if total > MAX_TOTAL_AUDIO_SECONDS:
        raise ValueError(f"the reference audios total {total:.1f} s; at most "
                         f"{MAX_TOTAL_AUDIO_SECONDS} s")
    return lengths
