"""Local speech recognition for talking to the agents (faster-whisper, runs offline on this computer).

The browser records through st.audio_input (it asks for microphone permission on first use); the audio is
transcribed here, on the local machine. Nothing is sent to a speech service. The model (~140 MB) downloads once.
"""
from __future__ import annotations

import hashlib
import io
import wave
from functools import lru_cache

import numpy as np

MODEL_SIZE = "base.en"
# Biases recognition toward this domain's vocabulary (callsigns, stands, runway names, team names).
VOCABULARY = ("ATC, Chief of Staff, AOCC, Flight Crew, Captain, MAYDAY, runway, stand, gate, TLDT, ELDT, SIBT, "
              "wake separation, holding, fuel reserve, turnaround, de-icing, METAR, PIREP, crosswind, United, Delta, "
              "American, UA1112, Denver, Dulles, negotiation, supervised, autonomous")


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def warm_up() -> None:
    """Download and load the model ahead of the first recording."""
    _model()


def fingerprint(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()[:16]


def _pcm16k(audio: bytes) -> np.ndarray:
    """WAV bytes (what st.audio_input records) -> mono float32 at 16 kHz, without any external decoder."""
    with wave.open(io.BytesIO(audio)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    x = (x - 128) / 128 if width == 1 else x / float(np.iinfo(dtype).max)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        n = int(len(x) * 16000 / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def transcribe(audio: bytes) -> str:
    segments, _ = _model().transcribe(_pcm16k(audio), language="en", initial_prompt=VOCABULARY,
                                      vad_filter=True, beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()
