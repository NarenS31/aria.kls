"""
Audio think-aloud listener for ARIA (optional mode).

ThinkAloudListener records the microphone in short chunks, transcribes speech
with Whisper, and builds a live transcript that can be fed to the
CognitiveStateAnalyzer together with prosodic audio features.

Behaviour:
  * start_listening()      — begin recording 3-second chunks in a background
                             thread; each chunk with speech is transcribed and
                             appended to the running transcript.
  * stop_listening()       — stop recording and return the full transcript.
  * get_live_transcript()  — return the transcript built so far.

Silence handling:
  * RMS energy threshold separates speech from silence; silent chunks are not
    transcribed.
  * A silence gap longer than 3 seconds is marked in the transcript as
    "[pause]" (a signal the analyzer treats as STUCK/CONFUSED).

Per-chunk audio features (also aggregated for the whole session):
  * speech_rate  — words per minute
  * pause_count  — number of pauses > 0.5 s
  * filler_rate  — um / uh / like per minute

`sounddevice` and `whisper` are heavy optional dependencies; they are imported
lazily so that importing this module never fails on a machine without a mic.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

SAMPLE_RATE = 16000          # Whisper expects 16 kHz mono
CHUNK_SECONDS = 3.0
SILENCE_RMS = 0.01           # RMS below this = silence (normalised float audio)
PAUSE_GAP_SECONDS = 3.0      # silence longer than this -> "[pause]"
WHISPER_MODEL = "small"

FILLER_RE = re.compile(r"\b(um+|uh+|erm+|like|you\s+know|hmm+)\b", re.IGNORECASE)


@dataclass
class AudioFeatures:
    speech_rate: float = 0.0      # words per minute
    pause_count: int = 0          # pauses > 0.5 s
    filler_rate: float = 0.0      # fillers per minute
    word_count: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "speech_rate": round(self.speech_rate, 1),
            "pause_count": self.pause_count,
            "filler_rate": round(self.filler_rate, 1),
            "word_count": self.word_count,
            "duration_seconds": round(self.duration_seconds, 1),
        }


class ThinkAloudListener:
    """Record + transcribe think-aloud audio and extract prosodic features."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = CHUNK_SECONDS,
        whisper_model: str = WHISPER_MODEL,
        silence_rms: float = SILENCE_RMS,
    ):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.whisper_model_name = whisper_model
        self.silence_rms = silence_rms

        self._sd = None
        self._np = None
        self._whisper_model = None

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._transcript_parts: list[str] = []
        self._word_count = 0
        self._pause_count = 0
        self._filler_count = 0
        self._speech_seconds = 0.0
        self._silence_run = 0.0
        self._start_time: Optional[float] = None

    # -- lazy dependency loading ------------------------------------

    def _ensure_deps(self) -> None:
        if self._sd is None:
            import sounddevice as sd  # noqa: F401
            import numpy as np
            self._sd = sd
            self._np = np
        if self._whisper_model is None:
            import whisper
            self._whisper_model = whisper.load_model(self.whisper_model_name)

    @staticmethod
    def is_available() -> bool:
        """True if the optional audio stack can be imported."""
        try:
            import sounddevice  # noqa: F401
            import whisper       # noqa: F401
            import numpy         # noqa: F401
            return True
        except Exception:
            return False

    # -- lifecycle ---------------------------------------------------

    def start_listening(self) -> None:
        """Begin recording + transcribing in a background thread."""
        if self._running:
            return
        self._ensure_deps()
        with self._lock:
            self._transcript_parts = []
            self._word_count = 0
            self._pause_count = 0
            self._filler_count = 0
            self._speech_seconds = 0.0
            self._silence_run = 0.0
            self._start_time = time.time()
            self._running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop_listening(self) -> str:
        """Stop recording and return the full transcript."""
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.chunk_seconds + 5)
            self._thread = None
        return self.get_live_transcript()

    def get_live_transcript(self) -> str:
        with self._lock:
            return " ".join(self._transcript_parts).strip()

    # -- recording loop ---------------------------------------------

    def _record_loop(self) -> None:
        np = self._np
        sd = self._sd
        frames = int(self.sample_rate * self.chunk_seconds)
        while True:
            with self._lock:
                if not self._running:
                    break
            try:
                audio = sd.rec(frames, samplerate=self.sample_rate,
                               channels=1, dtype="float32")
                sd.wait()
            except Exception:
                # Device error — stop gracefully.
                break
            chunk = audio.reshape(-1)
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0

            if rms < self.silence_rms:
                # Silence chunk: accumulate pause time.
                self._silence_run += self.chunk_seconds
                if self._silence_run >= PAUSE_GAP_SECONDS:
                    with self._lock:
                        if not self._transcript_parts or self._transcript_parts[-1] != "[pause]":
                            self._transcript_parts.append("[pause]")
                    self._silence_run = 0.0
                continue

            # Speech chunk: transcribe.
            self._silence_run = 0.0
            text = self._transcribe(chunk)
            if text:
                self._ingest(text)

    def _transcribe(self, chunk) -> str:
        try:
            result = self._whisper_model.transcribe(
                chunk, fp16=False, language="en",
            )
            return (result.get("text") or "").strip()
        except Exception:
            return ""

    def _ingest(self, text: str) -> None:
        words = re.findall(r"\w+", text)
        fillers = len(FILLER_RE.findall(text))
        # Count short intra-utterance pauses (commas / ellipses) as > 0.5 s gaps.
        short_pauses = text.count("...") + text.count(",")
        with self._lock:
            self._transcript_parts.append(text)
            self._word_count += len(words)
            self._filler_count += fillers
            self._pause_count += short_pauses
            self._speech_seconds += self.chunk_seconds

    # -- features ----------------------------------------------------

    def get_audio_features(self) -> dict:
        """Aggregate prosodic features over the session so far."""
        with self._lock:
            speech_min = max(self._speech_seconds / 60.0, 1e-6)
            duration = (time.time() - self._start_time) if self._start_time else 0.0
            feats = AudioFeatures(
                speech_rate=self._word_count / speech_min,
                pause_count=self._pause_count,
                filler_rate=self._filler_count / speech_min,
                word_count=self._word_count,
                duration_seconds=duration,
            )
        return feats.to_dict()

    @staticmethod
    def features_from_text(text: str, duration_seconds: float) -> dict:
        """Compute audio-style features from an already-transcribed string.

        Useful for tests / offline pipelines where no live mic is available.
        """
        words = re.findall(r"\w+", text)
        fillers = len(FILLER_RE.findall(text))
        pauses = text.count("[pause]") + text.count("...")
        minutes = max(duration_seconds / 60.0, 1e-6)
        return AudioFeatures(
            speech_rate=len(words) / minutes,
            pause_count=pauses,
            filler_rate=fillers / minutes,
            word_count=len(words),
            duration_seconds=duration_seconds,
        ).to_dict()


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if not ThinkAloudListener.is_available():
        print("Audio dependencies (sounddevice + whisper) not available.")
        print("Feature extraction from text still works:")
        demo = "Um, so first I need to... uh, wait, I don't get it. [pause]"
        print(json.dumps(ThinkAloudListener.features_from_text(demo, 8.0), indent=2))
        sys.exit(0)

    listener = ThinkAloudListener()
    print("Recording for 10 seconds — think out loud...")
    listener.start_listening()
    time.sleep(10)
    transcript = listener.stop_listening()
    print("\nTranscript:", transcript)
    print("Features:", json.dumps(listener.get_audio_features(), indent=2))
