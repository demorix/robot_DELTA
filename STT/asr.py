import os

import numpy as np
from dotenv import load_dotenv
from pywhispercpp.model import Model

_ENV_PATH = "data/.env"
load_dotenv(_ENV_PATH)


class ASR:
    """Whisper-cpp inference engine. Transcribes in-memory audio"""

    def __init__(self) -> None:
        model_path = os.environ["WHISPER_MODEL_PATH"]
        n_threads  = int(os.environ.get("WHISPER_THREADS", "3"))
        language   = os.environ.get("WHISPER_LANGUAGE", "uk")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Whisper model not found: {model_path}")

        self._model      = Model(model_path, n_threads=n_threads, language=language)
        self._samplerate = int(os.environ.get("SAMPLERATE", "16000"))

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribes raw audio to text entirely in memory — no temp files, no disk writes.

        Args:
            audio: int16 or float32 PCM samples at self._samplerate Hz.
                   Always pass a numpy array; never pass a file path.

        Returns:
            Transcribed string, whitespace-stripped. Empty string when nothing recognised.
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        segments = self._model.transcribe(audio, temperature=0.0)
        return " ".join(s.text.strip() for s in segments).strip()