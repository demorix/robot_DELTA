import os

import numpy as np
from dotenv import load_dotenv
from pywhispercpp.model import Model

_ENV_PATH = "data/.env"
load_dotenv(_ENV_PATH)


class ASR:

    def __init__(self) -> None:
        model_path = os.environ["WHISPER_MODEL_PATH"]
        n_threads  = int(os.environ.get("WHISPER_THREADS", "2"))
        language   = os.environ.get("WHISPER_LANGUAGE", "uk")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Whisper model not found: {model_path}")

        self._model = Model(model_path, n_threads=n_threads, language=language)

    def transcribe(self, audio: bytes) -> str:
        """

        Args:
            audio: int16 PCM bytes at self._samplerate Hz.
                   Always pass bytes; never pass a file path or numpy array.

        Returns:
            Transcribed string, whitespace-stripped. Empty string when nothing recognised.
        """
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments = self._model.transcribe(
            samples,
            temperature=0.0,
            no_context=True,
            single_segment=True,  # skip multi-segment overhead — commands are one utterance
            max_tokens=64,        # commands rarely exceed ~20 tokens; caps decoder iterations
        )
        return " ".join(s.text.strip() for s in segments).strip()