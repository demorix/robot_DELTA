import collections
import os

import numpy as np
import webrtcvad
from dotenv import load_dotenv
from pvrecorder import PvRecorder

from asr import ASR

_ENV_PATH = "data/.env"
load_dotenv(_ENV_PATH)

_SAMPLERATE        = int(os.environ.get("SAMPLERATE",             "16000"))
_BUFFER_SEC        = int(os.environ.get("BUFFER_SEC",             "5"))
_SLIDE_SEC         = float(os.environ.get("SLIDE_SEC",            "0.5"))
_VAD_MODE          = int(os.environ.get("VAD_MODE",               "2"))
_VAD_SPEECH_RATIO  = float(os.environ.get("VAD_SPEECH_RATIO",     "0.19"))
_VAD_ONSET_RATIO   = float(os.environ.get("VAD_ONSET_RATIO",      "0.30"))
_DEVICE_INDEX      = int(os.environ.get("AUDIO_DEVICE_INDEX",     "-1"))

_BUFFER_SAMPLES    = _BUFFER_SEC  * _SAMPLERATE
_SLIDE_SAMPLES     = int(_SLIDE_SEC * _SAMPLERATE)
_FRAME_30MS        = int(_SAMPLERATE * 0.030)   

_BUFFER_FRAMES     = _BUFFER_SAMPLES // _FRAME_30MS
_SLIDE_FRAMES      = _SLIDE_SAMPLES  // _FRAME_30MS
_FRAME_BYTES       = _FRAME_30MS * 2            

# Single VAD instance — both _has_speech and _trim_silence are called exclusively
# from the main loop; single-threaded execution makes this safe without locks.
_vad = webrtcvad.Vad(_VAD_MODE)


last_transcript: str = ""


def _has_speech(audio_bytes: bytes, ratio: float = _VAD_SPEECH_RATIO) -> bool:
    total  = 0
    speech = 0
    for i in range(0, len(audio_bytes) - _FRAME_BYTES, _FRAME_BYTES):
        if _vad.is_speech(audio_bytes[i : i + _FRAME_BYTES], _SAMPLERATE):
            speech += 1
        total += 1
    return total > 0 and (speech / total) >= ratio


_TRIM_MARGIN_FRAMES = 8   


def _trim_silence(audio_bytes: bytes) -> bytes:
    total_frames = len(audio_bytes) // _FRAME_BYTES

    speech_frames = [
        i
        for i in range(total_frames)
        if _vad.is_speech(audio_bytes[i * _FRAME_BYTES : (i + 1) * _FRAME_BYTES], _SAMPLERATE)
    ]

    if not speech_frames:
        return audio_bytes

    first_sample = max(0, speech_frames[0]  - _TRIM_MARGIN_FRAMES) * _FRAME_BYTES
    last_sample  = min(len(audio_bytes), (speech_frames[-1] + _TRIM_MARGIN_FRAMES + 1) * _FRAME_BYTES)
    return audio_bytes[first_sample:last_sample]


def _process(trimmed: bytes, asr: ASR) -> None:
    global last_transcript
    text = asr.transcribe(trimmed)
    if text:
        last_transcript = text
        print(f"[STT] {text}")
        # LLM integration point (enable when module is ready)
        # command = intent_inference(text)
        # execute(command)


def intent_inference(text: str) -> str:
    """Maps raw transcript to a structured command. Implemented by LLM module."""
    return text


def execute(command: str) -> None:
    """Dispatches a command. Implemented by execution module."""


def main() -> None:
    asr = ASR()

    ring          = collections.deque(maxlen=_BUFFER_FRAMES)
    accumulated   = 0
    phrase_active = False

    recorder = PvRecorder(frame_length=_FRAME_30MS, device_index=_DEVICE_INDEX)
    recorder.start()

    print("[STT] Listening…")
    try:
        while True:
            pcm = recorder.read()

            ring.append(np.asarray(pcm, dtype=np.int16).tobytes())

            # Gate onset to _SLIDE_SEC cadence — frequent checks waste CPU on silence.
            # Offset skips the gate: silence must be detected within one frame (30 ms)
            # of the recent window filling, not within the next slide boundary.
            accumulated += _FRAME_30MS
            if accumulated >= _SLIDE_SAMPLES:
                accumulated -= _SLIDE_SAMPLES
            elif not phrase_active:
                continue

            # _BUFFER_FRAMES warmup removed — VAD reads only the tail (recent),
            # so stale head-frames after ring.clear() never cause false onset.
            if len(ring) < _SLIDE_FRAMES:
                continue

            recent = b"".join(list(ring)[-_SLIDE_FRAMES:])

            if not phrase_active:
                # ONSET: check recent slice to catch short utterances.
                if _has_speech(recent, _VAD_ONSET_RATIO):
                    phrase_active = True

            else:
                if not _has_speech(recent):
                    phrase_active = False
                    trimmed = _trim_silence(b"".join(ring))
                    ring.clear()
                    accumulated = 0

                    try:
                        _process(trimmed, asr)
                    except Exception as exc:
                        print(f"[STT] transcription error: {exc}")

    finally:
        recorder.stop()
        recorder.delete()


if __name__ == "__main__":
    main()