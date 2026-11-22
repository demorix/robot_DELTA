import collections
import os
import threading

import numpy as np
import pvporcupine
import webrtcvad
from dotenv import load_dotenv
from pvrecorder import PvRecorder

from asr import ASR

_ENV_PATH = "data/.env"
load_dotenv(_ENV_PATH)

_SAMPLERATE        = int(os.environ.get("SAMPLERATE",             "16000"))
_BUFFER_SEC        = int(os.environ.get("BUFFER_SEC",             "5"))
_SLIDE_SEC         = int(os.environ.get("SLIDE_SEC",              "1"))
_VAD_MODE          = int(os.environ.get("VAD_MODE",               "2"))
_VAD_SPEECH_RATIO  = float(os.environ.get("VAD_SPEECH_RATIO",     "0.30"))
_WAKE_WINDOW_SEC   = int(os.environ.get("WAKE_WINDOW_SEC",        "5"))
_SENSITIVITY       = float(os.environ.get("WAKE_WORD_SENSITIVITY", "0.7"))
_DEVICE_INDEX      = int(os.environ.get("AUDIO_DEVICE_INDEX",     "-1"))
_MIN_COMMAND_SEC   = int(os.environ.get("MIN_COMMAND_SEC",         "2"))

_BUFFER_SAMPLES    = _BUFFER_SEC * _SAMPLERATE
_SLIDE_SAMPLES     = _SLIDE_SEC  * _SAMPLERATE
_FRAME_30MS        = int(_SAMPLERATE * 0.03)
_MIN_COMMAND_SAMPLES = _MIN_COMMAND_SEC * _SAMPLERATE

# Single VAD instance — webrtcvad.Vad() is not thread-safe but _has_speech
# is always called from the main loop thread, never from the worker thread.
_vad = webrtcvad.Vad(_VAD_MODE)


def _has_speech(audio_f32: np.ndarray) -> bool:
    """
    Returns True when the ratio of 30-ms speech frames exceeds _VAD_SPEECH_RATIO.

    Evaluates the full buffer for context rather than a single short frame,
    which eliminates the false-silence problem of single-frame VAD.
    """
    pcm16  = (audio_f32 * 32768.0).clip(-32768, 32767).astype(np.int16)
    total  = 0
    speech = 0
    for start in range(0, len(pcm16) - _FRAME_30MS, _FRAME_30MS):
        frame = pcm16[start : start + _FRAME_30MS].tobytes()
        if _vad.is_speech(frame, _SAMPLERATE):
            speech += 1
        total += 1
    return total > 0 and (speech / total) >= _VAD_SPEECH_RATIO


def _process(snapshot: np.ndarray, asr: ASR, lock: threading.Lock) -> None:
    """
    Runs STT → intent_inference → execute on a frozen buffer snapshot.

    Runs in a daemon thread so the audio-capture loop is never blocked.
    Releases the lock unconditionally in finally so the next trigger can fire.
    """
    try:
        text = asr.transcribe(snapshot)
        if text:
            # print(f"\n[Transcribed]: {text}")
            command = intent_inference(text)
            execute(command)
    finally:
        lock.release()


# функції виклику LLM
def intent_inference(text: str) -> str:
    return text


def execute(command: str) -> None:
    """Boundary stub — replace with the execution-module implementation."""


def main() -> None:
    asr = ASR()

    porcupine = pvporcupine.create(
        access_key    = os.environ["PORCUPINE_ACCESS_KEY"],
        keyword_paths = [os.environ["WAKE_WORD_PATH"]],
        sensitivities = [_SENSITIVITY],
    )

    frame_len   = porcupine.frame_length
    ring        = collections.deque(maxlen=_BUFFER_SAMPLES)
    accumulated = 0
    wake_at     = None   # sample index at which the last wake word fired
    total       = 0      # total samples consumed since process start
    proc_lock   = threading.Lock()

    recorder = PvRecorder(frame_length=frame_len, device_index=_DEVICE_INDEX)
    recorder.start()

    # print("\n 'Hey Ava'")
    try:
        while True:
            pcm = recorder.read()

            # Porcupine processes every frame at its native frame rate — this is
            # the lowest-latency path for wake-word detection available on RPi 4.
            if porcupine.process(pcm) >= 0:
                # Discard pre-wake audio so Whisper only sees the command.
                ring.clear()
                accumulated = 0
                wake_at = total
                # print("\nРозпізнаю команду...")

            # Normalize int16 PCM → float32 [-1, 1] and push into the ring buffer.
            # Multiplying by reciprocal is faster than dividing on ARM cores.
            ring.extend(np.asarray(pcm, dtype=np.float32) * (1.0 / 32768.0))
            total       += frame_len
            accumulated += frame_len

            # Gate: only evaluate on each sliding-window step, not every frame.
            if accumulated < _SLIDE_SAMPLES:
                continue
            accumulated -= _SLIDE_SAMPLES   # carry remainder to keep timing accurate

            # Skip until the ring buffer holds _BUFFER_SEC of audio.
            # When wake is active the buffer was just cleared, so a short
            # post-wake recording is expected — bypass the full-buffer requirement.
            wake_active = (
                wake_at is not None
                and (total - wake_at) <= _WAKE_WINDOW_SEC * _SAMPLERATE
            )
            if not wake_active and len(ring) < _BUFFER_SAMPLES:
                continue

            if not wake_active:
                continue

            # Wait until enough post-wake audio is captured for a complete command.
            # Triggering at SLIDE_SEC=1 risks transcribing a partial utterance and
            # burning the lock before the user finishes speaking.
            if wake_at is not None and (total - wake_at) < _MIN_COMMAND_SAMPLES:
                continue

            # Materialise a snapshot before VAD — avoids a race where the deque
            # mutates between the VAD call and the transcription snapshot.
            snapshot = np.array(ring, dtype=np.float32)

            # Gate 2 — VAD on the full buffer provides enough temporal context
            # to distinguish real speech from transient noise or clipping.
            if not _has_speech(snapshot):
                continue

            # Both gates passed: non-blocking lock prevents stacking a second
            # transcription while the previous one is still running.
            if proc_lock.acquire(blocking=False):
                wake_at = None   # consume the event; require a fresh wake word
                threading.Thread(
                    target=_process,
                    args=(snapshot, asr, proc_lock),
                    daemon=True,
                ).start()

    finally:
        recorder.stop()
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    main()