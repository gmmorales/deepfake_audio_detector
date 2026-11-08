import io
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf


SAMPLE_RATE = 16000
N_MELS = 128
TARGET_SHAPE = (128, 128)
WINDOW_SECONDS = 5.0


def audio_to_log_mel_windows(
    source: bytes | bytearray | io.BytesIO | str | Path,
    *,
    target_shape: tuple[int, int] = TARGET_SHAPE,
    window_seconds: float = WINDOW_SECONDS,
) -> np.ndarray:
    """Decode audio and convert it to normalized, fixed-size log-Mel windows."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise ValueError("Audio file is empty")
        source = io.BytesIO(source)

    audio, sample_rate = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError("Audio file contains no samples")

    window_size = int(sample_rate * window_seconds)
    windows = []
    for start in range(0, audio.size, window_size):
        window = audio[start : start + window_size]
        if window.size < window_size:
            window = np.pad(window, (0, window_size - window.size))

        mel_spectrogram = librosa.feature.melspectrogram(
            y=window,
            sr=sample_rate,
            n_mels=N_MELS,
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        value_range = float(np.ptp(mel_spectrogram_db))
        if value_range > 0:
            normalized_spectrogram = (
                mel_spectrogram_db - mel_spectrogram_db.min()
            ) / value_range
        else:
            normalized_spectrogram = np.zeros_like(mel_spectrogram_db)

        image = tf.image.resize(
            normalized_spectrogram[..., np.newaxis],
            target_shape,
        )
        windows.append(image.numpy().astype(np.float32, copy=False))

    return np.stack(windows)
