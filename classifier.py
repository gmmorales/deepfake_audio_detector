import io
import logging
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when prediction is requested without a trained model."""


class AudioClassifier:
    def __init__(
        self,
        model_path: str | Path | None = None,
        target_shape: tuple[int, int] = (128, 128),
        window_seconds: float = 5.0,
        physical_threshold: float = 0.4,
        digital_threshold: float = 0.6,
    ) -> None:
        if target_shape[0] <= 0 or target_shape[1] <= 0:
            raise ValueError("target_shape dimensions must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0 <= physical_threshold < digital_threshold <= 1:
            raise ValueError(
                "thresholds must satisfy 0 <= physical_threshold "
                "< digital_threshold <= 1"
            )

        self.target_shape = target_shape
        self.window_seconds = window_seconds
        self.physical_threshold = physical_threshold
        self.digital_threshold = digital_threshold
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).parent / "models" / "audio_model.keras"
        )

        if self.model_path.is_file():
            self.model = tf.keras.models.load_model(self.model_path)
        elif model_path is not None:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        else:
            self.model = None
            logger.warning(
                "No trained audio model found at %s; predictions are disabled",
                self.model_path,
            )

    def preprocess_audio(self, file_bytes: bytes) -> np.ndarray:
        """Convert all audio in the file to 5-second log-Mel model inputs."""
        if not file_bytes:
            raise ValueError("Audio file is empty")

        audio_stream = io.BytesIO(file_bytes)
        audio, sample_rate = librosa.load(audio_stream, sr=16000, mono=True)
        if audio.size == 0:
            raise ValueError("Audio file contains no samples")

        window_size = int(sample_rate * self.window_seconds)
        windows = []
        for start in range(0, audio.size, window_size):
            window = audio[start : start + window_size]
            if window.size < window_size:
                window = np.pad(window, (0, window_size - window.size))

            mel_spectrogram = librosa.feature.melspectrogram(
                y=window,
                sr=sample_rate,
                n_mels=128,
            )
            mel_spectrogram_db = librosa.power_to_db(
                mel_spectrogram,
                ref=np.max,
            )
            value_range = float(np.ptp(mel_spectrogram_db))
            if value_range > 0:
                normalized_spectrogram = (
                    mel_spectrogram_db - mel_spectrogram_db.min()
                ) / value_range
            else:
                normalized_spectrogram = np.zeros_like(mel_spectrogram_db)

            image = tf.image.resize(
                normalized_spectrogram[..., np.newaxis],
                self.target_shape,
            )
            windows.append(image.numpy().astype(np.float32, copy=False))

        return np.stack(windows)

    def predict(self, file_bytes: bytes) -> dict[str, str | float]:
        """Classify a recording using the mean digital score across its windows."""
        if self.model is None:
            raise ModelNotAvailableError(
                f"No trained model is available at {self.model_path}"
            )

        input_data = self.preprocess_audio(file_bytes)
        predictions = np.asarray(
            self.model.predict(input_data, verbose=0),
            dtype=np.float64,
        ).reshape(-1)
        if predictions.size != input_data.shape[0]:
            raise ValueError(
                "The model must return one digital score per audio window"
            )
        if not np.all(np.isfinite(predictions)) or np.any(
            (predictions < 0) | (predictions > 1)
        ):
            raise ValueError("The model must return finite scores between 0 and 1")

        digital_score = float(np.mean(predictions))
        if digital_score >= self.digital_threshold:
            label = "Digital"
        elif digital_score <= self.physical_threshold:
            label = "Físico"
        else:
            label = "Incierto"

        confidence = max(digital_score, 1.0 - digital_score)
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "raw_score": digital_score,
        }


classifier = AudioClassifier()
