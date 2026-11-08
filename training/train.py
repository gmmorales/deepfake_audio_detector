import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from training.audio_features import TARGET_SHAPE, audio_to_log_mel_windows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "audio_model.keras"
LABELS = {"fisico": 0, "físico": 0, "physical": 0, "digital": 1}
VALIDATION_FRACTION = 0.2


@dataclass(frozen=True)
class AudioRecord:
    path: Path
    relative_path: str
    label: int
    group_id: str


def load_records(metadata_path: Path, data_root: Path) -> list[AudioRecord]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    records = []
    group_labels: dict[str, int] = {}
    with metadata_path.open(encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        required_columns = {"relative_path", "label", "source_id"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Metadata CSV is missing required columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            relative_path = (row.get("relative_path") or "").strip()
            label_text = (row.get("label") or "").strip().casefold()
            if not relative_path or not label_text:
                raise ValueError(
                    f"Metadata row {line_number} must include relative_path and label"
                )
            if label_text not in LABELS:
                raise ValueError(
                    f"Unknown label {label_text!r} on metadata row {line_number}; "
                    "use 'fisico' or 'digital'"
                )

            path = (data_root / relative_path).resolve()
            try:
                path.relative_to(data_root.resolve())
            except ValueError as error:
                raise ValueError(
                    f"Audio path on metadata row {line_number} must be inside "
                    f"{data_root}"
                ) from error
            if not path.is_file():
                raise FileNotFoundError(
                    f"Audio file from metadata row {line_number} not found: {path}"
                )

            label = LABELS[label_text]
            group_id = (row.get("source_id") or "").strip()
            if not group_id:
                raise ValueError(
                    f"Metadata row {line_number} must include source_id to prevent "
                    "recordings from the same source crossing the train/validation split"
                )
            previous_label = group_labels.setdefault(group_id, label)
            if previous_label != label:
                raise ValueError(
                    f"Source {group_id!r} has recordings with conflicting labels"
                )

            records.append(
                AudioRecord(
                    path=path,
                    relative_path=relative_path,
                    label=label,
                    group_id=group_id,
                )
            )

    labels = {record.label for record in records}
    if labels != {0, 1}:
        raise ValueError(
            "Training requires at least one labeled recording in both "
            "'fisico' and 'digital' classes"
        )
    return records


def split_by_source(
    records: list[AudioRecord],
    seed: int,
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    groups_by_label: dict[int, dict[str, list[AudioRecord]]] = {0: {}, 1: {}}
    for record in records:
        groups_by_label[record.label].setdefault(record.group_id, []).append(record)

    random_generator = random.Random(seed)
    training_records = []
    validation_records = []
    for label, groups in groups_by_label.items():
        group_ids = list(groups)
        if len(group_ids) < 2:
            label_name = "digital" if label == 1 else "fisico"
            raise ValueError(
                f"At least two distinct source_id groups are required for "
                f"'{label_name}' to create a source-independent train/validation split"
            )
        if len(group_ids) < 5:
            label_name = "digital" if label == 1 else "fisico"
            logging.warning(
                "Only %d distinct sources are available for '%s'; "
                "validation metrics will be highly uncertain",
                len(group_ids),
                label_name,
            )
        random_generator.shuffle(group_ids)
        validation_count = max(1, round(len(group_ids) * VALIDATION_FRACTION))
        validation_count = min(validation_count, len(group_ids) - 1)
        validation_ids = set(group_ids[:validation_count])
        for group_id, group_records in groups.items():
            target = (
                validation_records if group_id in validation_ids else training_records
            )
            target.extend(group_records)

    return training_records, validation_records


def prepare_examples(
    records: list[AudioRecord],
) -> tuple[np.ndarray, np.ndarray, list[tuple[AudioRecord, np.ndarray]]]:
    features = []
    labels = []
    record_features = []
    for record in records:
        windows = audio_to_log_mel_windows(record.path)
        features.append(windows)
        labels.extend([record.label] * len(windows))
        record_features.append((record, windows))

    return (
        np.concatenate(features, axis=0),
        np.asarray(labels, dtype=np.float32),
        record_features,
    )


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*TARGET_SHAPE, 1)),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="digital_recall"),
        ],
    )
    return model


def report_record_metrics(
    model: tf.keras.Model,
    record_features: list[tuple[AudioRecord, np.ndarray]],
) -> None:
    true_labels = []
    predicted_labels = []
    for record, features in record_features:
        window_scores = model.predict(features, verbose=0).reshape(-1)
        true_labels.append(record.label)
        predicted_labels.append(int(float(np.mean(window_scores)) >= 0.5))

    true_labels_array = np.asarray(true_labels)
    predicted_labels_array = np.asarray(predicted_labels)
    digital_count = int(np.sum(true_labels_array == 1))
    correctly_detected_digital = int(
        np.sum((true_labels_array == 1) & (predicted_labels_array == 1))
    )
    accuracy = float(np.mean(true_labels_array == predicted_labels_array))
    digital_recall = (
        correctly_detected_digital / digital_count if digital_count else 0.0
    )
    print(f"Validation recording accuracy (threshold 0.5): {accuracy:.3f}")
    print(f"Validation digital recall: {digital_recall:.3f}")


def train_model(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    output_path: Path = DEFAULT_MODEL_PATH,
    data_root: Path = PROJECT_ROOT,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
) -> Path:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    tf.keras.utils.set_random_seed(seed)
    records = load_records(metadata_path, data_root)
    training_records, validation_records = split_by_source(records, seed)
    training_features, training_labels, _ = prepare_examples(training_records)
    validation_features, validation_labels, validation_record_features = (
        prepare_examples(validation_records)
    )

    model = build_model()
    model.fit(
        training_features,
        training_labels,
        validation_data=(validation_features, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved trained model to {output_path}")
    report_record_metrics(model, validation_record_features)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the TensorFlow audio classifier from data/metadata.csv."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        train_model(
            metadata_path=args.metadata,
            output_path=args.output,
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Training failed: {error}") from error


if __name__ == "__main__":
    main()
