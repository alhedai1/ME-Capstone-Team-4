import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

from capstone_robot.vision.bell_trigger_features import FEATURE_NAMES, BellFeatureConfig, RoiConfig, config_to_dict


def load_feature_csv(path):
    with Path(path).open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    missing = [name for name in ["label"] + FEATURE_NAMES if name not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.array([int(row["label"]) for row in rows], dtype=np.int32)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Train a shallow decision tree for the close-bell strike trigger.")
    parser.add_argument("--features", default=DEFAULT_DATA_DIR / "bell_features.csv", help="CSV from extract_bell_features.py")
    parser.add_argument("--out", default="models/bell_trigger.joblib", help="Model output path")
    parser.add_argument("--max-depth", type=int, default=3, help="Decision tree max depth, capped at 3")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out test fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-strike-weight",
        type=float,
        default=2.0,
        help="Class weight for no_strike; raise this to reduce false positives",
    )
    args = parser.parse_args()

    import joblib
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_text

    x, y = load_feature_csv(args.features)
    metadata_path = Path(args.features).with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    max_depth = min(3, max(1, args.max_depth))

    class_values, class_counts = np.unique(y, return_counts=True)
    stratify = y if len(class_values) == 2 and class_counts.min() >= 2 else None
    if len(y) >= 6 and stratify is not None:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )
    else:
        x_train, x_test, y_train, y_test = x, x, y, y
        print("Dataset is small; reporting metrics on the training set.")

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=args.random_state,
        class_weight={0: args.no_strike_weight, 1: 1.0},
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("Confusion matrix, rows=true [no_strike, strike], cols=pred [no_strike, strike]:")
    print(matrix)
    print(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print()
    print("Rules:")
    print(export_text(model, feature_names=FEATURE_NAMES))
    print("Feature importances:")
    for name, importance in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda item: item[1], reverse=True):
        print(f"{name}: {importance:.4f}")

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "roi": metadata.get("roi", config_to_dict(RoiConfig())),
        "feature_config": metadata.get("feature_config", config_to_dict(BellFeatureConfig())),
        "positive_label": 1,
        "negative_label": 0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
