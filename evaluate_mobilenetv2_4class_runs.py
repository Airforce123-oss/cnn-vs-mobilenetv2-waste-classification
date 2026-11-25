import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_pipeline_4class import test_ds, CLASS_NAMES


MODEL_FILES = [
    "mobilenetv2_4class_run1_best.keras",
    "mobilenetv2_4class_run2_best.keras",
    "mobilenetv2_4class_run3_best.keras",
]

RUN_NAMES = [
    "Run 1 (seed=42)",
    "Run 2 (seed=123)",
    "Run 3 (seed=456)",
]


def collect_true_labels(dataset):
    y_true = []

    for _, labels in dataset:
        y_true.extend(labels.numpy().tolist())

    return np.asarray(y_true, dtype=np.int64)


def evaluate_model(model_path, y_true):
    print("\n" + "=" * 72)
    print(f"MODEL: {model_path}")
    print("=" * 72)

    model = tf.keras.models.load_model(model_path, compile=False)

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Label/prediction length mismatch: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
    )

    print("\nTest Accuracy  :", f"{accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(
        "Macro Precision:",
        f"{macro_precision:.4f} ({macro_precision * 100:.2f}%)",
    )
    print(
        "Macro Recall   :",
        f"{macro_recall:.4f} ({macro_recall * 100:.2f}%)",
    )
    print(
        "Macro F1       :",
        f"{macro_f1:.4f} ({macro_f1 * 100:.2f}%)",
    )

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def main():
    print("=" * 72)
    print("MOBILENETV2 4-CLASS TEST-SET EVALUATION")
    print("=" * 72)

    print("\nClass order:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {i}: {class_name}")

    y_true = collect_true_labels(test_ds)

    print(f"\nTest samples: {len(y_true)}")

    if len(y_true) != 580:
        raise RuntimeError(
            f"Expected 580 test images, but found {len(y_true)}."
        )

    results = []

    for run_name, model_file in zip(RUN_NAMES, MODEL_FILES):
        result = evaluate_model(model_file, y_true)
        result["run"] = run_name
        result["model"] = model_file
        results.append(result)

    accuracies = np.array(
        [r["accuracy"] for r in results],
        dtype=np.float64,
    )

    macro_precisions = np.array(
        [r["macro_precision"] for r in results],
        dtype=np.float64,
    )

    macro_recalls = np.array(
        [r["macro_recall"] for r in results],
        dtype=np.float64,
    )

    macro_f1s = np.array(
        [r["macro_f1"] for r in results],
        dtype=np.float64,
    )

    accuracy_mean = np.mean(accuracies)
    accuracy_sd = np.std(accuracies, ddof=0)

    macro_precision_mean = np.mean(macro_precisions)
    macro_precision_sd = np.std(macro_precisions, ddof=0)

    macro_recall_mean = np.mean(macro_recalls)
    macro_recall_sd = np.std(macro_recalls, ddof=0)

    macro_f1_mean = np.mean(macro_f1s)
    macro_f1_sd = np.std(macro_f1s, ddof=0)

    print("\n" + "=" * 72)
    print("MOBILENETV2 4-CLASS SUMMARY")
    print("=" * 72)

    print("\nPer-run results:")
    print("-" * 72)

    print(
        f"{'Run':<18}"
        f"{'Accuracy':>12}"
        f"{'Macro P':>12}"
        f"{'Macro R':>12}"
        f"{'Macro F1':>12}"
    )

    print("-" * 72)

    for r in results:
        print(
            f"{r['run']:<18}"
            f"{r['accuracy']:>12.4f}"
            f"{r['macro_precision']:>12.4f}"
            f"{r['macro_recall']:>12.4f}"
            f"{r['macro_f1']:>12.4f}"
        )

    print("-" * 72)

    print(
        f"{'Mean ± SD':<18}"
        f"{accuracy_mean:>9.4f} ± {accuracy_sd:.4f}"
        f"{macro_precision_mean:>9.4f} ± {macro_precision_sd:.4f}"
        f"{macro_recall_mean:>9.4f} ± {macro_recall_sd:.4f}"
        f"{macro_f1_mean:>9.4f} ± {macro_f1_sd:.4f}"
    )

    print("\nRequested headline metrics:")
    print(f"Accuracy = {accuracy_mean:.4f} ± {accuracy_sd:.4f}")
    print(f"Macro F1 = {macro_f1_mean:.4f} ± {macro_f1_sd:.4f}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
