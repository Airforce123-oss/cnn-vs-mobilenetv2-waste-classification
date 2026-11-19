import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_pipeline import test_ds, CLASS_NAMES


# ============================================================
# DAFTAR MODEL
# ============================================================

models = {
    "CNN Run 1": "cnn_best.keras",
    "CNN Run 2": "cnn_run2_best.keras",
    "CNN Run 3": "cnn_run3_best.keras",
}


# ============================================================
# FUNGSI EVALUASI
# ============================================================

def evaluate_model(run_name, model_path):

    print("\n" + "=" * 70)
    print(run_name)
    print("=" * 70)

    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print(
        f"\nTest Accuracy : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro F1      : {macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(cm)

    return accuracy, macro_f1


# ============================================================
# EVALUASI SEMUA RUN
# ============================================================

results = {}

for run_name, model_path in models.items():

    accuracy, macro_f1 = evaluate_model(
        run_name,
        model_path
    )

    results[run_name] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


# ============================================================
# MEAN DAN STANDARD DEVIATION
# ============================================================

accuracies = [
    result["accuracy"]
    for result in results.values()
]

macro_f1_scores = [
    result["macro_f1"]
    for result in results.values()
]


accuracy_mean = np.mean(accuracies)
accuracy_std = np.std(
    accuracies,
    ddof=1
)

macro_f1_mean = np.mean(
    macro_f1_scores
)

macro_f1_std = np.std(
    macro_f1_scores,
    ddof=1
)


# ============================================================
# HASIL AKHIR
# ============================================================

print("\n" + "=" * 70)
print("HASIL AKHIR CNN 3 RUN")
print("=" * 70)

for run_name, result in results.items():

    print(
        f"{run_name:<12} | "
        f"Accuracy = {result['accuracy']:.4f} | "
        f"Macro F1 = {result['macro_f1']:.4f}"
    )

print("\n" + "-" * 70)

print(
    f"Accuracy Mean ± SD : "
    f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
)

print(
    f"Macro F1 Mean ± SD : "
    f"{macro_f1_mean:.4f} ± {macro_f1_std:.4f}"
)
