import importlib

try:
    tf = importlib.import_module("tensorflow")
except ImportError as exc:
    raise RuntimeError(
        "Dependency missing: tensorflow. Install it with 'pip install tensorflow'."
    ) from exc

try:
    sklearn_metrics = importlib.import_module("sklearn.metrics")
    classification_report = sklearn_metrics.classification_report
    confusion_matrix = sklearn_metrics.confusion_matrix
except ImportError as exc:
    raise RuntimeError(
        "Dependency missing: scikit-learn. Install it with 'pip install scikit-learn'."
    ) from exc

from data_pipeline import (
    test_ds,
    CLASS_NAMES,
)


MODEL_PATH = "cnn_best.keras"


print("=" * 60)
print("EVALUASI CNN BASELINE")
print("=" * 60)

print(f"Memuat model: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")


# ==========================================
# PREDIKSI TEST SET
# ==========================================

print()
print("=" * 60)
print("PREDIKSI TEST SET")
print("=" * 60)

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0,
    )

    predicted_labels = tf.argmax(
        predictions,
        axis=1,
    ).numpy()

    y_true.extend(labels.numpy().tolist())
    y_pred.extend(predicted_labels)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
)

print(report)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_true,
    y_pred,
)

print()

print(
    "                 "
    + " ".join(
        f"{name[:8]:>10}"
        for name in CLASS_NAMES
    )
)

for class_name, row in zip(CLASS_NAMES, cm):

    print(
        f"{class_name[:10]:>12} "
        + " ".join(
            f"{value:>10}"
            for value in row
        )
    )


# ==========================================
# TEST ACCURACY
# ==========================================

accuracy = sum(true == pred for true, pred in zip(y_true, y_pred)) / len(y_true)

print()
print("=" * 60)
print("TEST PERFORMANCE")
print("=" * 60)

print(
    f"Test Accuracy : {accuracy:.4f}"
)

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)

print()
print("Evaluasi CNN selesai.")