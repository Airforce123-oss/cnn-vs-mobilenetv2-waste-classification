import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from data_pipeline import test_ds


# ============================================================
# KONFIGURASI
# ============================================================

MODEL_PATH = "mobilenetv2_best.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("EVALUASI MOBILENETV2")
print("=" * 60)

print("\nMemuat model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")


# ============================================================
# PREDIKSI TEST SET
# ============================================================

print("\nMelakukan prediksi pada TEST SET...")

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0,
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1,
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)


# ============================================================
# MACRO F1
# ============================================================

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
)

print(report)


# ============================================================
# HASIL UTAMA
# ============================================================

print("\n" + "=" * 60)
print("HASIL UTAMA MOBILENETV2")
print("=" * 60)

print(
    f"Test Accuracy : {accuracy:.4f}"
)

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1      : {macro_f1:.4f}"
)

print(
    f"Macro F1      : {macro_f1 * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nClass order:")
print(CLASS_NAMES)

print("\nMatrix:")

print(cm)


# ============================================================
# SELESAI
# ============================================================

print("\n" + "=" * 60)
print("EVALUASI SELESAI")
print("=" * 60)
