import time

try:
    import tensorflow as tf  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required to run this training script. "
        "Please install the TensorFlow package in the active environment."
    ) from exc

from data_pipeline import (
    train_ds,
    val_ds,
    class_weights,
)

from model_cnn import model


# ============================================================
# KONFIGURASI TRAINING
# ============================================================

EPOCHS = 20

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "cnn_best.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("MULAI TRAINING CNN BASELINE")
print("=" * 60)

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

end_time = time.time()

training_time = end_time - start_time


# ============================================================
# HASIL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SELESAI")
print("=" * 60)

print(
    f"Waktu training : {training_time:.2f} detik"
)

print(
    f"Waktu training : {training_time / 60:.2f} menit"
)


# ============================================================
# SIMPAN MODEL TERAKHIR
# ============================================================

model.save("cnn_final.keras")

print("\nModel disimpan:")
print(" - cnn_best.keras")
print(" - cnn_final.keras")


# ============================================================
# HASIL TERBAIK
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)

print("\n" + "=" * 60)
print("PERFORMA TERBAIK VALIDASI")
print("=" * 60)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best Validation Loss     : "
    f"{best_val_loss:.4f}"
)