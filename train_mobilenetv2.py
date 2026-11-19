import time

import tensorflow as tf

from data_pipeline import (
    train_ds,
    val_ds,
    class_weights,
)

from model_mobilenetv2 import model


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
        "mobilenetv2_best.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("MULAI TRAINING MOBILENETV2")
print("=" * 60)

print("\nKonfigurasi:")
print("Model              : MobileNetV2")
print("Transfer Learning  : ImageNet")
print("Backbone Frozen    : Ya")
print("Epoch maksimum     :", EPOCHS)
print("Batch size         : 32")
print("Learning rate      : 0.001")
print("Class Weight       : Ya")

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

print(
    f"Waktu training : {training_time / 3600:.2f} jam"
)


# ============================================================
# SIMPAN MODEL TERAKHIR
# ============================================================

model.save("mobilenetv2_final.keras")

print("\nModel disimpan:")
print(" - mobilenetv2_best.keras")
print(" - mobilenetv2_final.keras")


# ============================================================
# HASIL TERBAIK VALIDASI
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
