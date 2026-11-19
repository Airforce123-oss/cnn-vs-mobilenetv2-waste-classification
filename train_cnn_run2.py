import os

# ============================================================
# SET SEED SEBELUM IMPORT MODEL
# ============================================================

SEED = 123

os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf

tf.keras.utils.set_random_seed(SEED)

# ============================================================
# IMPORT DATA DAN MODEL
# ============================================================

from data_pipeline import (
    train_ds,
    val_ds,
    class_weights,
)

from model_cnn import model


# ============================================================
# KONFIGURASI
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
        "cnn_run2_best.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("CNN RUN 2")
print("=" * 60)

print("Seed               :", SEED)
print("Epoch maksimum     :", EPOCHS)
print("Batch size         : 32")
print("Learning rate      : 0.001")
print("Class Weight       : Ya")

start_time = tf.timestamp()


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)


end_time = tf.timestamp()

training_time = float(end_time - start_time)


# ============================================================
# SIMPAN MODEL
# ============================================================

model.save("cnn_run2_final.keras")


# ============================================================
# HASIL
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)


print("\n" + "=" * 60)
print("CNN RUN 2 SELESAI")
print("=" * 60)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best Validation Loss     : "
    f"{best_val_loss:.4f}"
)

print(
    f"Waktu Training          : "
    f"{training_time / 60:.2f} menit"
)

print("\nModel:")
print(" - cnn_run2_best.keras")
print(" - cnn_run2_final.keras")
