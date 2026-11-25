import os

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf

tf.keras.utils.set_random_seed(SEED)

from data_pipeline_4class import (
    train_ds,
    val_ds,
    class_weights,
)

from model_cnn_4class import model


EPOCHS = 20

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
        "cnn_4class_run1_best.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]


print("\n" + "=" * 60)
print("CNN 4-CLASS RUN 1")
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

model.save("cnn_4class_run1_final.keras")

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)

print("\n" + "=" * 60)
print("CNN 4-CLASS RUN 1 SELESAI")
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
print(" - cnn_4class_run1_best.keras")
print(" - cnn_4class_run1_final.keras")
