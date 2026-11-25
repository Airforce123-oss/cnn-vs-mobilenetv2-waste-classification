import tensorflow as tf


# ============================================================
# KONFIGURASI
# ============================================================

IMG_SIZE = (224, 224)
NUM_CLASSES = 4


# ============================================================
# CNN BASELINE 4 KELAS
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(224, 224, 3)
        ),

        # Block 1
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
        ),
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 2
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
        ),
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 3
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
        ),
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 4
        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same",
        ),
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # Classifier
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu",
        ),

        tf.keras.layers.Dropout(
            0.5
        ),

        # Output 4 kelas
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax",
        ),
    ],
    name="CNN_Baseline_4Class",
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# TAMPILKAN ARSITEKTUR
# ============================================================

model.summary()
