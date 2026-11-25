from pathlib import Path

import tensorflow as tf


# ============================================================
# KONFIGURASI
# ============================================================

IMG_SIZE = (224, 224)
NUM_CLASSES = 4

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "mobilenet_v2_weights.h5"


# ============================================================
# MobileNetV2 ImageNet
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None,
)

# Memuat bobot ImageNet dari file lokal
base_model.load_weights(WEIGHTS_PATH)

# Bekukan backbone untuk feature extraction
base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="input_image",
)

# Data pipeline berada pada rentang 0-1.
# MobileNetV2 membutuhkan rentang -1 sampai 1.
x = tf.keras.layers.Rescaling(
    scale=2.0,
    offset=-1.0,
    name="mobilenet_preprocessing",
)(inputs)

x = base_model(
    x,
    training=False,
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu",
)(x)

x = tf.keras.layers.Dropout(0.5)(x)

# Output 4 kelas
outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="MobileNetV2_4Class",
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
