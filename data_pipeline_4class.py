from pathlib import Path

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = [
    "cardboard",
    "glass",
    "paper",
    "plastic",
]

print("Memuat dataset 4 kelas...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
        ),
    ],
    name="data_augmentation",
)


def preprocess_train(images, labels):
    images = tf.cast(images, tf.float32)
    images = data_augmentation(images, training=True)
    images = images / 255.0
    return images, labels


def preprocess_eval(images, labels):
    images = tf.cast(images, tf.float32)
    images = images / 255.0
    return images, labels


train_ds = train_ds.map(
    preprocess_train,
    num_parallel_calls=tf.data.AUTOTUNE,
)

val_ds = val_ds.map(
    preprocess_eval,
    num_parallel_calls=tf.data.AUTOTUNE,
)

test_ds = test_ds.map(
    preprocess_eval,
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# Class weights dihitung dari 4 kelas saja.
# Semua kelas yang dipakai eksperimen harus memiliki folder JPG/PNG/JPEG.
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

class_counts = {}

for class_index, class_name in enumerate(CLASS_NAMES):
    class_directory = TRAIN_DIR / class_name

    count = sum(
        len(list(class_directory.glob(pattern)))
        for pattern in IMAGE_EXTENSIONS
    )

    class_counts[class_index] = count


total_samples = sum(class_counts.values())
num_classes = len(CLASS_NAMES)

class_weights = {
    class_index: total_samples / (num_classes * count)
    for class_index, count in class_counts.items()
}


print()
print("========================================")
print("4-CLASS DATASET INFORMATION")
print("========================================")

print(f"Dataset directory : {DATA_DIR}")
print(f"Image size        : {IMG_SIZE}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Number of classes : {num_classes}")
print(f"Class names       : {CLASS_NAMES}")
print(f"Random seed       : {SEED}")

print()
print("Training class distribution:")

for class_index, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:10s}: "
        f"{class_counts[class_index]}"
    )

print()
print("Class weights:")

for class_index, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:10s}: "
        f"{class_weights[class_index]:.4f}"
    )


# Sanity check
images, labels = next(iter(train_ds))

print()
print("========================================")
print("SANITY CHECK")
print("========================================")

print("Image shape :", images.shape)
print("Label shape :", labels.shape)
print(
    "Pixel min   :",
    float(tf.reduce_min(images)),
)
print(
    "Pixel max   :",
    float(tf.reduce_max(images)),
)
print("Pixel dtype :", images.dtype)

print()
print("Data pipeline 4 kelas berhasil.")
