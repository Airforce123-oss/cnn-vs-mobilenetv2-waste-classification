import csv
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "validation"
TEST_DIR = BASE_DIR / "test"

MODEL_PAIRS = {
    "Run 1": {
        "cnn": BASE_DIR / "cnn_best.keras",
        "mobilenet": BASE_DIR / "mobilenetv2_best.keras",
    },
    "Run 2": {
        "cnn": BASE_DIR / "cnn_run2_best.keras",
        "mobilenet": BASE_DIR / "mobilenetv2_run2_best.keras",
    },
    "Run 3": {
        "cnn": BASE_DIR / "cnn_run3_best.keras",
        "mobilenet": BASE_DIR / "mobilenetv2_run3_best.keras",
    },
}

# ============================================================
# DATASET TANPA AUGMENTATION
# ============================================================

def load_clean_dataset(directory):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    ds = ds.map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds.prefetch(tf.data.AUTOTUNE)


print("=" * 80)
print("MEMUAT DATASET")
print("=" * 80)

train_clean_ds = load_clean_dataset(TRAIN_DIR)
val_clean_ds = load_clean_dataset(VAL_DIR)
test_clean_ds = load_clean_dataset(TEST_DIR)

print("Dataset siap.\n")


# ============================================================
# FUNGSI EVALUASI GENERALIZATION / OVERFITTING
# ============================================================

def evaluate_generalization(model, model_name):
    print("\n" + "=" * 80)
    print(f"GENERALIZATION ANALYSIS - {model_name}")
    print("=" * 80)

    train_loss, train_acc = model.evaluate(
        train_clean_ds,
        verbose=0,
    )

    val_loss, val_acc = model.evaluate(
        val_clean_ds,
        verbose=0,
    )

    test_loss, test_acc = model.evaluate(
        test_clean_ds,
        verbose=0,
    )

    train_val_acc_gap = train_acc - val_acc
    train_val_loss_gap = val_loss - train_loss
    val_test_acc_gap = val_acc - test_acc

    print(
        f"\nTrain Accuracy      : "
        f"{train_acc:.4f} ({train_acc * 100:.2f}%)"
    )

    print(
        f"Validation Accuracy : "
        f"{val_acc:.4f} ({val_acc * 100:.2f}%)"
    )

    print(
        f"Test Accuracy       : "
        f"{test_acc:.4f} ({test_acc * 100:.2f}%)"
    )

    print(f"\nTrain Loss          : {train_loss:.4f}")
    print(f"Validation Loss     : {val_loss:.4f}")
    print(f"Test Loss           : {test_loss:.4f}")

    print("\nGeneralization Gap:")

    print(
        f"Train - Validation Accuracy : "
        f"{train_val_acc_gap:.4f} "
        f"({train_val_acc_gap * 100:.2f} pp)"
    )

    print(
        f"Validation - Test Accuracy  : "
        f"{val_test_acc_gap:.4f} "
        f"({val_test_acc_gap * 100:.2f} pp)"
    )

    print(
        f"Validation - Train Loss     : "
        f"{train_val_loss_gap:.4f}"
    )

    return {
        "model": model_name,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "test_loss": test_loss,
        "train_val_accuracy_gap": train_val_acc_gap,
        "val_test_accuracy_gap": val_test_acc_gap,
        "train_val_loss_gap": train_val_loss_gap,
    }


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def get_predictions(model, dataset):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    return np.array(y_true), np.array(y_pred)


# ============================================================
# EXACT MCNEMAR TEST
# ============================================================

def exact_mcnemar_pvalue(b, c):
    """
    Exact two-sided McNemar test.

    b = CNN benar, MobileNetV2 salah
    c = CNN salah, MobileNetV2 benar
    """

    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)
    probability = 0.0

    for i in range(k + 1):
        probability += (
            math.comb(n, i) /
            (2 ** n)
        )

    p_value = min(1.0, 2.0 * probability)

    return p_value


def run_mcnemar(run_name, cnn_model_path, mobilenet_model_path):
    print("\n" + "=" * 80)
    print(f"MCNEMAR TEST - {run_name}")
    print("=" * 80)

    # compile=True diperlukan karena model digunakan untuk evaluate()
    cnn_model = tf.keras.models.load_model(
        cnn_model_path,
        compile=True,
    )

    mobilenet_model = tf.keras.models.load_model(
        mobilenet_model_path,
        compile=True,
    )

    y_true_cnn, y_pred_cnn = get_predictions(
        cnn_model,
        test_clean_ds,
    )

    y_true_mobile, y_pred_mobile = get_predictions(
        mobilenet_model,
        test_clean_ds,
    )

    # Sanity check:
    # kedua model harus dievaluasi pada data test yang sama
    if not np.array_equal(y_true_cnn, y_true_mobile):
        raise ValueError(
            "Urutan label test tidak sama antara CNN dan MobileNetV2."
        )

    correct_cnn = y_pred_cnn == y_true_cnn
    correct_mobile = y_pred_mobile == y_true_mobile

    # Discordant pairs
    b = int(np.sum(correct_cnn & ~correct_mobile))
    c = int(np.sum(~correct_cnn & correct_mobile))

    # Concordant pairs
    both_correct = int(
        np.sum(correct_cnn & correct_mobile)
    )

    both_wrong = int(
        np.sum(~correct_cnn & ~correct_mobile)
    )

    total = len(y_true_cnn)

    cnn_accuracy = np.mean(correct_cnn)
    mobile_accuracy = np.mean(correct_mobile)

    p_value = exact_mcnemar_pvalue(b, c)

    print(f"\nJumlah test samples : {total}")

    print(
        f"\nCNN Accuracy        : "
        f"{cnn_accuracy:.4f} ({cnn_accuracy * 100:.2f}%)"
    )

    print(
        f"MobileNetV2 Accuracy: "
        f"{mobile_accuracy:.4f} "
        f"({mobile_accuracy * 100:.2f}%)"
    )

    print("\nPaired Outcomes:")
    print(
        f"Both models correct                 : "
        f"{both_correct}"
    )

    print(
        f"Both models wrong                   : "
        f"{both_wrong}"
    )

    print(
        f"CNN correct, MobileNetV2 wrong      : "
        f"{b}"
    )

    print(
        f"CNN wrong, MobileNetV2 correct      : "
        f"{c}"
    )

    print(
        f"\nMcNemar exact p-value : "
        f"{p_value:.3e}"
    )

    if p_value < 0.05:
        interpretation = (
            "SIGNIFIKAN pada alpha 0.05: "
            "terdapat perbedaan pola kesalahan yang signifikan."
        )
    else:
        interpretation = (
            "TIDAK SIGNIFIKAN pada alpha 0.05: "
            "belum terdapat bukti statistik yang cukup "
            "untuk menyatakan perbedaan pola kesalahan."
        )

    print(interpretation)

    return {
        "run": run_name,
        "cnn_accuracy": cnn_accuracy,
        "mobilenet_accuracy": mobile_accuracy,
        "b_cnn_correct_mobile_wrong": b,
        "c_cnn_wrong_mobile_correct": c,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "p_value": p_value,
        "significant_alpha_0_05": p_value < 0.05,
    }


# ============================================================
# JALANKAN GENERALIZATION TEST
# ============================================================

generalization_results = []

for run_name, paths in MODEL_PAIRS.items():

    print("\n" + "#" * 80)
    print(f"{run_name}")
    print("#" * 80)

    # compile=True diperlukan untuk model.evaluate()
    cnn_model = tf.keras.models.load_model(
        paths["cnn"],
        compile=True,
    )

    mobilenet_model = tf.keras.models.load_model(
        paths["mobilenet"],
        compile=True,
    )

    cnn_result = evaluate_generalization(
        cnn_model,
        f"CNN {run_name}",
    )

    mobilenet_result = evaluate_generalization(
        mobilenet_model,
        f"MobileNetV2 {run_name}",
    )

    generalization_results.extend(
        [cnn_result, mobilenet_result]
    )


# ============================================================
# JALANKAN MCNEMAR TEST
# ============================================================

mcnemar_results = []

for run_name, paths in MODEL_PAIRS.items():

    result = run_mcnemar(
        run_name,
        paths["cnn"],
        paths["mobilenet"],
    )

    mcnemar_results.append(result)


# ============================================================
# SIMPAN HASIL GENERALIZATION
# ============================================================

generalization_csv = BASE_DIR / "generalization_results.csv"

with open(
    generalization_csv,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=generalization_results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(generalization_results)


# ============================================================
# SIMPAN HASIL MCNEMAR
# ============================================================

mcnemar_csv = BASE_DIR / "mcnemar_results.csv"

with open(
    mcnemar_csv,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=mcnemar_results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(mcnemar_results)


# ============================================================
# RINGKASAN AKHIR
# ============================================================

print("\n" + "=" * 80)
print("RINGKASAN PENGUJIAN TAMBAHAN")
print("=" * 80)

print("\nGENERALIZATION / OVERFITTING:")

for result in generalization_results:
    print(
        f"{result['model']:<25} | "
        f"Train Acc = {result['train_accuracy']:.4f} | "
        f"Val Acc = {result['validation_accuracy']:.4f} | "
        f"Test Acc = {result['test_accuracy']:.4f} | "
        f"Train-Val Gap = "
        f"{result['train_val_accuracy_gap']:.4f}"
    )


print("\nMCNEMAR TEST:")

for result in mcnemar_results:
    print(
        f"{result['run']:<8} | "
        f"CNN = {result['cnn_accuracy']:.4f} | "
        f"MobileNetV2 = {result['mobilenet_accuracy']:.4f} | "
        f"b = {result['b_cnn_correct_mobile_wrong']} | "
        f"c = {result['c_cnn_wrong_mobile_correct']} | "
        f"p = {result['p_value']:.6f}"
    )


print("\nFile hasil:")
print(f"- {generalization_csv}")
print(f"- {mcnemar_csv}")

print("\nSelesai.")