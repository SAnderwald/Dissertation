#!/usr/bin/env python3
"""Evaluate a single Conditional Variational Autoencoder (CVAE) for anomaly detection.

This script evaluates a pre-trained single CVAE model on specified data splits (50/50, 60/40, 90/10)
using mixed precision on a GPU. It processes test data from the UNCC CHAD dataset, computes performance
metrics (accuracy, precision, recall, F1-score, AUC-ROC, EER, MCC) across multiple thresholds, and
generates ROC and Precision-Recall curves as PNG files. The script is designed for a dissertation
project assessing anomaly detection performance on imbalanced video data.

Attributes:
    CONFIG (dict): Configuration dictionary containing hyperparameters and file paths, including
                   split-specific settings (e.g., NUM_SAMPLES, CHECKPOINT_DIR).
    SPLIT (str): The data split to evaluate on (e.g., '5050', '6040', '9010'), parsed from command-line.
    SPLIT_DIR (str): Directory mapping for the split (e.g., '5050' -> '5050', '6040' -> '60_40').

Args:
    --split (str): Required argument specifying the data split to evaluate on. Choices are '5050',
                   '6040', or '9010'.

Returns:
    None: The script saves evaluation metrics to a text file (e.g., evaluation_metrics_5050_single.txt),
          generates ROC and PR curves as PNG files (e.g., roc_curve_5050.png, pr_curve_5050.png),
          and logs the process to evaluate_single_cvae_{SPLIT}.log.

Raises:
    ValueError: If no valid test data is found in the test_split_subset.txt file.
    FileNotFoundError: If the checkpoint or test file is inaccessible.
    Exception: For other unforeseen errors during evaluation, metric computation, or file operations.

Example:
    python evaluate_single_cvae.py --split 5050

Notes:
    - Requires a GPU-enabled environment (e.g., tf_gpu Conda environment with TensorFlow 2.10, Python 3.10,
      and scikit-learn installed via `pip install scikit-learn`).
    - Assumes pre-trained model checkpoints in CHECKPOINT_DIR/epoch_3/cvae_0 format.
    - Test data should be preprocessed and listed in TEST_FILE path.
    - GPU memory is managed with tf.keras.backend.clear_session(), but no explicit memory limit is set.
    - Metrics are computed for thresholds [0.1, 0.3, 0.5, 0.7, 0.9], with 0.5 as the default evaluation point.

Author: Scott Anderwald
Date: 2025-07-03
"""

import tensorflow as tf
import numpy as np
import os
import logging
import sys
import subprocess
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, matthews_corrcoef
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Evaluate single CVAE for a specified split.")
parser.add_argument('--split', type=str, required=True, choices=['5050', '6040', '9010'], help="Specify the split: 5050, 6040, or 9010")
args = parser.parse_args()

split_dir_map = {'5050': '5050', '6040': '60_40', '9010': '90_10'}
SPLIT = args.split
SPLIT_DIR = split_dir_map[SPLIT]

CONFIG = {
    "MAIN_IMAGE_DIR": "/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/",
    "TEST_FILE": f"/home/sanderwald/Projects/dissertationProject/data/Splits/{SPLIT_DIR}/test_split_subset.txt",
    "CHECKPOINT_DIR": f"/home/sanderwald/Projects/dissertationProject/outputs_{SPLIT}_single_cvae_20250524",
    "OUTPUT_DIR": "/home/sanderwald/Projects/dissertationProject/",
    "IMG_SIZE": (360, 203),
    "BATCH_SIZE": 2,
    "LATENT_DIM": 64,
    "NUM_CVAES": 1,  # Single CVAE
    "NUM_SAMPLES": {"5050": 20600, "6040": 24720, "9010": 37080}[SPLIT],
    "FOCAL_GAMMA": 3.0,
    "FOCAL_ALPHA": 0.9,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"/home/sanderwald/Projects/dissertationProject/evaluate_single_cvae_{SPLIT}.log"),
        logging.StreamHandler()
    ]
)

problematic_images = set([
    '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/3_011_0/frame_03997.jpg',
    '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/3_011_0/frame_04022.jpg',
    '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/1_047_0/frame_02802.jpg',
])

tf.keras.backend.clear_session()

def load_test_data(file_path):
    logging.info(f"Loading test data from {file_path}")
    paths = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            path = line.strip()
            if path and path not in problematic_images and os.path.isfile(path):
                paths.append(path)
                full_dir = os.path.dirname(path)
                dir_name = os.path.basename(full_dir)
                label = 1 if '_1' in dir_name else 0
                labels.append(label)
            else:
                logging.warning(f"Skipping path {path}: Not a file or in problematic_images")
    logging.info(f"Loaded {len(paths)} test image paths")
    return paths, labels

def parse_function(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, CONFIG["IMG_SIZE"], method=tf.image.ResizeMethod.BILINEAR)
    image = image / 255.0
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, label

def create_test_dataset(paths, labels):
    logging.info("Creating test dataset")
    paths = np.array(paths, dtype=str)
    labels = np.array(labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(CONFIG["BATCH_SIZE"], drop_remainder=True)
    total_batches = len(paths) // CONFIG["BATCH_SIZE"]
    dataset = dataset.take(total_batches)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    logging.info(f"Dataset created successfully with {total_batches} batches")
    return dataset, total_batches

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, model_id=0):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.model_id = model_id
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 3)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu' if model_id == 0 else 'elu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu' if model_id == 0 else 'elu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu' if model_id == 0 else 'elu'),
        ])
        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(90 * 51 * 64, activation='relu' if model_id == 0 else 'elu'),
            tf.keras.layers.Reshape((90, 51, 64)),
            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu' if model_id == 0 else 'elu'),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu' if model_id == 0 else 'elu'),
            tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=1, padding='same', activation='relu' if model_id == 0 else 'elu'),
            tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=1, padding='same', activation='sigmoid'),
            tf.keras.layers.Cropping2D(((0, 1), (0, 1))),
            tf.keras.layers.Resizing(CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1]),
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'),
        ])

    def encode(self, x):
        h = self.encoder(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean), dtype=mean.dtype)
        return mean + tf.exp(0.5 * logvar) * eps

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=False):
        x, label = inputs
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        classification = self.classifier(z_mean)
        return reconstructed, z_mean, z_log_var, classification

def focal_loss(gamma=2.0, alpha=0.75):
    @tf.function(reduce_retracing=True)
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
        return tf.cast(tf.reduce_mean(loss), tf.float16)
    return focal_loss_fn

def compute_eer(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.argmin(np.abs(fnr - fpr))]
    eer = fpr[np.argmin(np.abs(fnr - fpr))]
    return eer

def evaluate(dataset, total_batches):
    logging.info("Starting evaluation")
    predictions = []
    true_labels = []
    recon_losses = []
    focal_losses = []
    models = []
    for i in range(CONFIG["NUM_CVAES"]):
        model_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "epoch_3", f"cvae_{i}")
        model = tf.keras.models.load_model(model_path, custom_objects={'CVAE': CVAE})
        logging.info(f"Loaded model {i} from {model_path} for evaluation")
        models.append(model)
        tf.keras.backend.clear_session()
    for batch_idx, (x_batch, y_batch) in enumerate(dataset):
        logging.info(f"Processing batch {batch_idx + 1}/{total_batches}")
        batch_preds = []
        batch_recon_losses = []
        batch_focal_losses = []
        for model in models:
            reconstructed, z_mean, _, classification = model((x_batch, None), training=False)
            batch_recon_loss = tf.reduce_mean(tf.keras.losses.mse(x_batch, reconstructed)).numpy()
            batch_focal_loss = focal_loss(gamma=CONFIG["FOCAL_GAMMA"], alpha=CONFIG["FOCAL_ALPHA"])(y_batch, classification).numpy()
            batch_recon_losses.append(batch_recon_loss)
            batch_focal_losses.append(batch_focal_loss)
            batch_preds.append(classification.numpy())
        batch_preds = np.mean(batch_preds, axis=0)
        batch_recon_loss = np.mean(batch_recon_losses)
        batch_focal_loss = np.mean(batch_focal_losses)
        predictions.extend(batch_preds.flatten())
        true_labels.extend(y_batch.numpy().flatten())
        recon_losses.append(batch_recon_loss)
        focal_losses.append(batch_focal_loss)
        tf.keras.backend.clear_session()
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    avg_recon_loss = np.mean(recon_losses)
    avg_focal_loss = np.mean(focal_losses)
    logging.info(f"Collected {len(predictions)} predictions and {len(true_labels)} true labels")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_metrics = {}
    for threshold in thresholds:
        predicted_labels = (predictions > threshold).astype(int)
        accuracy = np.mean(predicted_labels == true_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        auc_roc = roc_auc_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0.0
        eer = compute_eer(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predicted_labels)
        all_metrics[threshold] = (accuracy, precision, recall, f1, auc_roc, eer, mcc)
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)
    accuracy = np.mean(predicted_labels == true_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    auc_roc = roc_auc_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0.0
    eer = compute_eer(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    return (accuracy, precision, recall, f1, auc_roc, eer, mcc, avg_recon_loss, avg_focal_loss), all_metrics, true_labels, predictions

def generate_graphs(true_labels, predictions):
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='#1f77b4')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    roc_file = os.path.join(CONFIG["OUTPUT_DIR"], f"roc_curve_{SPLIT}.png")
    plt.savefig(roc_file)
    plt.close()
    logging.info(f"Saved ROC curve to {roc_file}")
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='#ff7f0e')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    pr_file = os.path.join(CONFIG["OUTPUT_DIR"], f"pr_curve_{SPLIT}.png")
    plt.savefig(pr_file)
    plt.close()
    logging.info(f"Saved Precision-Recall curve to {pr_file}")

def main():
    logging.info("Starting evaluation")
    try:
        paths, true_labels = load_test_data(CONFIG["TEST_FILE"])
        if not paths:
            raise ValueError("No valid test data found")
        test_dataset, total_batches = create_test_dataset(paths, true_labels)
        logging.info("Starting evaluation with multiple thresholds")
        (accuracy, precision, recall, f1, auc_roc, eer, mcc, avg_recon_loss, avg_focal_loss), all_metrics, true_labels, predictions = evaluate(test_dataset, total_batches)
        logging.info("Logging metrics for all thresholds")
        for threshold, (acc, prec, rec, f1, auc_roc, eer_val, mcc_val) in all_metrics.items():
            logging.info(f"\nMetrics for threshold {threshold}:")
            logging.info(f"Accuracy: {acc:.4f}")
            logging.info(f"Precision: {prec:.4f}")
            logging.info(f"Recall: {rec:.4f}")
            logging.info(f"F1-Score: {f1:.4f}")
            logging.info(f"AUC-ROC: {auc_roc:.4f}")
            logging.info(f"EER: {eer_val:.4f}")
            logging.info(f"MCC: {mcc_val:.4f}")
        logging.info(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
        logging.info(f"Average Focal Loss: {avg_focal_loss:.4f}")
        metrics_file = os.path.join(CONFIG["OUTPUT_DIR"], f"evaluation_metrics_{SPLIT}_single.txt")
        logging.info(f"Saving metrics to {metrics_file}")
        with open(metrics_file, 'w') as f:
            for threshold, (acc, prec, rec, f1, auc_roc, eer_val, mcc_val) in all_metrics.items():
                f.write(f"\nMetrics for threshold {threshold}:\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Precision: {prec:.4f}\n")
                f.write(f"Recall: {rec:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
                f.write(f"AUC-ROC: {auc_roc:.4f}\n")
                f.write(f"EER: {eer_val:.4f}\n")
                f.write(f"MCC: {mcc_val:.4f}\n")
            f.write(f"Average Reconstruction Loss: {avg_recon_loss:.4f}\n")
            f.write(f"Average Focal Loss: {avg_focal_loss:.4f}\n")
        logging.info("Generating binary classification graphs")
        generate_graphs(np.array(true_labels), np.array(predictions))
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise
    finally:
        tf.keras.backend.clear_session()
        logging.info("Exiting process to release GPU memory")
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            logging.info(f"GPU memory usage after run:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to log GPU memory usage after run: {str(e)}")
    sys.exit(0)

if __name__ == "__main__":
    main()
