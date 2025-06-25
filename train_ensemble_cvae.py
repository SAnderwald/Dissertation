#!/usr/bin/env python3
"""Train an ensemble of Conditional Variational Autoencoders (CVAEs) for anomaly detection.

This script trains an ensemble of CVAEs on specified data splits (5050, 6040, 9010) using
mixed precision training on a GPU. It supports reconstruction and classification tasks,
saving model checkpoints and attempting to generate loss curve plots. The script was
developed for a dissertation project to evaluate performance across different train-test
splits, with baseline AUC-ROC metrics (e.g., 0.7397 for 5050, 0.7355 for 6040).

Attributes:
    CONFIG (dict): Configuration dictionary containing hyperparameters and file paths.
    SPLIT (str): The data split to train on (e.g., '5050', '6040', '9010'), parsed from
                 command-line argument.
    SPLIT_DIR (str): Directory mapping for the split (e.g., '5050' -> '5050', '6040' -> '60_40').

Args:
    --split (str): Required argument specifying the data split to train on. Choices are
                   '5050', '6040', or '9010'.

Returns:
    None: The script saves model checkpoints to the CHECKPOINT_DIR and attempts to save
          loss curves as PNG files. Logs are written to train_ensemble_cvae_{SPLIT}.log.

Raises:
    FileNotFoundError: If the TRAIN_FILE is inaccessible.
    RuntimeError: If the batch count exceeds the expected total.
    Exception: For other unforeseen errors during training or saving.

Example:
    python train_ensemble_cvae.py --split 5050

Notes:
    - Requires a GPU-enabled environment (e.g., tf_gpu Conda environment).
    - Excludes problematic images listed in the problematic_images set.
    - Current issue: Loss curve PNG generation may fail due to empty loss data.

Author: Scott Anderwald
Date: 2025-06-25

"""







# train_ensemble_cvae.py
import tensorflow as tf
import numpy as np
import os
import logging
from tensorflow.keras import mixed_precision
import sys
import subprocess
import argparse

mixed_precision.set_global_policy('mixed_float16')

parser = argparse.ArgumentParser(description="Train ensemble CVAE for a specified split.")
parser.add_argument('--split', type=str, required=True, choices=['5050', '6040', '9010'], help="Specify the split: 5050, 6040, or 9010")
args = parser.parse_args()

split_dir_map = {'5050': '5050', '6040': '60_40', '9010': '90_10'}
SPLIT = args.split
SPLIT_DIR = split_dir_map[SPLIT]

CONFIG = {
    "MAIN_IMAGE_DIR": "/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/",
    "TRAIN_FILE": f"/home/sanderwald/Projects/dissertationProject/data/Splits/{SPLIT_DIR}/train_split_subset.txt",
    "CHECKPOINT_DIR": f"outputs_{SPLIT}_ensemble_cvae_subset_20250520_v2",
    "IMG_SIZE": (360, 203),
    "BATCH_SIZE": 4,
    "LATENT_DIM": 64,
    "RECON_LOSS_WEIGHT": 0.5,
    "KL_LOSS_WEIGHT": 0.01,
    "FOCAL_LOSS_WEIGHT": 5.0,
    "FOCAL_GAMMA": 3.0,
    "FOCAL_ALPHA": 0.9,
    "NUM_CVAES": 1,  # Reverted to 3 as per your dissertation
    "EPOCHS": 3,
    "NUM_SAMPLES": {"5050": 20600, "6040": 24720, "9010": 37080}[SPLIT]
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"/home/sanderwald/Projects/dissertationProject/train_ensemble_cvae_{SPLIT}.log"),
        logging.StreamHandler()
    ]
)

problematic_images = set([
    '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/3_011_0/frame_03997.jpg',
    '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/3_011_0/frame_04022.jpg',
    '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/1_047_0/frame_02802.jpg',
])

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]  # Reduced to 8000 MiB to prevent OOM
        )

tf.keras.backend.clear_session()

def load_image_paths_with_labels(file_path, max_samples):
    logging.info(f"Loading image paths from {file_path} with max {max_samples} samples")
    paths = []
    labels = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f]
    first_line = lines[0] if lines else ""
    if first_line.endswith('.jpg') and os.path.isfile(first_line):
        logging.info("Split file contains image paths")
        image_paths = lines
    else:
        logging.info("Split file contains directories; converting to image paths")
        image_paths = []
        for directory in lines:
            if os.path.isdir(directory):
                logging.info(f"Processing directory: {directory}")
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.jpg'):
                            full_path = os.path.join(root, file)
                            image_paths.append(full_path)
                            if len(image_paths) >= max_samples:
                                break
                    if len(image_paths) >= max_samples:
                        break
            else:
                logging.warning(f"Directory not found: {directory}")
    class_0_count = 0
    class_1_count = 0
    for i, path in enumerate(image_paths):
        if i >= max_samples:
            break
        if path and path not in problematic_images and os.path.isfile(path):
            paths.append(path)
            full_dir = os.path.dirname(path)
            dir_name = os.path.basename(full_dir)
            label = 1 if '_1' in dir_name else 0
            labels.append(label)
            if label == 0:
                class_0_count += 1
            else:
                class_1_count += 1
        else:
            logging.warning(f"Skipping path {path}: Not a file or in problematic_images")
    logging.info(f"Loaded {len(paths)} image paths")
    logging.info(f"Class distribution: Class 0 (negative): {class_0_count}, Class 1 (positive): {class_1_count}")
    return paths, labels

def parse_function(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, CONFIG["IMG_SIZE"], method=tf.image.ResizeMethod.BILINEAR)
    image = image / 255.0
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, label

def create_datasets():
    logging.info("Creating dataset with on-the-fly loading")
    paths, labels = load_image_paths_with_labels(CONFIG["TRAIN_FILE"], CONFIG["NUM_SAMPLES"])
    if len(paths) == 0:
        raise ValueError("No valid image paths found after filtering. Please check the train_split_subset.txt file.")
    paths = np.array(paths)
    labels = np.array(labels)
    indices = np.arange(len(paths))
    np.random.seed(42)
    np.random.shuffle(indices)
    paths = paths[indices]
    labels = labels[indices]
    train_ratio = 0.8
    num_train = int(len(paths) * train_ratio)
    train_paths, val_paths = paths[:num_train], paths[num_train:]
    train_labels, val_labels = labels[:num_train], labels[num_train:]
    logging.info(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_paths), reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(CONFIG["BATCH_SIZE"], drop_remainder=True)
    train_batches = len(train_paths) // CONFIG["BATCH_SIZE"]
    train_dataset = train_dataset.take(train_batches)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_dataset = val_dataset.shuffle(buffer_size=len(val_paths), reshuffle_each_iteration=True)
    val_dataset = val_dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(CONFIG["BATCH_SIZE"], drop_remainder=True)
    val_batches = len(val_paths) // CONFIG["BATCH_SIZE"]
    val_dataset = val_dataset.take(val_batches)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    logging.info(f"Training batches: {train_batches}, Validation batches: {val_batches}")
    return (train_dataset, train_batches), (val_dataset, val_batches)

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

def compute_kl_loss(z_mean, z_log_var):
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return tf.cast(kl_loss, tf.float16)

def focal_loss(gamma=2.0, alpha=0.75):
    @tf.function(reduce_retracing=True)
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
        return tf.cast(tf.reduce_mean(loss), tf.float16)
    return focal_loss_fn

@tf.function(reduce_retracing=True)
def train_step(model, x_batch, y_batch, optimizer):
    with tf.GradientTape() as tape:
        reconstructed, z_mean, z_log_var, classification = model((x_batch, y_batch), training=True)
        recon_loss = tf.reduce_mean(tf.keras.losses.mse(x_batch, reconstructed))
        kl_loss = compute_kl_loss(z_mean, z_log_var)
        focal_loss_val = focal_loss(gamma=CONFIG["FOCAL_GAMMA"], alpha=CONFIG["FOCAL_ALPHA"])(y_batch, classification)
        total_loss = (CONFIG["RECON_LOSS_WEIGHT"] * recon_loss +
                      CONFIG["KL_LOSS_WEIGHT"] * kl_loss +
                      CONFIG["FOCAL_LOSS_WEIGHT"] * focal_loss_val)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, recon_loss, kl_loss, focal_loss_val

def validate(models, dataset):
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_focal_loss = 0.0
    batch_count = 0
    for x_batch, y_batch in dataset:
        batch_loss = 0.0
        batch_recon_loss = 0.0
        batch_kl_loss = 0.0
        batch_focal_loss = 0.0
        for model in models:
            reconstructed, z_mean, z_log_var, classification = model((x_batch, y_batch), training=False)
            recon_loss = tf.reduce_mean(tf.keras.losses.mse(x_batch, reconstructed))
            kl_loss = compute_kl_loss(z_mean, z_log_var)
            focal_loss_val = focal_loss(gamma=CONFIG["FOCAL_GAMMA"], alpha=CONFIG["FOCAL_ALPHA"])(y_batch, classification)
            loss = (CONFIG["RECON_LOSS_WEIGHT"] * recon_loss +
                    CONFIG["KL_LOSS_WEIGHT"] * kl_loss +
                    CONFIG["FOCAL_LOSS_WEIGHT"] * focal_loss_val)
            batch_loss += loss
            batch_recon_loss += recon_loss
            batch_kl_loss += kl_loss
            batch_focal_loss += focal_loss_val
            tf.keras.backend.clear_session()
        total_loss += batch_loss / CONFIG["NUM_CVAES"]
        total_recon_loss += batch_recon_loss / CONFIG["NUM_CVAES"]
        total_kl_loss += batch_kl_loss / CONFIG["NUM_CVAES"]
        total_focal_loss += batch_focal_loss / CONFIG["NUM_CVAES"]
        batch_count += 1
    avg_loss = total_loss / batch_count
    avg_recon_loss = total_recon_loss / batch_count
    avg_kl_loss = total_kl_loss / batch_count
    avg_focal_loss = total_focal_loss / batch_count
    return avg_loss, avg_recon_loss, avg_kl_loss, avg_focal_loss

def main():
    logging.info("Starting training")
    try:
        (train_dataset, train_batches), (val_dataset, val_batches) = create_datasets()
        dummy_x = tf.zeros([CONFIG["BATCH_SIZE"], CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 3], dtype=tf.float32)
        dummy_y = tf.zeros([CONFIG["BATCH_SIZE"]], dtype=tf.int32)
        models = []
        for i in range(CONFIG["NUM_CVAES"]):
            model = CVAE(CONFIG["LATENT_DIM"], model_id=i)
            _ = model((dummy_x, dummy_y), training=True)
            models.append(model)
            logging.info(f"Initialized model {i}")
            tf.keras.backend.clear_session()
        optimizer = tf.keras.optimizers.Adam()
        all_trainable_vars = []
        for model in models:
            all_trainable_vars.extend(model.trainable_variables)
        optimizer.build(all_trainable_vars)
        logging.info(f"Optimizer built with {len(all_trainable_vars)} trainable variables")
        total_expected_batches = train_batches * CONFIG["EPOCHS"]
        global_batch_count = 0
        for epoch in range(1, CONFIG["EPOCHS"] + 1):
            logging.info(f"Starting epoch {epoch}/{CONFIG['EPOCHS']}")
            total_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_focal_loss = 0.0
            batch_count = 0
            for x_batch, y_batch in train_dataset:
                batch_count += 1
                global_batch_count += 1
                if global_batch_count > total_expected_batches:
                    logging.error(f"Exceeded expected total batches ({total_expected_batches}). Stopping training.")
                    raise RuntimeError("Batch count exceeded expected total")
                logging.info(f"Processing batch {batch_count}/{train_batches} (Global: {global_batch_count}/{total_expected_batches})")
                batch_loss = 0.0
                batch_recon_loss = 0.0
                batch_kl_loss = 0.0
                batch_focal_loss = 0.0
                for model in models:
                    loss, recon_loss, kl_loss, focal_loss_val = train_step(model, x_batch, y_batch, optimizer)
                    batch_loss += loss
                    batch_recon_loss += recon_loss
                    batch_kl_loss += kl_loss
                    batch_focal_loss += focal_loss_val
                    tf.keras.backend.clear_session()
                total_loss += batch_loss / CONFIG["NUM_CVAES"]
                total_recon_loss += batch_recon_loss / CONFIG["NUM_CVAES"]
                total_kl_loss += batch_kl_loss / CONFIG["NUM_CVAES"]
                total_focal_loss += batch_focal_loss / CONFIG["NUM_CVAES"]
                if batch_count % 500 == 0:
                    logging.info(f"Epoch {epoch}, Batch {batch_count}: Loss={batch_loss/CONFIG['NUM_CVAES']:.4f}, "
                                 f"Recon Loss={batch_recon_loss/CONFIG['NUM_CVAES']:.4f}, "
                                 f"KL Loss={batch_kl_loss/CONFIG['NUM_CVAES']:.4f}, "
                                 f"Focal Loss={batch_focal_loss/CONFIG['NUM_CVAES']:.4f}")
            avg_loss = total_loss / batch_count
            avg_recon_loss = total_recon_loss / batch_count
            avg_kl_loss = total_kl_loss / batch_count
            avg_focal_loss = total_focal_loss / batch_count
            logging.info(f"Epoch {epoch} Training Summary: Avg Loss={avg_loss:.4f}, "
                         f"Avg Recon Loss={avg_recon_loss:.4f}, "
                         f"Avg KL Loss={avg_kl_loss:.4f}, "
                         f"Avg Focal Loss={avg_focal_loss:.4f}")
            logging.info(f"Running validation for epoch {epoch}")
            val_loss, val_recon_loss, val_kl_loss, val_focal_loss = validate(models, val_dataset)
            logging.info(f"Epoch {epoch} Validation Summary: Avg Loss={val_loss:.4f}, "
                         f"Avg Recon Loss={val_recon_loss:.4f}, "
                         f"Avg KL Loss={val_kl_loss:.4f}, "
                         f"Avg Focal Loss={val_focal_loss:.4f}")
            checkpoint_dir = os.path.join(CONFIG["CHECKPOINT_DIR"], f"epoch_{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            for i, model in enumerate(models):
                model_path = os.path.join(checkpoint_dir, f"cvae_{i}")
                model.save(model_path)
                logging.info(f"Saved model {i} checkpoint to {model_path}")
                tf.keras.backend.clear_session()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
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
