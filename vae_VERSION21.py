#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised VAE with CNN Backbone and FED Focal Loss
Includes training, validation, testing, outputs (EER, metrics, hyperparameter heatmap, and visualizations).
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
# %%
# ---------------------------- Global Parameters ----------------------------
MAIN_IMAGE_DIR = '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images'
TRAIN_FILE = '/media/sanderwald/Project_Files/Dissertation_Project/data/Splits/50_50/train_split.txt'
TEST_FILE = '/media/sanderwald/Project_Files/Dissertation_Project/data/Splits/50_50/test_split.txt'
BATCH_SIZE = 8
IMG_SIZE = (1024, 576)
LATENT_DIM = 32
EPOCHS = 1
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
MODEL_SAVE_PATH = "vae_weights.h5"
OUTPUT_DIR = "outputs"

# ---------------------------- GPU Configuration ----------------------------
def configure_gpu():
    """Enable GPU memory growth to avoid memory allocation issues."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled successfully.")
        except RuntimeError as e:
            print(f"RuntimeError while configuring GPU: {e}")
    else:
        print("No GPU devices found.")

# ---------------------------- Model Plotting ----------------------------
def plot_and_save_model(model, file_name):
    """Plots and saves the model architecture."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, file_name)
    plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)
    print(f"Model architecture saved to {file_path}")

# ---------------------------- FED Focal Loss ----------------------------
def fed_focal_loss(gamma=2.0, alpha=0.25):
    """Implements the FED Focal Loss for binary classification."""
    def loss(y_true, y_pred):
        # Ensure y_true and y_pred have the same shape
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Calculate focal loss
        focal_loss = -alpha * (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred) - \
                     (1 - alpha) * y_pred ** gamma * (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Reduce mean over all dimensions
        return tf.reduce_mean(focal_loss, axis=[1, 2, 3])
    return loss

# ---------------------------- Data Processing ----------------------------
def read_image_paths(file_path, main_dir, split=None, purpose="Training"):
    """Reads image paths and splits data if required."""
    with open(file_path, 'r') as f:
        subdirectories = [line.strip() for line in f]

    all_image_paths = [
        file for subdir in subdirectories
        for file in glob.glob(os.path.join(main_dir, subdir, '*.jpg'))
    ]

    np.random.shuffle(all_image_paths)
    if split:
        val_size = int(len(all_image_paths) * split)
        if purpose == "Training":
            return all_image_paths[val_size:], all_image_paths[:val_size]
        return all_image_paths[:val_size]
    return all_image_paths

def load_image_with_label(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE) / 255.0
    label = tf.strings.regex_full_match(image_path, ".*_1.*")
    label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)  # Ensure correct shape
    return img, label

def create_dataset(paths, batch_size, shuffle=False):
    """Create a TensorFlow dataset for training or validation."""
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(load_image_with_label, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ---------------------------- Model Components ----------------------------
def build_encoder(input_shape):
    """Builds the encoder model."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')

    # Save encoder architecture
    plot_and_save_model(encoder, "encoder_model.png")
    return encoder

def build_decoder(latent_dim):
    """Builds the decoder model."""
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64 * 18 * 256, activation='relu')(latent_inputs)
    x = layers.Reshape((64, 18, 256))(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=(1, 2), padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(3, 3, strides=(2, 2), padding='same', activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # Save decoder architecture
    plot_and_save_model(decoder, "decoder_model.png")
    return decoder

# ---------------------------- VAE Class ----------------------------
class VAE(Model):
    """Variational Autoencoder Model."""
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_loss(self, inputs, reconstruction, z_mean, z_log_var):
        """Computes the VAE loss with Focal Loss + KL divergence."""
        focal_loss_fn = fed_focal_loss(gamma=2.0, alpha=0.25)
        reconstruction_loss = tf.reduce_mean(focal_loss_fn(inputs, reconstruction))
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1
        )
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        return total_loss

    def train_step(self, data):
        """Custom training step for the VAE."""
        inputs, _ = data
        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var = self(inputs)
            loss = self.compute_loss(inputs, reconstruction, z_mean, z_log_var)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        """Custom test step for validation."""
        inputs, _ = data
        reconstruction, z_mean, z_log_var = self(inputs)
        loss = self.compute_loss(inputs, reconstruction, z_mean, z_log_var)
        return {"loss": loss}

# ---------------------------- Training and Testing ----------------------------
if __name__ == "__main__":
    configure_gpu()

    # Load datasets
    train_paths, val_paths = read_image_paths(TRAIN_FILE, MAIN_IMAGE_DIR, split=VALIDATION_SPLIT)
    train_dataset = create_dataset(train_paths, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_paths, BATCH_SIZE)

    # Build and compile the VAE model
    encoder = build_encoder((*IMG_SIZE, 3))
    decoder = build_decoder(LATENT_DIM)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # Train the model
    vae.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # Save the model weights
    vae.save_weights(MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}.")

    # Save full VAE architecture
    plot_and_save_model(vae, "vae_model.png")

    # Test dataset
    test_image_paths = read_image_paths(TEST_FILE, MAIN_IMAGE_DIR)
    test_dataset = create_dataset(test_image_paths, BATCH_SIZE)

    # Evaluate the model on the test dataset
    true_labels, pred_probs = [], []
    for images, labels in tqdm(test_dataset, desc="Evaluating Model"):
        reconstruction, _, _ = vae(images)
        pred_probs.extend(tf.reduce_mean(reconstruction, axis=[1, 2, 3]).numpy())
        true_labels.extend(labels.numpy())

    # Calculate metrics
    auc = roc_auc_score(true_labels, pred_probs)
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    pred_binary = (np.array(pred_probs) >= eer_threshold).astype(int)
    accuracy = accuracy_score(true_labels, pred_binary)
    precision = precision_score(true_labels, pred_binary)
    recall = recall_score(true_labels, pred_binary)
    f1 = f1_score(true_labels, pred_binary)

    # Save metrics to text file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_file = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"ROC AUC: {auc:.4f}\n")
        f.write(f"EER: {eer:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    print(f"Metrics saved to {metrics_file}.")

    # Save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    roc_curve_file = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(roc_curve_file)
    print(f"ROC curve saved to {roc_curve_file}.")
