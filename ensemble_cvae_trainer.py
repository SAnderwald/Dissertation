#!/usr/bin/env python3
"""
Ensemble CVAE Training Script - Clean Version
Trains multiple CVAE models with different configurations for robust anomaly detection.
"""

import os
import sys
import json
import argparse
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow warnings and optimize for memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_disable_all_hlo_passes'
os.environ['TF_DISABLE_XLA'] = '1'
os.environ['TF_DISABLE_JIT'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
tf.keras.mixed_precision.set_global_policy('float32')

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )

# Base seed for reproducibility
BASE_SEED = 42

class SingleCVAE(tf.keras.Model):
    """Individual CVAE model for the ensemble."""
    
    def __init__(self, config, model_id: int = 0):
        super(SingleCVAE, self).__init__()
        self.config = config
        self.model_id = model_id
        self.latent_dim = config['latent_dim']
        
        # Set unique seed for this model
        model_seed = BASE_SEED + model_id * 100
        tf.random.set_seed(model_seed)
        
        # Model-specific regularization parameters
        l2_reg = config.get('l2_regularization', 1e-05)
        dropout_rates = config.get('dropout_rates', {})
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(config['img_size'][0], config['img_size'][1], 3)),
            
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.05)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.05)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.05)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder_dense', 0.15)),
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder_dense', 0.15)),
        ], name=f"encoder_{model_id}")
        
        # Latent space
        self.z_mean = tf.keras.layers.Dense(self.latent_dim, name=f"z_mean_{model_id}")
        self.z_log_var = tf.keras.layers.Dense(self.latent_dim, name=f"z_log_var_{model_id}")
        self.latent_dropout = tf.keras.layers.Dropout(dropout_rates.get('latent', 0.1))
        
        # Decoder
        decoder_reshape_size = 28 * 16 * 64
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.1)),
            
            tf.keras.layers.Dense(decoder_reshape_size, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Reshape((28, 16, 64)),
            
            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.05)),
            
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.05)),
            
            tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=1, padding='same', activation='sigmoid'),
            tf.keras.layers.Resizing(config['img_size'][0], config['img_size'][1]),
        ], name=f"decoder_{model_id}")
        
        # Classifier
        classifier_dropout = dropout_rates.get('classifier', 0.5)
        self.anomaly_classifier = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(classifier_dropout),
            
            tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(classifier_dropout * 0.8),
            
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(classifier_dropout * 0.6),
            
            tf.keras.layers.Dense(32, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Dropout(classifier_dropout * 0.4),
            
            tf.keras.layers.Dense(1, activation='linear',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        ], name=f"anomaly_classifier_{model_id}")
    
    def encode(self, x, training=False):
        h = self.encoder(x, training=training)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        if training:
            z_mean = self.latent_dropout(z_mean, training=training)
        return z_mean, z_log_var
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean), dtype=mean.dtype)
        return mean + tf.exp(0.5 * logvar) * eps
    
    def call(self, inputs, training=False):
        x, label = inputs
        z_mean, z_log_var = self.encode(x, training=training)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z, training=training)
        anomaly_logits = self.anomaly_classifier(z_mean, training=training)
        return reconstructed, z_mean, z_log_var, anomaly_logits
    
    def predict_anomaly(self, x, training=False):
        z_mean, z_log_var = self.encode(x, training=training)
        anomaly_logits = self.anomaly_classifier(z_mean, training=training)
        anomaly_logits_squeezed = tf.squeeze(anomaly_logits, axis=-1)
        anomaly_score = tf.nn.sigmoid(anomaly_logits_squeezed)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z, training=training)
        recon_error = tf.reduce_mean(tf.square(x - reconstructed), axis=[1, 2, 3])
        return anomaly_score, recon_error

class EnsembleCVAE:
    """Ensemble of CVAE models with different configurations."""
    
    def __init__(self, base_config, ensemble_size: int = 3):
        self.base_config = base_config
        self.ensemble_size = ensemble_size
        self.models = []
        self.model_configs = []
        self.model_weights = []  # For weighted averaging
        
        # Create diverse model configurations
        self._create_ensemble_configs()
        
        # Initialize models
        for i in range(ensemble_size):
            model = SingleCVAE(self.model_configs[i], model_id=i)
            self.models.append(model)
            self.model_weights.append(1.0)  # Equal weights initially
    
    def _create_ensemble_configs(self):
        """Create diverse configurations for ensemble members."""
        configs = []
        
        # Model variations
        variations = [
            # Model 0: Conservative regularization
            {
                'latent_dim': 128,
                'l2_regularization': 1e-05,
                'dropout_rates': {
                    'encoder': 0.05,
                    'encoder_dense': 0.15,
                    'decoder': 0.1,
                    'classifier': 0.5,
                    'latent': 0.1
                },
                'learning_rate': 1e-05,
                'loss_weights': {
                    'recon': 1.5,
                    'kl': 0.02,
                    'anomaly': 0.8,
                    'focal_alpha': 0.7,
                    'focal_gamma': 1.0
                }
            },
            # Model 1: Higher regularization
            {
                'latent_dim': 128,
                'l2_regularization': 5e-05,
                'dropout_rates': {
                    'encoder': 0.1,
                    'encoder_dense': 0.2,
                    'decoder': 0.15,
                    'classifier': 0.6,
                    'latent': 0.15
                },
                'learning_rate': 8e-06,
                'loss_weights': {
                    'recon': 1.8,
                    'kl': 0.025,
                    'anomaly': 0.9,
                    'focal_alpha': 0.75,
                    'focal_gamma': 1.2
                }
            },
            # Model 2: Different latent dimension
            {
                'latent_dim': 96,  # Different latent dimension
                'l2_regularization': 2e-05,
                'dropout_rates': {
                    'encoder': 0.07,
                    'encoder_dense': 0.18,
                    'decoder': 0.12,
                    'classifier': 0.55,
                    'latent': 0.12
                },
                'learning_rate': 1.2e-05,
                'loss_weights': {
                    'recon': 1.3,
                    'kl': 0.03,
                    'anomaly': 0.7,
                    'focal_alpha': 0.65,
                    'focal_gamma': 0.8
                }
            }
        ]
        
        for i in range(self.ensemble_size):
            config = self.base_config.copy()
            # Use predefined variations or cycle through them
            variation = variations[i % len(variations)]
            config.update(variation)
            config['img_size'] = self.base_config['img_size']
            config['batch_size'] = self.base_config['batch_size']
            config['epochs'] = self.base_config['epochs']
            config['patience'] = self.base_config['patience']
            
            self.model_configs.append(config)
            configs.append(config)
    
    def predict_ensemble(self, x, method='weighted_average'):
        """Make ensemble predictions."""
        all_anomaly_scores = []
        all_recon_errors = []
        
        for model in self.models:
            anomaly_score, recon_error = model.predict_anomaly(x, training=False)
            all_anomaly_scores.append(anomaly_score)
            all_recon_errors.append(recon_error)
        
        # Stack predictions
        anomaly_scores = tf.stack(all_anomaly_scores, axis=0)  # (ensemble_size, batch_size)
        recon_errors = tf.stack(all_recon_errors, axis=0)
        
        if method == 'weighted_average':
            # Weighted average
            weights = tf.constant(self.model_weights, dtype=tf.float32)
            weights = weights / tf.reduce_sum(weights)  # Normalize weights
            
            final_anomaly_score = tf.reduce_sum(anomaly_scores * weights[:, None], axis=0)
            final_recon_error = tf.reduce_sum(recon_errors * weights[:, None], axis=0)
            
        elif method == 'majority_vote':
            # Majority voting (threshold at 0.5)
            binary_preds = tf.cast(anomaly_scores > 0.5, tf.float32)
            final_anomaly_score = tf.reduce_mean(binary_preds, axis=0)
            final_recon_error = tf.reduce_mean(recon_errors, axis=0)
            
        else:  # simple average
            final_anomaly_score = tf.reduce_mean(anomaly_scores, axis=0)
            final_recon_error = tf.reduce_mean(recon_errors, axis=0)
        
        return final_anomaly_score, final_recon_error, anomaly_scores, recon_errors

class EnsembleDataLoader:
    """Data loader for ensemble training."""
    
    def __init__(self, split: str, img_size: Tuple[int, int] = (224, 128)):
        self.split = split
        self.img_size = img_size
        split_mapping = {'9010': '90_10', '6040': '60_40', '5050': '50_50'}
        self.split_dir = split_mapping.get(split, '90_10')
        base_path = f"/home/sanderwald/Projects/dissertationProject/data/Splits/{self.split_dir}"
        
        self.train_file = f"{base_path}/train_split.txt"
        self.val_file = f"{base_path}/val_split.txt"
        
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Training file not found: {self.train_file}")
        if not os.path.exists(self.val_file):
            raise FileNotFoundError(f"Validation file not found: {self.val_file}")
    
    def load_training_data(self, max_samples: int = 300000) -> Tuple[List[str], List[int], List[str], List[int]]:
        """Load training and validation data from split files."""
        
        # Load paths
        train_paths = []
        with open(self.train_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(line):
                    train_paths.append(line)
        
        val_paths = []
        with open(self.val_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(line):
                    val_paths.append(line)
        
        print(f"Found {len(train_paths)} train paths, {len(val_paths)} val paths")
        
        # Apply max_samples limit
        if max_samples:
            if len(train_paths) > max_samples:
                random.Random(BASE_SEED).shuffle(train_paths)
                train_paths = train_paths[:max_samples]
            
            max_val = max_samples // 5
            if len(val_paths) > max_val:
                random.Random(BASE_SEED).shuffle(val_paths)
                val_paths = val_paths[:max_val]
        
        # Load annotations
        json_path = "/home/sanderwald/Projects/dissertationProject/annotations.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotations not found: {json_path}")
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        lookup = {path: data['label'] == 1 for path, data in annotations.items() if 'label' in data}
        
        # Get labels
        def get_labels_for_paths(paths):
            valid_paths = []
            labels = []
            for path in paths:
                path_parts = path.split('/')
                if len(path_parts) >= 2:
                    rel_path = f"{path_parts[-2]}/{path_parts[-1]}"
                    if rel_path in lookup:
                        valid_paths.append(path)
                        labels.append(int(lookup[rel_path]))
            return valid_paths, labels
        
        train_paths, train_labels = get_labels_for_paths(train_paths)
        val_paths, val_labels = get_labels_for_paths(val_paths)
        
        print(f"Train: {len(train_paths)} images")
        print(f"Val: {len(val_paths)} images")
        
        return train_paths, train_labels, val_paths, val_labels
    
    def create_data_generator(self, paths: List[str], labels: List[int], batch_size: int = 20, model_seed: int = BASE_SEED):
        """Create data generator with model-specific seed for diversity."""
        def data_generator():
            epoch_seed = model_seed
            while True:
                indices = list(range(len(paths)))
                random.Random(epoch_seed).shuffle(indices)
                epoch_seed += 1
                
                batch_images = []
                batch_labels = []
                
                for idx in indices:
                    try:
                        img = tf.keras.preprocessing.image.load_img(paths[idx], target_size=self.img_size)
                        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                        batch_images.append(img)
                        batch_labels.append(labels[idx])
                        
                        if len(batch_images) == batch_size:
                            yield (np.array(batch_images), np.array(batch_labels, dtype=np.float32))
                            batch_images = []
                            batch_labels = []
                    except Exception as e:
                        continue
                
                if len(batch_images) > 0:
                    while len(batch_images) < batch_size:
                        random_idx = random.Random(epoch_seed).randint(0, len(paths) - 1)
                        try:
                            img = tf.keras.preprocessing.image.load_img(paths[random_idx], target_size=self.img_size)
                            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                            batch_images.append(img)
                            batch_labels.append(labels[random_idx])
                        except:
                            if batch_images:
                                batch_images.append(batch_images[-1])
                                batch_labels.append(batch_labels[-1])
                            else:
                                break
                    
                    if len(batch_images) == batch_size:
                        yield (np.array(batch_images), np.array(batch_labels, dtype=np.float32))
        
        return data_generator

class EnsembleLossTracker:
    """Enhanced loss tracking for ensemble training."""
    
    def __init__(self, output_dir: Path, ensemble_size: int):
        self.output_dir = output_dir
        self.ensemble_size = ensemble_size
        self.history = {}
        
        # Initialize history for each model
        for i in range(ensemble_size):
            self.history[f'model_{i}'] = {
                'epochs': [],
                'train_total': [],
                'train_recon': [],
                'train_kl': [],
                'train_anomaly': [],
                'val_total': [],
                'val_recon': [],
                'val_kl': [],
                'val_anomaly': []
            }
    
    def add_epoch(self, model_id: int, epoch: int, train_losses: dict, val_losses: dict):
        """Add epoch data for specific model."""
        model_key = f'model_{model_id}'
        self.history[model_key]['epochs'].append(epoch)
        self.history[model_key]['train_total'].append(np.mean(train_losses['total']))
        self.history[model_key]['train_recon'].append(np.mean(train_losses['recon']))
        self.history[model_key]['train_kl'].append(np.mean(train_losses['kl']))
        self.history[model_key]['train_anomaly'].append(np.mean(train_losses['anomaly']))
        self.history[model_key]['val_total'].append(np.mean(val_losses['total']))
        self.history[model_key]['val_recon'].append(np.mean(val_losses['recon']))
        self.history[model_key]['val_kl'].append(np.mean(val_losses['kl']))
        self.history[model_key]['val_anomaly'].append(np.mean(val_losses['anomaly']))
    
    def plot_ensemble_losses(self):
        """Create ensemble loss plots."""
        if not any(self.history[f'model_{i}']['epochs'] for i in range(self.ensemble_size)):
            print("No epochs to plot")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        # Create ensemble comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ensemble CVAE Training Analysis', fontsize=16, fontweight='bold')
        
        # 1. Total Loss Comparison
        for i in range(self.ensemble_size):
            model_key = f'model_{i}'
            if self.history[model_key]['epochs']:
                epochs = self.history[model_key]['epochs']
                train_loss = self.history[model_key]['train_total']
                val_loss = self.history[model_key]['val_total']
                
                color = colors[i % len(colors)]
                axes[0, 0].plot(epochs, train_loss, 'o-', label=f'Model {i} Train', 
                               color=color, alpha=0.7, linewidth=2)
                axes[0, 0].plot(epochs, val_loss, 's--', label=f'Model {i} Val', 
                               color=color, alpha=0.9, linewidth=2)
        
        axes[0, 0].set_title('Total Loss - All Models', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. KL Divergence Monitoring
        for i in range(self.ensemble_size):
            model_key = f'model_{i}'
            if self.history[model_key]['epochs']:
                epochs = self.history[model_key]['epochs']
                val_kl = self.history[model_key]['val_kl']
                
                color = colors[i % len(colors)]
                axes[0, 1].plot(epochs, val_kl, 'o-', label=f'Model {i}', 
                               color=color, linewidth=2, markersize=6)
        
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='KL=1.0 (Warning)')
        axes[0, 1].axhline(y=1.5, color='darkred', linestyle='--', alpha=0.7, label='KL=1.5 (Stop)')
        axes[0, 1].set_title('KL Divergence - Ensemble', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Final Performance Comparison
        final_performances = []
        model_names = []
        
        for i in range(self.ensemble_size):
            model_key = f'model_{i}'
            if self.history[model_key]['epochs']:
                final_val_loss = self.history[model_key]['val_total'][-1]
                final_performances.append(final_val_loss)
                model_names.append(f'Model {i}')
        
        if final_performances:
            bars = axes[1, 0].bar(model_names, final_performances, 
                                 color=colors[:len(final_performances)], alpha=0.7)
            axes[1, 0].set_title('Final Validation Loss - Ensemble', fontweight='bold')
            axes[1, 0].set_ylabel('Validation Loss')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, final_performances):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001, 
                               f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Ensemble Summary Statistics
        if final_performances:
            mean_performance = np.mean(final_performances)
            std_performance = np.std(final_performances)
            best_performance = np.min(final_performances)
            
            summary_text = f'Ensemble Size: {self.ensemble_size}\n'
            summary_text += f'Mean Val Loss: {mean_performance:.4f}\n'
            summary_text += f'Std Val Loss: {std_performance:.4f}\n'
            summary_text += f'Best Val Loss: {best_performance:.4f}\n'
            summary_text += f'Performance Diversity: {std_performance/mean_performance:.3f}'
            
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('Ensemble Statistics', fontweight='bold')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plots
        plots_dir = self.output_dir / "ensemble_logs"
        plots_dir.mkdir(exist_ok=True)
        
        plot_file = plots_dir / "ensemble_loss_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Ensemble plots saved: {plot_file}")
    
    def save_history(self):
        """Save ensemble loss history."""
        history_file = self.output_dir / "ensemble_logs" / "ensemble_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Ensemble history saved: {history_file}")

def train_step_safe(model, images, labels, optimizer, loss_fn):
    """Training step with gradient clipping."""
    with tf.GradientTape() as tape:
        outputs = model((images, labels), training=True)
        total_loss, recon_loss, kl_loss, anomaly_loss = loss_fn((images, labels), outputs)
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients = [tf.clip_by_norm(grad, 0.5) if grad is not None else grad for grad in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, recon_loss, kl_loss, anomaly_loss

def improved_focal_loss(alpha=0.7, gamma=1.0):
    """Conservative focal loss."""
    def focal_loss_fn(y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_logits = tf.squeeze(y_pred_logits, axis=-1)
        
        sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred_logits
        )
        
        y_pred = tf.nn.sigmoid(y_pred_logits)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1.0 - alpha)
        
        pt = tf.clip_by_value(pt, 1e-8, 1.0 - 1e-8)
        focal_weight = alpha_t * tf.pow(1 - pt, gamma)
        
        focal_loss = focal_weight * sigmoid_cross_entropy
        return tf.reduce_mean(focal_loss)
    return focal_loss_fn

def cvae_loss_function(model_config):
    """CVAE loss function."""
    loss_weights = model_config['loss_weights']
    focal_alpha = loss_weights['focal_alpha']
    focal_gamma = loss_weights['focal_gamma']
    
    focal_loss_fn = improved_focal_loss(focal_alpha, focal_gamma)
    
    def loss_fn(inputs, outputs):
        x, labels = inputs
        reconstructed, z_mean, z_log_var, anomaly_logits = outputs
        
        recon_loss = tf.reduce_mean(tf.square(x - reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.clip_by_value(kl_loss, 0.0, 1.0)
        anomaly_loss = focal_loss_fn(labels, anomaly_logits)
        
        total_loss = (loss_weights['recon'] * recon_loss + 
                     loss_weights['kl'] * kl_loss + 
                     loss_weights['anomaly'] * anomaly_loss)
        
        return total_loss, recon_loss, kl_loss, anomaly_loss
    
    return loss_fn

def create_timestamped_ensemble_dir(base_name: str = "ensemble_outputs", split: str = "9010", 
                                   num_samples: int = 300000, ensemble_size: int = 3) -> Path:
    """Create timestamped ensemble output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    samples_str = f"{num_samples//1000}k" if num_samples >= 1000 else str(num_samples)
    
    dir_name = f"{base_name}_{split}_{samples_str}_e{ensemble_size}_{timestamp}"
    ensemble_dir = Path(dir_name)
    
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    (ensemble_dir / "ensemble_models").mkdir(exist_ok=True)
    (ensemble_dir / "ensemble_logs").mkdir(exist_ok=True)
    
    # Create individual model directories
    for i in range(ensemble_size):
        (ensemble_dir / "ensemble_models" / f"model_{i}").mkdir(exist_ok=True)
    
    print(f"Created ensemble directory: {ensemble_dir}")
    return ensemble_dir

class EnsembleCVAETrainer:
    """Ensemble CVAE trainer."""
    
    def __init__(self, output_dir: Path, split: str, num_samples: int, 
                 img_size: Tuple[int, int] = (224, 128), ensemble_size: int = 3):
        self.output_dir = output_dir
        self.split = split
        self.num_samples = num_samples
        self.img_size = img_size
        self.ensemble_size = ensemble_size
        self.loss_tracker = EnsembleLossTracker(output_dir, ensemble_size)
        self.setup_base_config()
    
    def setup_base_config(self):
        """Setup base configuration."""
        self.base_config = {
            "name": "Ensemble CVAE",
            "latent_dim": 128,
            "img_size": list(self.img_size),
            "l2_regularization": 1e-05,
            "dropout_rates": {
                "encoder": 0.05,
                "encoder_dense": 0.15,
                "decoder": 0.1,
                "classifier": 0.5,
                "latent": 0.1
            },
            "learning_rate": 1e-05,
            "loss_weights": {
                "recon": 1.5,
                "kl": 0.02,
                "anomaly": 0.8,
                "focal_alpha": 0.7,
                "focal_gamma": 1.0
            },
            "batch_size": 16,
            "epochs": 5,
            "patience": 1
        }
    
    def save_ensemble_config(self, ensemble_cvae: EnsembleCVAE):
        """Save ensemble configuration."""
        config_file = self.output_dir / "ensemble_config.json"
        
        config_to_save = {
            "split": self.split,
            "num_samples": self.num_samples,
            "ensemble_size": self.ensemble_size,
            "img_size": self.base_config['img_size'],
            "base_config": self.base_config,
            "model_configs": ensemble_cvae.model_configs,
            "model_weights": ensemble_cvae.model_weights
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"Ensemble config saved")
    
    def train_ensemble(self, train_data_gen, val_data_gen, train_paths, train_labels, 
                      val_paths, val_labels, steps_per_epoch: int, val_steps: int) -> EnsembleCVAE:
        """Train ensemble of CVAE models."""
        print(f"\nTraining Ensemble of {self.ensemble_size} CVAE Models")
        print("=" * 60)
        
        # Create ensemble
        ensemble_cvae = EnsembleCVAE(self.base_config, self.ensemble_size)
        data_loader = EnsembleDataLoader(self.split, self.img_size)
        
        # Train each model in the ensemble
        for model_idx in range(self.ensemble_size):
            print(f"\nTraining Model {model_idx + 1}/{self.ensemble_size}")
            print("-" * 40)
            
            model = ensemble_cvae.models[model_idx]
            config = ensemble_cvae.model_configs[model_idx]
            
            # Build model
            dummy_x = tf.zeros([1, self.img_size[0], self.img_size[1], 3])
            dummy_y = tf.zeros([1])
            model((dummy_x, dummy_y), training=False)
            print(f"Model {model_idx} built successfully")
            
            # Create model-specific data generators with different seeds
            model_seed = BASE_SEED + model_idx * 1000
            model_train_gen = data_loader.create_data_generator(
                train_paths, train_labels, config['batch_size'], model_seed
            )
            model_val_gen = data_loader.create_data_generator(
                val_paths, val_labels, config['batch_size'], model_seed + 1
            )
            
            # Setup training
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config['learning_rate'])
            loss_fn = cvae_loss_function(config)
            
            epochs = config['epochs']
            patience = config['patience']
            
            print(f"Config: LR={config['learning_rate']}, LatentDim={config['latent_dim']}")
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Train this model
            for epoch in range(epochs):
                # Training
                train_losses = {'total': [], 'recon': [], 'kl': [], 'anomaly': []}
                step_count = 0
                
                print(f"Epoch {epoch + 1}/{epochs} - Training...", end=" ", flush=True)
                
                for batch_data in model_train_gen():
                    if step_count >= steps_per_epoch:
                        break
                        
                    images, labels = batch_data
                    total_loss, recon_loss, kl_loss, anomaly_loss = train_step_safe(
                        model, images, labels, optimizer, loss_fn
                    )
                    
                    if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
                        print(f"\nNaN/Inf detected in Model {model_idx} at epoch {epoch+1}")
                        break
                    
                    train_losses['total'].append(float(total_loss.numpy()))
                    train_losses['recon'].append(float(recon_loss.numpy()))
                    train_losses['kl'].append(float(kl_loss.numpy()))
                    train_losses['anomaly'].append(float(anomaly_loss.numpy()))
                    
                    step_count += 1
                    
                    if step_count % 100 == 0:
                        print(".", end="", flush=True)
                
                # Validation
                print(" Validating...", end=" ", flush=True)
                val_losses = {'total': [], 'recon': [], 'kl': [], 'anomaly': []}
                val_step_count = 0
                
                for batch_data in model_val_gen():
                    if val_step_count >= val_steps:
                        break
                        
                    images, labels = batch_data
                    outputs = model((images, labels), training=False)
                    total_loss, recon_loss, kl_loss, anomaly_loss = loss_fn((images, labels), outputs)
                    
                    val_losses['total'].append(float(total_loss.numpy()))
                    val_losses['recon'].append(float(recon_loss.numpy()))
                    val_losses['kl'].append(float(kl_loss.numpy()))
                    val_losses['anomaly'].append(float(anomaly_loss.numpy()))
                    
                    val_step_count += 1
                
                # Track losses
                self.loss_tracker.add_epoch(model_idx, epoch + 1, train_losses, val_losses)
                
                train_loss = np.mean(train_losses['total'])
                val_loss = np.mean(val_losses['total'])
                val_kl = np.mean(val_losses['kl'])
                
                if val_loss < best_val_loss and val_kl < 1.0:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, KL: {val_kl:.3f} (SAVED)")
                    
                    # Save model
                    model_path = self.output_dir / "ensemble_models" / f"model_{model_idx}" / "model_weights"
                    try:
                        model.save_weights(str(model_path))
                    except Exception as e:
                        print(f"Error saving model {model_idx}: {e}")
                    
                else:
                    patience_counter += 1
                    print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, KL: {val_kl:.3f} (patience: {patience_counter}/{patience})")
                    
                    if val_kl > 1.5 or patience_counter >= patience:
                        print(f"Early stopping Model {model_idx}")
                        break
            
            # Update model weight based on performance
            ensemble_cvae.model_weights[model_idx] = 1.0 / (best_val_loss + 1e-6)
            print(f"Model {model_idx} completed. Weight: {ensemble_cvae.model_weights[model_idx]:.4f}")
        
        # Normalize ensemble weights
        total_weight = sum(ensemble_cvae.model_weights)
        ensemble_cvae.model_weights = [w / total_weight for w in ensemble_cvae.model_weights]
        
        print(f"\nEnsemble Training Complete!")
        print(f"Final Model Weights: {[f'{w:.3f}' for w in ensemble_cvae.model_weights]}")
        
        # Create ensemble plots
        self.loss_tracker.plot_ensemble_losses()
        self.loss_tracker.save_history()
        
        return ensemble_cvae

def main():
    """Main ensemble training function."""
    parser = argparse.ArgumentParser(description='Train Ensemble CVAE Models')
    parser.add_argument('--split', type=str, default='9010',
                       choices=['9010', '6040', '5050'],
                       help='Data split to use for training')
    parser.add_argument('--num_samples', type=int, default=100000,
                       help='Maximum number of training samples to use')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 128],
                       help='Image dimensions [height, width]')
    parser.add_argument('--ensemble_size', type=int, default=3,
                       help='Number of models in ensemble (default: 3)')
    
    args = parser.parse_args()
    
    print("Ensemble CVAE Training")
    print("Multiple models with diverse configurations")
    print("Comprehensive ensemble visualization")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Max Samples: {args.num_samples:,}")
    print(f"Image Size: {args.img_size}")
    print(f"Ensemble Size: {args.ensemble_size}")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_timestamped_ensemble_dir("ensemble_outputs", args.split, 
                                                args.num_samples, args.ensemble_size)
    
    # Create trainer
    trainer = EnsembleCVAETrainer(output_dir, args.split, args.num_samples, 
                                 tuple(args.img_size), args.ensemble_size)
    
    # Load data
    print("Loading training data...")
    try:
        data_loader = EnsembleDataLoader(args.split, tuple(args.img_size))
        train_paths, train_labels, val_paths, val_labels = data_loader.load_training_data(args.num_samples)
        
        print(f"Data loaded successfully:")
        print(f"  Train: {len(train_paths)} samples")
        print(f"  Validation: {len(val_paths)} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create data generators (base generators, model-specific ones created during training)
    batch_size = trainer.base_config['batch_size']
    train_data_gen = data_loader.create_data_generator(train_paths, train_labels, batch_size)
    val_data_gen = data_loader.create_data_generator(val_paths, val_labels, batch_size)
    
    # Calculate steps
    steps_per_epoch = max(100, len(train_paths) // batch_size)
    val_steps = max(50, len(val_paths) // batch_size)
    
    print(f"Training setup:")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Expected time: {20 * args.ensemble_size}-{40 * args.ensemble_size} minutes")
    
    # Train ensemble
    try:
        print("\nStarting ensemble training...")
        ensemble_cvae = trainer.train_ensemble(
            train_data_gen, val_data_gen, train_paths, train_labels,
            val_paths, val_labels, steps_per_epoch, val_steps
        )
        
        # Save ensemble configuration
        trainer.save_ensemble_config(ensemble_cvae)
        
        print("Ensemble training completed successfully!")
        print(f"Ensemble saved in: {output_dir}")
        print(f"Ensemble plots saved in: {output_dir / 'ensemble_logs'}")
        print(f"Test with: python ensemble_cvae_tester.py --ensemble_dir {output_dir}")
        
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()