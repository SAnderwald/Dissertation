#!/usr/bin/env python3
"""
Single CVAE Training Script - Clean Version
Max 5 epochs with reduced regularization to allow more flexibility.
Compatible with single_cvae_tester.py for evaluation.
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

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class SingleCVAE(tf.keras.Model):
    """CVAE with reduced regularization."""
    
    def __init__(self, config):
        super(SingleCVAE, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        
        # Reduced regularization parameters
        l2_reg = config.get('l2_regularization', 1e-05)  # Reduced from 1e-04 to 1e-05
        dropout_rates = config.get('dropout_rates', {})
        
        # Encoder with reduced regularization
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(config['img_size'][0], config['img_size'][1], 3)),
            
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.05)),  # Reduced from 0.1
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.05)),  # Reduced from 0.1
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.05)),  # Reduced from 0.1
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder_dense', 0.15)),  # Reduced from 0.3
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder_dense', 0.15)),  # Reduced from 0.3
        ], name="encoder")
        
        # Latent space
        self.z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")
        self.latent_dropout = tf.keras.layers.Dropout(dropout_rates.get('latent', 0.1))  # Reduced from 0.2
        
        # Decoder with reduced regularization
        decoder_reshape_size = 28 * 16 * 64
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.1)),  # Reduced from 0.2
            
            tf.keras.layers.Dense(decoder_reshape_size, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Reshape((28, 16, 64)),
            
            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.05)),  # Reduced from 0.1
            
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.05)),  # Reduced from 0.1
            
            tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=1, padding='same', activation='sigmoid'),
            tf.keras.layers.Resizing(config['img_size'][0], config['img_size'][1]),
        ], name="decoder")
        
        # Classifier with reduced regularization
        classifier_dropout = dropout_rates.get('classifier', 0.5)  # Reduced from 0.7
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
        ], name="anomaly_classifier")
    
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

class SingleDataLoader:
    """Data loader with optional augmentation."""
    
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
                random.Random(RANDOM_SEED).shuffle(train_paths)
                train_paths = train_paths[:max_samples]
            
            max_val = max_samples // 5
            if len(val_paths) > max_val:
                random.Random(RANDOM_SEED).shuffle(val_paths)
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
    
    def create_data_generator(self, paths: List[str], labels: List[int], batch_size: int = 20):
        """Create data generator."""
        def data_generator():
            epoch_seed = RANDOM_SEED
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

class LossTracker:
    """Enhanced loss tracking and visualization."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.history = {
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
    
    def add_epoch(self, epoch: int, train_losses: dict, val_losses: dict):
        """Add epoch data to history."""
        self.history['epochs'].append(epoch)
        self.history['train_total'].append(np.mean(train_losses['total']))
        self.history['train_recon'].append(np.mean(train_losses['recon']))
        self.history['train_kl'].append(np.mean(train_losses['kl']))
        self.history['train_anomaly'].append(np.mean(train_losses['anomaly']))
        self.history['val_total'].append(np.mean(val_losses['total']))
        self.history['val_recon'].append(np.mean(val_losses['recon']))
        self.history['val_kl'].append(np.mean(val_losses['kl']))
        self.history['val_anomaly'].append(np.mean(val_losses['anomaly']))
    
    def plot_losses(self):
        """Create comprehensive loss plots."""
        if len(self.history['epochs']) < 1:
            print("No epochs to plot")
            return
        
        # Special handling for single epoch
        single_epoch = len(self.history['epochs']) == 1
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create comprehensive loss plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CVAE Training Loss Analysis', fontsize=16, fontweight='bold')
        
        epochs = self.history['epochs']
        
        # 1. Total Loss
        if single_epoch:
            # Use bar chart for single epoch
            axes[0, 0].bar(['Train', 'Val'], [self.history['train_total'][0], self.history['val_total'][0]], 
                          color=['blue', 'orange'], alpha=0.7)
            axes[0, 0].set_title(f'Total Loss (Epoch {epochs[0]})', fontweight='bold')
        else:
            axes[0, 0].plot(epochs, self.history['train_total'], 'o-', label='Train Total', linewidth=2, markersize=6)
            axes[0, 0].plot(epochs, self.history['val_total'], 's-', label='Val Total', linewidth=2, markersize=6)
            axes[0, 0].set_title('Total Loss', fontweight='bold')
            axes[0, 0].legend()
        
        axes[0, 0].set_xlabel('Epoch' if not single_epoch else 'Set')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Reconstruction Loss
        if single_epoch:
            axes[0, 1].bar(['Train', 'Val'], [self.history['train_recon'][0], self.history['val_recon'][0]], 
                          color=['blue', 'orange'], alpha=0.7)
            axes[0, 1].set_title(f'Reconstruction Loss (Epoch {epochs[0]})', fontweight='bold')
        else:
            axes[0, 1].plot(epochs, self.history['train_recon'], 'o-', label='Train Recon', linewidth=2, markersize=6)
            axes[0, 1].plot(epochs, self.history['val_recon'], 's-', label='Val Recon', linewidth=2, markersize=6)
            axes[0, 1].set_title('Reconstruction Loss', fontweight='bold')
            axes[0, 1].legend()
        
        axes[0, 1].set_xlabel('Epoch' if not single_epoch else 'Set')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. KL Divergence Loss
        if single_epoch:
            bars = axes[0, 2].bar(['Train', 'Val'], [self.history['train_kl'][0], self.history['val_kl'][0]], 
                                 color=['blue', 'orange'], alpha=0.7)
            axes[0, 2].set_title(f'KL Divergence Loss (Epoch {epochs[0]})', fontweight='bold')
            # Add threshold lines for single epoch
            axes[0, 2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='KL=1.0 (Early Stop)')
            axes[0, 2].axhline(y=1.5, color='darkred', linestyle='--', alpha=0.7, label='KL=1.5 (Force Stop)')
        else:
            axes[0, 2].plot(epochs, self.history['train_kl'], 'o-', label='Train KL', linewidth=2, markersize=6)
            axes[0, 2].plot(epochs, self.history['val_kl'], 's-', label='Val KL', linewidth=2, markersize=6)
            axes[0, 2].set_title('KL Divergence Loss', fontweight='bold')
            axes[0, 2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='KL=1.0 (Early Stop)')
            axes[0, 2].axhline(y=1.5, color='darkred', linestyle='--', alpha=0.7, label='KL=1.5 (Force Stop)')
            axes[0, 2].legend()
        
        axes[0, 2].set_xlabel('Epoch' if not single_epoch else 'Set')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Anomaly Classification Loss
        if single_epoch:
            axes[1, 0].bar(['Train', 'Val'], [self.history['train_anomaly'][0], self.history['val_anomaly'][0]], 
                          color=['blue', 'orange'], alpha=0.7)
            axes[1, 0].set_title(f'Anomaly Classification Loss (Epoch {epochs[0]})', fontweight='bold')
        else:
            axes[1, 0].plot(epochs, self.history['train_anomaly'], 'o-', label='Train Anomaly', linewidth=2, markersize=6)
            axes[1, 0].plot(epochs, self.history['val_anomaly'], 's-', label='Val Anomaly', linewidth=2, markersize=6)
            axes[1, 0].set_title('Anomaly Classification Loss', fontweight='bold')
            axes[1, 0].legend()
        
        axes[1, 0].set_xlabel('Epoch' if not single_epoch else 'Set')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Loss Components Comparison (Final Epoch)
        if len(epochs) > 0:
            final_train = [self.history['train_recon'][-1], self.history['train_kl'][-1], self.history['train_anomaly'][-1]]
            final_val = [self.history['val_recon'][-1], self.history['val_kl'][-1], self.history['val_anomaly'][-1]]
            
            x = np.arange(3)
            width = 0.35
            
            axes[1, 1].bar(x - width/2, final_train, width, label='Train', alpha=0.8)
            axes[1, 1].bar(x + width/2, final_val, width, label='Validation', alpha=0.8)
            axes[1, 1].set_title('Final Epoch Loss Components', fontweight='bold')
            axes[1, 1].set_xlabel('Loss Component')
            axes[1, 1].set_ylabel('Loss Value')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(['Reconstruction', 'KL Divergence', 'Anomaly'])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Learning Curves (Train vs Val) / Overfitting Monitor
        if single_epoch:
            # For single epoch, show the actual values instead of gap
            train_val_values = [self.history['train_total'][0], self.history['val_total'][0]]
            bars = axes[1, 2].bar(['Train Loss', 'Val Loss'], train_val_values, 
                                 color=['lightblue', 'lightcoral'], alpha=0.8)
            axes[1, 2].set_title(f'Train vs Val Comparison (Epoch {epochs[0]})', fontweight='bold')
            axes[1, 2].set_ylabel('Loss Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, train_val_values):
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001, 
                               f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[1, 2].plot(epochs, np.array(self.history['train_total']) - np.array(self.history['val_total']), 
                           'o-', label='Train-Val Gap', linewidth=2, markersize=6, color='purple')
            axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 2].set_title('Overfitting Monitor (Train-Val Gap)', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss Difference')
            axes[1, 2].legend()
        
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add text annotations with training info
        if single_epoch:
            textstr = f'Epochs: {len(epochs)} (Early Stop)\n'
            textstr += f'Val Loss: {self.history["val_total"][-1]:.4f}\n'
            textstr += f'KL Divergence: {self.history["val_kl"][-1]:.3f}\n'
            textstr += f'Status: {"Good" if self.history["val_kl"][-1] < 1.0 else "High KL"}'
        else:
            textstr = f'Epochs: {len(epochs)}\n'
            textstr += f'Best Val Loss: {min(self.history["val_total"]):.4f}\n'
            textstr += f'Final KL: {self.history["val_kl"][-1]:.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[1, 2].text(0.02, 0.98, textstr, transform=axes[1, 2].transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plots
        plots_dir = self.output_dir / "training_logs"
        plots_dir.mkdir(exist_ok=True)
        
        plot_file = plots_dir / "loss_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create a simplified single loss plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if single_epoch:
            # Bar chart for single epoch
            bars = ax.bar(['Training Loss', 'Validation Loss'], 
                         [self.history['train_total'][0], self.history['val_total'][0]], 
                         color=['blue', 'orange'], alpha=0.7, width=0.6)
            ax.set_title(f'CVAE Training Results - Epoch {epochs[0]}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Total Loss', fontsize=12)
            
            # Add value labels on bars
            for bar, value in zip(bars, [self.history['train_total'][0], self.history['val_total'][0]]):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, 
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                       
            # Add status text
            status_text = f"KL Divergence: {self.history['val_kl'][0]:.3f}"
            status_text += f"\nStatus: {'Converged' if self.history['val_kl'][0] < 1.0 else 'High KL'}"
            ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        else:
            ax.plot(epochs, self.history['train_total'], 'o-', label='Training Loss', linewidth=3, markersize=8)
            ax.plot(epochs, self.history['val_total'], 's-', label='Validation Loss', linewidth=3, markersize=8)
            ax.set_title('CVAE Training Progress', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Total Loss', fontsize=12)
            ax.legend(fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        simple_plot_file = plots_dir / "simple_loss_curve.png"
        plt.savefig(simple_plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Loss plots saved:")
        print(f"   Comprehensive: {plot_file}")
        print(f"   Simple: {simple_plot_file}")
        if single_epoch:
            print(f"   Single epoch plots generated (training stopped early)")
    
    def save_history(self):
        """Save loss history to JSON."""
        history_file = self.output_dir / "training_logs" / "loss_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Loss history saved: {history_file}")

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
    """CVAE loss function with conservative weights."""
    loss_weights = model_config['loss_weights']
    focal_alpha = loss_weights['focal_alpha']
    focal_gamma = loss_weights['focal_gamma']
    
    focal_loss_fn = improved_focal_loss(focal_alpha, focal_gamma)
    
    def loss_fn(inputs, outputs):
        x, labels = inputs
        reconstructed, z_mean, z_log_var, anomaly_logits = outputs
        
        recon_loss = tf.reduce_mean(tf.square(x - reconstructed))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.clip_by_value(kl_loss, 0.0, 1.0)  # Conservative clipping
        anomaly_loss = focal_loss_fn(labels, anomaly_logits)
        
        total_loss = (loss_weights['recon'] * recon_loss + 
                     loss_weights['kl'] * kl_loss + 
                     loss_weights['anomaly'] * anomaly_loss)
        
        return total_loss, recon_loss, kl_loss, anomaly_loss
    
    return loss_fn

def create_timestamped_single_dir(base_name: str = "outputs", split: str = "9010", num_samples: int = 300000) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    samples_str = f"{num_samples//1000}k" if num_samples >= 1000 else str(num_samples)
    
    dir_name = f"{base_name}_{split}_{samples_str}_{timestamp}"
    single_dir = Path(dir_name)
    
    single_dir.mkdir(parents=True, exist_ok=True)
    (single_dir / "best_model").mkdir(exist_ok=True)
    (single_dir / "model_config").mkdir(exist_ok=True)
    (single_dir / "training_logs").mkdir(exist_ok=True)
    
    print(f"Created output directory: {single_dir}")
    return single_dir

class SingleCVAETrainer:
    """Single CVAE trainer with reduced regularization and enhanced loss tracking."""
    
    def __init__(self, output_dir: Path, split: str, num_samples: int, img_size: Tuple[int, int] = (224, 128)):
        self.output_dir = output_dir
        self.split = split
        self.num_samples = num_samples
        self.img_size = img_size
        self.loss_tracker = LossTracker(output_dir)
        self.setup_config()
    
    def setup_config(self):
        """Setup configuration with reduced regularization."""
        self.model_config = {
            "name": "Single CVAE - Reduced Regularization",
            "latent_dim": 128,
            "img_size": list(self.img_size),
            "l2_regularization": 1e-05,  # Reduced from 1e-04
            "dropout_rates": {
                "encoder": 0.05,  # Reduced from 0.15
                "encoder_dense": 0.15,  # Reduced from 0.4
                "decoder": 0.1,  # Reduced from 0.2
                "classifier": 0.5,  # Reduced from 0.7
                "latent": 0.1  # Reduced from 0.2
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
    
    def save_config(self):
        """Save configuration."""
        config_file = self.output_dir / "config.json"
        
        config_to_save = {
            "split": self.split,
            "latent_dim": self.model_config['latent_dim'],
            "img_size": self.model_config['img_size'],
            "num_samples": self.num_samples,
            "focal_alpha": self.model_config['loss_weights']['focal_alpha'],
            "focal_gamma": self.model_config['loss_weights']['focal_gamma'],
            "anomaly_loss_weight": self.model_config['loss_weights']['anomaly'],
            "best_val_loss": 0.0,
            "epoch": 0,
            "batch_size": self.model_config['batch_size'],
            "learning_rate": self.model_config['learning_rate']
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        detailed_config_file = self.output_dir / "model_config" / "detailed_config.json"
        detailed_config = {
            "split": self.split,
            "num_samples": self.num_samples,
            "model_config": self.model_config
        }
        with open(detailed_config_file, 'w') as f:
            json.dump(detailed_config, f, indent=2)
        
        print(f"Config saved")
    
    def train_model(self, train_data_gen, val_data_gen, steps_per_epoch: int, val_steps: int) -> SingleCVAE:
        """Train model with max 5 epochs and comprehensive loss tracking."""
        print(f"\nTraining Single CVAE (Max 5 Epochs) with Loss Visualization")
        
        model = SingleCVAE(self.model_config)
        
        # Build model
        dummy_x = tf.zeros([1, self.img_size[0], self.img_size[1], 3])
        dummy_y = tf.zeros([1])
        model((dummy_x, dummy_y), training=False)
        print("Model built successfully")
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.model_config['learning_rate'])
        loss_fn = cvae_loss_function(self.model_config)
        
        epochs = self.model_config['epochs']
        batch_size = self.model_config['batch_size']
        patience = self.model_config['patience']
        
        print(f"Max epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Patience: {patience}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_losses = {'total': [], 'recon': [], 'kl': [], 'anomaly': []}
            step_count = 0
            
            print(f"Epoch {epoch + 1}/{epochs} - Training...", end=" ", flush=True)
            
            for batch_data in train_data_gen():
                if step_count >= steps_per_epoch:
                    break
                    
                images, labels = batch_data
                total_loss, recon_loss, kl_loss, anomaly_loss = train_step_safe(
                    model, images, labels, optimizer, loss_fn
                )
                
                if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss):
                    print(f"\nNaN/Inf detected at epoch {epoch+1}")
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
            
            for batch_data in val_data_gen():
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
            
            # Add to loss tracker
            self.loss_tracker.add_epoch(epoch + 1, train_losses, val_losses)
            
            train_loss = np.mean(train_losses['total'])
            val_loss = np.mean(val_losses['total'])
            val_kl = np.mean(val_losses['kl'])
            
            if val_loss < best_val_loss and val_kl < 1.0:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, KL: {val_kl:.3f} (SAVED)")
                
                try:
                    model_path = self.output_dir / "best_model" / "model_weights"
                    model.save_weights(str(model_path))
                    
                    config_file = self.output_dir / "config.json"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            current_config = json.load(f)
                        current_config['best_val_loss'] = float(best_val_loss)
                        current_config['epoch'] = epoch + 1
                        with open(config_file, 'w') as f:
                            json.dump(current_config, f, indent=2)
                            
                except Exception as e:
                    print(f"Error saving model: {e}")
                
            else:
                patience_counter += 1
                print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, KL: {val_kl:.3f} (patience: {patience_counter}/{patience})")
                
                if val_kl > 1.5 or patience_counter >= patience:
                    print(f"Early stopping triggered")
                    break
            
            # Create plots after each epoch
            self.loss_tracker.plot_losses()
        
        # Save final loss history
        self.loss_tracker.save_history()
        
        return model

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Single CVAE Model - Max 5 Epochs with Loss Visualization')
    parser.add_argument('--split', type=str, default='9010',
                       choices=['9010', '6040', '5050'],
                       help='Data split to use for training')
    parser.add_argument('--num_samples', type=int, default=100000,
                       help='Maximum number of training samples to use')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 128],
                       help='Image dimensions [height, width]')
    
    args = parser.parse_args()
    
    print("Single CVAE Training - Max 5 Epochs")
    print("Reduced regularization for better learning")
    print("Comprehensive loss visualization to PNG")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Max Samples: {args.num_samples:,}")
    print(f"Image Size: {args.img_size}")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_timestamped_single_dir("outputs", args.split, args.num_samples)
    
    # Create trainer
    trainer = SingleCVAETrainer(output_dir, args.split, args.num_samples, tuple(args.img_size))
    trainer.save_config()
    
    # Load data
    print("Loading training data...")
    try:
        data_loader = SingleDataLoader(args.split, tuple(args.img_size))
        train_paths, train_labels, val_paths, val_labels = data_loader.load_training_data(args.num_samples)
        
        print(f"Data loaded successfully:")
        print(f"   Train: {len(train_paths)} samples")
        print(f"   Validation: {len(val_paths)} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create data generators
    batch_size = trainer.model_config['batch_size']
    train_data_gen = data_loader.create_data_generator(train_paths, train_labels, batch_size)
    val_data_gen = data_loader.create_data_generator(val_paths, val_labels, batch_size)
    
    # Calculate steps
    steps_per_epoch = max(100, len(train_paths) // batch_size)
    val_steps = max(50, len(val_paths) // batch_size)
    
    print(f"Training setup:")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Expected time: 20-40 minutes")
    
    # Train model
    try:
        print("\nStarting training...")
        model = trainer.train_model(train_data_gen, val_data_gen, steps_per_epoch, val_steps)
        print("Training completed successfully!")
        print(f"Model saved in: {output_dir}")
        print(f"Loss plots saved in: {output_dir / 'training_logs'}")
        print(f"Test with: python single_cvae_tester.py --model_dir {output_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()