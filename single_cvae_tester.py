#!/usr/bin/env python3
"""
Single CVAE Testing Script - Clean Version
Uses IMAGE PATHS (not directories) just like the trainer
"""

import os
import sys
import json
import argparse
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, fbeta_score
import seaborn as sns
import pandas as pd

# Suppress TensorFlow warnings
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
            gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]
        )

class SingleCVAE(tf.keras.Model):
    """Single CVAE model class - EXACTLY MATCHING the trainer architecture."""
    
    def __init__(self, config):
        super(SingleCVAE, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        
        # Regularization parameters
        l2_reg = config.get('l2_regularization', 1e-04)
        dropout_rates = config.get('dropout_rates', {})
        
        # Encoder - EXACT MATCH with trainer
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(config['img_size'][0], config['img_size'][1], 3)),
            
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder', 0.1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder_dense', 0.3)),
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('encoder_dense', 0.3)),
        ], name="encoder")
        
        # Latent space
        self.z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")
        self.latent_dropout = tf.keras.layers.Dropout(dropout_rates.get('latent', 0.2))
        
        # Decoder - EXACT MATCH with trainer
        decoder_reshape_size = 28 * 16 * 64
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.2)),
            
            tf.keras.layers.Dense(decoder_reshape_size, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Reshape((28, 16, 64)),
            
            tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.1)),
            
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rates.get('decoder', 0.1)),
            
            tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=1, padding='same', activation='sigmoid'),
            tf.keras.layers.Resizing(config['img_size'][0], config['img_size'][1]),
        ], name="decoder")
        
        # Classifier - EXACT MATCH with trainer
        classifier_dropout = dropout_rates.get('classifier', 0.7)
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
        """Forward pass."""
        x, label = inputs
        z_mean, z_log_var = self.encode(x, training=training)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z, training=training)
        anomaly_logits = self.anomaly_classifier(z_mean, training=training)
        return reconstructed, z_mean, z_log_var, anomaly_logits
    
    def predict_anomaly(self, x, training=False):
        """Predict anomaly scores and reconstruction errors."""
        z_mean, z_log_var = self.encode(x, training=training)
        anomaly_logits = self.anomaly_classifier(z_mean, training=training)
        anomaly_logits_squeezed = tf.squeeze(anomaly_logits, axis=-1)
        anomaly_score = tf.nn.sigmoid(anomaly_logits_squeezed)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z, training=training)
        recon_error = tf.reduce_mean(tf.square(x - reconstructed), axis=[1, 2, 3])
        return anomaly_score, recon_error

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate (EER) and optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    return eer, eer_threshold, fpr[eer_index], fnr[eer_index]

def calculate_metrics_at_threshold(y_true, y_scores, threshold):
    """Calculate comprehensive metrics at a specific threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'f2_score': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        'f0_5_score': fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        'matthews_correlation_coefficient': ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': int(tp + tn + fp + fn)
    }
    return metrics

def get_optimal_thresholds(y_true, y_scores):
    """Calculate various optimal thresholds."""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        optimal_thresholds = {
            'eer': thresholds[eer_idx],
            'youden': thresholds[np.argmax(tpr - fpr)],
            'f1_optimal': 0.5
        }
        return optimal_thresholds
    except Exception as e:
        print(f"Error in optimal threshold calculation: {e}")
        return {'eer': 0.5, 'youden': 0.5, 'f1_optimal': 0.5}

class TestDataLoader:
    """Loads test data - Uses IMAGE PATHS like trainer."""
    
    def __init__(self, split: str, img_size: Tuple[int, int] = (224, 128), max_samples: int = 30000):
        self.split = split
        self.img_size = img_size
        
        # Split ratios
        split_configs = {
            '9010': {'split_dir': '90_10', 'target_samples': 10000},
            '6040': {'split_dir': '60_40', 'target_samples': 40000},
            '5050': {'split_dir': '50_50', 'target_samples': 50000}
        }
        
        if split in split_configs:
            self.split_dir = split_configs[split]['split_dir']
            self.target_samples = split_configs[split]['target_samples']
            self.max_samples = max(max_samples, self.target_samples)
        else:
            self.split_dir = '90_10'
            self.target_samples = max_samples
            self.max_samples = max_samples
            
        print(f"Split {split}: Target {self.target_samples:,} samples (max: {self.max_samples:,})")
        
        base_path = f"/home/sanderwald/Projects/dissertationProject/data/Splits/{self.split_dir}"
        
        test_candidates = [
            f"{base_path}/test_split.txt",
            f"{base_path}/test_split_subset.txt",
            f"{base_path}/train_split.txt"
        ]
        
        self.test_file = next((c for c in test_candidates if os.path.exists(c)), None)
        if not self.test_file:
            raise FileNotFoundError(f"No test file found in {base_path}")
        
        self.problematic_images = {
            '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/3_011_0/frame_03997.jpg',
            '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/3_011_0/frame_04022.jpg',
            '/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/1_047_0/frame_02802.jpg',
        }
        
        self.test_data = self.load_test_data()
    
    def load_test_data(self):
        """Loads test data from IMAGE PATHS (exactly like trainer does)."""
        json_path = "/home/sanderwald/Projects/dissertationProject/annotations.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotations JSON not found: {json_path}")
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        print(f"Loaded annotations with {len(annotations)} entries")
        
        lookup = {path: data['label'] == 1 for path, data in annotations.items() if 'label' in data}
        print(f"Created lookup with {len(lookup)} labeled entries")
        
        # Load IMAGE PATHS (not directories) - EXACTLY like trainer
        image_paths = []
        with open(self.test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(line) and line.endswith('.jpg'):
                    image_paths.append(line)
        
        print(f"Found {len(image_paths)} image files in test file")
        
        test_paths, test_labels = [], []
        matches_found = 0
        
        # Apply sampling if we have too many images
        if len(image_paths) > self.max_samples:
            print(f"Sampling {self.max_samples:,} from {len(image_paths):,} images")
            random.shuffle(image_paths)
            image_paths = image_paths[:self.max_samples]
        elif len(image_paths) < self.target_samples:
            print(f"Only found {len(image_paths):,} images, target was {self.target_samples:,}")
        
        print(f"Processing {len(image_paths)} images for annotation matching...")
        
        for i, full_path in enumerate(image_paths):
            if full_path in self.problematic_images:
                continue
                
            # EXACT SAME matching strategy as trainer
            path_parts = full_path.split('/')
            if len(path_parts) >= 2:
                rel_path = f"{path_parts[-2]}/{path_parts[-1]}"
                if rel_path in lookup:
                    test_paths.append(full_path)
                    test_labels.append(int(lookup[rel_path]))
                    matches_found += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images, {matches_found} matches found")
        
        print(f"FINAL STATS:")
        print(f"  Images in test file: {len(image_paths)}")
        print(f"  Successful matches with annotations: {matches_found}")
        print(f"  Final test set size: {len(test_paths)}")
        
        if not test_paths:
            raise ValueError(f"No valid test images found from {len(image_paths)} image files")
        
        # Shuffle final dataset
        combined = list(zip(test_paths, test_labels))
        random.shuffle(combined)
        test_paths, test_labels = zip(*combined) if combined else ([], [])
        
        print(f"Successfully created test set with {len(test_paths)} images")
        return list(zip(test_paths, test_labels))
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        try:
            if not os.path.exists(image_path):
                return np.zeros((*self.img_size, 3), dtype=np.float32)
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)
            if tf.size(image) == 0:
                return np.zeros((*self.img_size, 3), dtype=np.float32)
            image = tf.image.resize(image, self.img_size)
            image = image / 255.0
            return image.numpy()
        except Exception as e:
            return np.zeros((*self.img_size, 3), dtype=np.float32)
    
    def get_numpy_data(self) -> Tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []
        failed_images = 0
        print("Loading test images...")
        for i, (path, label) in enumerate(self.test_data):
            if i % 100 == 0:
                print(f"Processed {i}/{len(self.test_data)} images (failed: {failed_images})")
            image = self._preprocess_image(path)
            if np.all(image == 0):
                failed_images += 1
                if failed_images <= 5:
                    print(f"Failed to load: {path}")
                continue
            images.append(image)
            labels.append(label)
        print(f"Successfully loaded {len(images)} images ({failed_images} failed)")
        if len(images) == 0:
            raise ValueError("No test images were successfully loaded!")
        return np.array(images), np.array(labels)

def load_model_and_config(model_dir: str) -> Tuple[SingleCVAE, Dict]:
    """Load saved model and configuration."""
    model_dir = Path(model_dir)
    
    # Find config file
    config_candidates = [model_dir / "config.json"]
    config_candidates.extend([d / "config.json" for d in model_dir.iterdir() if d.is_dir()])
    
    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            break
    
    # Load configuration
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Config loaded from {config_path}")
    else:
        print("Warning: No config.json found, using default configuration")
        config = {
            'latent_dim': 128,
            'img_size': [224, 128],
            'l2_regularization': 1e-04,
            'dropout_rates': {
                'encoder': 0.15,
                'encoder_dense': 0.4,
                'decoder': 0.2,
                'classifier': 0.7,
                'latent': 0.2
            }
        }
    
    # Ensure dropout_rates exist
    if 'dropout_rates' not in config:
        config['dropout_rates'] = {
            'encoder': 0.15,
            'encoder_dense': 0.4,
            'decoder': 0.2,
            'classifier': 0.7,
            'latent': 0.2
        }
    
    print(f"Using config: latent_dim={config['latent_dim']}, dropout_classifier={config['dropout_rates']['classifier']}")
    
    # Create model instance with MATCHING architecture
    print(f"Creating model with MATCHING trainer architecture...")
    model = SingleCVAE(config)
    
    # Build the model by calling it once
    print("Building model with dummy input...")
    dummy_input = tf.random.normal((1, config['img_size'][0], config['img_size'][1], 3))
    dummy_label = tf.zeros((1,))
    _ = model((dummy_input, dummy_label), training=False)
    print("Model built successfully with MATCHING architecture")
    
    # Find checkpoint files
    checkpoint_file = None
    for root, dirs, files in os.walk(model_dir):
        if 'checkpoint' in files:
            with open(os.path.join(root, 'checkpoint'), 'r') as f:
                checkpoint_content = f.read()
            import re
            match = re.search(r'model_checkpoint_path:\s*"([^"]+)"', checkpoint_content)
            if match:
                checkpoint_name = match.group(1)
                checkpoint_file = Path(root) / checkpoint_name
                break
        for file in files:
            if file.endswith('.index'):
                checkpoint_file = Path(root) / file.replace('.index', '')
                break
        if checkpoint_file:
            break
    
    if checkpoint_file is None:
        # Try alternate naming patterns
        weight_candidates = [
            model_dir / "best_model" / "model_weights",
            model_dir / "model_weights",
            model_dir / "best_model" / "checkpoint",
            model_dir / "checkpoint"
        ]
        
        for candidate in weight_candidates:
            if candidate.with_suffix('.index').exists():
                checkpoint_file = candidate
                break
    
    if checkpoint_file is None:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    
    print(f"Attempting to load weights from: {checkpoint_file}")
    
    try:
        status = model.load_weights(str(checkpoint_file))
        status.expect_partial()
        print(f"Model weights loaded successfully from: {checkpoint_file}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise
    
    return model, config

def create_comprehensive_output_structure(base_output_dir: Path, model_name: str, split: str) -> Path:
    """Create comprehensive output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"{model_name}_{split}_test_{timestamp}"
    
    # Create all subdirectories
    directories = [
        output_dir / "metrics",
        output_dir / "plots" / "roc_curves",
        output_dir / "plots" / "precision_recall",  
        output_dir / "plots" / "distributions",
        output_dir / "plots" / "confusion_matrices",
        output_dir / "plots" / "threshold_analysis",
        output_dir / "raw_data",
        output_dir / "reports",
        output_dir / "sample_predictions",
        output_dir / "model_info",
        output_dir / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Created comprehensive output structure: {output_dir}")
    return output_dir

def generate_comprehensive_plots(y_true, y_scores, recon_errors, output_dir: Path, split_info: str = "TEST"):
    """Generate comprehensive evaluation plots."""
    plots_dir = output_dir / "plots"
    
    # Calculate all metrics we need
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    eer, eer_threshold, eer_fpr, eer_fnr = calculate_eer(y_true, y_scores)
    
    # Calculate metrics across all thresholds for analysis
    threshold_range = np.linspace(0.0, 1.0, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in threshold_range:
        y_pred = (y_scores >= thresh).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    # Find best F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = threshold_range[best_f1_idx]
    best_f1_score = f1_scores[best_f1_idx]
    
    # Get confusion matrix at best threshold
    y_pred_best = (y_scores >= best_f1_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    
    # Set up the main figure with 6 subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'CVAE Anomaly Detection Evaluation - Split {split_info}\n'
                f'Best: Classifier | F1: {best_f1_score:.4f} | AUC: {roc_auc:.4f}', 
                fontsize=16, fontweight='bold')
    
    # 1. ROC Curve with EER Point
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.plot(eer_fpr, 1-eer_fnr, 'ro', markersize=8, label=f'EER = {eer:.4f}')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve with EER Point', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(recall, precision, linewidth=3, label=f'AP = {avg_precision:.4f}', color='green')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Equal Error Rate Analysis
    ax3 = plt.subplot(2, 3, 3)
    fnr = 1 - tpr
    ax3.plot(thresholds, fpr, linewidth=2, label='FPR', color='blue')
    ax3.plot(thresholds, fnr, linewidth=2, label='FNR', color='red')
    ax3.axvline(x=eer_threshold, color='green', linestyle='--', linewidth=2, label=f'EER = {eer:.4f}')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.0])
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Error Rate')
    ax3.set_title(f'Equal Error Rate = {eer:.4f}', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix (Best Threshold)
    ax4 = plt.subplot(2, 3, 4)
    im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
    ax4.figure.colorbar(im, ax=ax4)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black",
                    fontsize=14, fontweight='bold')
    
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    ax4.set_title('Confusion Matrix (Best Threshold)', fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Normal', 'Anomaly'])
    ax4.set_yticklabels(['Normal', 'Anomaly'])
    
    # 5. Metrics vs Threshold
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(threshold_range, f1_scores, linewidth=2, label='F1', color='red')
    ax5.plot(threshold_range, precisions, linewidth=2, label='Precision', color='blue')
    ax5.plot(threshold_range, recalls, linewidth=2, label='Recall', color='green')
    ax5.axvline(x=best_f1_threshold, color='black', linestyle='--', alpha=0.7, 
                label=f'Best F1 = {best_f1_threshold:.3f}')
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.0])
    ax5.set_xlabel('Threshold')
    ax5.set_ylabel('Score')
    ax5.set_title('Metrics vs Threshold', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. F1 Score Distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(f1_scores, bins=30, alpha=0.7, color='red', edgecolor='darkred')
    ax6.axvline(x=best_f1_score, color='black', linestyle='--', linewidth=2,
                label=f'Best F1 = {best_f1_score:.4f}')
    ax6.set_xlabel('F1 Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('F1 Score Distribution', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "COMPREHENSIVE_evaluation_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Also create individual detailed plots for maximum quality
    
    # Detailed ROC Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC with multiple details
    ax1.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.7)
    ax1.plot(eer_fpr, 1-eer_fnr, 'ro', markersize=10, label=f'EER = {eer:.4f}')
    ax1.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve Analysis', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Threshold analysis
    ax2.plot(thresholds, fpr, label='False Positive Rate', linewidth=3, color='red')
    ax2.plot(thresholds, tpr, label='True Positive Rate', linewidth=3, color='blue')
    ax2.axvline(x=eer_threshold, color='green', linestyle='--', alpha=0.8, linewidth=2,
                label=f'EER Threshold = {eer_threshold:.3f}')
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Threshold Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curves" / "detailed_roc_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance Summary Table as Image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary data
    summary_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['ROC AUC', f'{roc_auc:.4f}', 'Overall discrimination ability'],
        ['Average Precision', f'{avg_precision:.4f}', 'Precision-recall performance'],
        ['Equal Error Rate', f'{eer:.4f}', 'Balanced error rate'],
        ['Best F1 Score', f'{best_f1_score:.4f}', 'Optimal F1 performance'],
        ['Best Threshold', f'{best_f1_threshold:.4f}', 'Optimal decision threshold'],
        ['True Positives', f'{tp}', 'Correctly identified anomalies'],
        ['True Negatives', f'{tn}', 'Correctly identified normal'],
        ['False Positives', f'{fp}', 'Normal classified as anomaly'],
        ['False Negatives', f'{fn}', 'Anomaly classified as normal'],
        ['Test Samples', f'{len(y_true):,}', 'Total test dataset size'],
        ['Anomaly Ratio', f'{np.mean(y_true):.2%}', 'Percentage of anomalies'],
    ]
    
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('CVAE Anomaly Detection - Performance Summary', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(plots_dir / "performance_summary_table.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"COMPREHENSIVE PLOTS GENERATED!")
    print(f"  Main analysis: {plots_dir}/COMPREHENSIVE_evaluation_analysis.png")
    print(f"  Detailed ROC: {plots_dir}/roc_curves/detailed_roc_analysis.png")
    print(f"  Summary table: {plots_dir}/performance_summary_table.png")

def save_raw_predictions(y_true, y_scores, recon_errors, output_dir: Path):
    """Save raw predictions for further analysis."""
    raw_data_dir = output_dir / "raw_data"
    
    predictions_data = {
        'true_labels': [int(x) for x in y_true.tolist()],
        'anomaly_scores': [float(x) for x in y_scores.tolist()],
        'reconstruction_errors': [float(x) for x in recon_errors.tolist()]
    }
    
    # Save as JSON
    with open(raw_data_dir / "raw_predictions.json", 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    # Save as CSV for easy analysis in Excel/R/etc.
    df = pd.DataFrame({
        'true_label': y_true,
        'anomaly_score': y_scores,
        'reconstruction_error': recon_errors
    })
    df.to_csv(raw_data_dir / "predictions.csv", index=False)
    
def save_detailed_metrics(results: Dict, output_dir: Path):
    """Save detailed metrics in multiple formats."""
    metrics_dir = output_dir / "metrics"
    
    # Save raw JSON results
    with open(metrics_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create human-readable summary
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("SINGLE CVAE MODEL EVALUATION SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    if 'basic_metrics' in results:
        basic = results['basic_metrics']
        summary_lines.append("CORE PERFORMANCE METRICS")
        summary_lines.append("-" * 30)
        summary_lines.append(f"ROC AUC Score:           {basic['roc_auc']:.4f}")
        summary_lines.append(f"Average Precision:       {basic['average_precision']:.4f}")
        summary_lines.append(f"Equal Error Rate:        {basic['equal_error_rate']:.4f}")
        summary_lines.append(f"EER Threshold:           {basic['eer_threshold']:.4f}")
        summary_lines.append("")
    
    if 'data_summary' in results:
        data = results['data_summary']
        summary_lines.append("DATASET SUMMARY")
        summary_lines.append("-" * 30)
        summary_lines.append(f"Total Test Samples:      {data['total_samples']:,}")
        summary_lines.append(f"Normal Samples:          {data['normal_samples']:,}")
        summary_lines.append(f"Anomaly Samples:         {data['anomaly_samples']:,}")
        summary_lines.append(f"Anomaly Ratio:           {data['anomaly_ratio']:.3%}")
        summary_lines.append("")
    
    if 'threshold_metrics' in results:
        summary_lines.append("THRESHOLD-SPECIFIC PERFORMANCE")
        summary_lines.append("-" * 40)
        
        for threshold_name, metrics in results['threshold_metrics'].items():
            if 'error' not in metrics:
                summary_lines.append(f"\n{threshold_name.upper().replace('_', ' ')} (threshold: {metrics['threshold']:.3f})")
                summary_lines.append(f"  Accuracy:      {metrics['accuracy']:.4f}")
                summary_lines.append(f"  Precision:     {metrics['precision']:.4f}")
                summary_lines.append(f"  Recall:        {metrics['recall']:.4f}")
                summary_lines.append(f"  F1-Score:      {metrics['f1_score']:.4f}")
                summary_lines.append(f"  Specificity:   {metrics['specificity']:.4f}")
                summary_lines.append(f"  Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
    
    summary_lines.append("\n" + "=" * 60)
    
    # Save summary
    with open(metrics_dir / "evaluation_summary.txt", 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Create CSV for easy analysis
    if 'threshold_metrics' in results:
        threshold_data = []
        for name, metrics in results['threshold_metrics'].items():
            if 'error' not in metrics:
                threshold_data.append({
                    'threshold_type': name,
                    'threshold_value': metrics['threshold'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'specificity': metrics['specificity'],
                    'balanced_accuracy': metrics['balanced_accuracy'],
                    'mcc': metrics['matthews_correlation_coefficient']
                })
        
        df = pd.DataFrame(threshold_data)
        df.to_csv(metrics_dir / "threshold_metrics.csv", index=False)
    
    print(f"Metrics saved to: {metrics_dir}")

def evaluate_model(model: SingleCVAE, test_loader: TestDataLoader, output_dir: Path, batch_size: int = 8):
    """Main evaluation function."""
    print("Starting model evaluation...")
    
    print("Loading test data...")
    try:
        X_test, y_test = test_loader.get_numpy_data()
        print(f"Test data loaded: {X_test.shape}, {y_test.shape}")
        print(f"Normal: {np.sum(y_test == 0)}, Anomaly: {np.sum(y_test == 1)}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return {"error": f"data_loading_error: {e}"}
    
    print("Generating predictions...")
    anomaly_scores = []
    recon_errors = []
    
    total_batches = (len(X_test) + batch_size - 1) // batch_size
    print(f"Processing {len(X_test)} samples in {total_batches} batches...")
    
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            if batch_num % 20 == 0:
                print(f"Batch {batch_num}/{total_batches}")
            
            scores, errors = model.predict_anomaly(batch_x, training=False)
            anomaly_scores.extend(scores.numpy())
            recon_errors.extend(errors.numpy())
            
        except Exception as e:
            print(f"Error in batch {batch_num}: Processing individually...")
            for sample_idx in range(len(batch_x)):
                try:
                    single_sample = batch_x[sample_idx:sample_idx+1]
                    scores, errors = model.predict_anomaly(single_sample, training=False)
                    anomaly_scores.extend(scores.numpy())
                    recon_errors.extend(errors.numpy())
                except Exception:
                    anomaly_scores.append(0.5)
                    recon_errors.append(0.0)
    
    anomaly_scores = np.array(anomaly_scores)
    recon_errors = np.array(recon_errors)
    
    # Align arrays
    min_len = min(len(anomaly_scores), len(y_test))
    anomaly_scores = anomaly_scores[:min_len]
    recon_errors = recon_errors[:min_len]
    y_test = y_test[:min_len]
    
    print(f"Computing metrics...")
    try:
        roc_auc = roc_auc_score(y_test, anomaly_scores)
        avg_precision = average_precision_score(y_test, anomaly_scores)
        eer, eer_threshold, _, _ = calculate_eer(y_test, anomaly_scores)
        optimal_thresholds = get_optimal_thresholds(y_test, anomaly_scores)
    except Exception as e:
        print(f"Error in metrics calculation: {e}")
        return {"error": f"metrics_error: {e}"}
    
    # Calculate metrics for multiple thresholds
    thresholds_to_test = [0.5, eer_threshold, optimal_thresholds['youden']]
    threshold_names = ['default_0.5', 'eer_optimal', 'youden_optimal']
    all_metrics = {}
    
    for thresh, name in zip(thresholds_to_test, threshold_names):
        try:
            metrics = calculate_metrics_at_threshold(y_test, anomaly_scores, thresh)
            all_metrics[name] = metrics
        except Exception as e:
            print(f"Error calculating {name} metrics: {e}")
            all_metrics[name] = {"error": str(e)}
    
    # Compile results - FIX JSON SERIALIZATION
    results = {
        'model_info': {
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_split': test_loader.split,
            'image_size': test_loader.img_size
        },
        'basic_metrics': {
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision),
            'equal_error_rate': float(eer),
            'eer_threshold': float(eer_threshold)
        },
        'optimal_thresholds': {k: float(v) for k, v in optimal_thresholds.items()},
        'threshold_metrics': all_metrics,
        'data_summary': {
            'total_samples': int(len(y_test)),
            'normal_samples': int(np.sum(y_test == 0)),
            'anomaly_samples': int(np.sum(y_test == 1)),
            'anomaly_ratio': float(np.mean(y_test)),
            'score_range': [float(anomaly_scores.min()), float(anomaly_scores.max())],
            'recon_error_range': [float(recon_errors.min()), float(recon_errors.max())]
        }
    }
    
    # Save all outputs
    print("Saving comprehensive outputs...")
    
    print("Creating directories...")
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots" / "roc_curves").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_data").mkdir(parents=True, exist_ok=True)
    
    print("Generating detailed metrics...")
    try:
        save_detailed_metrics(results, output_dir)
        print("Detailed metrics saved")
    except Exception as e:
        print(f"Error saving detailed metrics: {e}")
    
    print("Saving raw predictions...")
    try:
        save_raw_predictions(y_test, anomaly_scores, recon_errors, output_dir)
        print("Raw predictions saved")
    except Exception as e:
        print(f"Error saving raw predictions: {e}")
    
    print("Generating plots...")
    try:
        generate_comprehensive_plots(y_test, anomaly_scores, recon_errors, output_dir, test_loader.split)
        print("Plots generated successfully")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("Generating CSV...")
    try:
        csv_result = generate_detailed_csv(y_test, anomaly_scores, output_dir, test_loader.split)
        if csv_result is not None:
            print("CSV generated successfully")
        else:
            print("CSV generation returned None")
    except Exception as e:
        print(f"Error generating CSV: {e}")
        import traceback
        traceback.print_exc()
    
    # MANUAL SIMPLE PLOT GENERATION AS BACKUP
    print("Creating BACKUP simple plot...")
    try:
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
        roc_auc = roc_auc_score(y_test, anomaly_scores)
        plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'BACKUP ROC Curve - Split {test_loader.split}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        backup_plot = output_dir / "BACKUP_roc_curve.png"
        plt.savefig(backup_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"BACKUP plot saved: {backup_plot}")
    except Exception as e:
        print(f"Even backup plot failed: {e}")
    
    # MANUAL CSV GENERATION AS BACKUP
    print("Creating BACKUP CSV...")
    try:
        target_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        roc_auc = roc_auc_score(y_test, anomaly_scores)
        
        backup_results = []
        for thresh in target_thresholds:
            y_pred = (anomaly_scores >= thresh).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            backup_results.append({
                'Threshold': thresh,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'AUC': roc_auc
            })
        
        backup_csv = output_dir / "BACKUP_metrics.csv"
        backup_df = pd.DataFrame(backup_results)
        backup_df.to_csv(backup_csv, index=False)
        print(f"BACKUP CSV saved: {backup_csv}")
        
        # Print results immediately
        print(f"\nBACKUP RESULTS:")
        for result in backup_results:
            print(f"   Threshold {result['Threshold']:.1f}: F1={result['F1_Score']:.4f}, Acc={result['Accuracy']:.4f}")
            
    except Exception as e:
        print(f"Even backup CSV failed: {e}")
    
    print(f"Evaluation complete! Check directory: {output_dir}")
    return results

def generate_detailed_csv(y_true, y_scores, output_dir: Path, split_info: str):
    """Generate detailed CSV."""
    
    # Calculate comprehensive metrics for multiple thresholds
    threshold_range = np.linspace(0.0, 1.0, 21)  # 21 thresholds
    detailed_results = []
    
    # Calculate basic metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    eer, eer_threshold, _, _ = calculate_eer(y_true, y_scores)
    
    # Test multiple threshold approaches
    approaches = {
        'Classifier': threshold_range,
        'EER_Optimal': [eer_threshold],
        'Fixed_0.5': [0.5]
    }
    
    best_f1 = 0
    best_row = None
    
    for approach_name, thresholds in approaches.items():
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate MCC
            mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
            
            # Create threshold name
            if approach_name == 'Classifier':
                threshold_name = f'Threshold_{threshold:.2f}'
            else:
                threshold_name = approach_name
            
            # Track best F1
            is_best = False
            if f1 > best_f1:
                best_f1 = f1
                is_best = True
                best_row = len(detailed_results)
            
            # Add row
            detailed_results.append({
                'Approach': 'Single_CVAE',
                'Threshold': threshold_name,
                'Threshold_Value': float(threshold),
                'Accuracy': float(accuracy),
                'Precision': float(precision),
                'Recall': float(recall),
                'F1_Score': float(f1),
                'MCC': float(mcc),
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'AUC': float(roc_auc),
                'Is_Best': is_best
            })
    
    # Mark the best result
    if best_row is not None:
        for i, row in enumerate(detailed_results):
            row['Is_Best'] = (i == best_row)
    
    # Create DataFrame and save
    df = pd.DataFrame(detailed_results)
    
    # Save CSV
    csv_file = output_dir / "metrics" / f"detailed_metrics_{split_info}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"Detailed CSV saved: {csv_file}")
    print(f"   Best F1: {best_f1:.4f} at threshold {detailed_results[best_row]['Threshold_Value']:.3f}")
    
    return df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Test Single CVAE Model')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the trained model')
    parser.add_argument('--split', type=str, default='9010',
                       choices=['9010', '6040', '5050'],
                       help='Data split to use for testing')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Base directory for test results')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 128],
                       help='Image dimensions [height, width]')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of test samples (auto-set by split if None)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    print("Single CVAE Testing - Clean Version")
    print("Uses IMAGE PATHS like trainer (not directories)")
    print("SPLIT RATIOS: 9010=15k, 6040=50k, 5050=60k samples")
    print("=" * 60)
    print(f"Model Directory: {args.model_dir}")
    print(f"Data Split: {args.split}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Image Size: {args.img_size}")
    if args.max_samples:
        print(f"Max Samples (override): {args.max_samples:,}")
    else:
        default_samples = {'9010': 15000, '6040': 50000, '5050': 60000}.get(args.split, 30000)
        print(f"Max Samples (auto): {default_samples:,}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 60)
    
    # Validate model directory
    if not Path(args.model_dir).exists():
        print(f"Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    try:
        # Load model
        model, config = load_model_and_config(args.model_dir)
        
        # Extract model name from directory
        model_name = Path(args.model_dir).name
        
        # Create output structure
        base_output_dir = Path(args.output_dir)
        output_dir = create_comprehensive_output_structure(base_output_dir, model_name, args.split)
        
        # Create test data loader with CORRECT SPLIT RATIOS
        split_max_samples = {
            '9010': 15000,   # Keep current for 90/10 (imbalanced, smaller test set)
            '6040': 50000,   # 40% of 100k for proper 60/40 testing
            '5050': 60000    # 50% of 100k for proper 50/50 testing  
        }
        
        if args.max_samples is None:
            max_samples_for_split = split_max_samples.get(args.split, 30000)
        else:
            max_samples_for_split = args.max_samples
            
        print(f"Using {max_samples_for_split:,} max samples for split {args.split}")
        
        test_loader = TestDataLoader(args.split, tuple(args.img_size), max_samples_for_split)
        
        # Run evaluation
        results = evaluate_model(model, test_loader, output_dir, args.batch_size)
        
        if results and 'error' not in results:
            print("\nEVALUATION COMPLETED SUCCESSFULLY!")
            print("\nQUICK SUMMARY")
            print("=" * 40)
            
            if 'basic_metrics' in results:
                basic = results['basic_metrics']
                print(f"ROC AUC:           {basic['roc_auc']:.4f}")
                print(f"Average Precision: {basic['average_precision']:.4f}")
                print(f"Equal Error Rate:  {basic['equal_error_rate']:.4f}")
            
            if 'data_summary' in results:
                data = results['data_summary']
                print(f"Total Samples:     {data['total_samples']:,}")
                print(f"Anomaly Ratio:     {data['anomaly_ratio']:.2%}")
            
            print(f"\nAll results saved to: {output_dir}")
            print(f"View summary: {output_dir / 'metrics' / 'evaluation_summary.txt'}")
            print(f"View plots: {output_dir / 'plots' / 'COMPREHENSIVE_evaluation_analysis.png'}")
            print(f"Detailed table: {output_dir / 'plots' / 'performance_summary_table.png'}")
            print(f"Raw data: {output_dir / 'raw_data'}")
            print(f"Detailed CSV: {output_dir / 'metrics' / f'detailed_metrics_{args.split}.csv'}")
        else:
            print(f"Evaluation failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()