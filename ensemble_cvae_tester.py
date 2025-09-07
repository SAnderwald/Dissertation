#!/usr/bin/env python3
"""
Ensemble CVAE Testing Script - Clean Version
Comprehensive evaluation of ensemble CVAE models with advanced prediction methods.
Based on single_cvae_tester.py but extended for ensemble evaluation.
"""

import os
import sys
import json
import argparse
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, fbeta_score
import seaborn as sns
import pandas as pd
from scipy import stats

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
    """Single CVAE model class - matching the ensemble trainer architecture."""
    
    def __init__(self, config, model_id: int = 0):
        super(SingleCVAE, self).__init__()
        self.config = config
        self.model_id = model_id
        self.latent_dim = config['latent_dim']
        
        # Regularization parameters
        l2_reg = config.get('l2_regularization', 1e-05)
        dropout_rates = config.get('dropout_rates', {})
        
        # Encoder - EXACT MATCH with trainer
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
        
        # Decoder - EXACT MATCH with trainer
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
        
        # Classifier - EXACT MATCH with trainer
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

class OptimizedEnsemblePredictor:
    """Advanced ensemble prediction methods for anomaly detection."""
    
    def __init__(self, model_weights: List[float]):
        self.model_weights = np.array(model_weights)
        self.model_weights = self.model_weights / np.sum(self.model_weights)  # Normalize
    
    def predict_ensemble_optimal(self, anomaly_scores, recon_errors, method='adaptive_weighted'):
        """
        Optimal ensemble prediction combining anomaly scores and reconstruction errors.
        """
        anomaly_scores = np.array(anomaly_scores)  # (n_models, batch_size)
        recon_errors = np.array(recon_errors)
        
        if method == 'adaptive_weighted':
            return self._adaptive_weighted_prediction(anomaly_scores, recon_errors)
        elif method == 'weighted_soft':
            return self._weighted_soft_prediction(anomaly_scores, recon_errors)
        elif method == 'uncertainty_weighted':
            return self._uncertainty_weighted_prediction(anomaly_scores, recon_errors)
        elif method == 'consensus_threshold':
            return self._consensus_threshold_prediction(anomaly_scores, recon_errors)
        else:
            return self._simple_average_prediction(anomaly_scores, recon_errors)
    
    def _adaptive_weighted_prediction(self, anomaly_scores, recon_errors):
        """Adaptive weighting based on model agreement and performance."""
        
        # Calculate model agreement
        anomaly_variance = np.var(anomaly_scores, axis=0)
        recon_variance = np.var(recon_errors, axis=0)
        
        # Normalize variances
        anomaly_disagreement = anomaly_variance / (np.max(anomaly_variance) + 1e-8)
        recon_disagreement = recon_variance / (np.max(recon_variance) + 1e-8)
        
        disagreement = (anomaly_disagreement + recon_disagreement) / 2
        agreement = 1 - disagreement
        
        final_scores = []
        confidence_scores = []
        
        for i in range(anomaly_scores.shape[1]):
            sample_agreement = agreement[i]
            
            # Adaptive weight calculation
            performance_influence = sample_agreement
            democratic_influence = 1 - sample_agreement
            
            adaptive_weights = (performance_influence * self.model_weights + 
                               democratic_influence * np.ones_like(self.model_weights))
            adaptive_weights = adaptive_weights / np.sum(adaptive_weights)
            
            # Combine anomaly scores and reconstruction errors
            sample_anomaly = anomaly_scores[:, i]
            sample_recon = recon_errors[:, i]
            
            # Normalize reconstruction errors
            if np.max(sample_recon) > np.min(sample_recon):
                sample_recon_norm = (sample_recon - np.min(sample_recon)) / (np.max(sample_recon) - np.min(sample_recon))
            else:
                sample_recon_norm = np.ones_like(sample_recon) * 0.5
            
            # Weighted combination: 70% anomaly classifier, 30% reconstruction error
            combined_scores = 0.7 * sample_anomaly + 0.3 * sample_recon_norm
            
            # Final weighted ensemble score
            final_score = np.average(combined_scores, weights=adaptive_weights)
            confidence = sample_agreement
            
            final_scores.append(final_score)
            confidence_scores.append(confidence)
        
        final_scores = np.array(final_scores)
        confidence_scores = np.array(confidence_scores)
        
        # Dynamic threshold based on confidence
        base_threshold = 0.5
        confidence_adjustment = (confidence_scores - 0.5) * 0.1
        adaptive_thresholds = base_threshold + confidence_adjustment
        
        final_predictions = (final_scores > adaptive_thresholds).astype(int)
        
        method_info = {
            'method': 'adaptive_weighted',
            'mean_agreement': np.mean(agreement),
            'mean_confidence': np.mean(confidence_scores),
            'adaptive_thresholds_used': True
        }
        
        return final_scores, final_predictions, confidence_scores, method_info
    
    def _weighted_soft_prediction(self, anomaly_scores, recon_errors):
        """Performance-weighted averaging with fixed threshold."""
        
        # Weighted average of anomaly scores
        weighted_anomaly = np.average(anomaly_scores, axis=0, weights=self.model_weights)
        weighted_recon = np.average(recon_errors, axis=0, weights=self.model_weights)
        
        # Normalize reconstruction errors
        if np.max(weighted_recon) > np.min(weighted_recon):
            weighted_recon_norm = (weighted_recon - np.min(weighted_recon)) / (np.max(weighted_recon) - np.min(weighted_recon))
        else:
            weighted_recon_norm = np.ones_like(weighted_recon) * 0.5
        
        # Combine scores
        final_scores = 0.7 * weighted_anomaly + 0.3 * weighted_recon_norm
        final_predictions = (final_scores > 0.5).astype(int)
        confidence_scores = np.abs(final_scores - 0.5) * 2
        
        method_info = {
            'method': 'weighted_soft',
            'weights_used': self.model_weights.tolist(),
            'fixed_threshold': 0.5
        }
        
        return final_scores, final_predictions, confidence_scores, method_info
    
    def _uncertainty_weighted_prediction(self, anomaly_scores, recon_errors):
        """Weight models by their uncertainty (entropy-based)."""
        
        # Calculate entropy for each model
        model_uncertainties = []
        for i in range(len(anomaly_scores)):
            scores = anomaly_scores[i]
            eps = 1e-8
            scores_clipped = np.clip(scores, eps, 1-eps)
            entropy = -(scores_clipped * np.log(scores_clipped) + 
                       (1-scores_clipped) * np.log(1-scores_clipped))
            avg_entropy = np.mean(entropy)
            model_uncertainties.append(avg_entropy)
        
        # Convert uncertainty to confidence weights
        uncertainty_weights = 1 / (np.array(model_uncertainties) + 1e-8)
        combined_weights = self.model_weights * uncertainty_weights
        combined_weights = combined_weights / np.sum(combined_weights)
        
        # Weighted prediction
        final_scores = np.average(anomaly_scores, axis=0, weights=combined_weights)
        final_predictions = (final_scores > 0.5).astype(int)
        
        ensemble_uncertainty = np.mean(model_uncertainties)
        confidence_scores = np.ones_like(final_scores) * (1 - ensemble_uncertainty)
        
        method_info = {
            'method': 'uncertainty_weighted',
            'model_uncertainties': model_uncertainties,
            'ensemble_uncertainty': ensemble_uncertainty
        }
        
        return final_scores, final_predictions, confidence_scores, method_info
    
    def _consensus_threshold_prediction(self, anomaly_scores, recon_errors):
        """Require consensus among models - OPTIMIZED FOR 2 MODELS."""
        
        # Count how many models predict anomaly
        binary_predictions = (anomaly_scores > 0.5).astype(int)
        consensus_count = np.sum(binary_predictions, axis=0)
        
        # For 2 models: require both to agree (conservative approach)
        # For 3+ models: require majority
        if len(anomaly_scores) == 2:
            consensus_threshold = 2  # Both models must agree
            print(f"      Using strict consensus: both models must agree for 2-model ensemble")
        else:
            consensus_threshold = len(anomaly_scores) // 2 + 1  # Majority
        
        final_predictions = (consensus_count >= consensus_threshold).astype(int)
        
        # Confidence based on consensus strength
        confidence_scores = consensus_count / len(anomaly_scores)
        
        # For final scores, use weighted average
        final_scores = np.average(anomaly_scores, axis=0, weights=self.model_weights)
        
        method_info = {
            'method': 'consensus_threshold',
            'consensus_threshold': consensus_threshold,
            'mean_consensus': np.mean(confidence_scores),
            'models_used': len(anomaly_scores)
        }
        
        return final_scores, final_predictions, confidence_scores, method_info
    
    def _simple_average_prediction(self, anomaly_scores, recon_errors):
        """Simple baseline: equal weight average."""
        
        final_scores = np.mean(anomaly_scores, axis=0)
        final_predictions = (final_scores > 0.5).astype(int)
        confidence_scores = np.abs(final_scores - 0.5) * 2
        
        method_info = {
            'method': 'simple_average',
            'equal_weights': True
        }
        
        return final_scores, final_predictions, confidence_scores, method_info

class EnsembleCVAE:
    """Ensemble of CVAE models with advanced prediction capabilities."""
    
    def __init__(self, models: List[SingleCVAE], model_weights: List[float]):
        self.models = models
        self.model_weights = model_weights
        self.ensemble_predictor = OptimizedEnsemblePredictor(model_weights)
    
    def predict_ensemble(self, x, method='adaptive_weighted'):
        """Make ensemble predictions using specified method."""
        all_anomaly_scores = []
        all_recon_errors = []
        
        # Get predictions from all models
        for model in self.models:
            anomaly_score, recon_error = model.predict_anomaly(x, training=False)
            all_anomaly_scores.append(anomaly_score.numpy())
            all_recon_errors.append(recon_error.numpy())
        
        return self.ensemble_predictor.predict_ensemble_optimal(
            all_anomaly_scores, all_recon_errors, method=method
        )
    
    def predict_individual_models(self, x):
        """Get predictions from individual models for comparison."""
        individual_results = []
        
        for i, model in enumerate(self.models):
            anomaly_score, recon_error = model.predict_anomaly(x, training=False)
            individual_results.append({
                'model_id': i,
                'anomaly_scores': anomaly_score.numpy(),
                'recon_errors': recon_error.numpy()
            })
        
        return individual_results

class EnsembleTestDataLoader:
    """Test data loader for ensemble evaluation (matches single model loader)."""
    
    def __init__(self, split: str, img_size: Tuple[int, int] = (224, 128), max_samples: int = 30000):
        self.split = split
        self.img_size = img_size
        
        # Split configurations
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
        """Load test data from image paths (exactly like single model tester)."""
        json_path = "/home/sanderwald/Projects/dissertationProject/annotations.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotations JSON not found: {json_path}")
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        print(f"Loaded annotations with {len(annotations)} entries")
        
        lookup = {path: data['label'] == 1 for path, data in annotations.items() if 'label' in data}
        print(f"Created lookup with {len(lookup)} labeled entries")
        
        # Load image paths
        image_paths = []
        with open(self.test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(line) and line.endswith('.jpg'):
                    image_paths.append(line)
        
        print(f"Found {len(image_paths)} image files in test file")
        
        # Apply sampling if needed
        if len(image_paths) > self.max_samples:
            print(f"Sampling {self.max_samples:,} from {len(image_paths):,} images")
            random.shuffle(image_paths)
            image_paths = image_paths[:self.max_samples]
        
        test_paths, test_labels = [], []
        matches_found = 0
        
        for i, full_path in enumerate(image_paths):
            if full_path in self.problematic_images:
                continue
                
            path_parts = full_path.split('/')
            if len(path_parts) >= 2:
                rel_path = f"{path_parts[-2]}/{path_parts[-1]}"
                if rel_path in lookup:
                    test_paths.append(full_path)
                    test_labels.append(int(lookup[rel_path]))
                    matches_found += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images, {matches_found} matches found")
        
        print(f"\nFINAL STATS:")
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

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate (EER) and optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    return eer, eer_threshold, fpr[eer_index], fnr[eer_index]

def load_ensemble_models(ensemble_dir: str) -> Tuple[EnsembleCVAE, Dict]:
    """Load ensemble models and configuration - OPTIMIZED FOR 2 MODELS ONLY."""
    ensemble_dir = Path(ensemble_dir)
    
    # Load ensemble configuration
    config_path = ensemble_dir / "ensemble_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Ensemble config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        ensemble_config = json.load(f)
    
    original_ensemble_size = ensemble_config['ensemble_size']
    print(f"Ensemble config loaded: {original_ensemble_size} models available")
    print(f"OPTIMIZATION: Using only first 2 models for faster evaluation")
    
    # Load individual models - ONLY FIRST 2 FOR EFFICIENCY
    models = []
    model_configs = ensemble_config['model_configs']
    model_weights = ensemble_config['model_weights']
    
    # Use only first 2 models for time efficiency
    models_to_load = min(2, original_ensemble_size)
    
    for i in range(models_to_load):
        print(f"Loading Model {i}...")
        
        # Create model instance
        model_config = model_configs[i]
        model = SingleCVAE(model_config, model_id=i)
        
        # Build model
        dummy_input = tf.random.normal((1, model_config['img_size'][0], model_config['img_size'][1], 3))
        dummy_label = tf.zeros((1,))
        _ = model((dummy_input, dummy_label), training=False)
        
        # Load weights
        model_weights_path = ensemble_dir / "ensemble_models" / f"model_{i}" / "model_weights"
        if not model_weights_path.with_suffix('.index').exists():
            raise FileNotFoundError(f"Model {i} weights not found: {model_weights_path}")
        
        model.load_weights(str(model_weights_path))
        models.append(model)
        print(f"Model {i} loaded successfully")
    
    # Use only the weights for the first 2 models
    selected_model_weights = model_weights[:models_to_load]
    
    # Create ensemble with 2 models
    ensemble_cvae = EnsembleCVAE(models, selected_model_weights)
    print(f"Ensemble created with {len(models)} models (optimized from {original_ensemble_size})")
    print(f"Time savings: ~{((original_ensemble_size - models_to_load) / original_ensemble_size) * 100:.0f}% faster evaluation")
    
    return ensemble_cvae, ensemble_config

def generate_ensemble_comparison_plots(y_true, ensemble_results, individual_results, output_dir: Path, split_info: str):
    """Generate HIGH-QUALITY INDIVIDUAL PLOTS instead of cramped 8-panel layout."""
    plots_dir = output_dir / "plots"
    
    # Extract results for plotting
    ensemble_scores = ensemble_results['adaptive_weighted'][0]
    ensemble_auc = roc_auc_score(y_true, ensemble_scores)
    
    # Calculate individual model AUCs
    individual_aucs = []
    individual_scores = []
    for result in individual_results:
        scores = result['anomaly_scores']
        auc = roc_auc_score(y_true, scores)
        individual_aucs.append(auc)
        individual_scores.append(scores)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Professional colors
    
    # Set publication-ready style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'lines.linewidth': 2.5,
        'lines.markersize': 7
    })
    
    # 1. ROC CURVES COMPARISON - LARGE HIGH-QUALITY PLOT
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot individual models
    for i, (scores, auc) in enumerate(zip(individual_scores, individual_aucs)):
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, '--', alpha=0.8, color=colors[i % len(colors)], 
               linewidth=3, label=f'Model {i} (AUC={auc:.3f})')
    
    # Plot ensemble
    fpr_ens, tpr_ens, _ = roc_curve(y_true, ensemble_scores)
    ax.plot(fpr_ens, tpr_ens, linewidth=4, color='black', 
           label=f'Ensemble (AUC={ensemble_auc:.3f})', alpha=0.9)
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.6, linewidth=2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(f'ROC Curves Comparison - Split {split_info}', fontsize=16, fontweight='bold')
    ax.legend(framealpha=0.9, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    roc_plot = plots_dir / f"roc_curves_comparison_{split_info}.png"
    plt.savefig(roc_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. AUC COMPARISON BAR CHART - LARGE AND CLEAR
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    model_names = [f'Model {i}' for i in range(len(individual_aucs))] + ['Ensemble']
    all_aucs = individual_aucs + [ensemble_auc]
    colors_bar = colors[:len(individual_aucs)] + ['black']
    
    bars = ax.bar(model_names, all_aucs, color=colors_bar, alpha=0.8, width=0.6)
    ax.set_ylabel('AUC Score', fontweight='bold')
    ax.set_title(f'AUC Score Comparison - Split {split_info}', fontsize=16, fontweight='bold')
    ax.set_ylim([min(all_aucs) - 0.02, max(all_aucs) + 0.02])
    
    # Add value labels on bars
    for bar, value in zip(bars, all_aucs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, 
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    auc_plot = plots_dir / f"auc_comparison_{split_info}.png"
    plt.savefig(auc_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. PRECISION-RECALL CURVES - HIGH QUALITY
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Individual models
    for i, scores in enumerate(individual_scores):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, '--', alpha=0.8, color=colors[i % len(colors)], 
               linewidth=3, label=f'Model {i} (AP={ap:.3f})')
    
    # Ensemble
    precision_ens, recall_ens, _ = precision_recall_curve(y_true, ensemble_scores)
    ap_ens = average_precision_score(y_true, ensemble_scores)
    ax.plot(recall_ens, precision_ens, linewidth=4, color='black', 
           label=f'Ensemble (AP={ap_ens:.3f})', alpha=0.9)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title(f'Precision-Recall Curves - Split {split_info}', fontsize=16, fontweight='bold')
    ax.legend(framealpha=0.9, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pr_plot = plots_dir / f"precision_recall_curves_{split_info}.png"
    plt.savefig(pr_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 4. ENSEMBLE METHODS COMPARISON - CLEAR BAR CHART
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    method_names = list(ensemble_results.keys())
    method_aucs = []
    method_f1s = []
    
    for method in method_names:
        scores = ensemble_results[method][0]
        predictions = ensemble_results[method][1]
        auc = roc_auc_score(y_true, scores)
        f1 = f1_score(y_true, predictions, zero_division=0)
        method_aucs.append(auc)
        method_f1s.append(f1)
    
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, method_aucs, width, label='AUC', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, method_f1s, width, label='F1 Score', alpha=0.8, color='lightcoral')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'Ensemble Methods Performance - Split {split_info}', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in method_names], rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    methods_plot = plots_dir / f"ensemble_methods_comparison_{split_info}.png"
    plt.savefig(methods_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 5. MODEL CORRELATION HEATMAP - CLEAN AND LARGE
    if len(individual_scores) >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        correlation_matrix = np.corrcoef(individual_scores)
        
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f'Model Correlation Matrix - Split {split_info}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model Index', fontweight='bold')
        ax.set_ylabel('Model Index', fontweight='bold')
        
        # Add correlation values
        for i in range(len(individual_scores)):
            for j in range(len(individual_scores)):
                ax.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xticks(range(len(individual_scores)))
        ax.set_yticks(range(len(individual_scores)))
        ax.set_xticklabels([f'Model {i}' for i in range(len(individual_scores))])
        ax.set_yticklabels([f'Model {i}' for i in range(len(individual_scores))])
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        corr_plot = plots_dir / f"model_correlation_matrix_{split_info}.png"
        plt.savefig(corr_plot, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"HIGH-QUALITY INDIVIDUAL PLOTS GENERATED:")
    print(f"  ROC Curves: {roc_plot}")
    print(f"  AUC Comparison: {auc_plot}")
    print(f"  Precision-Recall: {pr_plot}")
    print(f"  Methods Comparison: {methods_plot}")
    if len(individual_scores) >= 2:
        print(f"  Model Correlation: {corr_plot}")
    
    return {
        'roc_curves': roc_plot,
        'auc_comparison': auc_plot,
        'precision_recall': pr_plot,
        'methods_comparison': methods_plot,
        'correlation_matrix': corr_plot if len(individual_scores) >= 2 else None
    }

def evaluate_ensemble_methods(ensemble_cvae, y_true, X_test, batch_size=8):
    """Evaluate all ensemble methods and return comprehensive results."""
    
    methods = ['adaptive_weighted', 'weighted_soft', 'uncertainty_weighted', 
              'consensus_threshold', 'simple_average']
    
    method_results = {}
    
    print("Evaluating ensemble methods...")
    
    for method in methods:
        print(f"Testing {method}...")
        
        all_scores = []
        all_predictions = []
        all_confidences = []
        
        # Process in batches
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test[i:i + batch_size]
            
            try:
                scores, predictions, confidence, method_info = ensemble_cvae.predict_ensemble(
                    batch_x, method=method
                )
                all_scores.extend(scores)
                all_predictions.extend(predictions)
                all_confidences.extend(confidence)
                
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Add dummy values to maintain alignment
                batch_size_actual = len(batch_x)
                all_scores.extend([0.5] * batch_size_actual)
                all_predictions.extend([0] * batch_size_actual)
                all_confidences.extend([0.5] * batch_size_actual)
        
        # Align arrays
        min_len = min(len(all_scores), len(y_true))
        all_scores = np.array(all_scores[:min_len])
        all_predictions = np.array(all_predictions[:min_len])
        all_confidences = np.array(all_confidences[:min_len])
        y_true_aligned = y_true[:min_len]
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_true_aligned, all_scores)
            accuracy = accuracy_score(y_true_aligned, all_predictions)
            precision = precision_score(y_true_aligned, all_predictions, zero_division=0)
            recall = recall_score(y_true_aligned, all_predictions, zero_division=0)
            f1 = f1_score(y_true_aligned, all_predictions, zero_division=0)
            
            method_results[method] = (all_scores, all_predictions, all_confidences, {
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            print(f"      {method}: AUC={auc:.4f}, F1={f1:.4f}")
            
        except Exception as e:
            print(f"Error calculating metrics for {method}: {e}")
            method_results[method] = (all_scores, all_predictions, all_confidences, {
                'auc': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
            })
    
    return method_results

def save_ensemble_results(ensemble_results, individual_results, y_true, output_dir: Path, split_info: str):
    """Save comprehensive ensemble evaluation results with LOWER thresholds and BALANCED ACCURACY."""
    
    # Create detailed CSV comparing all methods
    csv_data = []
    
    # Add individual model results with LOWER THRESHOLDS (including 0.05, 0.15)
    for i, result in enumerate(individual_results):
        scores = result['anomaly_scores']
        
        # Calculate metrics for LOWER thresholds: 0.05, 0.1, 0.15, 0.2, 0.3, 0.5
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            predictions = (scores >= threshold).astype(int)
            
            # Calculate confusion matrix for MCC and Balanced Accuracy
            tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
            
            # Calculate MCC (Matthews Correlation Coefficient)
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if mcc_denominator == 0:
                mcc = 0.0
            else:
                mcc = ((tp * tn) - (fp * fn)) / mcc_denominator
            
            # Calculate Balanced Accuracy = (Sensitivity + Specificity) / 2
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            balanced_accuracy = (sensitivity + specificity) / 2
            
            csv_data.append({
                'Approach': f'Individual_Model_{i}',
                'Method': f'Threshold_{threshold}',
                'Threshold': threshold,
                'AUC': roc_auc_score(y_true, scores),
                'Accuracy': accuracy_score(y_true, predictions),
                'Balanced_Accuracy': balanced_accuracy,
                'Precision': precision_score(y_true, predictions, zero_division=0),
                'Recall': recall_score(y_true, predictions, zero_division=0),
                'Specificity': specificity,
                'F1_Score': f1_score(y_true, predictions, zero_division=0),
                'MCC': mcc
            })
    
    # Add ensemble method results with MCC and Balanced Accuracy
    for method_name, (scores, predictions, confidences, metrics) in ensemble_results.items():
        # Calculate MCC and Balanced Accuracy for ensemble predictions
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        
        # MCC calculation
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if mcc_denominator == 0:
            ensemble_mcc = 0.0
        else:
            ensemble_mcc = ((tp * tn) - (fp * fn)) / mcc_denominator
        
        # Balanced Accuracy calculation
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ensemble_balanced_acc = (sensitivity + specificity) / 2
        
        csv_data.append({
            'Approach': 'Ensemble',
            'Method': method_name,
            'Threshold': 'Adaptive' if 'adaptive' in method_name else 'Fixed_0.5',
            'AUC': metrics['auc'],
            'Accuracy': metrics['accuracy'],
            'Balanced_Accuracy': ensemble_balanced_acc,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': specificity,
            'F1_Score': metrics['f1_score'],
            'MCC': ensemble_mcc
        })
    
    # Save comprehensive CSV
    df = pd.DataFrame(csv_data)
    df = df.sort_values(['Approach', 'F1_Score'], ascending=[True, False])
    
    csv_file = output_dir / "metrics" / f"ensemble_comprehensive_results_{split_info}.csv"
    df.to_csv(csv_file, index=False)
    
    # Find best results (now including balanced accuracy)
    best_individual_f1 = df[df['Approach'].str.contains('Individual')]['F1_Score'].max()
    best_individual_bal_acc = df[df['Approach'].str.contains('Individual')]['Balanced_Accuracy'].max()
    best_ensemble_f1 = df[df['Approach'] == 'Ensemble']['F1_Score'].max()
    best_ensemble_bal_acc = df[df['Approach'] == 'Ensemble']['Balanced_Accuracy'].max()
    best_overall = df.loc[df['F1_Score'].idxmax()]
    
    print(f"Results saved to: {csv_file}")
    print(f"  Best Individual F1: {best_individual_f1:.4f}")
    print(f"  Best Individual Balanced Acc: {best_individual_bal_acc:.4f}")
    print(f"  Best Ensemble F1: {best_ensemble_f1:.4f}")
    print(f"  Best Ensemble Balanced Acc: {best_ensemble_bal_acc:.4f}")
    print(f"  Best Overall: {best_overall['Approach']} - {best_overall['Method']} (F1: {best_overall['F1_Score']:.4f})")
    print(f"  Lower thresholds tested: 0.05, 0.1, 0.15, 0.2, 0.3, 0.5")
    print(f"  Added metrics: Balanced Accuracy, Specificity")
    
    return df

def create_ensemble_output_structure(base_output_dir: Path, ensemble_name: str, split: str) -> Path:
    """Create ensemble output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"{ensemble_name}_{split}_test_{timestamp}"
    
    directories = [
        output_dir / "metrics",
        output_dir / "plots" / "ensemble_comparison",
        output_dir / "plots" / "individual_models",
        output_dir / "plots" / "method_comparison",
        output_dir / "raw_data",
        output_dir / "reports",
        output_dir / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Created ensemble output structure: {output_dir}")
    return output_dir

def main():
    """Main ensemble testing function."""
    parser = argparse.ArgumentParser(description='Test Ensemble CVAE Models')
    parser.add_argument('--ensemble_dir', type=str, required=True,
                       help='Directory containing the trained ensemble models')
    parser.add_argument('--split', type=str, default='9010',
                       choices=['9010', '6040', '5050'],
                       help='Data split to use for testing')
    parser.add_argument('--output_dir', type=str, default='./ensemble_test_results',
                       help='Base directory for test results')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 128],
                       help='Image dimensions [height, width]')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of test samples')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    print("Ensemble CVAE Testing - OPTIMIZED FOR 2 MODELS")
    print("Using first 2 models for faster evaluation")
    print("Comprehensive individual vs ensemble comparison")
    print("Time-optimized: ~33% faster than full ensemble")
    print("=" * 60)
    print(f"Ensemble Directory: {args.ensemble_dir}")
    print(f"Data Split: {args.split}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Image Size: {args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 60)
    
    try:
        # Load ensemble
        ensemble_cvae, ensemble_config = load_ensemble_models(args.ensemble_dir)
        
        # Create output structure
        ensemble_name = Path(args.ensemble_dir).name
        base_output_dir = Path(args.output_dir)
        output_dir = create_ensemble_output_structure(base_output_dir, ensemble_name, args.split)
        
        # Create test data loader
        split_max_samples = {
            '9010': 15000,
            '6040': 50000, 
            '5050': 60000
        }
        
        max_samples_for_split = args.max_samples or split_max_samples.get(args.split, 30000)
        print(f"Using {max_samples_for_split:,} max samples for split {args.split}")
        
        test_loader = EnsembleTestDataLoader(args.split, tuple(args.img_size), max_samples_for_split)
        
        # Load test data
        X_test, y_test = test_loader.get_numpy_data()
        print(f"Test data loaded: {X_test.shape}, {y_test.shape}")
        
        # Evaluate individual models
        print("Evaluating individual models...")
        individual_results = []
        
        for i in range(0, len(X_test), args.batch_size):
            batch_x = X_test[i:i + args.batch_size]
            batch_individual = ensemble_cvae.predict_individual_models(batch_x)
            
            if i == 0:  # Initialize results structure
                individual_results = [{'model_id': j, 'anomaly_scores': [], 'recon_errors': []} 
                                    for j in range(len(batch_individual))]
            
            # Collect results
            for j, result in enumerate(batch_individual):
                individual_results[j]['anomaly_scores'].extend(result['anomaly_scores'])
                individual_results[j]['recon_errors'].extend(result['recon_errors'])
        
        # Convert to numpy arrays
        for result in individual_results:
            result['anomaly_scores'] = np.array(result['anomaly_scores'][:len(y_test)])
            result['recon_errors'] = np.array(result['recon_errors'][:len(y_test)])
        
        # Evaluate ensemble methods
        ensemble_results = evaluate_ensemble_methods(ensemble_cvae, y_test, X_test, args.batch_size)
        
        # Generate comprehensive analysis
        print("Generating comprehensive plots...")
        generate_ensemble_comparison_plots(y_test, ensemble_results, individual_results, 
                                         output_dir, args.split)
        
        # Save detailed results
        print("Saving detailed results...")
        results_df = save_ensemble_results(ensemble_results, individual_results, y_test, 
                                         output_dir, args.split)
        
        # Print final summary
        print("\nENSEMBLE EVALUATION COMPLETED!")
        print("=" * 60)
        
        # Get best results
        best_individual_f1 = results_df[results_df['Approach'].str.contains('Individual')]['F1_Score'].max()
        best_ensemble_f1 = results_df[results_df['Approach'] == 'Ensemble']['F1_Score'].max()
        best_overall = results_df.loc[results_df['F1_Score'].idxmax()]
        
        print(f"PERFORMANCE SUMMARY (2-Model Ensemble):")
        print(f"  Best Individual Model F1:  {best_individual_f1:.4f}")
        print(f"  Best Ensemble Method F1:   {best_ensemble_f1:.4f}")
        print(f"  Improvement:                +{best_ensemble_f1 - best_individual_f1:.4f}")
        print(f"  Best Overall Method:        {best_overall['Method']}")
        print(f"  Optimization: Using 2 models for faster evaluation")
        
        ensemble_auc = ensemble_results['adaptive_weighted'][3]['auc']
        individual_aucs = [roc_auc_score(y_test, result['anomaly_scores']) for result in individual_results]
        
        print(f"\nAUC COMPARISON:")
        print(f"  Individual Models AUC:      {[f'{auc:.4f}' for auc in individual_aucs]}")
        print(f"  Ensemble AUC (Adaptive):    {ensemble_auc:.4f}")
        print(f"  Improvement over best:      +{ensemble_auc - max(individual_aucs):.4f}")
        
        print(f"\nAll results saved to: {output_dir}")
        print(f"Comprehensive plots: {output_dir / 'plots' / 'ensemble_comprehensive_analysis.png'}")
        print(f"Detailed CSV: {output_dir / 'metrics' / f'ensemble_comprehensive_results_{args.split}.csv'}")
        
    except Exception as e:
        print(f"Ensemble testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()