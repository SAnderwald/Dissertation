import os
import json
import logging
import numpy as np
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tqdm import tqdm

mixed_precision.set_global_policy("mixed_float16")

CONFIG = {
    "MAIN_IMAGE_DIR": "/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images",
    "TRAIN_FILE": "/media/sanderwald/Project_Files/Dissertation_Project/data/Splits/50_50/train_split.txt",
    "TEST_FILE": "/media/sanderwald/Project_Files/Dissertation_Project/data/Splits/50_50/test_split.txt",
    "ANNOTATION_FILE": "/media/sanderwald/Project_Files/Dissertation_Project/data/annotations_new.json",
    "BATCH_SIZE": 8,
    "IMG_SIZE": (720, 405),
    "LATENT_DIM": 64,
    "EPOCHS": 5,  # Restore full epochs
    "LEARNING_RATE": 1e-4,
    "OUTPUT_DIR": "outputs",
    "TRAIN_LOG": "outputs/training_log.txt",
    "CHECKPOINT_DIR": "outputs/checkpoints",
    "VALIDATION_SPLIT": 0.2,
    "PATIENCE": 5
}

def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(input_shape, dtype=tf.float16)
    x = layers.Conv2D(16, 3, padding="same", activation="relu", dtype=tf.float16)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", dtype=tf.float16)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", dtype=tf.float16)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, -50.0, 50.0))(x)
    x = layers.Dense(64, activation="relu", dtype=tf.float16, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, -100.0, 100.0))(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", dtype=tf.float16, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    z_log_var_raw = layers.Dense(latent_dim, name="z_log_var_raw", dtype=tf.float16, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    z_log_var = layers.Lambda(lambda t: tf.clip_by_value(t, -5.0, 5.0), name="z_log_var")(z_log_var_raw)
    return Model(inputs, [z_mean, z_log_var], name="encoder")

def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,), dtype=tf.float16)
    x = layers.Dense(45 * 25 * 128, activation="relu", dtype=tf.float16)(latent_inputs)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 100.0))(x)
    x = layers.Reshape((45, 25, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu", dtype=tf.float16)(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 100.0))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", dtype=tf.float16)(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 100.0))(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", dtype=tf.float16)(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 100.0))(x)
    x = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid", dtype=tf.float16)(x)
    x = layers.Resizing(*CONFIG["IMG_SIZE"])(x)
    return Model(latent_inputs, x, name="decoder")

def build_bbox_predictor(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,), dtype=tf.float16)
    x = layers.Dense(64, activation="relu", dtype=tf.float16)(latent_inputs)
    x = layers.Dense(32, activation="relu", dtype=tf.float16)(x)
    bbox_outputs = layers.Dense(4, activation="sigmoid", dtype=tf.float16)(x)
    return Model(latent_inputs, bbox_outputs, name="bbox_predictor")

class CVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, bbox_predictor):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bbox_predictor = bbox_predictor
    
    def call(self, inputs, training=False):
        images, true_bboxes = inputs
        print(f"Debug - Input images shape: {images.shape}, true_bboxes shape: {true_bboxes.shape}")
        z_mean, z_log_var = self.encoder(images)
        print(f"Debug - z_mean shape: {z_mean.shape}, z_log_var shape: {z_log_var.shape}, z_log_var min: {tf.reduce_min(z_log_var)}, max: {tf.reduce_max(z_log_var)}")
        z = self.reparameterize(z_mean, z_log_var)
        print(f"Debug - z shape: {z.shape}, z min: {tf.reduce_min(z)}, z max: {tf.reduce_max(z)}")
        reconstructed = self.decoder(z)
        print(f"Debug - reconstructed shape: {reconstructed.shape}, min: {tf.reduce_min(reconstructed)}, max: {tf.reduce_max(reconstructed)}")
        predicted_bbox = self.bbox_predictor(z)
        print(f"Debug - predicted_bbox shape: {predicted_bbox.shape}")
        return reconstructed, predicted_bbox
    
    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean), dtype=tf.float16)
        eps = tf.clip_by_value(eps, -5.0, 5.0)
        print(f"Debug - eps min: {tf.reduce_min(eps)}, max: {tf.reduce_max(eps)}")
        return mean + tf.exp(0.5 * log_var) * eps
    
    def compute_loss(self, inputs, reconstructed, z_mean, z_log_var, predicted_bbox):
        images, true_bboxes = inputs
        print(f"Debug - images shape: {images.shape}, reconstructed shape: {reconstructed.shape}, true_bboxes shape: {true_bboxes.shape}, predicted_bbox shape: {predicted_bbox.shape}")
        true_bboxes_normalized = tf.stack([
            true_bboxes[:, 0] / CONFIG["IMG_SIZE"][1],
            true_bboxes[:, 1] / CONFIG["IMG_SIZE"][0],
            true_bboxes[:, 2] / CONFIG["IMG_SIZE"][1],
            true_bboxes[:, 3] / CONFIG["IMG_SIZE"][0]
        ], axis=1)
        pixel_count = CONFIG["IMG_SIZE"][0] * CONFIG["IMG_SIZE"][1] * 3 / 1000
        recon_diff = tf.cast(tf.square(images - reconstructed), tf.float32)
        print(f"Debug - recon_diff min: {tf.reduce_min(recon_diff)}, max: {tf.reduce_max(recon_diff)}")
        recon_sum = tf.reduce_sum(recon_diff, axis=[1, 2, 3])
        print(f"Debug - recon_sum min: {tf.reduce_min(recon_sum)}, max: {tf.reduce_max(recon_sum)}")
        recon_loss = tf.cast(tf.reduce_mean(recon_sum) / pixel_count, tf.float16)
        kl_loss = tf.cast(-0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)) * 0.1, tf.float16)
        bbox_loss = tf.cast(tf.reduce_mean(tf.reduce_sum(tf.square(true_bboxes_normalized - predicted_bbox), axis=1)), tf.float16)
        total_loss = recon_loss + kl_loss + bbox_loss
        return total_loss, recon_loss, kl_loss, bbox_loss

def get_image_paths_from_dirs(dir_list):
    image_paths = []
    for dir_name in dir_list:
        full_dir_path = os.path.join(CONFIG["MAIN_IMAGE_DIR"], dir_name)
        if not os.path.exists(full_dir_path):
            logging.warning(f"Skipping missing directory: {full_dir_path}")
            continue
        image_paths.extend([os.path.join(full_dir_path, f) for f in os.listdir(full_dir_path) 
                           if f.endswith('.jpg')])
    return image_paths

def create_tf_dataset(image_paths, annotation_dict, batch_size, shuffle=False):
    def generator():
        label_counts = {0: 0, 1: 0}
        TARGET_WIDTH, TARGET_HEIGHT = CONFIG["IMG_SIZE"][1], CONFIG["IMG_SIZE"][0]
        for idx, img_path in enumerate(image_paths):
            if idx % 10000 == 0:
                print(f"Processing image {idx+1}/{len(image_paths)}: {img_path}")
            try:
                img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
                img = tf.image.resize(img, CONFIG["IMG_SIZE"]) / 255.0
                img = tf.cast(img, tf.float16)
                if shuffle:
                    img = tf.image.random_flip_left_right(img)
                    if random.random() > 0.5:
                        img = tf.image.random_brightness(img, max_delta=0.1)
                
                key = img_path.replace(IMAGE_DIR, '').lstrip('/')
                annot = annotation_dict.get(key, {})
                orig_width, orig_height = annot.get("width", 1280), annot.get("height", 720)
                bbox = annot.get("bounding_boxes", [0.0, 0.0, 0.0, 0.0])
                label = annot.get("label", 0)
                
                scale_x = TARGET_WIDTH / orig_width
                scale_y = TARGET_HEIGHT / orig_height
                scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, 
                              bbox[2] * scale_x, bbox[3] * scale_y]
                
                scaled_bbox = [
                    max(0.0, min(scaled_bbox[0], TARGET_WIDTH)),
                    max(0.0, min(scaled_bbox[1], TARGET_HEIGHT)),
                    max(0.0, min(scaled_bbox[2], TARGET_WIDTH)),
                    max(0.0, min(scaled_bbox[3], TARGET_HEIGHT))
                ]
                scaled_bbox = tf.cast(scaled_bbox, tf.float16)
                
                label_counts[label] += 1
                yield img, scaled_bbox, label
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                continue
        print(f"Dataset label distribution: {label_counts}")

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=CONFIG["IMG_SIZE"] + (3,), dtype=tf.float16),
            tf.TensorSpec(shape=(4,), dtype=tf.float16),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=512)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model(model, train_ds, val_ds):
    optimizer = Adam(learning_rate=CONFIG["LEARNING_RATE"])
    best_val_loss = float('inf')
    patience_counter = 0
    factor = 0.5
    patience = 2

    for epoch in range(CONFIG["EPOCHS"]):
        total_loss_epoch = tf.constant(0.0, dtype=tf.float32)
        num_batches = 0
        print(f"Epoch {epoch + 1} started")
        
        for batch_idx, batch in enumerate(tqdm(train_ds, desc=f"Epoch {epoch + 1}")):
            images, true_bboxes, _ = batch
            print(f"Batch {batch_idx}: Images {images.shape}, Bboxes {true_bboxes.shape}")
            with tf.GradientTape() as tape:
                reconstructed, predicted_bbox = model((images, true_bboxes))
                total_loss, recon_loss, kl_loss, bbox_loss = model.compute_loss(
                    (images, true_bboxes), reconstructed, *model.encoder(images), predicted_bbox
                )
            gradients = tape.gradient(total_loss, model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 0.1) for g in gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss_epoch += tf.cast(total_loss, tf.float32)
            num_batches += 1
            print(f"Batch {batch_idx}: Total Loss: {total_loss:.4f}, Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}, BBox: {bbox_loss:.4f}")

        avg_train_loss = total_loss_epoch / num_batches if num_batches > 0 else 0
        val_loss = tf.constant(0.0, dtype=tf.float32)
        val_batches = 0
        for val_batch in val_ds:
            images, true_bboxes, _ = val_batch
            reconstructed, predicted_bbox = model((images, true_bboxes))
            total_loss, _, _, _ = model.compute_loss(
                (images, true_bboxes), reconstructed, *model.encoder(images), predicted_bbox
            )
            val_loss += tf.cast(total_loss, tf.float32)
            val_batches += 1
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        save_path = f"{CONFIG['CHECKPOINT_DIR']}/epoch_{epoch+1}.weights.h5"  # Simplified naming
        model.save_weights(save_path)
        print(f"Saved weights to: {save_path}")

def main():
    logging.basicConfig(level=logging.INFO, filename=CONFIG["TRAIN_LOG"], filemode='a',
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    val_split = CONFIG["VALIDATION_SPLIT"]
    all_dirs = Path(CONFIG["TRAIN_FILE"]).read_text().strip().split("\n") + \
              Path(CONFIG["TEST_FILE"]).read_text().strip().split("\n")
    random.shuffle(all_dirs)

    val_size = int(val_split * len(all_dirs))
    test_size = int(0.2 * len(all_dirs))
    train_size = len(all_dirs) - val_size - test_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(f"❌ Dataset too small for split {val_split}")

    train_dirs = all_dirs[:train_size]
    val_dirs = all_dirs[train_size:train_size + val_size]

    with open(CONFIG["ANNOTATION_FILE"], "r") as f:
        annotations = json.load(f)

    train_paths = get_image_paths_from_dirs(train_dirs)
    val_paths = get_image_paths_from_dirs(val_dirs)
    
    print(f"Train images: {len(train_paths)}, Val images: {len(val_paths)}")
    
    train_ds = create_tf_dataset(train_paths, annotations, CONFIG["BATCH_SIZE"], shuffle=True)
    val_ds = create_tf_dataset(val_paths, annotations, CONFIG["BATCH_SIZE"])

    encoder = build_encoder((*CONFIG["IMG_SIZE"], 3), CONFIG["LATENT_DIM"])
    decoder = build_decoder(CONFIG["LATENT_DIM"])
    bbox_predictor = build_bbox_predictor(CONFIG["LATENT_DIM"])
    model = CVAE(encoder, decoder, bbox_predictor)
    
    print(f"✅ Starting full training run for {CONFIG['EPOCHS']} epochs")
    train_model(model, train_ds, val_ds)

if __name__ == "__main__":
    IMAGE_DIR = CONFIG["MAIN_IMAGE_DIR"]
    main()