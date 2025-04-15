import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/sanderwald/Projects/dissertationProject/train_vae_fl_log_6040_best_latent64_epochs5_gamma2.txt"),
        logging.StreamHandler(),
    ],
)

CONFIG = {
    "MAIN_IMAGE_DIR": "/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/",
    "TRAIN_FILE": "/media/sanderwald/Project_Files/Dissertation_Project/data/Splits/60_40/60_40_train_split.txt",
    "TEST_FILE": "/media/sanderwald/Project_Files/Dissertation_Project/data/Splits/60_40/60_40_test_split.txt",
    "OUTPUT_DIR": "outputs_6040_focal_best_latent64_epochs5_gamma2",
    "IMG_SIZE": (360, 203),
    "BATCH_SIZE": 2,
    "LATENT_DIM": 64,
    "RECON_LOSS_WEIGHT": 0.5,
    "KL_LOSS_WEIGHT": 0.001,
    "FOCAL_LOSS_WEIGHT": 1.0,
    "FOCAL_GAMMA": 2.0,
    "FOCAL_ALPHA": 0.5,
    "EPOCHS": 5,
}

# Model Components
def build_encoder(input_shape, latent_dim):
    """Build the encoder network."""
    inputs = layers.Input(input_shape, dtype=tf.float32)
    x = layers.Conv2D(8, 3, padding="same", activation="relu", dtype=tf.float32)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu", dtype=tf.float32, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    x = layers.Dropout(0.3)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", dtype=tf.float32, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", dtype=tf.float32, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    return Model(inputs, [z_mean, z_log_var], name="encoder")

def build_decoder(latent_dim):
    """Build the decoder network."""
    latent_inputs = layers.Input(shape=(latent_dim,), dtype=tf.float32)
    x = layers.Dense(23 * 13 * 64, activation="relu", dtype=tf.float32)(latent_inputs)
    x = layers.Reshape((23, 13, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(3, 3, strides=(2, 2), padding="same", activation="sigmoid", dtype=tf.float32)(x)
    x = layers.Cropping2D(cropping=((4, 4), (2, 3)))(x)
    return Model(latent_inputs, x, name="decoder")

def focal_loss(y_true, y_pred):
    """Compute focal loss for binary classification."""
    alpha = CONFIG["FOCAL_ALPHA"]
    gamma = CONFIG["FOCAL_GAMMA"]
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    alpha = tf.cast(alpha, tf.float32)
    gamma = tf.cast(gamma, tf.float32)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    ce = -tf.math.log(pt)
    weight = alpha * tf.where(tf.equal(y_true, 1), 1.0, 1.0 - alpha) * tf.pow(1.0 - pt, gamma)
    return tf.reduce_mean(weight * ce)

# Supervised VAE Model with Classification Head
class CVAE(Model):
    def __init__(self, encoder, decoder, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.classifier = layers.Dense(1, activation='sigmoid', dtype=tf.float32, name="classifier")

    def call(self, inputs, training=False):
        images, true_bboxes = inputs
        z_mean, z_log_var = self.encoder(images, training=training)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z, training=training)
        class_logits = self.classifier(z_mean)
        return reconstructed, z_mean, z_log_var, class_logits

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim), dtype=tf.float32)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def train_step(self, data):
        images, true_bboxes, true_labels = data
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var, class_logits = self((images, true_bboxes), training=True)
            
            # Reconstruction loss
            pixel_count = CONFIG["IMG_SIZE"][0] * CONFIG["IMG_SIZE"][1] * 3 / 1000
            images = tf.cast(images, tf.float32)
            reconstructed = tf.cast(reconstructed, tf.float32)
            recon_diff = tf.square(images - reconstructed)
            recon_diff = tf.cast(recon_diff, tf.float32)
            recon_diff = tf.clip_by_value(recon_diff, 0.0, 10.0)
            recon_sum = tf.reduce_sum(recon_diff, axis=[1, 2, 3])
            recon_loss = tf.reduce_mean(recon_sum / pixel_count)
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            # Classification loss (focal loss)
            class_loss = focal_loss(true_labels, class_logits)
            
            # Cast weights to float32 to match the dtype of recon_loss and kl_loss
            recon_weight = tf.cast(CONFIG["RECON_LOSS_WEIGHT"], tf.float32)
            kl_weight = tf.cast(CONFIG["KL_LOSS_WEIGHT"], tf.float32)
            focal_weight = tf.cast(CONFIG["FOCAL_LOSS_WEIGHT"], tf.float32)
            
            # Compute total loss with consistent dtypes
            recon_term = recon_weight * recon_loss
            kl_term = kl_weight * kl_loss
            focal_term = focal_weight * class_loss
            total_loss = recon_term + kl_term + focal_term
        
        # Compute gradients and update weights with gradient clipping
        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else grad for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "class_loss": class_loss,
        }

# Data Loading
log_counter = 0

def load_image_paths_with_labels(dir_list):
    paths_0 = []
    paths_1 = []
    for dir_name in dir_list:
        full_dir_path = os.path.join(CONFIG["MAIN_IMAGE_DIR"], dir_name)
        logging.info(f"Checking directory: {full_dir_path}")
        if not os.path.exists(full_dir_path):
            logging.warning(f"Skipping missing directory: {full_dir_path}")
            continue
        all_files = os.listdir(full_dir_path)
        dir_images = [os.path.join(full_dir_path, f) for f in all_files if f.endswith('.jpg')]
        logging.info(f"Found {len(dir_images)} .jpg files in {full_dir_path}")
        if '_1' in dir_name:
            paths_1.extend(dir_images)
        else:
            paths_0.extend(dir_images)
    if not paths_0 and not paths_1:
        logging.error("No image paths found for directories")
        raise ValueError("No images found")
    return paths_0, paths_1

def create_tf_dataset(paths_0, paths_1):
    def parse_function(path):
        global log_counter
        try:
            path_str = path.numpy().decode('utf-8') if isinstance(path, tf.Tensor) else path.decode('utf-8') if isinstance(path, bytes) else path
            image = tf.io.read_file(path_str)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.ensure_shape(image, [None, None, 3])
            image = tf.image.resize(image, CONFIG["IMG_SIZE"], method=tf.image.ResizeMethod.BILINEAR)
            image = tf.ensure_shape(image, [CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 3])
            image = image / 255.0
            image = tf.cast(image, tf.float32)
            full_dir = os.path.dirname(path_str)
            dir_name = os.path.basename(full_dir)
            label = 1 if '_1' in dir_name else 0
            bbox = tf.convert_to_tensor([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
            log_counter += 1
            if log_counter % 10000 == 0:
                logging.info(f"Path: {path_str}, Full Dir: {full_dir}, Dir: {dir_name}, Assigned Label: {label}")
            label = tf.cast(label, tf.int32)
            path_tensor = tf.convert_to_tensor(path_str, dtype=tf.string)
            return tf.constant(True, dtype=tf.bool), image, bbox, label, path_tensor
        except Exception as e:
            logging.error(f"Error loading image {path}: {str(e)}. Skipping image.")
            default_image = tf.zeros([CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 3], dtype=tf.float32)
            default_bbox = tf.zeros([4], dtype=tf.float32)
            default_label = tf.constant(-1, dtype=tf.int32)
            path_str = path.numpy().decode('utf-8') if isinstance(path, tf.Tensor) else path.decode('utf-8') if isinstance(path, bytes) else path
            path_tensor = tf.convert_to_tensor(path_str, dtype=tf.string)
            return tf.constant(False, dtype=tf.bool), default_image, default_bbox, default_label, path_tensor

    dataset_0 = tf.data.Dataset.from_tensor_slices(paths_0)
    dataset_1 = tf.data.Dataset.from_tensor_slices(paths_1)
    dataset_0 = dataset_0.shuffle(buffer_size=len(paths_0))
    dataset_1 = dataset_1.shuffle(buffer_size=len(paths_1))
    dataset = tf.data.Dataset.sample_from_datasets(
        [dataset_0, dataset_1],
        weights=[0.5, 0.5],
        stop_on_empty_dataset=True
    )
    dataset = dataset.map(
        lambda path: tf.py_function(parse_function, [path], [tf.bool, tf.float32, tf.float32, tf.int32, tf.string]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.filter(lambda success, image, bbox, label, path: tf.logical_and(success, tf.not_equal(label, -1)))
    dataset = dataset.map(lambda success, image, bbox, label, path: (image, bbox, label, path))
    dataset = dataset.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
    return dataset

# Main training function
def main():
    logging.info("Starting training execution")

    # Set up GPU and memory growth
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU memory growth enabled")
            print(f"GPU Device: {physical_devices[0]}")
        except Exception as e:
            print(f"Failed to set memory growth or device: {e}, using default allocation")
            logging.error(f"Failed to set memory growth or device: {str(e)}", exc_info=True)

    # Define output directory
    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], f"gamma_{CONFIG['FOCAL_GAMMA']}_alpha_{CONFIG['FOCAL_ALPHA']}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # Build the model
    logging.info("Building encoder")
    encoder = build_encoder((CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 3), CONFIG["LATENT_DIM"])
    logging.info("Building decoder")
    decoder = build_decoder(CONFIG["LATENT_DIM"])
    logging.info("Building CVAE model")
    model = CVAE(encoder, decoder, CONFIG["LATENT_DIM"])

    # Set up optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer)

    # Read train directories
    train_file_path = CONFIG["TRAIN_FILE"]
    if not os.path.exists(train_file_path):
        logging.error(f"TRAIN_FILE does not exist: {train_file_path}")
        raise FileNotFoundError(f"TRAIN_FILE does not exist: {train_file_path}")
    train_dirs = list(set(Path(train_file_path).read_text().strip().split("\n")))
    logging.info(f"Total train dirs: {len(train_dirs)}")
    print(f"Total train dirs: {len(train_dirs)}")

    # Load train image paths
    paths_0, paths_1 = load_image_paths_with_labels(train_dirs)
    logging.info(f"Total _0 images: {len(paths_0)}, Total _1 images: {len(paths_1)}")
    print(f"Total _0 images: {len(paths_0)}, Total _1 images: {len(paths_1)}")

    # Create TensorFlow dataset for training
    train_ds = create_tf_dataset(paths_0, paths_1)

    # Train the model
    logging.info("Starting training")
    for epoch in range(CONFIG["EPOCHS"]):
        logging.info(f"Starting epoch {epoch + 1}/{CONFIG['EPOCHS']}")
        for batch_idx, batch in enumerate(train_ds):
            images, true_bboxes, true_labels, paths = batch
            batch_data = (images, true_bboxes, true_labels)
            losses = model.train_step(batch_data)
            if batch_idx % 100 == 0:
                logging.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx}: "
                    f"Loss: {losses['loss']:.4f}, "
                    f"Recon Loss: {losses['recon_loss']:.4f}, "
                    f"KL Loss: {losses['kl_loss']:.4f}, "
                    f"Class Loss: {losses['class_loss']:.4f}"
                )
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch + 1}_v18_focal.weights.h5")
        model.save_weights(checkpoint_path)
        logging.info(f"Saving checkpoint at epoch {epoch + 1}: {checkpoint_path}")
        logging.info(f"Epoch {epoch + 1} completed")

    logging.info("Training completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Main execution failed with error: {str(e)}", exc_info=True)
        raise