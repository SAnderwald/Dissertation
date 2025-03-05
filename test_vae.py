import os
import json
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import glob

CONFIG = {
    "MAIN_IMAGE_DIR": "/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images",
    "TEST_FILE": "/media/sanderwald/Project_Files/Dissertation_Project/data/Splits/50_50/test_split.txt",
    "ANNOTATION_FILE": "/media/sanderwald/Project_Files/Dissertation_Project/data/annotations_new.json",
    "BATCH_SIZE": 8,
    "IMG_SIZE": (720, 405),
    "LATENT_DIM": 64,
    "CHECKPOINT_DIR": "/home/sanderwald/Projects/dissertationProject/outputs/checkpoints"
}

def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(input_shape, dtype=tf.float32)
    x = layers.Conv2D(16, 3, padding="same", activation="relu", dtype=tf.float32)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, -50.0, 50.0))(x)
    x = layers.Dense(64, activation="relu", dtype=tf.float32, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, -100.0, 100.0))(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", dtype=tf.float32, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    z_log_var_raw = layers.Dense(latent_dim, name="z_log_var_raw", dtype=tf.float32, kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0))(x)
    z_log_var = layers.Lambda(lambda t: tf.clip_by_value(t, -5.0, 5.0), name="z_log_var")(z_log_var_raw)
    return Model(inputs, [z_mean, z_log_var], name="encoder")

def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,), dtype=tf.float32)
    x = layers.Dense(45 * 25 * 128, activation="relu", dtype=tf.float32)(latent_inputs)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 100.0))(x)
    x = layers.Reshape((45, 25, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 100.0))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 100.0))(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 100.0))(x)
    x = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid", dtype=tf.float32)(x)
    x = layers.Resizing(*CONFIG["IMG_SIZE"])(x)
    return Model(latent_inputs, x, name="decoder")

def build_bbox_predictor(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,), dtype=tf.float32)
    x = layers.Dense(64, activation="relu", dtype=tf.float32)(latent_inputs)
    x = layers.Dense(32, activation="relu", dtype=tf.float32)(x)
    bbox_outputs = layers.Dense(4, activation="sigmoid", dtype=tf.float32)(x)
    return Model(latent_inputs, bbox_outputs, name="bbox_predictor")

class CVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, bbox_predictor):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bbox_predictor = bbox_predictor
    
    def call(self, inputs, training=False):
        images, true_bboxes = inputs
        print(f"Raw inputs to call - images: {images.dtype}, true_bboxes: {true_bboxes.dtype}")
        images = tf.cast(images, tf.float32)
        true_bboxes = tf.cast(true_bboxes, tf.float32)
        print(f"After cast in call - images: {images.dtype}, true_bboxes: {true_bboxes.dtype}")
        z_mean, z_log_var = self.encoder(images)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        predicted_bbox = self.bbox_predictor(z)
        return reconstructed, predicted_bbox
    
    def reparameterize(self, mean, log_var):
        mean = tf.clip_by_value(mean, -100.0, 100.0)
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        eps = tf.random.normal(shape=tf.shape(mean), dtype=tf.float32)
        eps = tf.clip_by_value(eps, -5.0, 5.0)
        z = mean + tf.exp(0.5 * log_var) * eps
        return tf.clip_by_value(z, -100.0, 100.0)

def get_image_paths_from_dirs(dir_list):
    image_paths = []
    for dir_name in dir_list:
        full_dir_path = os.path.join(CONFIG["MAIN_IMAGE_DIR"], dir_name)
        if not os.path.exists(full_dir_path):
            print(f"Warning: Skipping missing directory: {full_dir_path}")
            continue
        image_paths.extend([os.path.join(full_dir_path, f) for f in os.listdir(full_dir_path) 
                           if f.endswith('.jpg')])
    return image_paths

def create_tf_dataset(image_paths, annotation_dict, batch_size):
    def generator():
        for img_path in image_paths:
            img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
            img = tf.image.resize(img, CONFIG["IMG_SIZE"]) / 255.0
            key = img_path.replace(IMAGE_DIR, '').lstrip('/')
            annot = annotation_dict.get(key, {})
            orig_width, orig_height = annot.get("width", 1280), annot.get("height", 720)
            bbox = annot.get("bounding_boxes", [0.0, 0.0, 0.0, 0.0])
            label = annot.get("label", 0)
            scale_x = CONFIG["IMG_SIZE"][1] / orig_width
            scale_y = CONFIG["IMG_SIZE"][0] / orig_height
            scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, 
                          bbox[2] * scale_x, bbox[3] * scale_y]
            scaled_bbox = [max(0.0, min(scaled_bbox[i], CONFIG["IMG_SIZE"][i % 2])) for i in range(4)]
            yield img, scaled_bbox, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=CONFIG["IMG_SIZE"] + (3,), dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    def cast_to_float32(images, bboxes, labels):
        images = tf.cast(images, tf.float32)
        bboxes = tf.cast(bboxes, tf.float32)
        print(f"Dataset cast - images: {images.dtype}, bboxes: {bboxes.dtype}")
        return images, bboxes, labels
    dataset = dataset.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def compute_iou(true_bboxes, predicted_bboxes):
    x1 = tf.maximum(true_bboxes[:, 0], predicted_bboxes[:, 0])
    y1 = tf.maximum(true_bboxes[:, 1], predicted_bboxes[:, 1])
    x2 = tf.minimum(true_bboxes[:, 2], predicted_bboxes[:, 2])
    y2 = tf.minimum(true_bboxes[:, 3], predicted_bboxes[:, 3])
    
    inter_area = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    true_area = (true_bboxes[:, 2] - true_bboxes[:, 0]) * (true_bboxes[:, 3] - true_bboxes[:, 1])
    pred_area = (predicted_bboxes[:, 2] - predicted_bboxes[:, 0]) * (predicted_bboxes[:, 3] - predicted_bboxes[:, 1])
    union_area = true_area + pred_area - inter_area
    
    return tf.reduce_mean(inter_area / (union_area + 1e-6))

def test_model(model, test_ds):
    print("Starting test_model with direct calls")
    total_recon_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    for batch in test_ds:
        images, true_bboxes, _ = batch
        print(f"Batch input dtypes - images: {images.dtype}, true_bboxes: {true_bboxes.dtype}")
        images = tf.cast(images, tf.float32)
        true_bboxes = tf.cast(true_bboxes, tf.float32)
        print(f"Before components - images: {images.dtype}, true_bboxes: {true_bboxes.dtype}")
        print(f"images min: {tf.reduce_min(images)}, max: {tf.reduce_max(images)}")
        
        z_mean, z_log_var = model.encoder(images, training=False)
        print(f"Encoder outputs - z_mean: {z_mean.dtype}, z_log_var: {z_log_var.dtype}")
        print(f"z_mean min: {tf.reduce_min(z_mean)}, max: {tf.reduce_max(z_mean)}")
        print(f"z_log_var min: {tf.reduce_min(z_log_var)}, max: {tf.reduce_max(z_log_var)}")
        z = model.reparameterize(z_mean, z_log_var)
        print(f"z min: {tf.reduce_min(z)}, z max: {tf.reduce_max(z)}")
        reconstructed = model.decoder(z, training=False)
        print(f"reconstructed min: {tf.reduce_min(reconstructed)}, max: {tf.reduce_max(reconstructed)}")
        predicted_bbox = model.bbox_predictor(z, training=False)
        print(f"Outputs - reconstructed: {reconstructed.dtype}, predicted_bbox: {predicted_bbox.dtype}")
        
        recon_diff = tf.square(images - reconstructed)
        print(f"recon_diff min: {tf.reduce_min(recon_diff)}, max: {tf.reduce_max(recon_diff)}")
        recon_sum = tf.reduce_sum(recon_diff, axis=[1, 2, 3])
        recon_loss = tf.reduce_mean(recon_sum)
        total_recon_loss += recon_loss
        
        iou = compute_iou(true_bboxes, predicted_bbox)
        total_iou += iou
        
        num_batches += 1
        print(f"Batch {num_batches}: Recon Loss: {recon_loss:.4f}, IoU: {iou:.4f}")
    
    avg_recon_loss = total_recon_loss / num_batches
    avg_iou = total_iou / num_batches
    print(f"\nTest Results - Avg Recon Loss: {avg_recon_loss:.4f}, Avg IoU: {avg_iou:.4f}")
    return avg_recon_loss, avg_iou

def main():
    test_dirs = Path(CONFIG["TEST_FILE"]).read_text().strip().split("\n")
    with open(CONFIG["ANNOTATION_FILE"], "r") as f:
        annotations = json.load(f)
    test_paths = get_image_paths_from_dirs(test_dirs)
    test_ds = create_tf_dataset(test_paths, annotations, CONFIG["BATCH_SIZE"])
    
    encoder = build_encoder((*CONFIG["IMG_SIZE"], 3), CONFIG["LATENT_DIM"])
    decoder = build_decoder(CONFIG["LATENT_DIM"])
    bbox_predictor = build_bbox_predictor(CONFIG["LATENT_DIM"])
    model = CVAE(encoder, decoder, bbox_predictor)
    
    # Force fresh model—comment out weights loading
    # checkpoint_dir = CONFIG["CHECKPOINT_DIR"]
    # weight_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.weights.h5"))
    # if not weight_files:
    #     print("No weights found—using fresh model.")
    # else:
    #     latest_weights = max(weight_files, key=os.path.getctime)
    #     print(f"Loading weights from: {latest_weights}")
    #     try:
    #         model.load_weights(latest_weights)
    #         print("Weights loaded successfully")
    #     except Exception as e:
    #         print(f"Failed to load weights: {e}—using fresh model.")
    print("Using fresh model—no weights loaded.")
    
    print("✅ Starting testing with fresh model")
    test_model(model, test_ds)

if __name__ == "__main__":
    IMAGE_DIR = CONFIG["MAIN_IMAGE_DIR"]
    main()