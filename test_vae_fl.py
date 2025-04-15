import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, matthews_corrcoef, precision_score, recall_score, accuracy_score, auc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/sanderwald/Projects/dissertationProject/test_log_fl_5050_best_latent64_epochs5_gamma2.txt"),
        logging.StreamHandler(),
    ],
)

CONFIG = {
    "MAIN_IMAGE_DIR": "/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/",
    "TEST_FILE": "/media/sanderwald/Project_Files/Dissertation_Project/data/Splits/50_50/test_split.txt",
    "OUTPUT_DIR": "outputs_5050_focal_best_latent64_epochs5_gamma2",
    "IMG_SIZE": (360, 203),
    "BATCH_SIZE": 2,
    "LATENT_DIM": 64,
    "FOCAL_GAMMA": 2.0,
    "FOCAL_ALPHA": 0.5,
}

# Model Components
def build_encoder(input_shape, latent_dim):
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
    latent_inputs = layers.Input(shape=(latent_dim,), dtype=tf.float32)
    x = layers.Dense(23 * 13 * 64, activation="relu", dtype=tf.float32)(latent_inputs)
    x = layers.Reshape((23, 13, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu", dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(3, 3, strides=(2, 2), padding="same", activation="sigmoid", dtype=tf.float32)(x)
    x = layers.Cropping2D(cropping=((4, 4), (2, 3)))(x)
    return Model(latent_inputs, x, name="decoder")

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

# Plotting Functions
def plot_roc_curve(y_true, y_scores, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()

def plot_confusion_matrix(conf_matrix, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

# Main testing function
def main():
    logging.info("Starting testing execution")

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

    # Build the model
    logging.info("Building encoder")
    encoder = build_encoder((CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 3), CONFIG["LATENT_DIM"])
    logging.info("Building decoder")
    decoder = build_decoder(CONFIG["LATENT_DIM"])
    logging.info("Building CVAE model")
    model = CVAE(encoder, decoder, CONFIG["LATENT_DIM"])

    # Build the model by passing a single batch through it
    logging.info("Creating TensorFlow test dataset to build model")
    test_file_path = CONFIG["TEST_FILE"]
    test_dirs = list(set(Path(test_file_path).read_text().strip().split("\n")))
    paths_0, paths_1 = load_image_paths_with_labels(test_dirs)
    test_ds = create_tf_dataset(paths_0, paths_1)
    for batch in test_ds.take(1):
        images, true_bboxes, true_labels, paths = batch
        _ = model((images, true_bboxes), training=False)
        break

    # Debug: Check the number of variables in the classifier layer
    logging.info(f"Classifier layer variables: {len(model.classifier.trainable_variables)}")

    # Load the checkpoint directly
    checkpoint_dir = os.path.join(CONFIG["OUTPUT_DIR"], f"gamma_{CONFIG['FOCAL_GAMMA']}_alpha_{CONFIG['FOCAL_ALPHA']}", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "epoch_5_v18_focal.weights.h5")
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    else:
        logging.error(f"No checkpoint found at {checkpoint_path}")
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Evaluate the model
    logging.info("Starting evaluation")
    all_true_labels = []
    all_recon_errors = []
    all_image_paths = []
    all_class_logits = []

    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], f"gamma_{CONFIG['FOCAL_GAMMA']}_alpha_{CONFIG['FOCAL_ALPHA']}", "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(test_ds):
        try:
            images, true_bboxes, true_labels, paths = batch
            reconstructed, z_mean, z_log_var, class_logits = model((images, true_bboxes), training=False)

            # Compute reconstruction error
            pixel_count = CONFIG["IMG_SIZE"][0] * CONFIG["IMG_SIZE"][1] * 3 / 1000
            recon_diff = tf.square(images - reconstructed)
            recon_diff = tf.cast(recon_diff, tf.float32)
            recon_diff = tf.clip_by_value(recon_diff, 0.0, 10.0)
            recon_sum = tf.reduce_sum(recon_diff, axis=[1, 2, 3])
            recon_error = recon_sum / pixel_count

            all_true_labels.extend(true_labels.numpy())
            all_recon_errors.extend(recon_error.numpy())
            all_image_paths.extend(paths.numpy())
            all_class_logits.extend(class_logits.numpy().flatten())

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"Processed batch {batch_idx + 1}")

        except Exception as e:
            logging.error(f"Error at batch {batch_idx + 1}: {str(e)}. Skipping batch.")
            continue

    # Save combined data
    np.save(os.path.join(output_dir, "true_labels.npy"), np.array(all_true_labels))
    np.save(os.path.join(output_dir, "recon_errors.npy"), np.array(all_recon_errors))
    np.save(os.path.join(output_dir, "image_paths.npy"), np.array(all_image_paths))
    np.save(os.path.join(output_dir, "class_logits.npy"), np.array(all_class_logits))

    # Compute EER threshold using reconstruction errors
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_recon_errors)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_idx]
    logging.info(f"EER threshold: {eer_threshold:.4f}")

    # Compute metrics using reconstruction error
    pred_labels = (all_recon_errors > eer_threshold).astype(int)
    conf_matrix = confusion_matrix(all_true_labels, pred_labels)
    accuracy = accuracy_score(all_true_labels, pred_labels)
    precision = precision_score(all_true_labels, pred_labels)
    recall = recall_score(all_true_labels, pred_labels)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mcc = matthews_corrcoef(all_true_labels, pred_labels)
    roc_auc = auc(fpr, tpr)
    precision_pr, recall_pr, _ = precision_recall_curve(all_true_labels, all_recon_errors)
    pr_auc = auc(recall_pr, precision_pr)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    fpr_value = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0]) if (conf_matrix[0, 1] + conf_matrix[0, 0]) > 0 else 0
    fnr_value = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    logging.info("Metrics using Reconstruction Error:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"MCC: {mcc:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    logging.info(f"PR AUC: {pr_auc:.4f}")
    logging.info(f"EER: {fpr[eer_idx]:.4f}")
    logging.info(f"Specificity: {specificity:.4f}")
    logging.info(f"FPR: {fpr_value:.4f}")
    logging.info(f"FNR: {fnr_value:.4f}")
    logging.info(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    logging.info(f"Confusion Matrix: \n{conf_matrix}")

    # Save metrics and plots
    np.save(os.path.join(output_dir, "pred_labels.npy"), pred_labels)
    np.save(os.path.join(output_dir, "pred_scores.npy"), np.array(all_recon_errors))
    np.save(os.path.join(output_dir, "confusion_matrix.npy"), conf_matrix)
    np.save(os.path.join(output_dir, "metrics.npy"), {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "eer": fpr[eer_idx],
        "specificity": specificity,
        "fpr": fpr_value,
        "fnr": fnr_value,
        "balanced_accuracy": balanced_accuracy,
    })

    plot_roc_curve(all_true_labels, all_recon_errors, output_dir)
    plot_precision_recall_curve(all_true_labels, all_recon_errors, output_dir)
    plot_confusion_matrix(conf_matrix, output_dir)

    logging.info("Evaluation completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Main execution failed with error: {str(e)}", exc_info=True)
        raise