# Dissertation Project - Single and Ensemble CVAE Training

## Overview
This repository contains the code and data for training both single and ensemble Conditional Variational Autoencoders (CVAEs) as part of a dissertation project. The study evaluates anomaly detection performance across data splits: `5050` (50% train/50% test), `6040` (60% train/40% test), and `9010` (90% train/10% test). The single CVAE provides baseline results (e.g., AUC-ROC 0.7397 for `5050`, 0.7355 for `6040`), while the ensemble CVAE enhances robustness. Loss curves are intended as PNG outputs for both models.

## Requirements
- **Python**: 3.8 or higher
- **TensorFlow**: 2.6 or higher (with GPU support)
- **NumPy**: For numerical operations
- **Matplotlib**: For generating loss curve plots
- **Conda Environment**: `tf_gpu` with NVIDIA GPU drivers and CUDA toolkit (version 12.4 compatible)

## Setup
1. Activate the Conda environment: `conda activate tf_gpu`
2. Install required dependencies: `pip install tensorflow numpy matplotlib`
3. Verify GPU setup: `nvidia-smi` (ensure CUDA and cuDNN are configured).

## Usage
- **Train Single CVAE**: Run the single CVAE training script for a specific split (assuming `train_single_cvae.py` or similar exists; adjust path if different): `python train_single_cvae.py --split 5050`, `python train_single_cvae.py --split 6040`, `python train_single_cvae.py --split 9010` (Note: If the single CVAE script differs, e.g., no `--split` argument, refer to its documentation or modify accordingly).
- **Train Ensemble CVAE**: Run the ensemble training for a specific split: `python train_ensemble_cvae.py --split 5050`, `python train_ensemble_cvae.py --split 6040`, `python train_ensemble_cvae.py --split 9010`.

## Configuration
- **Data Directory**: `/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/`
- **Train Files**: Located at `/home/sanderwald/Projects/dissertationProject/data/Splits/{5050,60_40,90_10}/train_split_subset.txt`
- **Checkpoints**: Single CVAE saved to a configurable directory (e.g., `outputs_single_{SPLIT}_...` if defined), Ensemble CVAE saved to `outputs_{SPLIT}_ensemble_cvae_subset_20250520_v2/epoch_{1,2,3}/cvae_{0,1,2}`
- **Loss Curves**: Intended output to `/home/sanderwald/Projects/dissertationProject/loss_curve_{SPLIT}.png` for both models (current issue with ensemble PNG generation—see Troubleshooting).

## Parameters
- `BATCH_SIZE`: 4 (reduce to 1 if memory issues arise for either model)
- `LATENT_DIM`: 64
- `EPOCHS`: 3
- `NUM_CVAES`: 1 for single CVAE, configurable to 3 for ensemble CVAE (adjust in `CONFIG` if needed)
- `NUM_SAMPLES`: 20600 (`5050`), 24720 (`6040`), 37080 (`9010`)
- `RECON_LOSS_WEIGHT`: 0.5
- `KL_LOSS_WEIGHT`: 0.01
- `FOCAL_LOSS_WEIGHT`: 5.0
- `FOCAL_GAMMA`: 3.0
- `FOCAL_ALPHA`: 0.9
- **GPU Memory Limit**: 8000MB (reduce to 6000MB if out-of-memory errors occur)

## Data Preprocessing
- **Input Images**: Processed JPEGs from `MAIN_IMAGE_DIR`, resized to 360x203 pixels.
- **Labeling**: Derived from directory names (e.g., `_1` indicates positive class, otherwise negative).
- **Problematic Images**: Excluded set includes `/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/3_011_0/frame_03997.jpg`, `/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/3_011_0/frame_04022.jpg`, and `/media/sanderwald/Project_Files/Dissertation_Project/data/Processed/Images/1_047_0/frame_02802.jpg`.
- **Splits**: Text files list image paths or directories, with 80% train/20% validation per split.

## Evaluation Metrics
- **AUC-ROC**: Primary metric for classification performance (e.g., 0.7397 for `5050` single CVAE).
- **Loss Components**: Reconstruction loss (MSE), KL divergence loss, and focal loss tracked per epoch for both models.
- **Validation**: Performed after each epoch on the 20% validation set.

## Output
- **Model Checkpoints**: Single CVAE saved post-training (location depends on script), Ensemble CVAE saved per epoch in `CHECKPOINT_DIR` (e.g., `outputs_5050_ensemble_cvae_subset_20250520_v2/epoch_3/cvae_0`).
- **Loss Curves**: PNG plots of training and validation loss for each split and model type. Ensemble PNG generation is currently unresolved.

## Notes
- Mixed precision training enhances efficiency for both models.
- GPU memory growth is enabled with a configurable limit.
- Logs saved to `/home/sanderwald/Projects/dissertationProject/train_ensemble_cvae_{SPLIT}.log` for ensemble, and a similar path for single (if applicable).

## Future Improvements
- Automate split iteration for both single and ensemble scripts.
- Fix ensemble plot generation with fallback options.
- Standardize paths across single and ensemble scripts.
- Introduce sketetal algorithm

## Contact
For issues, contact the author 

