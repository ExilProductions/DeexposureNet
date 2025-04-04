# DeexposureNet

This project implements **DeexposureNet**, a deep learning model designed to correct overexposed images. Overexposure occurs when too much light washes out details in a photograph, and DeexposureNet uses a convolutional neural network to restore these images by learning to map overexposed patches to their properly exposed counterparts. The model is trained on pairs of original and overexposed image patches and can process full images by dividing them into patches, correcting each one, and stitching them back together.

## Installation

To set up this project, ensure you have **Python 3.x** installed. The following Python packages are required:

- `torch` - For building and training the neural network.
- `torchvision` - For image transformations and utilities.
- `pillow` - For image loading and manipulation.
- `tqdm` - For progress bars during training and processing.

Install these packages using pip:

```bash
pip install torch torchvision pillow tqdm
```

## Project Structure

The project is organized as follows:

- **`data/`**: Contains the `download_dataset.bat` script (for Windows) and will store the dataset in `data/original`.
- **`train-model.py`**: Script to train the model.
- **`test-model.py`**: Script to test the model on a single overexposed image.
- **`prepare-dataset.py`**: Script to generate overexposed images from original ones.

## Dataset Preparation

The dataset consists of pairs of original and overexposed images. Here's how to prepare it:

### 1. Download the Original Images

- **For Windows Users**:
  1. Navigate to the `data/` directory:
     ```bash
     cd data
     ```
  2. Run the provided batch script:
     ```bash
     download_dataset.bat
     ```
     This downloads a ZIP file from [Hugging Face](https://huggingface.co/datasets/mebix/images/blob/main/txt2img-images.zip), extracts it, and places the images in `data/original`.

- **For Linux/macOS Users or Manual Download**:
  1. Download the ZIP file from [this link](https://huggingface.co/datasets/mebix/images/blob/main/txt2img-images.zip).
  2. Extract it to `data/original`. Ensure the images are directly under `data/original` (not nested in another folder).

### 2. Generate Overexposed Images

After obtaining the original images, generate their overexposed versions:

```bash
python prepare-dataset.py
```

This script reads images from `data/original`, applies brightness and saturation enhancements to simulate overexposure, and saves the results to `data/overexposed`. The filenames are preserved to match the original images, which is necessary for the dataset class.

## Training

To train the DeexposureNet model:

1. Ensure the `data/original` and `data/overexposed` directories contain the respective images.
2. Run the training script:

```bash
python train-model.py
```

### What Happens During Training:
- The script loads image pairs using `DeexposureNetDataset`, extracting random 128x128 patches.
- It trains the model for **20 epochs** using:
  - **Batch size**: 8
  - **Optimizer**: Adam (learning rate = 0.0002)
  - **Loss function**: Mean Squared Error (MSE)
- The trained model weights are saved to `saved_models/DeexposureNet.pth`.

You can modify hyperparameters (e.g., epochs, batch size, learning rate) in `train-model.py` if needed.

## Additional Information

- **Model Architecture**: DeexposureNet is a simple encoder-decoder network with convolutional layers for feature extraction and transposed convolutions for reconstruction.
- **Image Requirements**: The model expects RGB images. Ensure input images for inference are in this format.
- **Troubleshooting**:
  - If the dataset download fails, verify the URL or manually extract the ZIP.
  - If training or testing fails due to missing files, check that all directories (`data/original`, `data/overexposed`, `saved_models`) are correctly populated.
