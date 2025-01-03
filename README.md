# Autoencoder for Image Reconstruction Using EfficientNet

## Overview
This project implements an advanced autoencoder designed to reconstruct food images with high fidelity. The model leverages EfficientNet-B0 as the encoder and a custom-designed decoder to achieve efficient feature extraction and detailed image reconstruction.

## Features
- **EfficientNet-B0 Encoder**: Utilizes a pre-trained EfficientNet-B0 model to extract high-level features.
- **Custom Decoder**: Designed with transpose convolutions, instance normalization, Leaky ReLU activation, and dropout for effective reconstruction.
- **Food-101 Dataset**: Uses a comprehensive dataset of 101,000 food images across 101 categories.
- **Streamlit Application**: A user-friendly interface to upload images and view reconstruction results.

## Dataset
The project uses the [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/):
- **Number of images**: 101,000
- **Categories**: 101 food types
- **Split**: 75,750 for training and 25,250 for testing
- **Preprocessing**: Images resized to 224x224 pixels and normalized to [0, 1]

## Model Architecture
### Encoder
- Pre-trained EfficientNet-B0
- Extracts meaningful features from input images

### Bottleneck
- Compresses features into a latent representation

### Decoder
- **Transpose Convolutions**: For upsampling features
- **Instance Normalization**: Stabilizes feature distributions
- **Leaky ReLU**: Handles non-linearities effectively
- **Dropout**: Reduces overfitting

## Training Pipeline
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW with weight decay
- **Learning Rate Scheduler**: Cosine Annealing LR
- **Epochs**: 10
- **Hardware**: Trained on a GPU-enabled environment

## Results
- **Loss**: Steadily declined across epochs
- **Pixel Accuracy**: Achieved 72% (AVG Accuracy for 10th epochs)

## Application
A Streamlit-based web application is included to interact with the model:
1. Upload an image of food.
2. View the original and reconstructed images side by side.
3. Visit [Effecient Net - Image Autoencoder](https://image-reconstruction.streamlit.app/)

## Installation
1. Clone the repository:
   ```bash
   [git clone https://github.com/yourusername/food-image-autoencoder.git](https://github.com/AndreasKrist/ImageReconstructionEffecientNet.git)
   cd ImageReconstructionEffecientNet
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Food-101 dataset and place it in the appropriate directory.(automatically downloaded from provided code)

## Usage
### Training
To train the model, run:
```bash
python main.ipynb
```

### Inference
To test the model, run:
```bash
python test.py
```

### Streamlit App
To launch the web application:
```bash
streamlit run app.py
```

## Future Work
- Implementing GAN-based reconstruction models for higher fidelity
- Exploring larger datasets for broader generalization
- Optimizing training for faster convergence

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The creators of the Food-101 dataset
- The developers of EfficientNet and PyTorch
- Open-source contributors and the deep learning community
