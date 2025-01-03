import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import streamlit as st

# Define the Advanced Autoencoder model
class AdvancedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use EfficientNet as encoder
        efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(efficientnet.features))

        # Advanced decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Process image using the trained model
def process_image(model, image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)

    output = output.squeeze().permute(1, 2, 0).numpy()
    output = output * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    output = (output * 255).clip(0, 255).astype(np.uint8)
    return output

# Streamlit application
def run_app():
    st.title('Effecient Net - Image Autoencoder')

    # Load the model
    model = AdvancedAutoencoder()
    model.load_state_dict(torch.load('autoencoder_epoch_9.pth', map_location='cpu'))
    model.eval()

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image)

        with col2:
            st.subheader("Reconstructed")
            reconstructed = process_image(model, image)
            st.image(reconstructed)

if __name__ == '__main__':
    run_app()
