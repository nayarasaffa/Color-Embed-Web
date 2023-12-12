import streamlit as st
import cv2
import albumentations
import albumentations.pytorch
import torch
import numpy as np
from PIL import Image
from io import BytesIO

import Color2Embed_pytorch.color2embed.config as config
from Color2Embed_pytorch.color2embed.models import Color2Embed

# Load the pre-trained model
model = Color2Embed(config.COLOR_EMBEDDING_DIM)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('Color2Embed_weights.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()
model = model.to(device)

# Function to convert RGBA to RGB
def convert_rgba_to_rgb(image):
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

# Function to predict colors
def predict_colors(target_image, ref_image, model):
    transform_torch = albumentations.Compose(
        [
            albumentations.ToFloat(),
            albumentations.pytorch.ToTensorV2(),
        ]
    )

    # read images
    MAX_SIZE = 1024
    ref_image = convert_rgba_to_rgb(np.array(ref_image))
    target_image = convert_rgba_to_rgb(np.array(target_image))

    delta = max(target_image.shape) / MAX_SIZE
    if delta > 1.0:
        target_image = cv2.resize(target_image, (0, 0), fx=1 / delta, fy=1 / delta)
    color_src = ref_image
    grayscale_dst = target_image

    # convert to input
    color_src = cv2.resize(color_src, grayscale_dst.shape[:2], interpolation=cv2.INTER_LINEAR)
    color_src = transform_torch(image=color_src)['image']

    grayscale_dst = cv2.cvtColor(grayscale_dst, cv2.COLOR_BGR2LAB)
    grayscale_dst = transform_torch(image=grayscale_dst)['image']

    grayscale_dst = grayscale_dst[0, ...].unsqueeze(0)

    # predict ab channels
    with torch.no_grad():
        pab = model(grayscale_dst.to(device).unsqueeze(0), color_src.to(device).unsqueeze(0))

    # create predicted image
    merged_lab = torch.cat((grayscale_dst.to(device).unsqueeze(0), pab), 1)
    prgb = merged_lab.cpu().numpy()
    prgb = np.transpose(prgb, (0, 2, 3, 1))
    for i in range(prgb.shape[0]):
        prgb[i] = cv2.cvtColor(np.clip(prgb[i] * 255, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

        merged_lab[i] = transform_torch(image=prgb[i])['image'].to(device) / 255.

    result_image = np.transpose((merged_lab.cpu().numpy()[0] * 255).astype(np.uint8), (1, 2, 0))

    return result_image

# Streamlit UI
st.title("Color2Embed Web App")

# File Upload
uploaded_target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
uploaded_ref_image = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"])

# Check if images are uploaded
if uploaded_target_image and uploaded_ref_image:
    target_image = Image.open(uploaded_target_image)
    ref_image = Image.open(uploaded_ref_image)

    # Display images
    st.image([target_image, ref_image], caption=["Target Image", "Reference Image"], use_column_width=True)

    # Predict colors and display result
    result_image = predict_colors(target_image, ref_image, model)
    st.image(result_image, caption="Result Image", use_column_width=True)
