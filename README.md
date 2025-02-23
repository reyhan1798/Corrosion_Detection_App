# Corrosion_Detection_App
The web app link is https://huggingface.co/spaces/rey1995/Steel_Corrosion_Detection_App_with_Severity_Scores

Detailed Explanation of the Corrosion Detection App Code

1. Overview
   
The app is designed to:

Allow users to upload one or more images.

Use a deep learning model (HRNet) to detect and analyze corrosion in the images.

Display the processed images with corrosion probability maps.

Calculate and display a corrosion severity score for each image.

Provide an option to download all results (processed images and severity scores) in a ZIP file.

2. Code Breakdown

2.1. Imports

import streamlit as st

import torch

import os

import json

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

from torchvision import transforms

from utils import is_image_file, pil_loader, process_images, normalize_tensor

from HRNet import HRNet_dropout, HRNet_var

import cv2

import zipfile

import io

Streamlit: Used to create the web app interface.

Torch: PyTorch is used for loading and running the deep learning model.

PIL (Pillow): For image loading and processing.

NumPy: For numerical operations on image data.

Matplotlib: For visualizing the processed images.

Torchvision Transforms: For preprocessing images before feeding them into the model.

HRNet: Custom model architecture for corrosion detection.

Zipfile and io: For creating a downloadable ZIP file of results.

2.2. Streamlit Configuration

st.set_page_config(page_title="Corrosion Detection App", layout="wide")

Sets the page title and layout for the Streamlit app.

2.3. Main Function

The main() function is the core of the app. It handles:

User input (image uploads).

Image processing and corrosion detection.

Displaying results and providing a download option.

2.3.1. User Interface

st.title("🛠️ Corrosion Detection App")

st.markdown("Upload an image to analyze corrosion severity.")

uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

Displays a title and instructions.

Provides a file uploader for users to upload images.

2.3.2. Argument Class


class Args:

    model = "fold8_epoch100.pt"  # Path to the trained model
    
    gt = None
    
    target = 1
    
    n_MC = 24  # Number of Monte Carlo samples
    
    out_res = None
    
    thresh = 0.75  # Threshold for corrosion detection
    
    factor = None
    
A simple class to store configuration arguments for the model and processing.

2.3.3. Severity Calculation

def calculate_severity(out, threshold=0.75):

    out_np = out.cpu().numpy()
    
    if out_np.ndim > 2:
    
        out_np = out_np.mean(axis=0)  # Average across channels if multi-channel
        
    corroded_pixels = out_np > threshold
    
    total_pixels = out_np.size
    
    corroded_pixel_count = np.sum(corroded_pixels)
    
    severity = (corroded_pixel_count / total_pixels) * 10
    
    severity = min(max(severity, 1), 10)
    
    return round(severity, 1)
    
Converts the model output tensor to a NumPy array.

Creates a binary mask of corroded pixels based on the threshold.

Calculates the percentage of corroded pixels and scales it to a severity score between 1 and 10.

2.3.4. Model and Hypes File Loading

hypesfile = "hypes.json"

with open(hypesfile, 'r') as f:

    hypes = json.load(f)
    
modelfile = hypes['model']

Loads the configuration file (hypes.json) and extracts the model path.

2.3.5. Model Initialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if hypes['arch']['config'] == 'HRNet_do':

    seg_model = HRNet_dropout(config=hypes).to(device)
    
elif hypes['arch']['config'] == 'HRNet_var':

    seg_model = HRNet_var(config=hypes).to(device)
    
pretrained_dict = torch.load(modelfile, map_location=device)

seg_model.load_state_dict(pretrained_dict)

Initializes the model (HRNet with dropout or variational inference) and loads the pre-trained weights.

2.3.6. Image Processing

input_transforms = transforms.Compose([

    transforms.Resize(input_res),
    
    transforms.ToTensor(),
    
    transforms.Normalize(hypes['data']['pop_mean'], hypes['data']['pop_std0'])
    
])

image_tensor = input_transforms(image).unsqueeze(dim=0).to(device)

Preprocesses the uploaded image:

Resizes it to the required input resolution.

Converts it to a tensor.

Normalizes it using mean and standard deviation from the hypes file.

2.3.7. Model Inference

with torch.no_grad():

    seg_model.eval()
    
    out = []
    
    for j in range(args.n_MC):
    
        outDict = seg_model(image_tensor)
        
        out.append(outDict['out'].squeeze().detach())
    out = torch.stack(out)
    
    out = normalize_tensor(out)
    
Runs the model in evaluation mode.

Performs Monte Carlo sampling (n_MC times) to estimate uncertainty (if using dropout or variational inference).

Normalizes the output tensor.

2.3.8. Visualization

fig, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(out.mean(dim=0).cpu().numpy(), cmap="jet")

plt.colorbar(im, ax=ax, label="Corrosion Probability")

ax.set_title("Processed Image (Corrosion Detection)")

ax.axis("off")

st.pyplot(fig)

Displays the processed image with a color map indicating corrosion probability.

2.3.9. Severity Display

severity_score = calculate_severity(out, threshold=0.5)

st.subheader(f"Estimated Corrosion Severity: {severity_score} / 10")

st.progress(severity_score / 10)

Calculates and displays the corrosion severity score as a value between 1 and 10.

Shows a progress bar representing the severity.

2.3.10. Results Download

if results:

    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
    
        severity_text = "\n".join(f"{filename}: {severity}/10" for filename, severity, _ in results)
        
        zipf.writestr("corrosion_severity_scores.txt", severity_text)
        
        for filename, severity, img_buf in results:
        
            zipf.writestr(f"{filename}_processed.png", img_buf.getvalue())
            
    zip_buffer.seek(0)
    
    st.download_button(label="📥 Download All Results", data=zip_buffer, file_name="corrosion_results.zip", mime="application/zip")
    
Collects all results (processed images and severity scores) into a ZIP file.

Provides a download button for the user.

3. Workflow
   
User Uploads Images:

The user uploads one or more images via the Streamlit file uploader.

Image Preprocessing:

Each image is resized, converted to a tensor, and normalized.

Model Inference:

The preprocessed image is passed through the HRNet model to generate a corrosion probability map.

Severity Calculation:

The severity score is calculated based on the percentage of corroded pixels.

Visualization:

The processed image and severity score are displayed to the user.

Results Download:

The user can download all processed images and severity scores in a ZIP file.

4. Key Features

Multiple Image Upload: Users can upload and process multiple images at once.

Uncertainty Estimation: Monte Carlo sampling is used to estimate uncertainty in the model's predictions.

Interactive Interface: Streamlit provides an intuitive and interactive user interface.

Downloadable Results: Users can download all results for further analysis.
