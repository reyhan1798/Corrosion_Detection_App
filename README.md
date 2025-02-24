# 🛠️ Corrosion Detection App

**Live Web App:** [Steel Corrosion Detection App with Severity Scores](https://huggingface.co/spaces/rey1995/Steel_Corrosion_Detection_App_with_Severity_Scores)

## 📌 Overview
The Corrosion Detection App is a **deep learning-powered tool** designed to analyze and detect corrosion in **steel structures**. It allows users to:

✅ Upload one or multiple images.
✅ Detect corrosion using an **HRNet-based model**.
✅ Display the processed images with **corrosion probability maps**.
✅ Calculate and show a **severity score (1 to 10)** for each image.
✅ Download all results (processed images + severity scores) in a **ZIP file**.

---
## 🏗️ Code Breakdown

### 📥 1. Imports
```python
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
```
### 🖥️ 2. Streamlit Configuration
```python
st.set_page_config(page_title="Corrosion Detection App", layout="wide")
```
Configures the **web app layout and title**.

---
## 🎯 3. Main Functionality

### 📂 3.1. User Interface
```python
st.title("🛠️ Corrosion Detection App")
st.markdown("Upload an image to analyze corrosion severity.")

uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
```
Provides a **file uploader** for image selection.

### ⚙️ 3.2. Model Configuration
```python
class Args:
    model = "fold8_epoch100.pt"
    gt = None
    target = 1
    n_MC = 24  # Number of Monte Carlo samples
    out_res = None
    thresh = 0.75  # Threshold for corrosion detection
    factor = None
```
Defines **configuration parameters** for the model.

### 🧮 3.3. Corrosion Severity Calculation
```python
def calculate_severity(out, threshold=0.75):
    out_np = out.cpu().numpy()
    if out_np.ndim > 2:
        out_np = out_np.mean(axis=0)  # Average across channels
    corroded_pixels = out_np > threshold
    total_pixels = out_np.size
    corroded_pixel_count = np.sum(corroded_pixels)
    severity = (corroded_pixel_count / total_pixels) * 10
    severity = min(max(severity, 1), 10)
    return round(severity, 1)
```
✅ Converts **model output** to NumPy.
✅ Computes **corrosion severity** (scaled from **1 to 10**).

### 🔍 3.4. Model Loading
```python
hypesfile = "hypes.json"
with open(hypesfile, 'r') as f:
    hypes = json.load(f)
modelfile = hypes['model']
```
Loads the **model configuration** from `hypes.json`.

### 🚀 3.5. Model Initialization
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hypes['arch']['config'] == 'HRNet_do':
    seg_model = HRNet_dropout(config=hypes).to(device)
elif hypes['arch']['config'] == 'HRNet_var':
    seg_model = HRNet_var(config=hypes).to(device)
pretrained_dict = torch.load(modelfile, map_location=device)
seg_model.load_state_dict(pretrained_dict)
```
✅ Loads the **pre-trained HRNet model**.
✅ Supports **dropout** and **variational inference**.

### 🖼️ 3.6. Image Processing
```python
input_transforms = transforms.Compose([
    transforms.Resize(input_res),
    transforms.ToTensor(),
    transforms.Normalize(hypes['data']['pop_mean'], hypes['data']['pop_std0'])
])
image_tensor = input_transforms(image).unsqueeze(dim=0).to(device)
```
✅ **Resizes**, **converts**, and **normalizes** the input image.

### 📊 3.7. Model Inference
```python
with torch.no_grad():
    seg_model.eval()
    out = []
    for j in range(args.n_MC):
        outDict = seg_model(image_tensor)
        out.append(outDict['out'].squeeze().detach())
    out = torch.stack(out)
    out = normalize_tensor(out)
```
✅ Runs the **HRNet model** on the input image.
✅ Performs **Monte Carlo sampling** for uncertainty estimation.

### 🎨 3.8. Visualization
```python
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(out.mean(dim=0).cpu().numpy(), cmap="jet")
plt.colorbar(im, ax=ax, label="Corrosion Probability")
ax.set_title("Processed Image (Corrosion Detection)")
ax.axis("off")
st.pyplot(fig)
```
✅ Displays the **corrosion probability map** using `matplotlib`.

### 📊 3.9. Severity Display
```python
severity_score = calculate_severity(out, threshold=0.5)
st.subheader(f"Estimated Corrosion Severity: {severity_score} / 10")
st.progress(severity_score / 10)
```
✅ Shows a **severity score** along with a **progress bar**.

### 📥 3.10. Download Results
```python
if results:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        severity_text = "\n".join(f"{filename}: {severity}/10" for filename, severity, _ in results)
        zipf.writestr("corrosion_severity_scores.txt", severity_text)
        for filename, severity, img_buf in results:
            zipf.writestr(f"{filename}_processed.png", img_buf.getvalue())
    zip_buffer.seek(0)
    st.download_button(label="📥 Download All Results", data=zip_buffer, file_name="corrosion_results.zip", mime="application/zip")
```
✅ **Compiles all results** into a ZIP file.
✅ Provides a **download button** for the user.

---
## 🔄 Workflow
1️⃣ **User Uploads Images** via Streamlit.
2️⃣ **Preprocessing**: Image is resized, converted to tensor, and normalized.
3️⃣ **Model Inference**: The HRNet model generates a corrosion probability map.
4️⃣ **Severity Calculation**: Corrosion severity is computed.
5️⃣ **Visualization**: The processed image and severity are displayed.
6️⃣ **Download Results**: Users can download images + severity scores in a ZIP file.

---
## ✨ Features
✅ **Multiple Image Upload**: Process several images at once.
✅ **Monte Carlo Uncertainty Estimation**.
✅ **Interactive UI**: Built with **Streamlit**.
✅ **Downloadable Reports**.

---
## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
---
### 🎉 *Enjoy corrosion detection with deep learning!*
---

