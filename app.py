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

st.set_page_config(page_title="Corrosion Detection App", layout="wide")

def main():
    st.title("🛠️ Corrosion Detection App")
    st.markdown("Upload an image to analyze corrosion severity.")
    
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    results = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_column_width=True)

            device = torch.device("cpu")

            # Define arguments as a class
            class Args:
                model = "fold8_epoch100.pt"  # Adjust path if needed
                image = uploaded_file
                gt = None
                target = 1
                n_MC = 24
                out_res = None
                thresh = 0.75
                factor = None

            args = Args()

            def calculate_severity(out, threshold=0.75):
                # Convert tensor to numpy for processing
                out_np = out.cpu().numpy()

                # Create a binary mask (1 for corrosion, 0 for non-corrosion)
                corroded_pixels = out_np > threshold

                # Count total and corroded pixels
                total_pixels = out_np.size
                corroded_pixel_count = np.sum(corroded_pixels)

                # Compute severity based on corroded area percentage
                severity = (corroded_pixel_count / total_pixels) * 10

                # Ensure score is between 1 and 10
                severity = min(max(severity, 1), 10)

                return round(severity, 1)

            hypesfile = "hypes.json"
            with open(hypesfile, 'r') as f:
                hypes = json.load(f)

            modelfile = hypes['model']
            if args.out_res is not None:
                if len(args.out_res) == 1:
                    input_res = [args.out_res[0], args.out_res[0]]
                elif len(args.out_res) > 2:
                    print('out res must be length 2')
                    exit()
                else:
                    input_res = args.out_res
            else:
                input_res = hypes['arch']['image_shape'][1:3]
            num_classes = hypes['arch']['num_classes']
            class_colors = hypes['data']['class_colours']
            class_labels = hypes['data']['class_labels']

            input_transforms = transforms.Compose([
                transforms.Resize(input_res),
                transforms.ToTensor(),
                transforms.Normalize(hypes['data']['pop_mean'], hypes['data']['pop_std0'])
            ])

            image_orig = image
            image = input_transforms(image_orig)
            image = image.unsqueeze(dim=0).to(device)

            if hypes['arch']['config'] == 'HRNet_do':
                seg_model = HRNet_dropout(config=hypes).to(device)
                bayesMethod = 'dropout'
            elif hypes['arch']['config'] == 'HRNet_var':
                seg_model = HRNet_var(config=hypes).to(device)
                bayesMethod = 'variational'

            if hypes['arch']['config'][:5] == 'HRNet':
                pretrained_dict = torch.load(modelfile, map_location=device, weights_only=False)
                if 'state_dict' in pretrained_dict:
                    pretrained_dict = pretrained_dict['state_dict']

                prefix = "module."
                keys = sorted(pretrained_dict.keys())
                for key in keys:
                    if key.startswith(prefix):
                        newkey = key[len(prefix):]
                        pretrained_dict[newkey] = pretrained_dict.pop(key)

                if "_metadata" in pretrained_dict:
                    metadata = pretrained_dict["_metadata"]
                    for key in list(metadata.keys()):
                        if len(key) == 0:
                            continue
                        newkey = key[len(prefix):]
                        metadata[newkey] = metadata.pop(key)

                seg_model.load_state_dict(pretrained_dict)
                seg_model.to(device)
            else:
                seg_model.load_state_dict(torch.load(modelfile, map_location=device))

            weights_factor = hypes['data']['class_weights']
            with st.spinner("Processing image..."):
                with torch.no_grad():
                    seg_model.train(False)

                    out = []
                    var = []
                    for j in range(args.n_MC):
                        with torch.no_grad():
                            outDict = seg_model(image)
                            out.append(outDict['out'].squeeze().detach())
                            var.append(outDict['logVar'].squeeze().detach())
                        print(' ' + '>' * j + 'X' + '<' * (args.n_MC - j - 1), end="\r", flush=True)

                    out = torch.stack(out)
                    var = torch.stack(var)
                    varmax = var.max()
                    varmin = var.min()
                    out = normalize_tensor(out)
                    var = normalize_tensor(var) * (varmax - varmin)

                    processed_image = out.mean(dim=0).cpu().numpy()
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(processed_image, cmap="jet")
                    plt.colorbar(im, ax=ax, label="Corrosion Probability")
                    ax.set_title("Processed Image (Corrosion Detection)")
                    ax.axis("off")
                    st.pyplot(fig)

                    severity_score = calculate_severity(out, threshold=0.5)
                    st.subheader(f"Estimated Corrosion Severity: {severity_score} / 10")
                    st.progress(severity_score / 10)

                    # Save results
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format="png")
                    img_buffer.seek(0)
                    results.append((uploaded_file.name, severity_score, img_buffer))

        # Download button (appears after all images are processed)
        if results:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                # Create a single file with all severity scores
                severity_text = "\n".join(f"{filename}: {severity}/10" for filename, severity, _ in results)
                zipf.writestr("corrosion_severity_scores.txt", severity_text)
                for filename, severity, img_buf in results:
                    zipf.writestr(f"{filename}_processed.png", img_buf.getvalue())
            zip_buffer.seek(0)
            st.download_button(
                label="📥 Download All Results",
                data=zip_buffer,
                file_name="corrosion_results.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()