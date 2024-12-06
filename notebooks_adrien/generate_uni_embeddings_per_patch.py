import os
from pathlib import Path
import pickle
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
import timm
import h5py

from huggingface_hub import login
login(token='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')
from huggingface_hub import hf_hub_download

# Configure model and transformation
MODEL_PATH = "/mnt/HDD8TO/models/"
LOCAL_DIR = '/mnt/HDD8TO/data/hest'
PATCH_SIZE_LV0 = 1024

# Download the model if needed (uncomment if required)
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=MODEL_PATH, force_download=True)

# Initialize the encoder model
encoder_model = timm.create_model(
    "vit_large_patch16_224",
    img_size=224,
    patch_size=16,
    init_values=1e-5,
    num_classes=0,
    dynamic_img_size=True
)
encoder_model.load_state_dict(
    torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location="cpu"),
    strict=True
)
encoder_model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Locate .h5 and corresponding .tif files
h5_files = list(Path(LOCAL_DIR).rglob('*.h5'))
tif_files = list(Path(LOCAL_DIR).rglob('*.tif'))

# Initialize container for embeddings
all_embeddings = {}

# Process each .h5 file
for h5_file in h5_files[:5]:
    basefile = os.path.basename(h5_file).replace(".h5", "")
    image_fname = next((f for f in tif_files if basefile in f.name), None)

    if not image_fname:
        print(f"No matching .tif file found for {h5_file}")
        continue

    print(f"Processing file: {h5_file}")

    # Open the .h5 file
    with h5py.File(h5_file, 'r') as file:
        if "img" not in file or "coords" not in file:
            print(f"Missing required keys in {h5_file.name}")
            continue

        image_patches = file["img"][:]
        coords = file["coords"][:]

    # Generate embeddings for image patches
    features = []
    for i in tqdm(range(image_patches.shape[0]), desc=f"Extracting patches from {h5_file.name}"):
        patch = Image.fromarray(image_patches[i])
        patch_tensor = transform(patch).unsqueeze(dim=0)  # Prepare input tensor

        with torch.inference_mode():
            features_per_patch = encoder_model(patch_tensor)

        features.append(features_per_patch)

    # Save embeddings and coordinates
    all_embeddings[basefile] = {
        "features": features,
        "coords": coords
    }

    # Save all embeddings to a pickle file
    output_file = f"embeddings_{basefile}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_embeddings, f)

    print(f"Embeddings saved to {output_file}")
