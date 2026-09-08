import os

import clip
import pandas as pd
import torch
from PIL import Image

# Install clip
"""
Do it manually, it didnt work from requirement-clip
activate env, python -m pip install git+https://github.com/openai/CLIP.git
Also run manually, activate env, python src/evaluate_clip_test_from_file.py
"""

# Paths
input_file = "results/perturbation/perturbation_metrics.csv"
image_dir = "data/flickr8k/images"
output_file = "results/perturbation/perturbation_metrics_with_clipscore.csv"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("CLIP model loaded")

# Load saved predictions
df = pd.read_csv(input_file)
print("Total rows:", len(df))


# CLIPScore function
def calculate_clipscore(image_path, caption):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Same prefix used in our earlier CLIPScore evaluation
    text = "A photo depicts " + caption
    text_input = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (image_features @ text_features.T).item()

    # Official-style CLIPScore scaling
    score = 2.5 * max(similarity, 0)
    return score


# Calculate scores
clip_scores = []

for i, row in df.iterrows():
    image_name = row["image"]
    caption = str(row["perturbed_caption"]).strip()
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print("Missing image:", image_path)
        clip_scores.append(None)
        continue

    score = calculate_clipscore(image_path, caption)
    clip_scores.append(score)
    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(df)}")


# Add scores
df["CLIPScore"] = clip_scores

# Save
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)

# Summary
print("\nCLIPScore evaluation complete")
print("-" * 60)
print("Average CLIPScore:", round(df["CLIPScore"].mean(), 4))
print("Missing scores:", df["CLIPScore"].isna().sum())
print("\nSaved to:")
print(output_file)

# Preview
print(
    "\n",
    df[["image", "perturbation_type", "perturbed_caption", "CLIPScore"]]
    .head(10)
    .to_string(index=False),
)
