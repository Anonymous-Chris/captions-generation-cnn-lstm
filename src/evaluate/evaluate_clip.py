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
predictions_file = "results/test_predictions.csv"
image_folder = "data/flickr8k/images"

overall_output = "results/clipscore_results.csv"
per_image_output = "results/test_predictions_with_clipscore.csv"


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("CLIP model loaded")

# Load saved predictions
df = pd.read_csv(predictions_file)
print("Number of predictions:", len(df))

# Calculate CLIPScore
clip_scores = []

for i, row in df.iterrows():
    image_name = row["image"]
    generated_caption = str(row["generated_caption"])

    image_path = os.path.join(image_folder, image_name)

    # Make sure image exists
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        clip_scores.append(None)
        continue

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    # Prepare caption

    # Prompt used by CLIPScore
    prompted_caption = "A photo depicts " + generated_caption
    text = clip.tokenize([prompted_caption], truncate=True).to(device)

    # Get image and text embeddings
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Normalize text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (image_features @ text_features.T).item()

        # CLIPScore
        clip_score = 2.5 * max(similarity, 0)

    clip_scores.append(clip_score)

    # Progress

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / " f"{len(df)} images")

# Add score to dataframe
df["CLIPScore"] = clip_scores

# Calculate average CLIPScore
valid_scores = df["CLIPScore"].dropna()
average_clipscore = valid_scores.mean()

# Print result
print()
print("Average CLIPScore:", average_clipscore)


# Save overall score
os.makedirs("results", exist_ok=True)

overall_results = pd.DataFrame({"metric": ["CLIPScore"], "score": [average_clipscore]})
overall_results.to_csv(overall_output, index=False)

# Save per-image CLIPScore
df.to_csv(per_image_output, index=False)

# Done
print()
print("Overall CLIPScore saved to:", overall_output)

print("Per-image CLIPScores saved to:", per_image_output)
