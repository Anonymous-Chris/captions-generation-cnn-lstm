import os

import pandas as pd

# Settings
input_file = "results/predictions/test_predictions.csv"
output_dir = "results/perturbation"
output_file = f"{output_dir}/perturbation_sample.csv"

sample_size = 100
random_seed = 42

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load test predictions
df = pd.read_csv(input_file)
print("Total test images:", len(df))

# Randomly select 100 images
sample_df = df.sample(n=sample_size, random_state=random_seed).copy()

# Extract first human reference
# Need to use regex or else it does regex matching
sample_df["original_caption"] = (
    sample_df["references"].str.split(" ||| ", regex=False).str[0]
)
# print(sample_df["original_caption"])
# Print all length
# print(sample_df.head(10).to_string(index=False))

# Keep required columns
sample_df = sample_df[["image", "original_caption"]]

# Save
sample_df.to_csv(output_file, index=False)

# Print
print()
print("Perturbation sample created")
print("-" * 50)
print("Images:", len(sample_df))
print("Random seed:", random_seed)
print("Saved to:", output_file)
print()
print(sample_df.head(10).to_string(index=False))
