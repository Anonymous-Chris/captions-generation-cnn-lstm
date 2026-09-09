import os

import pandas as pd
from PIL import Image
from polos.models import download_model, load_checkpoint

PROJECT = "/content/drive/MyDrive/polos-project"

INPUT = os.path.join(
    PROJECT, "results", "perturbation", "perturbation_metrics_with_clipscore.csv"
)

IMAGE_DIR = os.path.join(PROJECT, "data", "flickr8k", "images")

OUTPUT = os.path.join(
    PROJECT, "results", "perturbation", "perturbation_metrics_with_polos.csv"
)

df = pd.read_csv(INPUT)

print("Rows:", len(df))
print("Loading POLOS...")

model_path = download_model("polos")
model = load_checkpoint(model_path)

all_scores = []

chunk_size = 50

for start_idx in range(0, len(df), chunk_size):
    end_idx = min(start_idx + chunk_size, len(df))
    print(f"\nEvaluating {start_idx + 1} to " f"{end_idx} of {len(df)}")
    data = []

    for _, row in df.iloc[start_idx:end_idx].iterrows():
        image_path = os.path.join(IMAGE_DIR, row["image"])
        references = [ref.strip() for ref in str(row["references"]).split("|||")]
        data.append(
            {
                "img": Image.open(image_path).convert("RGB"),
                "mt": str(row["perturbed_caption"]).strip(),
                "refs": references,
            }
        )

    _, scores = model.predict(data, batch_size=8, cuda=True)
    all_scores.extend(scores)
    print(f"Finished {end_idx}/{len(df)}")

df["polos"] = all_scores

df.to_csv(OUTPUT, index=False)

print("\nDone.")
print("Rows:", len(df))
print("Average POLOS:", df["polos"].mean())
print("\nAverage POLOS by perturbation type:")
print(df.groupby("perturbation_type")["polos"].agg(["count", "mean"]).round(4))

print("\nSaved:")
print(OUTPUT)
