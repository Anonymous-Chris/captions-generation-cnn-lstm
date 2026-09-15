import os
import time

import pandas as pd
from PIL import Image
from polos.models import download_model, load_checkpoint

PROJECT = "/content/drive/MyDrive/polos-project"

PREDICTIONS = os.path.join(PROJECT, "results", "test_predictions.csv")
IMAGE_DIR = os.path.join(PROJECT, "data", "flickr8k", "images")
OUTPUT = os.path.join(PROJECT, "results", "test_predictions_with_polos.csv")

print("Loading predictions...")
df = pd.read_csv(PREDICTIONS)

start = time.time()

print("Loading POLOS model...")
model_path = download_model("polos")

print("Before load_checkpoint")
model = load_checkpoint(model_path)
print("After load_checkpoint")

print("Load time:", time.time() - start, "seconds")

all_scores = []

chunk_size = 50

for start_idx in range(0, len(df), chunk_size):

    end_idx = min(start_idx + chunk_size, len(df))

    print(f"\nEvaluating images {start_idx + 1} to {end_idx} " f"of {len(df)}...")

    data = []

    for _, row in df.iloc[start_idx:end_idx].iterrows():

        image_path = os.path.join(IMAGE_DIR, row["image"])

        references = [ref.strip() for ref in row["references"].split("|||")]

        item = {
            "img": Image.open(image_path).convert("RGB"),
            "mt": row["generated_caption"].strip(),
            "refs": references,
        }

        data.append(item)

    eval_start = time.time()

    _, scores = model.predict(data, batch_size=8, cuda=True)

    all_scores.extend(scores)

    print(f"Finished {end_idx}/{len(df)}")

    print("Chunk time:", round(time.time() - eval_start, 2), "seconds")

    print("Average POLOS so far:", round(sum(all_scores) / len(all_scores), 4))


df["polos"] = all_scores

df.to_csv(OUTPUT, index=False)

print("\nDone.")

print("Average POLOS score:", df["polos"].mean())

print("Saved to:", OUTPUT)
