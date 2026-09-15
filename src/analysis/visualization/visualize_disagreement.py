import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Settings
image_dir = "data/flickr8k/images"
high_clip_low_bleu_file = "results/disagreement/high_clip_low_bleu4.csv"
low_clip_high_bleu_file = "results/disagreement/low_clip_high_bleu4.csv"

# Number of examples to inspect
n = 5

# Load results
high_clip = pd.read_csv(high_clip_low_bleu_file)
low_clip = pd.read_csv(low_clip_high_bleu_file)


# Display examples
def show_examples(df, title):
    for i, row in df.head(n).iterrows():
        image_path = os.path.join(image_dir, row["image"])
        image = Image.open(image_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"{title}\n"
            f"BLEU-4: {row['BLEU-4']:.3f} | "
            f"CLIPScore: {row['CLIPScore']:.3f}"
        )

        print("\n" + "=" * 80)
        print("IMAGE:")
        print(row["image"])
        print("\nGENERATED:")
        print(row["generated_caption"])
        print("\nREFERENCES:")
        references = row["references"].split(" ||| ")

        for number, reference in enumerate(references, start=1):
            print(f"{number}. {reference}")
        print(f"\nBLEU-4: {row['BLEU-4']:.4f}")
        print(f"CLIPScore: {row['CLIPScore']:.4f}")
        plt.show()


# High CLIP / Low BLEU
print("\n\nHIGH CLIP / LOW BLEU-4")
show_examples(high_clip, "High CLIPScore / Low BLEU-4")

# Low CLIP / High BLEU
print("\n\nLOW CLIP / HIGH BLEU-4")
show_examples(low_clip, "Low CLIPScore / High BLEU-4")
