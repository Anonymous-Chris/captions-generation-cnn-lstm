import string

import pandas as pd

# Files
perturbation_file = "results/perturbation/perturbed_captions.csv"
captions_file = "data/flickr8k/captions.txt"
output_file = "results/perturbation/perturbed_captions_with_references.csv"


def clean_caption(caption):
    """
    Apply the same basic cleaning used for the image-captioning dataset.
    """
    caption = str(caption).lower()

    # Remove punctuation
    caption = caption.translate(str.maketrans("", "", string.punctuation))

    # Keep alphabetic words only
    words = caption.split()
    words = [word for word in words if word.isalpha()]
    return " ".join(words)


# Load perturbations
perturb_df = pd.read_csv(perturbation_file)
print("Perturbation rows:", len(perturb_df))

# Load Flickr8k captions
captions_df = pd.read_csv(captions_file)
print("Caption rows:", len(captions_df))

# Clean Flickr8k captions
captions_df["clean_caption"] = captions_df["caption"].apply(clean_caption)

# Build image -> list of CLEAN captions
caption_groups = captions_df.groupby("image")["clean_caption"].apply(list).to_dict()

# Attach references
reference_rows = []

for _, row in perturb_df.iterrows():
    image = row["image"]
    original_caption = clean_caption(row["original_caption"])
    all_references = caption_groups.get(image, [])

    # Make copy so we remove ONLY ONE matching caption
    remaining_references = all_references.copy()
    if original_caption in remaining_references:
        remaining_references.remove(original_caption)
    else:
        print(
            f"WARNING: Original caption not found for {image}: " f"{original_caption}"
        )
    reference_rows.append(" ||| ".join(remaining_references))

perturb_df["references"] = reference_rows

# Count references
perturb_df["reference_count"] = perturb_df["references"].apply(
    lambda x: len(x.split(" ||| ")) if isinstance(x, str) and x.strip() else 0
)

# Save
perturb_df.to_csv(output_file, index=False)

# Summary
print("\nReferences attached")
print("-" * 60)

print(perturb_df["reference_count"].value_counts().sort_index())
print("\nSaved to:")
print(output_file)

# Preview
print("\nExample rows")
print("-" * 60)
print(
    perturb_df[
        [
            "image",
            "perturbation_type",
            "original_caption",
            "perturbed_caption",
            "reference_count",
        ]
    ]
    .head(15)
    .to_string(index=False)
)
