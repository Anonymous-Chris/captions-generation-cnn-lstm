import pandas as pd

# Files
perturbation_file = "results/perturbation/perturbed_captions.csv"
captions_file = "data/flickr8k/captions.txt"
output_file = "results/perturbation/perturbed_captions_with_references.csv"

# Load perturbations
perturb_df = pd.read_csv(perturbation_file)
print("Perturbation rows:", len(perturb_df))

# Load flickr8k captions
captions_df = pd.read_csv(captions_file)
print("Caption rows:", len(captions_df))

# Build image -> list of captions
grouped = captions_df.groupby("image")["caption"]
caption_lists = grouped.apply(list)
caption_groups = caption_lists.to_dict()
# caption_groups = captions_df.groupby("image")["caption"].apply(list).to_dict()
# print(caption_groups)
print(perturb_df)
# Attach references
reference_rows = []
for _, row in perturb_df.iterrows():
    image = row["image"]
    original_caption = str(row["original_caption"]).strip()
    all_references = caption_groups.get(image, [])

    # Get only captions that are different from original caption
    remaining_references = [
        str(caption).strip()
        for caption in all_references
        if str(caption).strip() != original_caption
    ]
    reference_rows.append(" ||| ".join(remaining_references))

perturb_df["references"] = reference_rows

# Count number of references
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
