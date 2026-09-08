import pandas as pd
from pycocoevalcap.cider.cider import Cider

# Load saved predictions
df = pd.read_csv("results/test_predictions.csv")

# Ground-truth references
gts = {}

# Generated captions
res = {}

for i, row in df.iterrows():
    references = row["references"].split(" ||| ")
    generated = row["generated_caption"]

    # pycocoevalcap expects dictionaries keyed by image ID
    gts[i] = references
    res[i] = [generated]


# Create CIDEr scorer
scorer = Cider()

# Calculate CIDEr
cider_score, individual_scores = scorer.compute_score(gts, res)

print("CIDEr:", cider_score)

# Save overall CIDEr score
results = pd.DataFrame({"metric": ["CIDEr"], "score": [cider_score]})

results.to_csv("results/evaluate_cider_results.csv", index=False)

print("CIDEr result saved.")

# Optional: save score for every image
df["CIDEr"] = individual_scores

df.to_csv("results/test_predictions_with_cider.csv", index=False)

print("Per-image CIDEr scores saved.")
