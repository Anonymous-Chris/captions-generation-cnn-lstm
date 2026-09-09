import os

import pandas as pd

# Files
input_file = "results/perturbation/perturbation_metrics_with_polos.csv"
delta_output_file = "results/perturbation/perturbation_deltas.csv"
summary_output_file = "results/perturbation/perturbation_summary.csv"

# Load data
df = pd.read_csv(input_file)
print("Total rows:", len(df))

# Metrics
metrics = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr",
    "CLIPScore",
    "polos",
]

# Get original scores for each image
original_df = df[df["perturbation_type"] == "original"][["image"] + metrics].copy()
# print(original_df)

# Rename metric columns
original_df = original_df.rename(
    columns={metric: f"original_{metric}" for metric in metrics}
)
# print(original_df.head(10).to_string())
print("Original caption rows:", len(original_df))

# Keep only perturbed rows
perturbed_df = df[df["perturbation_type"] != "original"].copy()
print("Perturbed rows:", len(perturbed_df))

# Merge original scores
result_df = perturbed_df.merge(original_df, on="image", how="left")
# print(result_df.head(1).to_string())

# Validate originals
missing_originals = result_df["original_CLIPScore"].isna().sum()
print("Rows missing original scores:", missing_originals)

# Calculate deltas
for metric in metrics:
    result_df[f"delta_{metric}"] = result_df[metric] - result_df[f"original_{metric}"]

print(result_df.head(1).to_string())
# Save detailed deltas
os.makedirs(os.path.dirname(delta_output_file), exist_ok=True)
result_df.to_csv(delta_output_file, index=False)

# Build summary
summary_rows = []

for perturbation_type, group in result_df.groupby("perturbation_type"):
    row = {"perturbation_type": perturbation_type, "n": len(group)}
    for metric in metrics:
        delta_column = f"delta_{metric}"
        row[f"mean_delta_{metric}"] = group[delta_column].mean()
        row[f"median_delta_{metric}"] = group[delta_column].median()
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Sort perturbations
preferred_order = [
    "object_replacement",
    "person_replacement",
    "color_replacement",
    "action_replacement",
    "negation",
    "hallucination",
]

summary_df["sort_order"] = summary_df["perturbation_type"].apply(
    lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order)
)

summary_df = (
    summary_df.sort_values("sort_order")
    .drop(columns=["sort_order"])
    .reset_index(drop=True)
)

# Save summary
summary_df.to_csv(summary_output_file, index=False)

# Print compact mean-delta table
display_columns = ["perturbation_type", "n"]
for metric in metrics:
    display_columns.append(f"mean_delta_{metric}")

print("\nMean score change by perturbation")
print("-" * 120)
print(summary_df[display_columns].round(4).to_string(index=False))

# Overall sensitivity
print("\nOverall mean delta by metric")
print("-" * 60)

overall_deltas = {}
for metric in metrics:
    overall_deltas[metric] = result_df[f"delta_{metric}"].mean()
overall_series = pd.Series(overall_deltas).sort_values()
print(overall_series.round(4))

# Finished
print("\nSaved detailed results to:")
print(delta_output_file)
print("\nSaved summary to:")
print(summary_output_file)
