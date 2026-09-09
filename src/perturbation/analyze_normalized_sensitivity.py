import os

import numpy as np
import pandas as pd

# Files
input_file = "results/perturbation/perturbation_deltas.csv"
output_dir = "results/perturbation"
summary_file = f"{output_dir}/normalized_sensitivity_summary.csv"

# Settings
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

perturbation_order = [
    "object_replacement",
    "person_replacement",
    "color_replacement",
    "action_replacement",
    "negation",
    "hallucination",
]

bootstrap_iterations = 10000
random_seed = 42

rng = np.random.default_rng(random_seed)

# Load data
df = pd.read_csv(input_file)
print("Perturbed rows:", len(df))

# Calculate standard deviation of ORIGINAL scores
# We only want one original score per image.

original_columns = [f"original_{metric}" for metric in metrics]

original_scores = (
    df[["image"] + original_columns].drop_duplicates(subset="image").copy()
)

print("Unique original images:", len(original_scores))

# Original-score standard deviations
original_std = {}
print("\nOriginal-score standard deviations")
print("-" * 60)

for metric in metrics:
    column = f"original_{metric}"
    std = original_scores[column].std(ddof=1)
    original_std[metric] = std
    print(f"{metric:<12} {std:.4f}")

# Normalize deltas
# normalized delta =
# raw delta / SD(original metric scores)
#
# Example:
#
# delta_CLIPScore = -0.05
# original SD     =  0.10
#
# normalized = -0.50
#
# Meaning: decrease of half an SD.

for metric in metrics:
    delta_column = f"delta_{metric}"
    normalized_column = f"normalized_delta_{metric}"
    df[normalized_column] = df[delta_column] / original_std[metric]


# Bootstrap confidence interval
def bootstrap_mean_ci(values, iterations=10000, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan

    bootstrap_means = np.empty(iterations)
    for i in range(iterations):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = sample.mean()
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return lower, upper


# Build summary
summary_rows = []
for perturbation_type in perturbation_order:
    group = df[df["perturbation_type"] == perturbation_type]
    if len(group) == 0:
        continue
    for metric in metrics:
        raw_column = f"delta_{metric}"
        normalized_column = f"normalized_delta_{metric}"
        raw_values = group[raw_column].dropna().values
        normalized_values = group[normalized_column].dropna().values
        raw_mean = np.mean(raw_values)
        raw_median = np.median(raw_values)
        normalized_mean = np.mean(normalized_values)
        normalized_median = np.median(normalized_values)
        ci_lower, ci_upper = bootstrap_mean_ci(
            normalized_values, iterations=bootstrap_iterations
        )

        summary_rows.append(
            {
                "perturbation_type": perturbation_type,
                "metric": metric,
                "n": len(normalized_values),
                "mean_raw_delta": raw_mean,
                "median_raw_delta": raw_median,
                "mean_normalized_delta": normalized_mean,
                "median_normalized_delta": normalized_median,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
            }
        )
summary_df = pd.DataFrame(summary_rows)

# Save detailed normalized data
normalized_output = f"{output_dir}/" "perturbation_normalized_deltas.csv"
df.to_csv(normalized_output, index=False)

# Save summary
summary_df.to_csv(summary_file, index=False)

# Print compact table
pivot = summary_df.pivot(
    index="perturbation_type", columns="metric", values="mean_normalized_delta"
)

# Preserve order
pivot = pivot.reindex(perturbation_order)
pivot = pivot.reindex(columns=metrics)

print("\nMean normalized sensitivity")
print("(delta divided by SD of original scores)")
print("-" * 120)
print(pivot.round(3).to_string())

# CLIPScore confidence intervals
print("\nCLIPScore normalized sensitivity " "with 95% bootstrap CI")
print("-" * 80)

clip_summary = summary_df[summary_df["metric"] == "CLIPScore"][
    [
        "perturbation_type",
        "n",
        "mean_raw_delta",
        "mean_normalized_delta",
        "ci_95_lower",
        "ci_95_upper",
    ]
]

print(clip_summary.round(4).to_string(index=False))

# BLEU-4 confidence intervals
print("\nBLEU-4 normalized sensitivity " "with 95% bootstrap CI")
print("-" * 80)

bleu4_summary = summary_df[summary_df["metric"] == "BLEU-4"][
    [
        "perturbation_type",
        "n",
        "mean_raw_delta",
        "mean_normalized_delta",
        "ci_95_lower",
        "ci_95_upper",
    ]
]

print(bleu4_summary.round(4).to_string(index=False))

# POLOS confidence intervals
print("\nPOLOS normalized sensitivity with 95% bootstrap CI")
print("-" * 80)

polos_summary = summary_df[summary_df["metric"] == "polos"][
    [
        "perturbation_type",
        "n",
        "mean_raw_delta",
        "mean_normalized_delta",
        "ci_95_lower",
        "ci_95_upper",
    ]
]

print(polos_summary.round(4).to_string(index=False))

# Finished
print("\nSaved:")
print(normalized_output)
print(summary_file)
