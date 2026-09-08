import os

import pandas as pd

# Load data
input_file = "results/per_image_metrics.csv"
output_dir = "results/disagreement"

os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_file)
print(f"Total images: {len(df)}")

# Metrics to compare against CLIPScore
reference_metrics = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr",
]

# CLIPScore thresholds
clip_low = df["CLIPScore"].quantile(0.20)
clip_high = df["CLIPScore"].quantile(0.80)
print("\nCLIPScore Thresholds")
print("-" * 50)

print(f"Low CLIPScore:  <= {clip_low:.4f}")
print(f"High CLIPScore: >= {clip_high:.4f}")

# Save CLIPScore thresholds
clip_thresholds = pd.DataFrame(
    [
        {
            "Metric": "CLIPScore",
            "Low_Threshold_20th_Percentile": clip_low,
            "High_Threshold_80th_Percentile": clip_high,
        }
    ]
)

clip_thresholds.to_csv("results/disagreement/clipscore_thresholds.csv", index=False)

# Store summary information
summary = []

# Compare CLIPScore against every reference metric

for metric in reference_metrics:
    metric_low = df[metric].quantile(0.20)
    metric_high = df[metric].quantile(0.80)

    # High CLIPScore / Low reference metric
    high_clip_low_metric = df[
        (df["CLIPScore"] >= clip_high) & (df[metric] <= metric_low)
    ].copy()

    # Low CLIPScore / High reference metric
    low_clip_high_metric = df[
        (df["CLIPScore"] <= clip_low) & (df[metric] >= metric_high)
    ].copy()

    # Sort strongest disagreements first
    high_clip_low_metric = high_clip_low_metric.sort_values(
        ["CLIPScore", metric], ascending=[False, True]
    )

    low_clip_high_metric = low_clip_high_metric.sort_values(
        ["CLIPScore", metric], ascending=[True, False]
    )

    # Clean metric name for filenames
    metric_filename = metric.lower().replace("-", "")

    # Save disagreement examples
    high_clip_low_metric.to_csv(
        f"{output_dir}/high_clip_low_{metric_filename}.csv", index=False
    )

    low_clip_high_metric.to_csv(
        f"{output_dir}/low_clip_high_{metric_filename}.csv", index=False
    )

    # Pearson correlation
    correlation = df[["CLIPScore", metric]].corr(method="pearson").iloc[0, 1]

    # Add result to summary
    summary.append(
        {
            "Metric": metric,
            "Pearson_Correlation": correlation,
            "Low_Threshold": metric_low,
            "High_Threshold": metric_high,
            "High_CLIP_Low_Metric": len(high_clip_low_metric),
            "Low_CLIP_High_Metric": len(low_clip_high_metric),
        }
    )

# Create summary DataFrame
summary_df = pd.DataFrame(summary)

# Sort from weakest CLIPScore correlation to strongest
summary_df = summary_df.sort_values("Pearson_Correlation")

# Save summary
summary_df.to_csv(f"{output_dir}/disagreement_summary.csv", index=False)

# Print clean summary table
print("\nMetric Disagreement Summary")
print("-" * 90)

print(
    summary_df.round(
        {"Pearson_Correlation": 3, "Low_Threshold": 4, "High_Threshold": 4}
    ).to_string(index=False)
)
print("\nFiles saved to:")
print(output_dir)
