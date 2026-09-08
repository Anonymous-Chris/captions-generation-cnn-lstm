import matplotlib.pyplot as plt
import pandas as pd

# Load per-image metrics
input_file = "results/per_image_metrics.csv"
df = pd.read_csv(input_file)

# Metrics to compare
metrics = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr",
    "CLIPScore",
]

# Calculate Pearson correlation
correlation = df[metrics].corr(method="pearson")

# Print as table
print()
print("Pearson Correlation Matrix")
print()

print(correlation.round(3).to_string())

# Save correlation table
correlation.round(4).to_csv("results/metric_correlation.csv")
print()
print("Correlation table saved to:", "results/metric_correlation.csv")

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = ax.imshow(correlation, vmin=-1, vmax=1)

# Axis labels
ax.set_xticks(range(len(metrics)))
ax.set_yticks(range(len(metrics)))
ax.set_xticklabels(metrics, rotation=45, ha="right")
ax.set_yticklabels(metrics)

# Add correlation values inside cells
for i in range(len(metrics)):
    for j in range(len(metrics)):
        value = correlation.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center")

# Color bar
colorbar = fig.colorbar(heatmap, ax=ax)
colorbar.set_label("Pearson Correlation")

# Title
ax.set_title("Correlation Between Image Caption Evaluation Metrics")

# Layout
fig.tight_layout()

# Save heatmap
plt.savefig("results/metric_correlation.png", dpi=300, bbox_inches="tight")
print("Correlation heatmap saved to:", "results/metric_correlation.png")

# Show plot
plt.show()
