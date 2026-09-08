from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Files
input_file = "results/perturbation/normalized_sensitivity_summary.csv"
output_file = "results/perturbation/perturbation_sensitivity.png"
Path("results/perturbation").mkdir(parents=True, exist_ok=True)

# Load results
df = pd.read_csv(input_file)
print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())

# Metrics to plot
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

# Perturbation order
perturbation_order = [
    "object_replacement",
    "person_replacement",
    "color_replacement",
    "action_replacement",
    "negation",
    "hallucination",
]

pretty_names = {
    "object_replacement": "Object",
    "person_replacement": "Person",
    "color_replacement": "Color",
    "action_replacement": "Action",
    "negation": "Negation",
    "hallucination": "Hallucination",
}

# Pivot normalized mean values
plot_df = df.pivot(
    index="perturbation_type",
    columns="metric",
    values="mean_normalized_delta",
)

plot_df = plot_df.reindex(perturbation_order)
plot_df = plot_df[metrics]

# Create heatmap
fig, ax = plt.subplots(figsize=(11, 5.5))
image = ax.imshow(
    plot_df.values,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-1.5,
    vmax=1.5,
)

# Axis labels
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, rotation=35, ha="right")
ax.set_yticks(range(len(perturbation_order)))
ax.set_yticklabels([pretty_names[p] for p in perturbation_order])

# Add values inside cells
for i in range(len(plot_df.index)):
    for j in range(len(plot_df.columns)):
        value = plot_df.iloc[i, j]
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9,
        )

# Colorbar
colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Normalized Score Change (Δ / SD)")

# Labels
ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Perturbation Type")
ax.set_title("Sensitivity of Image Captioning Metrics " "to Controlled Perturbations")

# Layout and save
plt.tight_layout()
plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print("\nSaved figure:")
print(output_file)

plt.show()
