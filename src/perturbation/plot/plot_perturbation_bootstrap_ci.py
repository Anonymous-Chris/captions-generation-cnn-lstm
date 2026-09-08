import matplotlib.pyplot as plt
import pandas as pd

input_file = "results/perturbation/normalized_sensitivity_summary.csv"
output_file = "results/perturbation/perturbation_bootstrap_ci.png"

df = pd.read_csv(input_file)

# Keep two representative metrics for a clean comparison
metrics = ["BLEU-4", "CLIPScore"]

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

plot_df = df[df["metric"].isin(metrics)].copy()

plot_df["perturbation_type"] = pd.Categorical(
    plot_df["perturbation_type"],
    categories=perturbation_order,
    ordered=True,
)

plot_df = plot_df.sort_values(["perturbation_type", "metric"])

fig, ax = plt.subplots(figsize=(9, 6))

offsets = {
    "BLEU-4": -0.12,
    "CLIPScore": 0.12,
}

for metric in metrics:
    metric_df = plot_df[plot_df["metric"] == metric]
    y_positions = [i + offsets[metric] for i in range(len(perturbation_order))]
    means = metric_df["mean_normalized_delta"].values
    lower = means - metric_df["ci_95_lower"].values
    upper = metric_df["ci_95_upper"].values - means

    ax.errorbar(
        means,
        y_positions,
        xerr=[lower, upper],
        fmt="o",
        capsize=4,
        label=metric,
    )

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

ax.set_yticks(range(len(perturbation_order)))
ax.set_yticklabels([pretty_names[p] for p in perturbation_order])
ax.invert_yaxis()
ax.set_xlabel("Normalized Score Change (Δ / SD)")
ax.set_ylabel("Perturbation Type")
ax.set_title(
    "Metric Sensitivity to Controlled Perturbations\n"
    "Mean Effect with 95% Bootstrap Confidence Intervals"
)
ax.legend()

plt.tight_layout()
plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print("Saved:")
print(output_file)

plt.show()
