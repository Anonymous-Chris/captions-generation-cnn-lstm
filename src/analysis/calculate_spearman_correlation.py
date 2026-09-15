import pandas as pd

INPUT = "results/metrics/per_image_metrics_with_polos.csv"

PEARSON_OUTPUT = "results/analysis/pearson_correlation.csv"
SPEARMAN_OUTPUT = "results/analysis/spearman_correlation.csv"

df = pd.read_csv(INPUT)

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

# Pearson correlation
pearson = df[metrics].corr(method="pearson")

# Spearman correlation
spearman = df[metrics].corr(method="spearman")

print("\n=== PEARSON CORRELATION ===\n")
print(pearson.round(3))

print("\n=== SPEARMAN CORRELATION ===\n")
print(spearman.round(3))

pearson.to_csv(PEARSON_OUTPUT)
spearman.to_csv(SPEARMAN_OUTPUT)

print("\nSaved:")
print(PEARSON_OUTPUT)
print(SPEARMAN_OUTPUT)
