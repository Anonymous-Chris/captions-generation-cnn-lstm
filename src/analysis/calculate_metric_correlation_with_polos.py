import pandas as pd

INPUT = "results/metrics/per_image_metrics_with_polos.csv"
OUTPUT = "results/analysis/metric_correlation_with_polos.csv"

df = pd.read_csv(INPUT)

metric_columns = [
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

correlation = df[metric_columns].corr()

print(correlation.round(3))

correlation.to_csv(OUTPUT)

print("\nSaved to:", OUTPUT)
