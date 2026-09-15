import pandas as pd

PROJECT = "/content/drive/MyDrive/polos-project"

metrics_path = f"{PROJECT}/results/metrics/per_image_metrics.csv"
polos_path = f"{PROJECT}/results/test_predictions_with_polos.csv"

metrics = pd.read_csv(metrics_path)
polos = pd.read_csv(polos_path)

# Keep only image + POLOS score
polos = polos[["image", "polos"]]

# Merge using image filename
merged = metrics.merge(polos, on="image", how="left", validate="one_to_one")

print("Rows:", len(merged))
print("Missing POLOS scores:", merged["polos"].isna().sum())
print("Average POLOS:", merged["polos"].mean())

# Save updated metrics
output = f"{PROJECT}/results/per_image_metrics_with_polos.csv"
merged.to_csv(output, index=False)

print("Saved:", output)
