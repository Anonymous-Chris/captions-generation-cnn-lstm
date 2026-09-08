import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

# Load saved predictions
predictions_file = "results/test_predictions.csv"
df = pd.read_csv(predictions_file)

# Prepare references and generated captions
all_references = []
all_predictions = []


for _, row in df.iterrows():
    # Split the 5 reference captions
    references = row["references"].split(" ||| ")

    # Tokenize references
    references = [caption.split() for caption in references]

    # Tokenize generated caption
    prediction = row["generated_caption"].split()
    all_references.append(references)
    all_predictions.append(prediction)

# print(all_predictions)
# print(all_references)

# BLEU scores
smoothing = SmoothingFunction().method1
bleu_1 = corpus_bleu(
    all_references,
    all_predictions,
    weights=(1.0, 0, 0, 0),
    smoothing_function=smoothing,
)

bleu_2 = corpus_bleu(
    all_references,
    all_predictions,
    weights=(0.5, 0.5, 0, 0),
    smoothing_function=smoothing,
)

bleu_3 = corpus_bleu(
    all_references,
    all_predictions,
    weights=(1 / 3, 1 / 3, 1 / 3, 0),
    smoothing_function=smoothing,
)

bleu_4 = corpus_bleu(
    all_references,
    all_predictions,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=smoothing,
)

print("BLEU-1:", bleu_1)
print("BLEU-2:", bleu_2)
print("BLEU-3:", bleu_3)
print("BLEU-4:", bleu_4)

# Save results
results = pd.DataFrame(
    {
        "metric": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"],
        "score": [bleu_1, bleu_2, bleu_3, bleu_4],
    }
)

results.to_csv("results/evaluate_bleu_results.csv", index=False)
print()
print("Metrics saved to results/evaluation_metrics.csv")
