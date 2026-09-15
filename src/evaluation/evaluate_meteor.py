import nltk
import pandas as pd
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet")
nltk.download("omw-1.4")

# Load predictions
df = pd.read_csv("results/predictions/test_predictions.csv")

meteor_scores = []

for _, row in df.iterrows():
    # Split the 5 reference captions
    references = row["references"].split(" ||| ")

    # Tokenize references
    references = [caption.split() for caption in references]

    # Tokenize generated caption
    generated = row["generated_caption"].split()
    score = meteor_score(references, generated)
    meteor_scores.append(score)

# Average across all test images
average_meteor = sum(meteor_scores) / len(meteor_scores)
print("METEOR:", average_meteor)
results = pd.DataFrame({"metric": ["METEOR"], "score": [average_meteor]})

results.to_csv("results/metrics/evaluate_meteor_results.csv", index=False)
print("METEOR results saved")
