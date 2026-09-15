import pandas as pd
from rouge_score import rouge_scorer

df = pd.read_csv("results/predictions/test_predictions.csv")

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rouge_scores = []

for _, row in df.iterrows():
    references = row["references"].split(" ||| ")
    generated = row["generated_caption"]

    # Score generated caption against all 5 references
    scores = [
        scorer.score(reference, generated)["rougeL"].fmeasure
        for reference in references
    ]

    # Use best-matching reference
    rouge_scores.append(max(scores))

average_rouge_l = sum(rouge_scores) / len(rouge_scores)

results = pd.DataFrame({"metric": ["ROUGE-L"], "score": [average_rouge_l]})

results.to_csv("results/metrics/evaluate_rouge_results.csv", index=False)

print("ROUGE-L:", average_rouge_l)
print("ROUGE-L result saved.")
