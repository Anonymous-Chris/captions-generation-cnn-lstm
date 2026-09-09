import pandas as pd

INPUT = "results/metric_disagreement_cases.csv"
OUTPUT = "results/case_studies.csv"

df = pd.read_csv(INPUT)

# Group 1: Semantic metrics much higher

semantic_high = df.sort_values("group_disagreement", ascending=False).head(5).copy()

semantic_high["case_type"] = "Semantic High / Traditional Low"


# Group 2: Traditional metrics much higher

traditional_high = df.sort_values("group_disagreement", ascending=True).head(5).copy()

traditional_high["case_type"] = "Traditional High / Semantic Low"


# Group 3: Strong agreement
# Both groups give relatively high scores

df["agreement_strength"] = df["traditional_score"] + df["semantic_score"]

agreement = (
    df[abs(df["group_disagreement"]) < 0.05]
    .sort_values("agreement_strength", ascending=False)
    .head(5)
    .copy()
)

agreement["case_type"] = "Strong Agreement"


# Combine

cases = pd.concat([semantic_high, traditional_high, agreement], ignore_index=True)

columns = [
    "case_type",
    "image",
    "generated_caption",
    "references",
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr",
    "CLIPScore",
    "polos",
    "traditional_score",
    "semantic_score",
    "group_disagreement",
]

cases = cases[columns]

cases.to_csv(OUTPUT, index=False)


# Display

for i, row in cases.iterrows():

    print("\n" + "=" * 80)

    print("CASE:", i + 1)
    print("TYPE:", row["case_type"])
    print("IMAGE:", row["image"])

    print("\nGENERATED:")
    print(row["generated_caption"])

    print("\nREFERENCES:")

    references = str(row["references"]).split("|||")

    for j, reference in enumerate(references, 1):
        print(f"{j}. {reference.strip()}")

    print("\nMETRICS:")

    print(
        f"BLEU-4:    {row['BLEU-4']:.3f}\n"
        f"METEOR:    {row['METEOR']:.3f}\n"
        f"ROUGE-L:   {row['ROUGE-L']:.3f}\n"
        f"CIDEr:     {row['CIDEr']:.3f}\n"
        f"CLIPScore: {row['CLIPScore']:.3f}\n"
        f"POLOS:     {row['polos']:.3f}"
    )


print("\n" + "=" * 80)
print("Saved:", OUTPUT)
