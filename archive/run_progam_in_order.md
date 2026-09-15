Copy this entire block at once:

````md
# How to Run the Project

This project uses three separate environments:

1. Main environment
   - Caption generation
   - BLEU
   - METEOR
   - ROUGE-L
   - CIDEr
   - Metric analysis
   - Perturbation analysis

2. CLIP environment
   - CLIPScore evaluation
   - CLIPScore perturbation evaluation

3. POLOS environment
   - Google Colab / Python 3.9
   - POLOS evaluation
   - POLOS perturbation evaluation

---

# STEP 0 - Optional Model Training

Run this only if you want to retrain the captioning model.

## Main Environment

```bash
python3 src/training/extract_features.py
python3 src/training/train_model.py
python3 src/training/plot_training.py
```

---

# STEP 1 - Generate Captions and Run Traditional Metrics

## Main Environment

```bash
python3 src/generation/generate_captions_for_test_images.py

python3 src/evaluation/evaluate_bleu.py
python3 src/evaluation/evaluate_meteor.py
python3 src/evaluation/evaluate_rouge.py
python3 src/evaluation/evaluate_cider.py
```

Main prediction output:

```text
results/predictions/test_predictions.csv
```

CIDEr output:

```text
results/predictions/test_predictions_with_cider.csv
```

---

# STEP 2 - Run CLIPScore

Switch to the CLIP environment.

## CLIP Environment

```bash
python3 src/evaluation/evaluate_clip.py
```

Expected outputs:

```text
results/metrics/clipscore_results.csv
results/predictions/test_predictions_with_clipscore.csv
```

---

# STEP 3 - Combine Main Metrics

Switch back to the main environment.

## Main Environment

```bash
python3 src/evaluation/evaluate_per_image_metrics.py
```

Output:

```text
results/metrics/per_image_metrics.csv
```

This combines:

- BLEU
- METEOR
- ROUGE-L
- CIDEr
- CLIPScore

---

# STEP 4 - Run POLOS

POLOS is run separately in Google Colab / Python 3.9.

## POLOS / Google Colab

Run:

```text
src/evaluation/evaluate_polos.py
```

After POLOS finishes, copy the resulting file back into the local project.

Expected output:

```text
results/predictions/test_predictions_with_polos.csv
```

---

# STEP 5 - Combine POLOS With Other Metrics

Switch back to the main environment.

## Main Environment

```bash
python3 src/analysis/combine_metrics_data_with_polos.py
```

Output:

```text
results/metrics/per_image_metrics_with_polos.csv
```

---

# STEP 6 - Run Main Metric Analysis

## Main Environment

```bash
python3 src/analysis/analyze_metric_correlation.py
python3 src/analysis/calculate_metric_correlation_with_polos.py
python3 src/analysis/calculate_spearman_correlation.py

python3 src/analysis/analyze_metric_disagreement.py
python3 src/analysis/analyze_all_metric_disagreements.py
python3 src/analysis/analyze_rank_disagreement.py
python3 src/analysis/select_case_studies.py

python3 src/analysis/visualization/visualize_disagreement.py
```

Important outputs:

```text
results/analysis/metric_correlation.csv
results/analysis/metric_correlation_with_polos.csv
results/analysis/pearson_correlation.csv
results/analysis/spearman_correlation.csv
results/analysis/metric_pair_disagreements.csv
results/analysis/metric_disagreement_cases.csv
results/analysis/rank_disagreement.csv
results/analysis/rank_pair_disagreement.csv
results/case_studies/case_studies.csv
```

---

# STEP 7 - Create Perturbation Dataset

## Main Environment

```bash
python3 src/perturbation/create_perturbation_sample.py
python3 src/perturbation/generate_perturbations.py
python3 src/perturbation/attach_perturbation_references.py

python3 src/perturbation/evaluate/evaluate_perturbation_metrics.py
```

Important outputs:

```text
results/perturbation/perturbation_sample.csv
results/perturbation/perturbed_captions.csv
results/perturbation/perturbed_captions_with_references.csv
results/perturbation/perturbation_metrics.csv
```

---

# STEP 8 - Evaluate Perturbations With CLIPScore

Switch to the CLIP environment.

## CLIP Environment

```bash
python3 src/perturbation/evaluate/evaluate_perturbation_clipscore.py
```

Output:

```text
results/perturbation/perturbation_metrics_with_clipscore.csv
```

---

# STEP 9 - Evaluate Perturbations With POLOS

Run this in Google Colab / Python 3.9.

## POLOS / Google Colab

Run:

```text
src/perturbation/evaluate/evaluate_polos_perturbation.py
```

After POLOS finishes, copy the output back into the local project.

Expected output:

```text
results/perturbation/perturbation_metrics_with_polos.csv
```

---

# STEP 10 - Analyze Perturbation Sensitivity

Switch back to the main environment.

## Main Environment

```bash
python3 src/perturbation/analyze_perturbation_sensitivity.py
python3 src/perturbation/analyze_normalized_sensitivity.py

python3 src/perturbation/plot/plot_perturbation_sensitivity.py
python3 src/perturbation/plot/plot_perturbation_bootstrap_ci.py
```

Important outputs:

```text
results/perturbation/perturbation_deltas.csv
results/perturbation/perturbation_summary.csv
results/perturbation/perturbation_normalized_deltas.csv
results/perturbation/normalized_sensitivity_summary.csv
results/perturbation/perturbation_sensitivity.png
results/perturbation/perturbation_bootstrap_ci.png
```

---

# Complete Workflow Summary

```text
MAIN ENVIRONMENT

Generate captions
        |
        v
BLEU
METEOR
ROUGE-L
CIDEr
        |
        v
CLIP ENVIRONMENT
        |
        v
CLIPScore
        |
        v
MAIN ENVIRONMENT
        |
        v
Combine per-image metrics
        |
        v
POLOS / GOOGLE COLAB
        |
        v
POLOS scores
        |
        v
MAIN ENVIRONMENT
        |
        v
Combine all metrics
        |
        v
Correlation analysis
Disagreement analysis
Rank analysis
Case studies
        |
        v
Create perturbations
        |
        v
Traditional perturbation metrics
        |
        +----------------------+
        |                      |
        v                      v
CLIP ENVIRONMENT          POLOS / COLAB
CLIPScore                 POLOS
        |                      |
        +----------+-----------+
                   |
                   v
            MAIN ENVIRONMENT
                   |
                   v
      Perturbation sensitivity
                   |
                   v
      Normalized sensitivity
                   |
                   v
       Bootstrap confidence
                   |
                   v
              Final plots
```

# Important

Do not run CLIPScore scripts in the main environment.

Run:

```bash
python3 src/evaluation/evaluate_clip.py
```

only in the CLIP environment.

Run:

```bash
python3 src/perturbation/evaluate/evaluate_perturbation_clipscore.py
```

only in the CLIP environment.

Do not run POLOS scripts in the main environment.

Run POLOS in Google Colab / Python 3.9.

Main POLOS script:

```text
src/evaluation/evaluate_polos.py
```

Perturbation POLOS script:

```text
src/perturbation/evaluate/evaluate_polos_perturbation.py
```

After running CLIP or POLOS, copy their output files back into the correct `results/` directories before continuing with the main analysis.
````
