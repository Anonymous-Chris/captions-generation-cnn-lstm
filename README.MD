# Image Captioning and Evaluation Study

This project implements and evaluates a CNN-LSTM image captioning system on the Flickr8k dataset. In addition to standard caption-generation evaluation, the project compares traditional lexical metrics with semantic evaluation metrics, studies cases where metrics disagree, and evaluates metric robustness using controlled caption perturbations.

## Project Overview

The project has four main goals:

1. Train and use an image captioning model.
2. Evaluate generated captions using multiple automatic metrics.
3. Compare agreement and disagreement between lexical and semantic metrics.
4. Test metric sensitivity using controlled caption perturbations.

The image captioning model uses image features extracted with a pretrained CNN and an LSTM-based caption generator. Generated captions are evaluated using:

- BLEU
- METEOR
- ROUGE-L
- CIDEr
- CLIPScore
- POLOS

The project also includes:

- Per-image metric analysis
- Pearson and Spearman correlation analysis
- Metric disagreement analysis
- Rank disagreement analysis
- Case-study selection
- Controlled perturbation experiments
- Perturbation sensitivity analysis
- Bootstrap confidence interval visualization
- A simple UI for caption generation

---

## Project Structure

```text
.
├── archive/
│   └── initial_implementation/
│
├── models/
│   ├── best_image_caption_model.keras
│   ├── max_length.txt
│   └── tokenizer.pkl
│
├── results/
│   ├── analysis/
│   ├── case_studies/
│   ├── disagreement/
│   ├── metrics/
│   ├── perturbation/
│   ├── plots/
│   ├── predictions/
│   └── training/
│
├── src/
│   ├── analysis/
│   ├── evaluation/
│   ├── generation/
│   ├── perturbation/
│   ├── training/
│   └── ui/
│
├── README.md
└── results.txt
```

### Important folders

- `models/` contains the trained captioning model, tokenizer, and maximum caption length.
- `src/training/` contains feature extraction, training, and training-history plotting scripts.
- `src/generation/` contains caption-generation scripts.
- `src/evaluation/` contains the evaluation scripts for BLEU, METEOR, ROUGE-L, CIDEr, CLIPScore, POLOS, and per-image metric aggregation.
- `src/analysis/` contains correlation, disagreement, rank-disagreement, case-study, and visualization scripts.
- `src/perturbation/` contains the robustness experiment pipeline.
- `results/` contains generated predictions, metrics, plots, disagreement cases, and perturbation outputs.
- `archive/` contains older implementations and experimental files that are no longer part of the main pipeline.

---

## Dataset

The main experiments use Flickr8k.

The original dataset preparation used:

- 8,000 images
- 5 captions per image
- Training split: 6,000 images
- Validation split: 1,000 images
- Test split: 1,000 images

Caption preprocessing included:

- Lowercasing
- Removing punctuation
- Removing unsuitable tokens
- Adding `startseq`
- Adding `endseq`

The tokenizer vocabulary contained approximately 8,778 tokens.

The maximum caption sequence length was 37.

---

## Model Architecture

The image-captioning model follows a CNN-LSTM encoder-decoder style architecture.

### Image branch

```text
Image
  ↓
Pretrained VGG16 feature extraction
  ↓
4096-dimensional feature vector
  ↓
Dropout
  ↓
Dense(256)
```

### Caption branch

```text
Caption tokens
  ↓
Embedding(8778, 256)
  ↓
Dropout
  ↓
LSTM(256)
```

### Fusion

The image and caption representations are combined and passed through dense layers:

```text
Image representation
        +
Caption representation
        ↓
Dense(256)
        ↓
Dense(vocabulary_size, softmax)
```

The trained model is stored at:

```text
models/best_image_caption_model.keras
```

The tokenizer is stored at:

```text
models/tokenizer.pkl
```

The maximum sequence length is stored at:

```text
models/max_length.txt
```

---

# Setup

## 1. Create a Python environment

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

## 2. Install dependencies

Install the dependencies required by your local environment.

Typical packages used by this project include:

```bash
pip install tensorflow numpy pandas matplotlib nltk rouge-score pillow tqdm scikit-learn
```

CLIPScore and POLOS may require additional packages or a separate environment depending on the system.

If POLOS is run in Google Colab, use the POLOS setup files stored in the archived Colab setup directory as a reference.

---

# Running the Project

Run all commands from the project root.

Example:

```text
Image Generation/
```

Do not run the commands from inside `src/`.

This project uses three separate environments:

1. **Main environment**
   - Caption generation
   - BLEU
   - METEOR
   - ROUGE-L
   - CIDEr
   - Per-image metric aggregation
   - Correlation analysis
   - Disagreement analysis
   - Rank analysis
   - Perturbation generation
   - Perturbation analysis

2. **CLIP environment**
   - CLIPScore evaluation
   - CLIPScore perturbation evaluation

3. **POLOS environment**
   - Google Colab / Python 3.9
   - POLOS evaluation
   - POLOS perturbation evaluation

---

## STEP 0 - Optional Model Training

Run this only if you want to regenerate image features or retrain the captioning model.

### Main Environment

```bash
python3 src/training/extract_features.py
python3 src/training/train_model.py
python3 src/training/plot_training.py
```

Important outputs include:

```text
models/best_image_caption_model.keras
models/tokenizer.pkl
models/max_length.txt
results/training/training_history.csv
```

If the trained model already exists, skip this step.

---

## STEP 1 - Generate Captions and Run Traditional Metrics

### Main Environment

Generate test captions:

```bash
python3 src/generation/generate_captions_for_test_images.py
```

Then evaluate with the traditional metrics:

```bash
python3 src/evaluation/evaluate_bleu.py
python3 src/evaluation/evaluate_meteor.py
python3 src/evaluation/evaluate_rouge.py
python3 src/evaluation/evaluate_cider.py
```

Important outputs include:

```text
results/predictions/test_predictions.csv
results/predictions/test_predictions_with_cider.csv

results/metrics/evaluate_bleu_results.csv
results/metrics/evaluate_meteor_results.csv
results/metrics/evaluate_rouge_results.csv
results/metrics/evaluate_cider_results.csv
```

---

## STEP 2 - Run CLIPScore

Switch to the separate CLIP environment.

### CLIP Environment

```bash
python3 src/evaluation/evaluate_clip.py
```

Expected outputs:

```text
results/metrics/clipscore_results.csv
results/predictions/test_predictions_with_clipscore.csv
```

Do not run this script in the main environment if CLIP is not installed there.

---

## STEP 3 - Combine Main Per-Image Metrics

After the CLIP output has been generated, switch back to the main environment.

### Main Environment

```bash
python3 src/evaluation/evaluate_per_image_metrics.py
```

Output:

```text
results/metrics/per_image_metrics.csv
```

This combines the main per-image scores for:

- BLEU
- METEOR
- ROUGE-L
- CIDEr
- CLIPScore

---

## STEP 4 - Run POLOS

POLOS is run separately in Google Colab / Python 3.9.

### POLOS / Google Colab

Run the POLOS evaluation using:

```text
src/evaluation/evaluate_polos.py
```

After POLOS finishes, copy the resulting output back into the local project.

Expected local output:

```text
results/predictions/test_predictions_with_polos.csv
```

Do not run POLOS in the main local environment if the required POLOS dependencies are not installed there.

---

## STEP 5 - Combine POLOS With the Other Metrics

Switch back to the main environment.

### Main Environment

```bash
python3 src/analysis/combine_metrics_data_with_polos.py
```

Output:

```text
results/metrics/per_image_metrics_with_polos.csv
```

This creates the combined per-image metric table used for the later POLOS-inclusive analyses.

---

## STEP 6 - Run Main Metric Analysis

### Main Environment

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

Important outputs include:

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
results/plots/metric_correlation.png
```

Disagreement-specific outputs are stored under:

```text
results/disagreement/
```

---

# Perturbation Experiment

The perturbation experiment also uses the three-environment workflow.

---

## STEP 7 - Create and Evaluate the Perturbation Dataset

### Main Environment

Create the perturbation sample:

```bash
python3 src/perturbation/create_perturbation_sample.py
```

Generate controlled perturbations:

```bash
python3 src/perturbation/generate_perturbations.py
```

Attach reference captions:

```bash
python3 src/perturbation/attach_perturbation_references.py
```

Evaluate the traditional perturbation metrics:

```bash
python3 src/perturbation/evaluate/evaluate_perturbation_metrics.py
```

Important outputs include:

```text
results/perturbation/perturbation_sample.csv
results/perturbation/perturbed_captions.csv
results/perturbation/perturbed_captions_with_references.csv
results/perturbation/perturbation_metrics.csv
```

---

## STEP 8 - Evaluate Perturbations With CLIPScore

Switch to the CLIP environment.

### CLIP Environment

```bash
python3 src/perturbation/evaluate/evaluate_perturbation_clipscore.py
```

Output:

```text
results/perturbation/perturbation_metrics_with_clipscore.csv
```

---

## STEP 9 - Evaluate Perturbations With POLOS

Run the POLOS perturbation evaluation in Google Colab / Python 3.9.

### POLOS / Google Colab

Run:

```text
src/perturbation/evaluate/evaluate_polos_perturbation.py
```

After POLOS finishes, copy the output back into the local project.

Expected local output:

```text
results/perturbation/perturbation_metrics_with_polos.csv
```

---

## STEP 10 - Analyze Perturbation Sensitivity

Once the CLIPScore and POLOS perturbation results have been copied back into the project, return to the main environment.

### Main Environment

```bash
python3 src/perturbation/analyze_perturbation_sensitivity.py
python3 src/perturbation/analyze_normalized_sensitivity.py

python3 src/perturbation/plot/plot_perturbation_sensitivity.py
python3 src/perturbation/plot/plot_perturbation_bootstrap_ci.py
```

Important outputs include:

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

## Environment Notes

Do not run the following CLIP scripts in the main environment unless CLIP is installed there:

```bash
python3 src/evaluation/evaluate_clip.py
python3 src/perturbation/evaluate/evaluate_perturbation_clipscore.py
```

Run those in the dedicated CLIP environment.

Do not run the following POLOS scripts in the main local environment unless POLOS is installed there:

```text
src/evaluation/evaluate_polos.py
src/perturbation/evaluate/evaluate_polos_perturbation.py
```

Run those in Google Colab / Python 3.9.

After running CLIP or POLOS, copy their output files back into the appropriate `results/` directories before continuing with downstream analysis.

It is important that BLEU, METEOR, ROUGE-L, CIDEr, CLIPScore, and POLOS are all evaluated using the same generated caption set before final analysis.

---

# Main Experimental Findings

The captioning baseline produced the following approximate overall scores on the test set:

```text
BLEU-1:     0.51–0.55
BLEU-2:     0.34–0.37
BLEU-3:     0.21–0.24
BLEU-4:     0.13–0.15
METEOR:     0.36
ROUGE-L:    0.42
CIDEr:      0.37
CLIPScore:  0.65
```

The exact BLEU values varied slightly across evaluation runs, but the overall pattern was consistent.

The central research result was not the absolute captioning score. The main result was that lexical and semantic evaluation metrics were related but not interchangeable.

## Correlation With CLIPScore

Approximate Pearson correlations with CLIPScore were:

```text
BLEU-1    0.510
BLEU-2    0.499
BLEU-3    0.446
BLEU-4    0.386
METEOR    0.495
ROUGE-L   0.540
CIDEr     0.536
```

These moderate positive correlations indicate that traditional reference-based metrics and semantic image-text evaluation agree to some extent, but they are clearly measuring different properties of caption quality.

BLEU-4 had one of the weakest correlations with CLIPScore:

```text
BLEU-4 vs CLIPScore ≈ 0.386
```

This is consistent with the fact that BLEU-4 rewards exact multi-word overlap, while CLIPScore focuses more on semantic alignment between the generated caption and the image.

## POLOS Findings

POLOS showed a much stronger relationship with CLIPScore than the traditional lexical metrics.

Approximate correlations were:

```text
Pearson:
CLIPScore vs POLOS ≈ 0.803

Spearman:
CLIPScore vs POLOS ≈ 0.816
```

POLOS also showed moderate Pearson correlations with the traditional metrics:

```text
BLEU-1   vs POLOS ≈ 0.634
BLEU-2   vs POLOS ≈ 0.640
BLEU-3   vs POLOS ≈ 0.587
BLEU-4   vs POLOS ≈ 0.509
METEOR   vs POLOS ≈ 0.629
ROUGE-L  vs POLOS ≈ 0.668
CIDEr    vs POLOS ≈ 0.668
```

This suggests that POLOS behaves more like a learned semantic metric than a purely lexical overlap metric.

At the same time, the correlation between POLOS and CLIPScore was not perfect. This indicates that even semantic metrics are not identical and may capture different aspects of caption quality.

## Metric Disagreement

The disagreement analysis identified captions where traditional and semantic metrics made very different judgments.

Examples included:

```text
High CLIPScore / Low BLEU
Low CLIPScore / High BLEU
High POLOS / Low traditional scores
Low POLOS / Higher traditional scores
```

One semantic-high / traditional-low example was:

```text
Generated:
a black dog is running in the grass

BLEU-4:    0.042
CLIPScore: 0.823
POLOS:     0.722
```

The generated caption was semantically reasonable, but its exact wording differed from the reference captions. As a result, BLEU-4 was low while both semantic metrics were high.

Another example showed the reverse pattern:

```text
Generated:
a man sitting on a table with a man in a blue shirt

BLEU-4:    0.306
CLIPScore: 0.400
POLOS:     0.160
```

Here, lexical overlap was relatively stronger, but the semantic metrics suggested that the caption did not align well with the actual image.

These cases support the conclusion that lexical overlap can sometimes overestimate caption quality, while semantic metrics can recognize meaning that traditional overlap metrics miss.

## Rank Disagreement

Rank analysis showed that the metrics also ordered captions differently.

The strongest rank disagreements frequently involved CLIPScore or POLOS compared with traditional metrics.

This matters because two metrics can have a moderate overall correlation while still strongly disagreeing about which individual captions are best or worst.

## Controlled Perturbation Findings

The perturbation experiment was designed to test metric sensitivity to known semantic errors.

Perturbation categories included:

- object replacement
- attribute replacement
- action replacement
- negation
- hallucination

For each perturbation, the project measures the score change between the original and perturbed caption.

The traditional metrics, CLIPScore, and POLOS are all included in this experiment.

The purpose is to answer:

> When a caption is deliberately made semantically incorrect, which evaluation metrics detect the change most reliably?

The perturbation pipeline produces:

```text
results/perturbation/perturbation_metrics.csv
results/perturbation/perturbation_metrics_with_clipscore.csv
results/perturbation/perturbation_metrics_with_polos.csv
results/perturbation/perturbation_deltas.csv
results/perturbation/perturbation_normalized_deltas.csv
results/perturbation/normalized_sensitivity_summary.csv
results/perturbation/perturbation_sensitivity.png
results/perturbation/perturbation_bootstrap_ci.png
```

POLOS is therefore not only used in correlation analysis; it is also part of the controlled robustness experiment.

## Overall Interpretation

The overall study supports the following conclusion:

> No single automatic metric captures all aspects of image-caption quality.

Traditional metrics such as BLEU, METEOR, ROUGE-L, and CIDEr measure similarity to human-written reference captions.

CLIPScore measures image-caption semantic alignment.

POLOS provides an additional learned semantic evaluation signal and showed especially strong agreement with CLIPScore while still differing from it.

The combined evidence from correlation analysis, disagreement analysis, rank analysis, and controlled perturbation testing supports a multi-metric evaluation framework rather than relying on a single score.

## Research Contribution

The main contribution of the project is the evaluation framework rather than a novel caption-generation architecture.

The study combines:

- a reproducible CNN-LSTM captioning baseline
- traditional lexical metrics
- learned semantic metrics
- Pearson correlation analysis
- Spearman rank correlation
- per-image disagreement analysis
- rank disagreement analysis
- qualitative case studies
- controlled semantic perturbations
- normalized sensitivity analysis
- bootstrap confidence intervals

The research question therefore moves beyond:

> How high is the captioning score?

and instead asks:

> What does each metric actually measure, where do the metrics disagree, and how robustly do they respond to known semantic errors?

## Limitations

The current study has several limitations:

- Flickr8k is relatively small.
- CNN-LSTM is an older baseline compared with transformer-based vision-language models.
- Automatic metrics are not substitutes for human judgment.
- The perturbation dataset can be expanded.
- Results should be replicated across additional datasets and captioning architectures.

## Future Work

Future extensions include:

- Flickr30k
- MSCOCO
- transformer-based captioning models
- human evaluation
- larger perturbation sets
- additional learned evaluation metrics
- error-type-specific sensitivity analysis
- cross-model comparison of metric behavior
