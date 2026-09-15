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

The trained captioning model achieved approximately:

```text
BLEU-1: 0.5540
BLEU-2: 0.3703
BLEU-3: 0.2387
BLEU-4: 0.1488
METEOR: approximately 0.36
CIDEr: approximately 0.36
CLIPScore: approximately 0.64
```

Earlier BLEU evaluation also produced approximately:

```text
BLEU-1: 0.5258
BLEU-2: 0.3518
BLEU-3: 0.2201
BLEU-4: 0.1351
```

The metric-comparison analysis showed that lexical metrics and semantic/image-aware metrics are related but not interchangeable.

Example Pearson correlations with CLIPScore included approximately:

```text
BLEU-4   0.386
BLEU-3   0.446
METEOR   0.495
BLEU-2   0.499
BLEU-1   0.510
CIDEr    0.536
ROUGE-L  0.540
```

This suggests that metrics based primarily on reference-caption word overlap do not always agree with semantic image-text alignment.

The disagreement analysis identified captions that received relatively strong lexical scores while receiving weaker semantic scores, as well as the reverse.

One observed example had approximately:

```text
BLEU-4:    0.3058
CLIPScore: 0.4003
```

This is useful evidence that a caption can share words or phrases with the references without aligning equally well with the actual visual content.

The perturbation experiment extends this analysis by deliberately modifying captions and measuring how each metric responds to semantic errors such as object replacements, action changes, attribute changes, negation, and hallucination.

The overall research question is therefore not simply:

> Which metric gives the highest score?

Instead, the project asks:

> What properties of caption quality does each metric measure, where do the metrics disagree, and which metrics are most sensitive to meaningful semantic errors?

---

# Notes

- Run scripts from the project root.
- Some scripts depend on files produced by earlier stages.
- POLOS may require a separate Python environment or Google Colab setup.
- Do not delete intermediate results unless you plan to regenerate them.
- The `archive/` directory contains historical implementations and is not part of the current main pipeline.
