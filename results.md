# IMAGE CAPTIONING EVALUATION PROJECT

### PROCESS AND RESULTS SUMMARY

## 1. PROJECT PURPOSE

This project studies automatic evaluation methods for image captioning.

The work began with the implementation of a CNN-LSTM image captioning system. The generated captions were then evaluated using both traditional lexical evaluation metrics and semantic evaluation metrics.

The main research motivation is that image-caption evaluation metrics do not always measure the same property.

Traditional reference-based metrics generally reward overlap between a generated caption and one or more human-written reference captions. Semantic and image-aware metrics can instead measure whether the generated caption is semantically compatible with the image or with learned language representations.

Therefore, a caption may receive a relatively strong lexical score while still being semantically weak, or it may describe an image correctly using wording that differs from the human references and consequently receive a weaker lexical score.

The project investigates this behavior through:

1. Image caption generation
2. Traditional metric evaluation
3. Semantic metric evaluation
4. Per-image metric comparison
5. Pearson correlation analysis
6. Spearman correlation analysis
7. Metric disagreement analysis
8. Rank disagreement analysis
9. Qualitative case studies
10. Controlled caption perturbation experiments

## 2. DATASET

The main dataset is Flickr8k.

The dataset contains approximately 8,000 images, with multiple human captions associated with each image.

The working split used in the project was:

Training images:   6,000
Validation images: 1,000
Test images:       1,000

Caption preprocessing included normalization, punctuation removal, tokenization, and the insertion of sequence boundary tokens.

The tokenizer vocabulary contained approximately:

8,778 tokens

Maximum caption sequence length:

37 tokens

## 3. IMAGE FEATURE EXTRACTION

Visual features were extracted using a pretrained VGG16 convolutional neural network.

The final classification layer was removed, and the penultimate representation was used as the image feature vector.

Image feature dimensionality:

4096

These visual feature vectors were then used as input to the caption-generation model.

## 4. CAPTIONING MODEL

The model uses two input branches.

### IMAGE BRANCH

4096-dimensional image feature

### |
Dropout

### |
Dense layer with 256 units

### CAPTION BRANCH

Caption token sequence

### |
Embedding layer

### |
Dropout

### |
LSTM with 256 units

### FUSION

The image representation and caption representation are combined.

The combined representation is passed through a dense layer and finally through a vocabulary-sized softmax output layer.

The model predicts the next word in the caption sequence.

Important model settings included:

Embedding dimension: 256
LSTM hidden size:    256
Dropout:             0.5
Maximum length:      37

## 5. TRAINING

Training used a validation set and early stopping.

A checkpoint was used to retain the model with the best validation loss.

A representative training run showed training loss decreasing from approximately:

4.8

to approximately:

2.5

Validation loss decreased to approximately:

3.36

The final model is stored as:

models/best_image_caption_model.keras

## 6. CAPTION GENERATION

The trained model generates captions for the Flickr8k test images.

The test predictions are stored in:

results/predictions/test_predictions.csv

These generated captions are used as the common input for the evaluation pipeline.

## 7. EVALUATION METRICS

The project evaluates captions with the following metrics.

### BLEU
Measures n-gram overlap between generated and reference captions.

BLEU-1 focuses primarily on unigram overlap.

BLEU-2, BLEU-3, and BLEU-4 progressively require longer matching word sequences.

### METEOR
Uses word-level matching while incorporating more flexible alignment than simple n-gram precision.

### ROUGE-L
Uses the longest common subsequence between the generated and reference captions.

CIDEr
Measures consensus between a generated caption and multiple human captions using weighted n-gram representations.

CLIPScore
Uses a pretrained vision-language model to estimate semantic compatibility between an image and its generated caption.

Unlike traditional reference-based metrics, CLIPScore can directly consider the image.

### POLOS
A learned semantic evaluation metric used as an additional comparison metric.

The purpose of including multiple metrics is not to declare one metric universally correct. Instead, the goal is to study how different metrics behave and where they disagree.

## 8. OVERALL CAPTIONING RESULTS

A later BLEU evaluation produced approximately:

BLEU-1: 0.5540
BLEU-2: 0.3703
BLEU-3: 0.2387
BLEU-4: 0.1488

An earlier evaluation produced approximately:

BLEU-1: 0.5258
BLEU-2: 0.3518
BLEU-3: 0.2201
BLEU-4: 0.1351

Additional approximate overall results included:

METEOR:   0.36
CIDEr:    0.36
CLIPScore: 0.64

These results indicate that the model produces captions with moderate lexical agreement with the Flickr8k references.

The decline from BLEU-1 to BLEU-4 is expected because matching longer n-gram sequences is substantially more difficult than matching individual words.

## 9. PER-IMAGE METRIC TABLE

The project combines metric scores at the individual image level.

The main combined tables are:

results/metrics/per_image_metrics.csv

and

results/metrics/per_image_metrics_with_polos.csv

Per-image analysis is essential because aggregate averages hide cases where metrics behave very differently.

Two metrics may have similar average behavior while strongly disagreeing on particular examples.

## 10. CORRELATION ANALYSIS

Pearson and Spearman correlation analyses were performed between the different metrics.

Pearson correlation measures linear relationships.

Spearman correlation measures similarity in ranking.

Example Pearson correlations with CLIPScore were approximately:

BLEU-4   : 0.386
BLEU-3   : 0.446
METEOR   : 0.495
BLEU-2   : 0.499
BLEU-1   : 0.510
CIDEr    : 0.536
ROUGE-L  : 0.540

These correlations are positive, meaning that better captions according to traditional metrics often also receive better CLIPScore values.

However, none of these relationships are perfect.

The lower correlations with higher-order BLEU metrics are especially interesting.

BLEU-4 depends heavily on exact multi-word overlap with reference captions, whereas CLIPScore measures image-text semantic alignment.

Therefore, a semantically appropriate caption may receive a relatively low BLEU-4 score simply because its phrasing differs from the references.

## 11. METRIC DISAGREEMENT ANALYSIS

To understand this behavior in more detail, the project explicitly identifies disagreement cases.

The CLIPScore distribution was divided using percentile-based thresholds.

Example thresholds were approximately:

Low CLIPScore threshold:  0.5454
High CLIPScore threshold: 0.7467

This allowed the creation of groups such as:

High CLIPScore / Low BLEU
Low CLIPScore / High BLEU

Similar comparisons were performed for:

BLEU-1
BLEU-2
BLEU-3
BLEU-4

### METEOR

### ROUGE-L
CIDEr

The generated disagreement files are stored under:

results/disagreement/

## 12. EXAMPLE DISAGREEMENT CASE

One identified caption had approximately:

BLEU-4:    0.3058
CLIPScore: 0.4003

The image involved a child in a medieval-fair-like scene.

The relatively stronger BLEU-4 score indicates that the generated caption shared useful phrases or word sequences with human references.

However, the considerably lower CLIPScore suggests that the generated caption was not as well aligned with the visual content of the image.

This illustrates one of the central observations of the study:

Lexical overlap and visual-semantic correctness are related, but they are not equivalent.

## 13. RANK DISAGREEMENT

The project also compares metric rankings.

Instead of only asking whether two metrics have similar raw values, rank analysis asks whether they order captions similarly.

For example:

Metric A may rank Caption X among the best captions.

Metric B may rank the same caption much lower.

These differences are stored in:

results/analysis/rank_disagreement.csv

and

results/analysis/rank_pair_disagreement.csv

Rank disagreement provides another way to show that evaluation metrics emphasize different properties of caption quality.

## 14. CASE STUDIES

Representative disagreement examples are selected for qualitative analysis.

These are stored in:

results/case_studies/case_studies.csv

Qualitative case studies allow manual inspection of:

The image
The reference captions
The generated caption
The lexical metric scores
The semantic metric scores

This makes it possible to interpret why particular metrics agree or disagree.

## 15. CONTROLLED PERTURBATION EXPERIMENT

Correlation alone cannot fully determine which metric is more sensitive to meaningful caption errors.

For this reason, a controlled perturbation experiment was added.

Starting from valid captions, modified captions are created by introducing known semantic errors.

Perturbation categories include:

### ATTRIBUTE REPLACEMENT

Example:

Original:
a man wearing a red shirt

Perturbed:
a man wearing a blue shirt

### OBJECT REPLACEMENT

Example:

Original:
a dog running through grass

Perturbed:
a horse running through grass

### ACTION REPLACEMENT

Example:

Original:
a boy is jumping

Perturbed:
a boy is sleeping

### NEGATION

Example:

Original:
a man is riding a bicycle

Perturbed:
a man is not riding a bicycle

### HALLUCINATION

Information that is not supported by the image is inserted into the caption.

## 16. WHY PERTURBATION TESTING MATTERS

Suppose a caption is deliberately made factually incorrect.

An effective evaluation metric should normally reduce its score.

However, some lexical metrics may remain relatively high if most of the words in the perturbed caption still overlap with the reference captions.

Example:

Original:
a dog is running through the grass

Perturbed:
a cat is running through the grass

Most words remain unchanged.

A word-overlap metric may penalize only the single changed noun.

A semantic or image-aware metric has the possibility of recognizing that the central object identity is incorrect.

The perturbation experiment therefore tests metric robustness more directly than simple correlation analysis.

## 17. PERTURBATION PIPELINE

The perturbation workflow is:

1. Select source captions
2. Generate controlled perturbations
3. Attach reference captions
4. Evaluate standard metrics
5. Evaluate CLIPScore
6. Evaluate POLOS
7. Calculate score changes
8. Normalize score changes
9. Aggregate sensitivity by perturbation type
10. Estimate confidence intervals
11. Visualize results

Important output files include:

results/perturbation/perturbation_sample.csv
results/perturbation/perturbed_captions.csv
results/perturbation/perturbed_captions_with_references.csv
results/perturbation/perturbation_metrics.csv
results/perturbation/perturbation_metrics_with_clipscore.csv
results/perturbation/perturbation_metrics_with_polos.csv
results/perturbation/perturbation_deltas.csv
results/perturbation/perturbation_normalized_deltas.csv
results/perturbation/perturbation_summary.csv
results/perturbation/normalized_sensitivity_summary.csv
results/perturbation/perturbation_sensitivity.png
results/perturbation/perturbation_bootstrap_ci.png

## 18. SCORE DELTAS

For each perturbation, the project compares the metric score before and after the semantic modification.

Conceptually:

delta = perturbed_score - original_score

If the perturbation makes the caption worse, a useful metric should generally produce a negative score change.

A larger reduction indicates greater sensitivity to the introduced semantic error.

Because different metrics operate on different numerical scales, normalized score changes are also calculated.

This makes sensitivity comparisons across metrics more meaningful.

## 19. BOOTSTRAP CONFIDENCE INTERVALS

Perturbation sensitivity is also examined with bootstrap confidence intervals.

Instead of relying only on a single average score change, bootstrapping repeatedly resamples the observations and estimates the uncertainty around the measured sensitivity.

This helps answer:

Is the observed metric response stable?

Or could it be caused by a small number of particular examples?

The resulting visualization is stored in:

results/perturbation/perturbation_bootstrap_ci.png

## 20. MAIN INTERPRETATION

The central result of this project is not that lexical metrics are useless or that semantic metrics should always replace them.

Instead, the experiments show that different evaluation metrics capture different aspects of caption quality.

Traditional metrics are useful for measuring similarity to human-written reference captions.

Semantic and image-aware metrics provide complementary information about whether a generated caption is meaningfully aligned with the image.

Therefore, evaluating captioning systems with only one metric can hide important weaknesses.

A stronger evaluation methodology uses multiple complementary metrics and investigates disagreement rather than reporting only a single aggregate score.

## 21. RESEARCH CONTRIBUTION OF THE PROJECT

The project goes beyond simply training an image-captioning model and reporting BLEU scores.

The evaluation framework includes:

Multiple traditional metrics
Semantic evaluation metrics
Per-image analysis
Metric correlation analysis
Rank correlation
Metric disagreement detection
Qualitative case studies
Controlled semantic perturbations
Normalized sensitivity analysis
Bootstrap confidence intervals

This changes the research focus from:

"How high is the captioning score?"

to:

"How reliable are the evaluation metrics, what do they measure, and how do they respond to specific semantic errors?"

## 22. LIMITATIONS

The current study has several limitations.

The captioning architecture is based on CNN-LSTM rather than newer transformer-based vision-language architectures.

The primary dataset is Flickr8k, which is relatively small compared with Flickr30k and MSCOCO.

The caption decoder uses a comparatively simple generation strategy.

Automatic evaluation metrics themselves are imperfect proxies for human judgment.

The controlled perturbation dataset is also limited by the number and type of perturbations introduced.

These limitations create several natural directions for future work.

## 23. FUTURE WORK

Possible extensions include:

Use beam-search decoding instead of only greedy decoding.

Evaluate on Flickr30k.

Evaluate on MSCOCO.

Compare CNN-LSTM with transformer-based captioning models.

Add additional learned evaluation metrics.

Perform human evaluation.

Increase the number of controlled perturbation examples.

Study metric sensitivity separately for objects, attributes, actions, negation, and hallucination.

Compare metric behavior across multiple captioning architectures.

Investigate whether newer multimodal evaluation models align better with human judgments.

## 24. FINAL SUMMARY

The project implemented an image-captioning system and then built an extensive evaluation framework around it.

The generated captions achieved approximately:

BLEU-1:    0.554
BLEU-2:    0.370
BLEU-3:    0.239
BLEU-4:    0.149
METEOR:    0.36
CIDEr:     0.36
CLIPScore: 0.64

Correlation analysis showed moderate but imperfect relationships between traditional evaluation metrics and CLIPScore.

Example correlations with CLIPScore ranged from approximately:

0.386 for BLEU-4

to:

0.540 for ROUGE-L

The disagreement analysis demonstrated that lexical overlap and semantic/image alignment can produce substantially different evaluations for individual captions.

The controlled perturbation experiment provides a further test of metric reliability by measuring whether metrics correctly penalize known semantic errors.

The main conclusion is that image-caption evaluation benefits from combining lexical, semantic, image-aware, and robustness-based analyses rather than relying on a single automatic metric.
