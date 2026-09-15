# Image Captioning Evaluation Research — Interview Guide

## 30-Second Research Summary

My research focused on evaluating the reliability of automatic metrics for image captioning.

I first implemented a CNN-LSTM image captioning system on Flickr8k to generate a controlled set of captions. I then evaluated those captions using traditional reference-based metrics such as BLEU, METEOR, ROUGE-L, and CIDEr, along with semantic metrics such as CLIPScore and POLOS.

The main research question was whether lexical overlap metrics and semantic evaluation metrics measure the same aspects of caption quality.

To investigate that, I performed correlation analysis, rank-disagreement analysis, qualitative case studies, and controlled perturbation experiments involving object changes, attribute changes, action changes, negation, and hallucination.

The main finding was that the metrics are related but not interchangeable, which suggests that image-caption evaluation is stronger when multiple complementary metrics are used.

---

# Research Motivation

A common problem in image captioning research is that model quality is often summarized using a small number of automatic metrics.

However, different metrics measure different things.

Traditional metrics such as BLEU or ROUGE primarily compare generated captions with human reference captions.

Semantic metrics such as CLIPScore can instead evaluate whether the generated caption is semantically aligned with the image.

This creates an important research question:

> If two captions have similar lexical overlap with the references, do they also have similar semantic quality?

Or more generally:

> Do traditional lexical metrics and semantic image-text metrics evaluate the same dimensions of caption quality?

My project was designed to investigate that question experimentally.

---

# Research Objective

The objective of the study was to compare automatic image-caption evaluation metrics and determine:

1. How strongly the metrics correlate.
2. Whether they rank captions similarly.
3. Which types of captions produce strong metric disagreement.
4. How sensitive each metric is to controlled semantic errors.
5. Whether semantic metrics provide information that is not captured by lexical overlap metrics.

---

# Experimental Pipeline

The overall research pipeline was:

```text
Flickr8k Dataset
      |
      v
CNN-LSTM Captioning Model
      |
      v
Generated Captions
      |
      +-----------------------------+
      |                             |
      v                             v
Traditional Metrics          Semantic Metrics
BLEU                         CLIPScore
METEOR                       POLOS
ROUGE-L
CIDEr
      |                             |
      +-------------+---------------+
                    |
                    v
             Metric Comparison
                    |
         +----------+----------+
         |          |          |
         v          v          v
   Correlation   Ranking   Disagreement
    Analysis     Analysis     Cases
                    |
                    v
          Controlled Perturbations
                    |
                    v
          Sensitivity Analysis
```

---

# Captioning Model as the Experimental Baseline

The captioning model was used as a baseline system to generate captions for evaluation.

The main research contribution was not the use of a novel captioning architecture. Instead, the captioning model provided a reproducible source of generated captions for the metric study.

## Image Encoder

I used a pretrained VGG16 convolutional neural network as the visual encoder.

The CNN converts an image into a numerical feature representation.

The extracted feature vector had 4096 dimensions.

In research terms:

> VGG16 was used as a fixed visual feature extractor so that the experiment could focus on evaluation behavior rather than introducing additional complexity from end-to-end visual model training.

## Language Decoder

An LSTM was used as the caption decoder.

The decoder receives:

- the image representation
- the previously generated caption tokens

and predicts the next token.

The decoding process is sequential:

```text
startseq
    ↓
a
    ↓
a dog
    ↓
a dog is
    ↓
a dog is running
    ↓
a dog is running in the grass
    ↓
endseq
```

---

# Why CNN-LSTM Was Appropriate for This Study

A useful way to explain this in an interview is:

> I intentionally used a well-understood CNN-LSTM baseline because my main research question was about evaluation metrics rather than proposing a new caption-generation architecture. Using a standard baseline helped isolate the evaluation problem.

The architecture can be summarized as:

```text
CNN  -> visual representation
LSTM -> language generation
```

---

# Dataset and Experimental Split

The experiments used Flickr8k.

The dataset contains approximately:

```text
8,000 images
5 human-written captions per image
```

The experimental split was approximately:

```text
Training:   6,000 images
Validation: 1,000 images
Testing:    1,000 images
```

Caption preprocessing included:

- lowercasing
- punctuation removal
- tokenization
- start and end sequence tokens

The resulting tokenizer vocabulary contained approximately:

```text
8,778 tokens
```

The maximum caption length was:

```text
37 tokens
```

---

# Research Question 1:
# How Do the Evaluation Metrics Differ?

I evaluated generated captions using six metrics:

```text
BLEU
METEOR
ROUGE-L
CIDEr
CLIPScore
POLOS
```

These can be divided broadly into two categories.

## Traditional / Reference-Based Metrics

```text
BLEU
METEOR
ROUGE-L
CIDEr
```

These metrics primarily compare a generated caption with one or more human-written reference captions.

## Semantic / Learned Metrics

```text
CLIPScore
POLOS
```

These metrics rely more heavily on learned representations and semantic information.

---

# BLEU

BLEU measures n-gram overlap between generated captions and reference captions.

For example:

```text
Reference:
a dog is running through the grass

Generated:
a dog is running in the grass
```

Because many word sequences overlap, BLEU is relatively high.

BLEU-1 through BLEU-4 progressively evaluate longer n-grams.

Research interpretation:

> BLEU mainly captures lexical similarity rather than full semantic correctness.

---

# METEOR

METEOR also compares generated and reference captions but allows more flexible word matching than strict n-gram precision.

Research interpretation:

> METEOR can capture some linguistic variation that BLEU may miss, but it is still fundamentally reference-based.

---

# ROUGE-L

ROUGE-L evaluates the longest common subsequence between a generated caption and its references.

Research interpretation:

> ROUGE-L captures sequence-level overlap and sentence structure similarity.

---

# CIDEr

CIDEr was developed specifically for image captioning.

It compares a generated caption against multiple human references using weighted n-gram statistics.

Research interpretation:

> CIDEr attempts to measure consensus with human descriptions and gives greater importance to informative phrases.

---

# CLIPScore

CLIPScore differs conceptually from the traditional metrics.

Instead of relying only on reference captions, it evaluates semantic compatibility between the generated caption and the image.

Research interpretation:

> CLIPScore provides an image-grounded semantic signal that is not directly available from purely lexical reference-based metrics.

---

# POLOS

POLOS was included as an additional learned semantic evaluation metric.

Research interpretation:

> Including POLOS allowed me to compare multiple learned semantic approaches rather than relying on only one semantic metric.

---

# Research Question 2:
# Do the Metrics Agree?

To answer this, I performed correlation analysis.

I calculated both:

```text
Pearson correlation
Spearman correlation
```

## Pearson Correlation

Pearson measures linear association between metric scores.

Research explanation:

> If two metrics have a high Pearson correlation, captions that score highly on one metric also tend to score highly on the other.

## Spearman Correlation

Spearman evaluates rank agreement.

Research explanation:

> Spearman tells me whether two metrics order captions similarly even if their numerical scales are different.

---

# Correlation Results

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

Interpretation:

> The correlations were positive but only moderate. This indicates that traditional metrics and semantic image-text evaluation capture overlapping but distinct properties of caption quality.

One particularly interesting observation was the lower correlation between BLEU-4 and CLIPScore.

```text
BLEU-4 vs CLIPScore ≈ 0.386
```

A reasonable interpretation is:

> BLEU-4 is strongly affected by exact multi-word overlap, while CLIPScore is more sensitive to semantic image-caption alignment.

---

# Research Question 3:
# Where Do Metrics Disagree?

Correlation gives an aggregate picture, but it does not show individual failure cases.

So I performed per-image disagreement analysis.

I identified examples such as:

```text
High CLIPScore / Low BLEU

Low CLIPScore / High BLEU
```

These cases help reveal what each metric is emphasizing.

---

# Example: Semantic High, Lexical Low

Example generated caption:

```text
a black dog is running in the grass
```

The caption received approximately:

```text
BLEU-4:    0.042
CLIPScore: 0.823
POLOS:     0.722
```

Interpretation:

> The generated caption used wording that differed from the reference captions, so BLEU-4 was low, but the semantic metrics judged the caption to be well aligned with the image.

This is evidence that low lexical overlap does not always imply poor semantic quality.

---

# Example: Lexical High, Semantic Low

Another case had approximately:

```text
BLEU-4:    0.306
CLIPScore: 0.400
POLOS:     0.160
```

The generated caption shared lexical patterns with the reference descriptions but was not strongly aligned with the image content.

Interpretation:

> Lexical similarity can sometimes produce a relatively strong score even when the generated caption contains semantic errors.

---

# Rank Disagreement Analysis

I also analyzed how differently metrics ranked captions.

This is important because two metrics can have moderate correlation while still disagreeing significantly on individual rankings.

Research explanation:

> I compared normalized caption rankings across metrics to identify cases where one metric considered a caption strong while another metric ranked it much lower.

The largest rank disagreements frequently involved CLIPScore or POLOS compared with traditional metrics.

This further supported the hypothesis that semantic metrics capture different properties from lexical overlap metrics.

---

# Why Correlation Alone Was Not Enough

Correlation analysis is observational.

A correlation can tell us that two metrics behave differently, but it does not directly tell us which metric is responding appropriately to a known semantic error.

That motivated the second phase of the study:

> controlled perturbation experiments.

---

# Research Question 4:
# How Sensitive Are Metrics to Controlled Semantic Errors?

I deliberately modified captions to introduce known semantic errors.

Then I measured how the metric scores changed.

This gives a more controlled experimental setup because the original and perturbed captions differ in a known way.

---

# Perturbation Categories

## Object Replacement

Original:

```text
A dog is running through the grass.
```

Perturbed:

```text
A cat is running through the grass.
```

Research purpose:

> Test whether the metric notices that the central object identity has changed.

---

## Attribute Replacement

Original:

```text
A man is wearing a red shirt.
```

Perturbed:

```text
A man is wearing a blue shirt.
```

Research purpose:

> Test sensitivity to incorrect visual attributes.

---

## Action Replacement

Original:

```text
A boy is jumping into a pool.
```

Perturbed:

```text
A boy is sleeping near a pool.
```

Research purpose:

> Test sensitivity to incorrect actions.

---

## Negation

Original:

```text
A man is riding a bicycle.
```

Perturbed:

```text
A man is not riding a bicycle.
```

Research purpose:

> Test whether the metric can recognize a major semantic reversal caused by a very small lexical change.

---

## Hallucination

Original:

```text
A dog is running in a field.
```

Perturbed:

```text
A dog is running in a field next to a red car.
```

Research purpose:

> Test whether the metric penalizes information that is not supported by the image.

---

# Why Perturbation Experiments Are Important

Consider this pair:

```text
Original:
A dog is running through the grass.

Perturbed:
A cat is running through the grass.
```

Lexically, almost the entire sentence is unchanged.

Only one word changes:

```text
dog -> cat
```

However, semantically, the caption may now be incorrect.

This gives a controlled way to test whether a metric is sensitive to meaning rather than only word overlap.

Interview explanation:

> I used perturbations as a controlled robustness test. Instead of only observing natural metric disagreement, I introduced known semantic errors and measured whether each metric responded appropriately.

---

# Sensitivity Measurement

For each perturbation, I calculated the change in metric score.

Conceptually:

```text
delta = perturbed_score - original_score
```

If the perturbation makes the caption worse, a useful metric should generally produce a negative score change.

The larger the decrease, the more sensitive the metric is to that type of error.

---

# Why Normalize the Score Changes?

Different metrics operate on different numerical scales.

For example:

```text
BLEU may operate roughly between 0 and 1
CIDEr can exceed 1
CLIPScore has its own score distribution
```

Therefore, raw score changes are not directly comparable.

I calculated normalized sensitivity values to make comparisons across metrics more meaningful.

---

# Bootstrap Confidence Intervals

I also used bootstrap confidence intervals.

Research explanation:

> I repeatedly resampled the perturbation results to estimate uncertainty around the average metric sensitivity.

This allowed me to distinguish between:

```text
a consistent metric response
```

and

```text
a result driven by only a few examples
```

---

# Main Findings

The main finding was:

> Traditional lexical metrics and semantic metrics capture different but complementary aspects of caption quality.

The correlation analysis showed that the metrics are related, but their correlations are far from perfect.

The disagreement analysis showed concrete cases where:

```text
semantic quality was high but lexical overlap was low
```

and cases where:

```text
lexical overlap was relatively high but semantic quality was weak
```

The perturbation experiments provided a more controlled way to evaluate metric sensitivity to semantic errors.

---

# Research Contribution

A concise interview answer:

> The main contribution of my project was an evaluation framework rather than a new caption-generation architecture. I combined lexical and semantic evaluation metrics, analyzed their correlations and ranking disagreements, identified qualitative failure cases, and then designed controlled perturbation experiments to evaluate how sensitive each metric was to semantic errors.

A slightly stronger research framing:

> My contribution was to treat caption evaluation itself as the research problem. Rather than reporting only aggregate BLEU scores, I investigated metric reliability through correlation analysis, disagreement analysis, rank analysis, and controlled robustness experiments.

---

# Results Summary

The captioning baseline achieved approximately:

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

The exact BLEU values varied slightly between evaluation runs.

The more important result for the research question was the moderate agreement between lexical and semantic metrics.

For example:

```text
BLEU-4 vs CLIPScore    ≈ 0.386
ROUGE-L vs CLIPScore   ≈ 0.540
CIDEr vs CLIPScore     ≈ 0.536
```

This supports the conclusion that the metrics should not be treated as interchangeable.

---

# Why Different Environments Were Used

Some research libraries had conflicting dependency requirements.

The workflow was separated into:

## Main Environment

```text
Caption generation
BLEU
METEOR
ROUGE-L
CIDEr
Statistical analysis
Perturbation analysis
```

## CLIP Environment

```text
CLIPScore
```

## Google Colab / Python 3.9

```text
POLOS
```

Research-oriented explanation:

> Because some evaluation libraries required incompatible dependency versions, I isolated them in reproducible environments and used structured CSV outputs as interfaces between stages of the experiment.

This is a better interview answer than saying only that the packages did not work together.

---

# Limitations

The study has several limitations.

## Dataset Size

Flickr8k is relatively small compared with Flickr30k or MSCOCO.

## Captioning Architecture

CNN-LSTM is a useful baseline, but it is older than modern transformer-based vision-language architectures.

## Automatic Evaluation

Automatic metrics are still proxies for human judgment.

## Perturbation Coverage

The perturbation set can be expanded to include more examples and more diverse semantic errors.

## Generalizability

The results should be validated across multiple datasets and captioning architectures before making broad conclusions.

---

# Future Research

Possible next steps include:

1. Replicate the experiment on Flickr30k and MSCOCO.
2. Compare CNN-LSTM with transformer-based captioning systems.
3. Add human evaluation as an external reference.
4. Expand controlled perturbation categories.
5. Compare additional learned evaluation metrics.
6. Analyze object, action, attribute, negation, and hallucination sensitivity separately.
7. Study whether metric behavior changes across different caption-generation architectures.
8. Investigate agreement between automatic metrics and human preference judgments.

---

# Common Research Interview Questions

## What was the research problem?

> I investigated whether traditional lexical evaluation metrics and semantic evaluation metrics measure the same aspects of image-caption quality.

---

## What was your hypothesis?

> My hypothesis was that lexical and semantic metrics would be positively related but would show substantial disagreement on captions where wording and semantic image alignment diverged.

---

## How did you test the hypothesis?

> I used three types of analysis: correlation analysis, per-image disagreement and ranking analysis, and controlled semantic perturbation experiments.

---

## Why was correlation analysis useful?

> It quantified the overall relationship between metrics and showed that they were related but not equivalent.

---

## Why did you need disagreement analysis if you already had correlation?

> Correlation summarizes average behavior. Disagreement analysis identifies individual cases where the metrics make very different judgments, which is important for understanding metric failure modes.

---

## Why did you add perturbation experiments?

> Natural disagreement cases are observational. Perturbations let me introduce a known semantic error and directly measure how each metric responds, which makes the analysis more controlled.

---

## What is the strongest part of the research?

> I think the strongest part is the combination of aggregate statistical analysis with controlled perturbation testing. Correlations show broad relationships, while perturbations test metric behavior under known semantic errors.

---

## What is the main result?

> The metrics are positively correlated, but semantic and lexical metrics are not interchangeable. They capture different aspects of caption quality, so a multi-metric evaluation framework is more informative than relying on a single score.

---

## Was the CNN-LSTM architecture your main contribution?

> No. I treated CNN-LSTM as a reproducible baseline. The main contribution was the evaluation study built around the generated captions.

---

## Why use an older CNN-LSTM model?

> Using a standard baseline helped isolate the evaluation problem. If I had simultaneously introduced a complex new captioning architecture, it would have been harder to separate model behavior from metric behavior.

---

## What would make the work stronger for publication?

> I would extend it to larger datasets, evaluate modern transformer-based captioning models, add human judgments, expand the perturbation dataset, and statistically compare metric sensitivity across error categories.

---

# 1-Minute Research Interview Answer

> My research focused on the reliability of automatic evaluation metrics for image captioning. I first implemented a CNN-LSTM captioning baseline on Flickr8k using pretrained VGG16 features and an LSTM decoder. I then evaluated the generated captions using traditional reference-based metrics such as BLEU, METEOR, ROUGE-L, and CIDEr, as well as semantic metrics such as CLIPScore and POLOS. The main question was whether these metrics measure the same aspects of caption quality. I performed Pearson and Spearman correlation analysis, rank-disagreement analysis, and qualitative case studies. I also designed controlled perturbation experiments where I changed objects, attributes, actions, introduced negation, or added hallucinated information, and measured how each metric responded. The main finding was that lexical and semantic metrics are related but not interchangeable, so using multiple complementary metrics gives a more reliable evaluation of caption quality.

---

# 2-Minute Research Interview Answer

> My project investigated the reliability of automatic metrics for image-caption evaluation. I used a CNN-LSTM captioning model as a controlled baseline rather than treating the generation architecture itself as the main contribution. The model used pretrained VGG16 features as the visual representation and an LSTM decoder to generate captions on Flickr8k.
>
> I evaluated the generated captions using BLEU, METEOR, ROUGE-L, CIDEr, CLIPScore, and POLOS. My main research question was whether reference-based lexical metrics and semantic image-text metrics evaluate captions in the same way.
>
> I first performed Pearson and Spearman correlation analysis. The correlations were positive but moderate. For example, BLEU-4 and CLIPScore had a Pearson correlation of about 0.39, which suggested that exact lexical overlap and semantic image alignment were capturing different properties.
>
> I then performed per-image disagreement and ranking analysis. This revealed cases where a caption received a low BLEU score but high semantic scores because it described the image correctly using different wording, and the opposite case where lexical overlap was relatively strong but semantic alignment was poor.
>
> To make the experiment more controlled, I created semantic perturbations such as object replacement, attribute replacement, action replacement, negation, and hallucination. I measured how much each metric score changed after introducing these known errors and used normalized sensitivity and bootstrap confidence intervals to analyze the robustness of the results.
>
> The main conclusion was that no single automatic metric captures all dimensions of caption quality. Traditional and semantic metrics provide complementary information, so a stronger evaluation framework should use multiple metrics and analyze where they disagree.

---

# Very Short Research Version

If an interviewer asks only:

## "Tell me about your research."

Say:

> I studied the reliability of automatic evaluation metrics for image captioning. I used a CNN-LSTM captioning system as a baseline, evaluated its captions with both lexical and semantic metrics, analyzed correlations and ranking disagreements, and then designed controlled perturbation experiments to test how sensitive each metric was to semantic errors. The main finding was that lexical overlap and semantic image alignment capture different aspects of caption quality, so multi-metric evaluation is more reliable than using a single score.
