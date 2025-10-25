# Results

## DeBERTa Predictions

Contains complete test set predictions and calibration metrics for three datasets:

### Prediction Files

Each dataset has a predictions JSON file with the following structure:

- `example_id`: Sequential identifier
- `text`: Input text
- `gold_label`: Ground truth label (integer)
- `predicted_label`: Model prediction (integer)
- `logits`: Raw model outputs (before softmax)
- `probabilities`: Softmax probabilities for each class
- `entropy`: Prediction entropy (uncertainty measure)
- `max_prob`: Maximum probability (confidence score)
- `correct`: Boolean indicating if prediction matches gold label

**Files:**
- `imdb_deberta_predictions.json` (12,500 samples)
- `jigsaw_deberta_predictions.json` (97,320 samples)
- `fever_deberta_predictions.json` (9,999 samples)

### Metrics Files

Each dataset has a metrics JSON file containing:

- Overall performance: accuracy, precision, recall, F1-score
- Calibration metrics: ECE (Expected Calibration Error), Brier score
- Per-class metrics: precision, recall, F1-score, support for each class
- Average entropy and confidence scores

**Files:**
- `imdb_deberta_metrics.json`
- `jigsaw_deberta_metrics.json`
- `fever_deberta_metrics.json`

### Calibration Plots

Contains reliability diagrams and confusion matrices for each dataset:

- Calibration plots show model confidence vs actual accuracy
- Confusion matrices show prediction patterns across classes

**Files:**
- `{dataset}_calibration.png` - Reliability diagrams
- `{dataset}_confusion_matrix.png` - Normalized confusion matrices

## LLM Annotations (Qwen2.5-7B-Instruct)

Complete LLM annotation results for three datasets using Qwen2.5-7B-Instruct model.

### Dataset Results

Each dataset contains:
- `{dataset}_llm_annotations.json` - Complete annotation results
- `{dataset}_agreement_metrics.json` - Performance metrics
- `{dataset}_hard_cases.json` - Difficult samples for analysis

**Datasets:**
- `imdb/` - Sentiment analysis (12,500 samples, 95.7% accuracy)
- `fever/` - Fact verification (9,999 samples, 53.8% accuracy)  
- `jigsaw/` - Toxicity detection (9,900 samples, 76.8% accuracy)

### Annotation Format

Each annotation includes:
- `text`: Input text
- `gold_label`: Ground truth label
- `llm_label`: LLM prediction
- `llm_confidence`: Verbal confidence (low/medium/high)
- `llm_rationale`: Explanation (when applicable)
- `is_valid`: JSON parsing success
- `raw_output`: Raw LLM response

### LLM Performance Summary

| Dataset | Samples | Accuracy | F1-Score | Kappa | Confidence |
|---------|---------|----------|----------|-------|------------|
| IMDB    | 12,500  | 95.7%    | 0.957    | 0.915 | 0.873      |
| FEVER   | 9,999   | 53.8%    | 0.543    | 0.308 | 0.615      |
| Jigsaw  | 9,900   | 76.8%    | 0.323    | 0.228 | 0.836      |

### Usage

```bash
# Run LLM annotation
python scripts/run_llm_annotations_7b.py --datasets imdb fever jigsaw

# Stratified sampling for Jigsaw
python scripts/sample_jigsaw_stratified.py
```

### Model Configuration

- **Model**: Qwen2.5-7B-Instruct (7B parameters)
- **Batch Size**: 32 samples
- **Temperature**: 0.1 (near-deterministic)
- **Max Tokens**: 12-24 (task-dependent)
- **Optimization**: TF32, SDPA attention, length-based batching

## Reproducing Results

To regenerate all predictions and metrics:

```bash
# DeBERTa predictions
python scripts/save_deberta_predictions.py

# LLM annotations
python scripts/run_llm_annotations_7b.py --datasets all
```

Requires trained models in `outputs/{dataset}/deberta-v3-base/run1/best_model_calibrated/`

## Model Performance Summary

| Dataset | DeBERTa Accuracy | DeBERTa F1-Score | DeBERTa ECE | DeBERTa Brier | LLM Accuracy | LLM F1-Score |
|---------|------------------|------------------|-------------|---------------|--------------|--------------|
| IMDB    | 95.14%           | 0.9518           | 0.0371      | 0.0430        | 95.7%        | 0.957        |
| Jigsaw  | 95.09%           | 0.6772           | 0.0129      | 0.0374        | 76.8%        | 0.323        |
| FEVER   | 64.52%           | 0.6449           | 0.0566      | 0.1563        | 53.8%        | 0.543        |

Lower ECE and Brier scores indicate better calibration.