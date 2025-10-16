# LLM Annotation Reliability Analysis Framework

A comprehensive framework for analyzing LLM annotation reliability across multiple datasets with predictive modeling for confidence scores and interactive visualization dashboard.

## Overview

This project implements two main tracks:
- **Track 1**: Predictive Modeling for Confidence - Train trust score classifiers to predict when LLM annotations are reliable
- **Track 2**: Dashboard for Annotation Analysis - Interactive visualization tool for exploring confidence regions and annotation quality

## Datasets

- **IMDb**: Sentiment analysis dataset (positive/negative)
- **Jigsaw**: Toxicity classification dataset (toxic/non-toxic)
- **CrisisBench**: Crisis tweet classification dataset (crisis/not_crisis)
- **FEVER**: Fact verification dataset (supports/refutes)

## Project Structure

```
1684Proj/
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── test_datasets.py             # Test script for dataset loading
├── data/
│   └── dataset_loader.py         # Dataset loading utilities
├── llm/                          # LLM annotation module
│   ├── annotator.py              # QwenAnnotator for local inference
│   └── prompts.py                # Structured prompt templates
├── models/                       # Supervised baseline models
│   ├── baseline_models.py        # Pre-trained transformer classifiers
│   └── text_features.py          # Text feature extraction
├── evaluation/                   # Trust scoring and metrics
│   ├── trust_scorer.py           # Trust score classifier
│   ├── hard_cases.py             # Hard case identification
│   └── metrics.py                # Evaluation metrics
├── dashboard/                    # Interactive visualization
│   ├── app.py                    # Main Plotly Dash application
│   └── components/               # Dashboard UI components
│       ├── overview.py           # High-level statistics
│       ├── explorer.py           # Filterable annotation table
│       ├── confidence_regions.py # Confidence threshold visualization
│       ├── fairness.py           # Subgroup performance analysis
│       └── export.py             # Export functionality
├── pipeline/                     # End-to-end workflows
│   ├── annotation_pipeline.py    # Complete annotation workflow
│   └── experiment_runner.py      # Experiment orchestration
├── utils/                        # Utility functions
│   ├── validation.py             # JSON schema validation
│   └── visualization.py          # Plotting utilities
├── scripts/                      # Executable scripts
│   ├── run_annotation.py         # Run LLM annotation
│   ├── train_trust_scorer.py     # Train trust classifier
│   └── launch_dashboard.py       # Start dashboard
├── tests/                        # Test suite
│   ├── test_structure.py         # Structure validation
│   ├── test_llm.py              # LLM module tests
│   ├── test_baseline_models.py   # Baseline model tests
│   ├── test_evaluation.py        # Evaluation module tests
│   ├── test_dashboard.py         # Dashboard component tests
│   ├── test_pipeline.py          # Pipeline tests
│   └── test_utils.py             # Utility function tests
├── results/                      # Generated outputs
└── models/                       # Saved model weights
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Test Dataset Loading
```bash
python test_datasets.py
```

### Load Datasets Programmatically
```python
from data.dataset_loader import load_dataset_by_name

# Load IMDb dataset
imdb_data = load_dataset_by_name('imdb')
print(f"Train: {len(imdb_data['train'])} samples")
print(f"Dev: {len(imdb_data['dev'])} samples") 
print(f"Test: {len(imdb_data['test'])} samples")
```

## Features

### LLM Annotation (Track 1)
- **Local Qwen-7B inference** with deterministic generation
- **Structured JSON output** with label, confidence, and rationale
- **Batch processing** with progress tracking
- **Schema validation** for output quality control
- **Multiple task support** (sentiment, toxicity, crisis, fact verification)

### Trust Score Prediction
- **Feature extraction** combining LLM confidence, supervised model disagreement, text features
- **Multiple classifiers** (Logistic Regression, Random Forest, Gradient Boosting)
- **Threshold optimization** on development set
- **Confidence regions** for accept/review decisions

### Supervised Baselines
- **Pre-trained transformers** (DeBERTa-v3, BERT, RoBERTa)
- **Probability distributions** and entropy calculation
- **Multi-task support** across all datasets
- **No fine-tuning** for initial baseline comparison

### Interactive Dashboard (Track 2)
- **Plotly Dash interface** with multi-page layout
- **Confidence region visualization** with threshold analysis
- **Filterable annotation explorer** with detailed views
- **Fairness metrics** and subgroup analysis
- **Export functionality** for human review cases
- **Real-time filtering** and drill-down capabilities

### Evaluation Metrics
- **Agreement metrics**: Krippendorff's alpha, Cohen's kappa
- **Calibration**: Expected Calibration Error (ECE), Brier score
- **Fairness**: Subgroup error rates and bias analysis
- **Trust prediction**: ROC-AUC, precision-recall curves

## Configuration

The `config.py` file contains:

### Datasets
- IMDb: Sentiment analysis (positive/negative)
- Jigsaw: Toxicity classification (toxic/non-toxic)
- CrisisBench: Crisis classification (crisis/not_crisis)
- FEVER: Fact verification (supports/refutes)

### Model Configurations
- **Supervised Models**: DeBERTa-v3-base, BERT-base
- **LLM Models**: Qwen-7B, Qwen-14B
- **Trust Scorers**: Logistic Regression, Random Forest, Gradient Boosting

### LLM Settings
- Temperature: 0.1 (deterministic)
- Max tokens: 100
- JSON output format with validation

## Dataset Format

Each dataset is loaded with the following structure:
- **text**: Input text for classification
- **label**: Ground truth label (0 or 1)
- **Splits**: Stratified train/dev/test splits (70%/15%/15%)

## Dependencies

### Core Dependencies
- `torch` - PyTorch for deep learning
- `transformers` - HuggingFace transformers
- `datasets` - HuggingFace datasets
- `scikit-learn` - Machine learning utilities
- `pandas`/`numpy` - Data manipulation

### Dashboard Dependencies
- `plotly` - Interactive plotting
- `dash` - Web application framework
- `dash-bootstrap-components` - UI components

### Additional Dependencies
- `nltk` - Natural language processing
- `textstat` - Readability metrics
- `krippendorff` - Agreement metrics
- `matplotlib`/`seaborn` - Static plotting
- `requests` - HTTP requests
- `tqdm` - Progress bars

## Testing

Run the complete test suite:
```bash
# Test all modules
python tests/test_structure.py
python tests/test_llm.py
python tests/test_baseline_models.py
python tests/test_evaluation.py
python tests/test_dashboard.py
python tests/test_pipeline.py
python tests/test_utils.py

# Run all tests at once
python -m pytest tests/ -v
```
