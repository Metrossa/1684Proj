# LLM Annotation Reliability Analysis - Dataset Repository

This repository contains the dataset loading infrastructure for analyzing LLM annotation reliability across multiple datasets.

## Datasets

- **IMDb**: Sentiment analysis dataset
- **Jigsaw**: Toxicity classification dataset  
- **CrisisBench**: Crisis tweet classification dataset

## Project Structure

```
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── test_datasets.py         # Test script for dataset loading
├── data/
│   └── dataset_loader.py     # Dataset loading utilities
└── models/                   # Directory for saved models
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

## Configuration

The `config.py` file contains:

### Datasets
- IMDb: Sentiment analysis (positive/negative)
- Jigsaw: Toxicity classification (toxic/non-toxic)
- CrisisBench: Crisis classification (crisis/not_crisis)

### BERT Models
- distilbert-base-uncased
- microsoft/deberta-v3-base

### LLM Models (Qwen)
- Qwen/Qwen-7B
- Qwen/Qwen-14B

## Dataset Format

Each dataset is loaded with the following structure:
- **text**: Input text for classification
- **label**: Ground truth label (0 or 1)
- **Splits**: Stratified train/dev/test splits (70%/15%/15%)

## Dependencies

- `torch` - PyTorch for deep learning
- `transformers` - HuggingFace transformers
- `datasets` - HuggingFace datasets
- `scikit-learn` - Machine learning utilities
- `pandas`/`numpy` - Data manipulation
- `requests` - HTTP requests
- `tqdm` - Progress bars
