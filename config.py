"""
Configuration file for LLM annotation reliability analysis project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, DASHBOARD_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configurations
DATASETS = {
    "imdb": {
        "name": "IMDb Sentiment",
        "task": "sentiment",
        "classes": ["negative", "positive"],
        "split_ratios": {"train": 0.7, "dev": 0.15, "test": 0.15}
    },
    "jigsaw": {
        "name": "Jigsaw Toxicity",
        "task": "toxicity",
        "classes": ["non-toxic", "toxic"],
        "split_ratios": {"train": 0.7, "dev": 0.15, "test": 0.15}
    },
    "crisisbench": {
        "name": "CrisisBench",
        "task": "crisis_classification",
        "classes": ["not_crisis", "crisis"],
        "split_ratios": {"train": 0.7, "dev": 0.15, "test": 0.15}
    }
}

# Model configurations
BERT_MODELS = {
    "bert-base-uncased": "bert-base-uncased",
    "bert-large-uncased": "bert-large-uncased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "roberta-base": "roberta-base",
    "deberta-v3-base": "microsoft/deberta-v3-base"
}

# LLM configurations
LLM_MODELS = {
    "qwen-7b": "Qwen/Qwen-7B",
    "qwen-14b": "Qwen/Qwen-14B",
    "qwen-32b": "Qwen/Qwen-32B",
    "qwen-72b": "Qwen/Qwen-72B"
}

# LLM settings
LLM_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 100,
    "top_p": 0.9,
    "model_name": "qwen-7b"  # Default Qwen model
}

# Evaluation settings (for dataset splitting)
EVALUATION = {
    "random_seed": 42
}