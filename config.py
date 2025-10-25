"""
Configuration file for LLM annotation reliability analysis project.
"""
import os
from pathlib import Path

# Fix OpenMP library conflict (required for PyTorch with conda)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
    # CrisisBench English subsets (loaded from Hugging Face: QCRI/CrisisBench-english)
    # Note: The exact class names are taken from the dataset features at runtime.
    "crisisbench_humanitarian": {
        "name": "CrisisBench (Humanitarian)",
        "task": "crisis_humanitarian",
        "classes": [
            "infrastructure_and_utilities_damage",
            "not_humanitarian",
            "injured_or_dead_people",
            "sympathy_and_support",
            "donation_and_volunteering",
            "response_efforts",
            "caution_and_advice",
            "requests_or_needs",
            "affected_individual",
            "displaced_and_evacuations",
            "missing_and_found_people"
        ],
        "split_ratios": {"train": 0.7, "dev": 0.15, "test": 0.15}
    },
    "crisisbench_informativeness": {
        "name": "CrisisBench (Informativeness)",
        "task": "crisis_informativeness",
        "classes": ["informative", "not_informative"],
        "split_ratios": {"train": 0.7, "dev": 0.15, "test": 0.15}
    },
    "fever": {
        "name": "FEVER Fact Verification",
        "task": "fact_verification",
        "classes": ["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"],
        "split_ratios": None  # Pre-split dataset
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
    "qwen-14b": "Qwen/Qwen3-14B",
    "qwen-7b": "models/Qwen2.5-7B-Instruct",
    "qwen-7b-awq": "models/Qwen2.5-7B-Instruct-AWQ"
}

# LLM configurations
LLM_CONFIG_7B_TRANSFORMERS = {
    "model_name": "qwen-7b",
    "temperature": 0.1,
    "top_p": 0.7,
    "top_k": 10,
    "repetition_penalty": 1.05,
    "max_tokens": 24,
    "do_sample": True,
    "seed": 42
}

LLM_CONFIG_7B_VLLM = {
    "model_name": "qwen-7b-awq",
    "temperature": 0.1,
    "do_sample": False,
    "max_tokens": 12,
    "repetition_penalty": 1.0,
    "seed": 42
}

LLM_BATCH_SIZE_7B = 32
LLM_CONFIG_7B = LLM_CONFIG_7B_TRANSFORMERS
LLM_CONFIG = LLM_CONFIG_7B_TRANSFORMERS

ANNOTATION_CONFIG = {
    "save_full_text": False,
    "text_preview_length": 100,
    "save_text_preview_for_hard_cases": True,
    "rationale_strategy": "selected",
    "rationale_conditions": {
        "low_confidence": True,
        "disagreement": True,
        "baseline_uncertain": True,
        "audit_sample_rate": 0.05
    },
    "rationale_max_tokens": 48, 
    "rationale_max_words": 12
}

EVALUATION = {
    "random_seed": 42
}