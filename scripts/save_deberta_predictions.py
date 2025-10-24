"""
Save DeBERTa model predictions and calibration metrics.

This script:
1. Loads trained DeBERTa models for IMDb, Jigsaw, and FEVER
2. Runs inference on test sets
3. Saves predictions with logits, probabilities, entropy, etc.
4. Calculates and saves calibration metrics
5. Generates calibration plots
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from models.baseline_models import SupervisedClassifier
from data.dataset_loader import load_dataset_by_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check device availability
if torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    logger.info("GPU not available, using CPU")

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def compute_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute categorical entropy for probability distributions.
    
    Args:
        probs: Probability array of shape [N, C]
        eps: Small value to avoid log(0)
        
    Returns:
        Entropy array of shape [N]
    """
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels (integers)
        y_prob: Predicted probabilities [N, C]
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    # Get max probabilities and predictions
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier Score (mean squared error between probabilities and one-hot labels).
    
    Args:
        y_true: True labels (integers)
        y_prob: Predicted probabilities [N, C]
        
    Returns:
        Brier score
    """
    # Convert to one-hot
    n_classes = y_prob.shape[1]
    y_true_onehot = np.zeros_like(y_prob)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1.0
    
    # Compute mean squared error
    return float(np.mean((y_prob - y_true_onehot) ** 2))


def generate_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, 
                             dataset_name: str, save_path: Path, n_bins: int = 15):
    """
    Generate and save calibration (reliability) diagram.
    
    Args:
        y_true: True labels (integers)
        y_prob: Predicted probabilities [N, C]
        dataset_name: Name of dataset
        save_path: Path to save plot
        n_bins: Number of bins
    """
    # Get max probabilities and predictions
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            bin_confidences.append(confidences[in_bin].mean())
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_counts.append(prop_in_bin)
        else:
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(0.0)
            bin_counts.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot calibration curve
    ax.plot(bin_confidences, bin_accuracies, 'o-', label='Model Calibration', 
            linewidth=2, markersize=8, color='steelblue')
    
    # Add histogram of confidence values
    ax2 = ax.twinx()
    ax2.bar(bin_confidences, bin_counts, width=1.0/n_bins, alpha=0.3, 
            color='gray', label='Sample Count')
    ax2.set_ylabel('Count', fontsize=12)
    
    # Formatting
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Calibration Plot - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved calibration plot to {save_path}")


def generate_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                   class_names: List[str], dataset_name: str, 
                                   save_path: Path):
    """
    Generate and save confusion matrix heatmap.
    
    Args:
        y_true: True labels (integers)
        y_pred: Predicted labels (integers)
        class_names: List of class names
        dataset_name: Name of dataset
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_path}")


def save_predictions(dataset_name: str, model_path: Path, output_dir: Path):
    """
    Load model, run inference, and save predictions with metrics.
    
    Args:
        dataset_name: Name of dataset (imdb, jigsaw, fever)
        model_path: Path to trained model
        output_dir: Directory to save outputs
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    data_splits = load_dataset_by_name(dataset_name)
    test_df = data_splits['test']
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    logger.info(f"Test set size: {len(test_texts)}")
    
    # Get class information
    dataset_config = config.DATASETS[dataset_name]
    class_names = dataset_config['classes']
    num_classes = len(class_names)
    
    # Create label mapping
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    logger.info(f"Using device: {DEVICE}")
    classifier = SupervisedClassifier(
        model_name=str(model_path),
        task_type=dataset_config['task'],
        num_labels=num_classes,
        multilabel=False,
        id2label=id2label,
        label2id=label2id,
        max_length=256,
        device=DEVICE
    )
    classifier.load_trained_model(model_path)
    
    # Run inference
    logger.info("Running inference on test set...")
    predictions = classifier.predict_batch(test_texts, batch_size=32)
    
    # Extract prediction data
    output_data = {
        'example_id': list(range(len(test_texts))),
        'text': test_texts,
        'gold_label': test_labels,
        'predicted_label': [],
        'logits': [],
        'probabilities': [],
        'entropy': [],
        'max_prob': [],
        'correct': []
    }
    
    for i, pred in enumerate(predictions):
        output_data['predicted_label'].append(pred['label'])
        output_data['logits'].append(pred['logits'])
        output_data['probabilities'].append(pred['probabilities'])
        output_data['entropy'].append(pred['entropy'])
        output_data['max_prob'].append(pred['confidence'])
        output_data['correct'].append(pred['label'] == test_labels[i])
    
    # Save predictions to JSON
    predictions_file = output_dir / f"{dataset_name}_deberta_predictions.json"
    logger.info(f"Saving predictions to {predictions_file}...")
    
    with open(predictions_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Verify predictions
    logger.info("\nVerifying predictions...")
    logger.info(f"Sample count: {len(output_data['example_id'])} (expected: {len(test_texts)})")
    
    # Check probability sums
    prob_sums = [sum(probs) for probs in output_data['probabilities']]
    avg_prob_sum = np.mean(prob_sums)
    logger.info(f"Average probability sum: {avg_prob_sum:.6f} (expected: ~1.0)")
    
    # Verify entropy calculation
    probs_array = np.array(output_data['probabilities'])
    computed_entropy = compute_entropy(probs_array)
    saved_entropy = np.array(output_data['entropy'])
    entropy_diff = np.abs(computed_entropy - saved_entropy).max()
    logger.info(f"Entropy calculation verified (max diff: {entropy_diff:.8f})")
    
    # Calculate metrics
    logger.info("\nCalculating metrics...")
    y_true = np.array(test_labels)
    y_pred = np.array(output_data['predicted_label'])
    y_prob = np.array(output_data['probabilities'])
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle different averaging based on number of classes
    if num_classes == 2:
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calibration metrics
    ece = compute_ece(y_true, y_prob, n_bins=15)
    brier_score = compute_brier_score(y_true, y_prob)
    
    # Per-class metrics
    per_class_metrics = {}
    for class_idx, class_name in enumerate(class_names):
        # Binary mask for this class
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        if y_true_binary.sum() > 0:  # Only compute if class exists in test set
            class_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            class_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            class_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        else:
            class_precision = class_recall = class_f1 = 0.0
        
        per_class_metrics[class_name] = {
            'precision': float(class_precision),
            'recall': float(class_recall),
            'f1_score': float(class_f1),
            'support': int(y_true_binary.sum())
        }
    
    # Create metrics dictionary
    metrics = {
        'dataset': dataset_name,
        'model_path': str(model_path),
        'test_size': len(test_texts),
        'num_classes': num_classes,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'ece': float(ece),
        'brier_score': float(brier_score),
        'per_class_metrics': per_class_metrics,
        'average_entropy': float(np.mean(output_data['entropy'])),
        'average_confidence': float(np.mean(output_data['max_prob']))
    }
    
    # Save metrics
    metrics_file = output_dir / f"{dataset_name}_deberta_metrics.json"
    logger.info(f"Saving metrics to {metrics_file}...")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print metrics summary
    logger.info("\n" + "="*60)
    logger.info("METRICS SUMMARY")
    logger.info("="*60)
    logger.info(f"Accuracy:      {accuracy:.4f}")
    logger.info(f"Precision:     {precision:.4f}")
    logger.info(f"Recall:        {recall:.4f}")
    logger.info(f"F1-Score:      {f1:.4f}")
    logger.info(f"ECE:           {ece:.4f}")
    logger.info(f"Brier Score:   {brier_score:.4f}")
    logger.info("="*60)
    
    # Generate calibration plot
    calibration_dir = output_dir / "calibration_plots"
    calibration_dir.mkdir(exist_ok=True)
    
    calibration_plot_path = calibration_dir / f"{dataset_name}_calibration.png"
    generate_calibration_plot(y_true, y_prob, dataset_name, calibration_plot_path)
    
    # Generate confusion matrix
    confusion_matrix_path = calibration_dir / f"{dataset_name}_confusion_matrix.png"
    generate_confusion_matrix_plot(y_true, y_pred, class_names, dataset_name, 
                                   confusion_matrix_path)
    
    logger.info(f"\nCompleted processing {dataset_name.upper()}\n")


def main():
    """Main function to save predictions and metrics for all datasets."""
    # Create output directory
    output_dir = project_root / "results" / "deberta_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Define datasets and their model paths
    datasets = {
        'imdb': project_root / "outputs" / "imdb" / "deberta-v3-base" / "run1" / "best_model_calibrated",
        'jigsaw': project_root / "outputs" / "jigsaw" / "deberta-v3-base" / "run1" / "best_model_calibrated",
        'fever': project_root / "outputs" / "fever" / "deberta-v3-base" / "run1" / "best_model_calibrated"
    }
    
    # Check which models exist
    existing_datasets = {}
    for dataset_name, model_path in datasets.items():
        if model_path.exists():
            existing_datasets[dataset_name] = model_path
            logger.info(f"Found model for {dataset_name}: {model_path}")
        else:
            logger.warning(f"Model not found for {dataset_name}: {model_path}")
    
    if not existing_datasets:
        logger.error("No trained models found! Please train models first.")
        return
    
    # Process each dataset
    for dataset_name, model_path in existing_datasets.items():
        try:
            save_predictions(dataset_name, model_path, output_dir)
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}", exc_info=True)
            continue
    
    logger.info("\n" + "="*60)
    logger.info("ALL PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - Predictions: *_deberta_predictions.json")
    logger.info(f"  - Metrics: *_deberta_metrics.json")
    logger.info(f"  - Plots: calibration_plots/")


if __name__ == "__main__":
    main()

