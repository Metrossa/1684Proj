"""
Evaluation script for trained baseline models.

Usage:
    python scripts/eval_baseline_model.py \
        --dataset imdb \
        --load_dir outputs/imdb/deberta-v3-base/run1/best_model \
        --batch_size 16
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import load_dataset_by_name
from models.baseline_models import SupervisedClassifier
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    dataset_name: str,
    load_dir: str,
    batch_size: int = 16,
    split: str = "test",
):
    """
    Evaluate a trained model on a dataset.
    
    Args:
        dataset_name: Dataset name
        load_dir: Directory containing trained model
        batch_size: Batch size for evaluation
        split: Which split to evaluate on ('test', 'dev', or 'train')
    """
    logger.info(f"=" * 80)
    logger.info(f"Evaluating model on {dataset_name} ({split} split)")
    logger.info(f"=" * 80)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    data = load_dataset_by_name(dataset_name)
    
    if split not in data:
        raise ValueError(f"Split '{split}' not found in dataset")
    
    eval_df = data[split]
    logger.info(f"{split.capitalize()} size: {len(eval_df)}")
    
    # Load trained model
    load_path = Path(load_dir)
    if not load_path.exists():
        raise ValueError(f"Model directory not found: {load_path}")
    
    logger.info(f"Loading model from: {load_path}")
    
    # Get task configuration from dataset
    dataset_config = config.DATASETS[dataset_name]
    task_type = dataset_config['task']
    num_labels = eval_df['label'].nunique()
    multilabel = task_type == "toxicity"
    
    # Create model instance
    model = SupervisedClassifier(
        task_type=task_type,
        num_labels=num_labels,
        multilabel=multilabel,
    )
    
    # Load trained weights
    model.load_trained_model(load_path)
    
    # Prepare evaluation data
    eval_texts = eval_df['text'].astype(str).tolist()
    eval_labels = eval_df['label'].tolist()
    
    # Make predictions
    logger.info(f"Making predictions on {len(eval_texts)} samples...")
    predictions = model.predict_batch(eval_texts, batch_size=batch_size)
    
    # Calculate metrics
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    
    correct = 0
    for pred, true_label in zip(predictions, eval_labels):
        if multilabel:
            pred_labels = set(pred['label']) if isinstance(pred['label'], list) else {pred['label']}
            true_labels = set([true_label]) if isinstance(true_label, int) else set(true_label)
            if pred_labels == true_labels:
                correct += 1
        else:
            if pred['label'] == true_label:
                correct += 1
    
    accuracy = correct / len(eval_labels)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Calculate average entropy and confidence
    avg_entropy = sum(p['entropy'] for p in predictions) / len(predictions)
    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
    
    logger.info(f"Average Entropy: {avg_entropy:.4f}")
    logger.info(f"Average Confidence: {avg_confidence:.4f}")
    
    # Calculate detailed metrics using sklearn
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        if not multilabel:
            true_labels_array = np.array(eval_labels)
            pred_labels_array = np.array([p['label'] for p in predictions])
            
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(true_labels_array, pred_labels_array))
            
            logger.info("\nConfusion Matrix:")
            logger.info("\n" + str(confusion_matrix(true_labels_array, pred_labels_array)))
    except Exception as e:
        logger.warning(f"Could not generate detailed metrics: {e}")
    
    # Save evaluation results
    results_file = Path(load_dir).parent / f"eval_{split}_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Model: {load_path}\n")
        f.write(f"Samples: {len(eval_labels)}\n\n")
        f.write("Results:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Average Entropy: {avg_entropy:.4f}\n")
        f.write(f"  Average Confidence: {avg_confidence:.4f}\n")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("Evaluation completed!")
    
    return {
        "accuracy": accuracy,
        "avg_entropy": avg_entropy,
        "avg_confidence": avg_confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained baseline models")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["imdb", "jigsaw", "crisisbench_humanitarian", "crisisbench_informativeness", "fever"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        required=True,
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to evaluate on (default: test)"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        dataset_name=args.dataset,
        load_dir=args.load_dir,
        batch_size=args.batch_size,
        split=args.split,
    )


if __name__ == "__main__":
    main()

