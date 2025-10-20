"""
Training script for baseline supervised models.

This script demonstrates how to fine-tune DeBERTa-v3-base on different datasets:
- IMDb (sentiment analysis)
- Jigsaw (toxicity classification)
- CrisisBench (humanitarian & informativeness)

Usage:
    python scripts/train_baseline_models.py --dataset imdb --epochs 3 --batch_size 16
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import load_dataset_by_name
from models.baseline_models import SupervisedClassifier
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model_for_dataset(
    dataset_name: str,
    model_name: str = "microsoft/deberta-v3-base",
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    early_stopping_patience: int = 3,
    eval_steps: int = 500,
    seed: int = 42,
    save_dir: str = None,
    fit_temperature: bool = False,
):
    """
    Train a baseline model on a specific dataset.
    
    Args:
        dataset_name: Name of dataset (imdb, jigsaw, crisisbench_humanitarian, etc.)
        model_name: Model name or path (default: microsoft/deberta-v3-base)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_length: Maximum sequence length
        early_stopping_patience: Patience for early stopping
        eval_steps: Evaluate every N steps
        seed: Random seed for reproducibility
        save_dir: Directory to save trained model
        fit_temperature: Whether to fit temperature scaling after training
    """
    logger.info(f"=" * 80)
    logger.info(f"Training model for dataset: {dataset_name}")
    logger.info(f"=" * 80)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    data = load_dataset_by_name(dataset_name)
    
    train_df = data['train']
    dev_df = data['dev']
    test_df = data['test']
    
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Dev size: {len(dev_df)}")
    logger.info(f"Test size: {len(test_df)}")
    
    # Get dataset configuration
    dataset_config = config.DATASETS[dataset_name]
    task_type = dataset_config['task']
    
    # Determine number of labels
    num_labels = train_df['label'].nunique()
    logger.info(f"Task type: {task_type}")
    logger.info(f"Number of labels: {num_labels}")
    
    # Check if multi-label (labels are arrays/lists, not single integers)
    # For multilabel, each label should be a list or array, not a scalar
    first_label = train_df['label'].iloc[0]
    multilabel = isinstance(first_label, (list, np.ndarray)) and len(first_label) > 1
    
    # Create model
    logger.info(f"Initializing model: {model_name}...")
    model = SupervisedClassifier(
        model_name=model_name,
        task_type=task_type,
        num_labels=num_labels,
        multilabel=multilabel,
        max_length=max_length,
    )
    
    # Load pre-trained model
    logger.info(f"Loading pre-trained model: {model_name}...")
    model.load_model()
    
    # Prepare training data
    train_texts = train_df['text'].astype(str).tolist()
    train_labels = train_df['label'].tolist()
    
    val_texts = dev_df['text'].astype(str).tolist()
    val_labels = dev_df['label'].tolist()
    
    # Set up save directory
    if save_dir is None:
        save_dir = config.RESULTS_DIR / "trained_models" / dataset_name
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models will be saved to: {save_dir}")
    
    # Train model
    logger.info("Starting training...")
    history = model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir,
        eval_steps=eval_steps,
        seed=seed,
    )
    
    # Print training summary
    logger.info("\n" + "=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    
    # Extract validation metrics if available
    if history.get('val_metrics') and len(history['val_metrics']) > 0:
        val_losses = [m['loss'] for m in history['val_metrics']]
        val_macro_f1s = [m.get('macro_f1', 0) for m in history['val_metrics']]
        logger.info(f"Best val loss: {min(val_losses):.4f}")
        logger.info(f"Best val macro F1: {max(val_macro_f1s):.4f}")
        
        # Log best calibration metrics if available
        best_metrics = max(history['val_metrics'], key=lambda x: x.get('macro_f1', 0))
        if 'ece' in best_metrics:
            logger.info(f"Best model ECE: {best_metrics['ece']:.4f}")
        elif 'ece_macro' in best_metrics:
            logger.info(f"Best model ECE (macro): {best_metrics['ece_macro']:.4f}")
        if 'brier' in best_metrics:
            logger.info(f"Best model Brier: {best_metrics['brier']:.4f}")
        elif 'brier_macro' in best_metrics:
            logger.info(f"Best model Brier (macro): {best_metrics['brier_macro']:.4f}")
    
    # Load best model for evaluation
    best_model_path = save_dir / "best_model"
    if best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path}")
        model.load_trained_model(best_model_path)
    
    # Optionally fit temperature scaling on validation set
    optimal_temp = None
    if fit_temperature:
        logger.info("\nFitting temperature scaling on validation set...")
        
        # Get validation predictions with logits
        val_results = model.predict_batch(val_texts, batch_size=batch_size)
        val_logits = np.array([r['logits'] for r in val_results])
        val_labels_array = np.array(val_labels)
        
        # Fit temperature
        try:
            optimal_temp = model.fit_temperature(val_logits, val_labels_array, max_iter=50)
            logger.info(f"Optimal temperature: {optimal_temp:.4f}")
            
            # Save calibrated model
            calibrated_model_path = save_dir / "best_model_calibrated"
            model.save_model(calibrated_model_path)
            logger.info(f"Calibrated model saved to {calibrated_model_path}")
        except Exception as e:
            logger.warning(f"Failed to fit temperature: {e}")
            optimal_temp = None
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_texts = test_df['text'].astype(str).tolist()
    test_labels = test_df['label'].tolist()
    
    # Make predictions on test set
    test_results = model.predict_batch(test_texts, batch_size=batch_size)
    
    # Calculate test accuracy
    correct = 0
    for pred, true_label in zip(test_results, test_labels):
        if multilabel:
            # For multi-label, check if prediction matches
            pred_labels = set(pred['label']) if isinstance(pred['label'], list) else {pred['label']}
            true_labels = set([true_label]) if isinstance(true_label, int) else set(true_label)
            if pred_labels == true_labels:
                correct += 1
        else:
            if pred['label'] == true_label:
                correct += 1
    
    test_accuracy = correct / len(test_labels)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save test results
    results_file = save_dir / "training_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Task type: {task_type}\n")
        f.write(f"Number of labels: {num_labels}\n")
        f.write(f"Multi-label: {multilabel}\n\n")
        f.write("Training Configuration:\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Max length: {max_length}\n\n")
        f.write("Results:\n")
        f.write(f"  Final train loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final train accuracy: {history['train_accuracy'][-1]:.4f}\n")
        
        # Write validation metrics if available
        if history.get('val_metrics') and len(history['val_metrics']) > 0:
            val_losses = [m['loss'] for m in history['val_metrics']]
            val_macro_f1s = [m.get('macro_f1', 0) for m in history['val_metrics']]
            f.write(f"  Best val loss: {min(val_losses):.4f}\n")
            f.write(f"  Best val macro F1: {max(val_macro_f1s):.4f}\n")
            
            # Write calibration metrics
            best_metrics = max(history['val_metrics'], key=lambda x: x.get('macro_f1', 0))
            if 'ece' in best_metrics:
                f.write(f"  Best model ECE: {best_metrics['ece']:.4f}\n")
            if 'brier' in best_metrics:
                f.write(f"  Best model Brier: {best_metrics['brier']:.4f}\n")
            if 'kappa' in best_metrics:
                f.write(f"  Best model Kappa: {best_metrics['kappa']:.4f}\n")
        
        f.write(f"  Test accuracy: {test_accuracy:.4f}\n")
        
        # Write temperature scaling info
        if optimal_temp is not None:
            f.write(f"\nTemperature Scaling:\n")
            f.write(f"  Optimal temperature: {optimal_temp:.4f}\n")
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"Training completed successfully!")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train baseline models on datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["imdb", "jigsaw", "crisisbench_humanitarian", "crisisbench_informativeness", "fever"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Model name or path (default: microsoft/deberta-v3-base)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (default: 3)"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save trained model (default: results/trained_models/{dataset})"
    )
    parser.add_argument(
        "--fit_temperature",
        action="store_true",
        help="Fit temperature scaling on validation set after training"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train on all datasets"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Train on all datasets
        datasets = ["imdb", "jigsaw", "crisisbench_humanitarian", "crisisbench_informativeness", "fever"]
        for dataset_name in datasets:
            try:
                train_model_for_dataset(
                    dataset_name=dataset_name,
                    model_name=args.model,
                    batch_size=args.batch_size,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    max_length=args.max_length,
                    early_stopping_patience=args.early_stopping_patience,
                    eval_steps=args.eval_steps,
                    seed=args.seed,
                    save_dir=args.save_dir,
                    fit_temperature=args.fit_temperature,
                )
            except Exception as e:
                logger.error(f"Error training on {dataset_name}: {e}")
                continue
    else:
        # Train on single dataset
        train_model_for_dataset(
            dataset_name=args.dataset,
            model_name=args.model,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            early_stopping_patience=args.early_stopping_patience,
            eval_steps=args.eval_steps,
            seed=args.seed,
            save_dir=args.save_dir,
            fit_temperature=args.fit_temperature,
        )


if __name__ == "__main__":
    main()

