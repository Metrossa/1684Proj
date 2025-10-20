"""
Supervised baseline models for comparison with LLM annotations.

This module provides pre-trained transformer classifiers (DeBERTa, BERT, etc.)
for generating baseline predictions, probabilities, and entropy scores.

Features
- Single-label (binary/multiclass) and multi-label support
- Batched inference with torch.no_grad and smart device placement
- Entropy & confidence computation
- Optional per-class thresholds for multi-label
- Optional temperature scaling for calibration
- DataFrame helper for bulk prediction
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optional transformers import
try:
    from transformers import (
        AutoModelForSequenceClassification, 
        AutoTokenizer, 
        AutoConfig,
        get_linear_schedule_with_warmup,
    )
    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic_cudnn: bool = True, set_hash: bool = True) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        deterministic_cudnn: Whether to make cuDNN deterministic (may reduce performance)
        set_hash: Whether to set PYTHONHASHSEED environment variable
    """
    import os
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if set_hash:
        os.environ["PYTHONHASHSEED"] = str(seed)


def _binary_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Bernoulli entropy for multi-label probs (per class).
    H(p) = - p log p - (1-p) log(1-p)
    Returns tensor of shape (..., C)
    """
    p = torch.clamp(p, eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def _categorical_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Categorical entropy for single-label probs (sum over classes)."""
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)


def _compute_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
    multilabel: bool = False
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities, shape [N, C]
        y_true: True labels (one-hot or multi-hot), shape [N, C]
        n_bins: Number of bins for calibration
        multilabel: Whether this is multi-label classification
        
    Returns:
        Dictionary with ECE metrics
    """
    if multilabel:
        # Per-class ECE for multi-label
        n_classes = probs.shape[1]
        ece_per_class = []
        
        for c in range(n_classes):
            p_c = probs[:, c]
            y_c = y_true[:, c]
            
            # Bin predictions
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece_c = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (p_c > bin_lower) & (p_c <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_c[in_bin].mean()
                    avg_confidence_in_bin = p_c[in_bin].mean()
                    ece_c += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            ece_per_class.append(float(ece_c))
        
        return {
            "ece_macro": float(np.mean(ece_per_class)),
            "ece_per_class": ece_per_class
        }
    else:
        # Single-label: use max probability and correctness
        p_max = probs.max(axis=1)
        y_pred = probs.argmax(axis=1)
        y_true_labels = y_true.argmax(axis=1)
        correct = (y_pred == y_true_labels).astype(float)
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (p_max > bin_lower) & (p_max <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = p_max[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {"ece": float(ece)}


def _compute_brier(
    probs: np.ndarray,
    y_true: np.ndarray,
    multilabel: bool = False
) -> Dict[str, Any]:
    """
    Compute Brier Score (mean squared error between probabilities and labels).
    
    Args:
        probs: Predicted probabilities, shape [N, C]
        y_true: True labels (one-hot or multi-hot), shape [N, C]
        multilabel: Whether this is multi-label classification
        
    Returns:
        Dictionary with Brier score metrics
    """
    # Brier score: mean((p - y)^2)
    squared_diff = (probs - y_true) ** 2
    
    if multilabel:
        # Per-class Brier score
        brier_per_class = squared_diff.mean(axis=0).tolist()
        return {
            "brier_macro": float(np.mean(brier_per_class)),
            "brier_per_class": brier_per_class
        }
    else:
        # Overall Brier score
        return {"brier": float(squared_diff.mean())}


def _compute_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multilabel: bool = False
) -> Dict[str, Any]:
    """
    Compute Cohen's Kappa coefficient.
    
    Args:
        y_true: True labels (one-hot or multi-hot), shape [N, C]
        y_pred: Predicted labels (one-hot or multi-hot), shape [N, C]
        multilabel: Whether this is multi-label classification
        
    Returns:
        Dictionary with Kappa metrics
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        logger.warning("sklearn not available, returning kappa=0.0")
        if multilabel:
            n_classes = y_true.shape[1]
            return {
                "kappa_macro": 0.0,
                "kappa_per_class": [0.0] * n_classes
            }
        return {"kappa": 0.0}
    
    if multilabel:
        # Per-class Kappa
        n_classes = y_true.shape[1]
        kappa_per_class = []
        
        for c in range(n_classes):
            try:
                kappa_c = cohen_kappa_score(y_true[:, c], y_pred[:, c])
                kappa_per_class.append(float(kappa_c))
            except Exception:
                kappa_per_class.append(0.0)
        
        return {
            "kappa_macro": float(np.mean(kappa_per_class)),
            "kappa_per_class": kappa_per_class
        }
    else:
        # Overall Kappa
        y_true_labels = y_true.argmax(axis=1)
        y_pred_labels = y_pred.argmax(axis=1)
        try:
            kappa = cohen_kappa_score(y_true_labels, y_pred_labels)
            return {"kappa": float(kappa)}
        except Exception:
            return {"kappa": 0.0}


@dataclass
class PredictOutput:
    label: Union[int, List[int]]
    probabilities: List[float]
    entropy: float
    confidence: float
    logits: Optional[List[float]] = None
    thresholded: Optional[bool] = None  # for multi-label


# -----------------------------------------------------------------------------
# PyTorch Dataset for Training
# -----------------------------------------------------------------------------
class TextDataset(Dataset):
    """Simple PyTorch Dataset wrapper for text classification."""
    
    def __init__(self, texts: List[str], labels: Union[List[int], List[List[int]]], multilabel: bool = False):
        self.texts = texts
        self.labels = labels
        self.multilabel = multilabel
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, Union[int, List[int]]]:
        return self.texts[idx], self.labels[idx]


# -----------------------------------------------------------------------------
# Supervised Classifier
# -----------------------------------------------------------------------------
class SupervisedClassifier:
    """Wrapper for pre-trained transformer classifiers."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        task_type: str = "sentiment",
        num_labels: Optional[int] = None,
        multilabel: bool = False,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        max_length: int = 256,
        temperature: float = 1.0,
        thresholds: Optional[List[float]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.task_type = task_type
        self.multilabel = multilabel
        self.max_length = max_length
        self.temperature = temperature
        self._thresholds = thresholds

        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)

        self.model = None
        self.tokenizer = None
        self.config = None

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        logger.info(
            f"Init SupervisedClassifier(model={model_name}, task={task_type}, multilabel={multilabel}, device={self.device})"
        )

        if not HAS_TRANSFORMERS:
            logger.warning("transformers not available - class will run in placeholder mode")

    # -----------------------------
    # Loading
    # -----------------------------
    def load_model(self) -> None:
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers is required to load a model")

        # Prepare config
        cfg_kwargs = {}
        if self.num_labels is not None:
            cfg_kwargs["num_labels"] = self.num_labels
        if self.id2label is not None:
            cfg_kwargs["id2label"] = {int(k): v for k, v in self.id2label.items()}
        if self.label2id is not None:
            cfg_kwargs["label2id"] = self.label2id
        if self.multilabel:
            cfg_kwargs["problem_type"] = "multi_label_classification"
        else:
            cfg_kwargs["problem_type"] = "single_label_classification"

        self.config = AutoConfig.from_pretrained(self.model_name, **cfg_kwargs)
        if self.num_labels is None:
            self.num_labels = int(self.config.num_labels)

        # Try to load fast tokenizer, fall back to slow if it fails
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        except (ValueError, ImportError, AttributeError) as e:
            logger.warning(f"Fast tokenizer loading failed: {e}. Trying alternative loading method...")
            try:
                # Try loading with explicit trust_remote_code and use_fast=False
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    use_fast=False,
                    trust_remote_code=True
                )
            except Exception as e2:
                logger.warning(f"Alternative loading also failed: {e2}. Using SPM tokenizer directly.")
                # Last resort: try loading the specific tokenizer class
                from transformers import DebertaV2Tokenizer
                self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        self.model.eval().to(self.device)
        logger.info("Model and tokenizer loaded.")

        # Default thresholds for multi-label if not provided: 0.5 each
        if self.multilabel and self._thresholds is None:
            self._thresholds = [0.5] * self.num_labels

    # -----------------------------
    # Model saving and loading
    # -----------------------------
    def save_model(
        self,
        save_path: Union[str, Path],
        best_dev_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save the fine-tuned model and tokenizer to disk with extended metadata."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model to save. Train or load a model first.")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Collect version information
        versions = {
            "torch": torch.__version__,
        }
        if HAS_TRANSFORMERS:
            try:
                from transformers import __version__ as transformers_version
                versions["transformers"] = transformers_version
            except Exception:
                pass
        
        # Save extended metadata
        metadata = {
            "task_type": self.task_type,
            "multilabel": self.multilabel,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "thresholds": self._thresholds,
            "id2label": self.id2label,
            "label2id": self.label2id,
            # Reproducibility
            "seed": getattr(self, '_training_seed', 42),
            "device": str(self.device),
            "versions": versions,
            # Best dev metrics
            "best_dev_metrics": best_dev_metrics if best_dev_metrics else {},
        }
        with open(save_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_trained_model(self, model_path: Union[str, Path], subfolder: Optional[str] = None) -> None:
        """
        Load a previously fine-tuned model from disk or HuggingFace Hub.
        
        Args:
            model_path: Local path (e.g., "outputs/imdb/model") or 
                       HuggingFace repo (e.g., "username/model-name")
            subfolder: Subfolder within the repo (e.g., "best_model_calibrated")
                      Only used when loading from HuggingFace Hub
        """
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers is required to load a model")
        
        # Check if this is a HuggingFace Hub repo or local path
        model_path_str = str(model_path)
        is_hub_repo = "/" in model_path_str and not Path(model_path).exists()
        
        if is_hub_repo:
            # Loading from HuggingFace Hub
            logger.info(f"Loading model from HuggingFace Hub: {model_path_str}")
            
            # Try to download and load metadata from hub
            try:
                from huggingface_hub import hf_hub_download
                metadata_filename = "training_metadata.json"
                if subfolder:
                    metadata_filename = f"{subfolder}/training_metadata.json"
                
                metadata_file = hf_hub_download(
                    repo_id=model_path_str,
                    filename=metadata_filename,
                    repo_type="model"
                )
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                self.task_type = metadata.get("task_type", self.task_type)
                self.multilabel = metadata.get("multilabel", self.multilabel)
                self.num_labels = metadata.get("num_labels", self.num_labels)
                self.max_length = metadata.get("max_length", self.max_length)
                self.temperature = metadata.get("temperature", self.temperature)
                self._thresholds = metadata.get("thresholds", self._thresholds)
                self.id2label = metadata.get("id2label", self.id2label)
                self.label2id = metadata.get("label2id", self.label2id)
            except Exception as e:
                logger.warning(f"Could not load metadata from hub: {e}")
            
            # Load model and tokenizer from hub
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_str, 
                subfolder=subfolder,
                use_fast=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path_str,
                subfolder=subfolder
            )
        else:
            # Loading from local path
            model_path = Path(model_path)
            
            # Load metadata if available
            metadata_path = model_path / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.task_type = metadata.get("task_type", self.task_type)
                self.multilabel = metadata.get("multilabel", self.multilabel)
                self.num_labels = metadata.get("num_labels", self.num_labels)
                self.max_length = metadata.get("max_length", self.max_length)
                self.temperature = metadata.get("temperature", self.temperature)
                self._thresholds = metadata.get("thresholds", self._thresholds)
                self.id2label = metadata.get("id2label", self.id2label)
                self.label2id = metadata.get("label2id", self.label2id)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.model.eval().to(self.device)
        
        if self.num_labels is None:
            self.num_labels = int(self.model.config.num_labels)
        
        logger.info(f"Model loaded from {model_path_str}")

    # -----------------------------
    # Training
    # -----------------------------
    def train(
        self,
        train_texts: List[str],
        train_labels: Union[List[int], List[List[int]]],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[Union[List[int], List[List[int]]]] = None,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        eval_steps: int = 500,
        save_steps: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
        early_stopping_patience: int = 3,
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the model on training data.
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels (int for single-label, List[int] for multi-label)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio for learning rate schedule
            weight_decay: Weight decay for AdamW
            max_grad_norm: Max gradient norm for clipping
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps (None = don't save intermediate)
            save_dir: Directory to save checkpoints
            early_stopping_patience: Stop if validation doesn't improve for N evaluations (based on macro_f1)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with training history (loss, metrics, etc.)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load_model() first")
        
        # Set seed for reproducibility
        set_seed(seed)
        self._training_seed = seed
        
        # Prepare dataset
        train_dataset = TextDataset(train_texts, train_labels, self.multilabel)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Windows compatibility
            generator=generator,
            pin_memory=(self.device.type == "cuda")
        )
        
        # Validation setup
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = TextDataset(val_texts, val_labels, self.multilabel)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(self.device.type == "cuda")
            )
        
        # Optimizer and scheduler (use torch.optim.AdamW)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # AMP setup (only for CUDA)
        use_amp = (self.device.type == "cuda")
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        # Training loop
        self.model.train()
        global_step = 0
        best_val_metric = float('-inf')  # macro_f1
        best_dev_metrics = {}
        patience_counter = 0
        
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_metrics": [],  # Store all validation metrics
            "learning_rate": []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_texts, batch_labels in progress_bar:
                # Tokenize
                encoding = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                
                # Prepare labels
                if self.multilabel:
                    # Multi-label: convert to float tensor of shape (batch, num_labels)
                    labels_tensor = torch.zeros(len(batch_labels), self.num_labels)
                    for i, label_list in enumerate(batch_labels):
                        # Handle different label formats
                        if isinstance(label_list, (int, np.integer)):
                            label_list = [label_list]
                        elif isinstance(label_list, torch.Tensor):
                            if label_list.ndim == 0:  # 0-d tensor (scalar)
                                label_list = [label_list.item()]
                            else:
                                label_list = label_list.tolist()
                        elif not isinstance(label_list, (list, tuple)):
                            label_list = [int(label_list)]
                        
                        for lbl in label_list:
                            lbl_int = int(lbl) if isinstance(lbl, torch.Tensor) else lbl
                            labels_tensor[i, lbl_int] = 1.0
                    labels_tensor = labels_tensor.to(self.device)
                else:
                    # Single-label
                    if isinstance(batch_labels, torch.Tensor):
                        labels_tensor = batch_labels.detach().clone().to(self.device)
                    else:
                        labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
                
                # Forward pass with optional AMP
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**encoding, labels=labels_tensor)
                        loss = outputs.loss
                        logits = outputs.logits
                else:
                    outputs = self.model(**encoding, labels=labels_tensor)
                    loss = outputs.loss
                    logits = outputs.logits
                
                # Backward pass
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                scheduler.step()
                
                # Metrics
                epoch_loss += loss.item()
                
                if self.multilabel:
                    # Multi-label accuracy: exact match ratio
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    epoch_correct += (preds == labels_tensor).all(dim=1).sum().item()
                else:
                    # Single-label accuracy
                    preds = torch.argmax(logits, dim=-1)
                    epoch_correct += (preds == labels_tensor).sum().item()
                
                epoch_total += len(batch_labels)
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0]
                })
                
                # Evaluation
                if val_loader is not None and global_step % eval_steps == 0:
                    val_metrics = self._evaluate(val_loader)
                    history["val_metrics"].append(val_metrics)
                    
                    # Log key metrics
                    log_msg = f"Step {global_step}: Val Loss={val_metrics['loss']:.4f}, Macro F1={val_metrics.get('macro_f1', 0.0):.4f}"
                    if 'ece' in val_metrics:
                        log_msg += f", ECE={val_metrics['ece']:.4f}"
                    elif 'ece_macro' in val_metrics:
                        log_msg += f", ECE={val_metrics['ece_macro']:.4f}"
                    if 'brier' in val_metrics:
                        log_msg += f", Brier={val_metrics['brier']:.4f}"
                    elif 'brier_macro' in val_metrics:
                        log_msg += f", Brier={val_metrics['brier_macro']:.4f}"
                    logger.info(log_msg)
                    
                    # Early stopping based on macro_f1
                    current_metric = val_metrics.get("macro_f1", 0.0)
                    if current_metric > best_val_metric:
                        best_val_metric = current_metric
                        best_dev_metrics = val_metrics.copy()
                        patience_counter = 0
                        # Save best model
                        if save_dir is not None:
                            best_model_path = Path(save_dir) / "best_model"
                            self.save_model(best_model_path, best_dev_metrics=best_dev_metrics)
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered at step {global_step}")
                            break
                    
                    self.model.train()
                
                # Save checkpoint
                if save_steps is not None and save_dir is not None and global_step % save_steps == 0:
                    checkpoint_path = Path(save_dir) / f"checkpoint-{global_step}"
                    self.save_model(checkpoint_path)
            
            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            avg_acc = epoch_correct / epoch_total
            history["train_loss"].append(avg_loss)
            history["train_accuracy"].append(avg_acc)
            history["learning_rate"].append(scheduler.get_last_lr()[0])
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
            )
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                break
        
        # Save final model
        if save_dir is not None:
            final_model_path = Path(save_dir) / "final_model"
            self.save_model(final_model_path)
        
        logger.info("Training completed!")
        return history
    
    def _evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set with comprehensive metrics."""
        self.model.eval()
        total_loss = 0.0
        
        # Collect all predictions and labels
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                # Tokenize
                encoding = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                
                # Prepare labels
                if self.multilabel:
                    labels_tensor = torch.zeros(len(batch_labels), self.num_labels)
                    for i, label_list in enumerate(batch_labels):
                        # Handle different label formats
                        if isinstance(label_list, (int, np.integer)):
                            label_list = [label_list]
                        elif isinstance(label_list, torch.Tensor):
                            if label_list.ndim == 0:  # 0-d tensor (scalar)
                                label_list = [label_list.item()]
                            else:
                                label_list = label_list.tolist()
                        elif not isinstance(label_list, (list, tuple)):
                            label_list = [int(label_list)]
                        
                        for lbl in label_list:
                            lbl_int = int(lbl) if isinstance(lbl, torch.Tensor) else lbl
                            labels_tensor[i, lbl_int] = 1.0
                    labels_tensor = labels_tensor.to(self.device)
                else:
                    if isinstance(batch_labels, torch.Tensor):
                        labels_tensor = batch_labels.detach().clone().to(self.device)
                    else:
                        labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
                
                # Forward pass
                outputs = self.model(**encoding, labels=labels_tensor)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                # Get probabilities
                if self.multilabel:
                    probs = torch.sigmoid(logits)
                else:
                    probs = F.softmax(logits, dim=-1)
                
                # Store for metrics computation
                all_probs.append(probs.cpu().numpy())
                if self.multilabel:
                    all_labels.append(labels_tensor.cpu().numpy())
                else:
                    # Convert to one-hot for consistent metric computation
                    labels_onehot = torch.zeros(len(batch_labels), self.num_labels)
                    labels_onehot.scatter_(1, labels_tensor.cpu().unsqueeze(1), 1.0)
                    all_labels.append(labels_onehot.numpy())
        
        # Concatenate all batches
        all_probs = np.concatenate(all_probs, axis=0)  # [N, C]
        all_labels = np.concatenate(all_labels, axis=0)  # [N, C]
        
        # Get predictions using thresholds (for multilabel) or argmax (for single-label)
        if self.multilabel:
            # Use self._thresholds if available, else 0.5
            thresholds = self._thresholds if self._thresholds else [0.5] * self.num_labels
            thresholds_array = np.array(thresholds).reshape(1, -1)
            all_preds = (all_probs >= thresholds_array).astype(float)
        else:
            # Single-label: convert argmax to one-hot
            pred_labels = all_probs.argmax(axis=1)
            all_preds = np.zeros_like(all_probs)
            all_preds[np.arange(len(pred_labels)), pred_labels] = 1.0
        
        # Compute metrics
        metrics = {"loss": total_loss / len(val_loader)}
        
        # Accuracy (for single-label or exact match for multilabel)
        if self.multilabel:
            exact_match = (all_preds == all_labels).all(axis=1).mean()
            metrics["exact_match"] = float(exact_match)
        else:
            accuracy = (all_preds.argmax(axis=1) == all_labels.argmax(axis=1)).mean()
            metrics["accuracy"] = float(accuracy)
        
        # Macro F1 (works for both single and multilabel)
        try:
            from sklearn.metrics import f1_score
            if self.multilabel:
                macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            else:
                macro_f1 = f1_score(
                    all_labels.argmax(axis=1),
                    all_preds.argmax(axis=1),
                    average='macro',
                    zero_division=0
                )
            metrics["macro_f1"] = float(macro_f1)
        except Exception as e:
            logger.warning(f"Could not compute macro_f1: {e}")
            metrics["macro_f1"] = 0.0
        
        # Calibration metrics: ECE, Brier, Kappa
        try:
            ece_metrics = _compute_ece(all_probs, all_labels, n_bins=15, multilabel=self.multilabel)
            metrics.update(ece_metrics)
        except Exception as e:
            logger.warning(f"Could not compute ECE: {e}")
        
        try:
            brier_metrics = _compute_brier(all_probs, all_labels, multilabel=self.multilabel)
            metrics.update(brier_metrics)
        except Exception as e:
            logger.warning(f"Could not compute Brier: {e}")
        
        try:
            kappa_metrics = _compute_kappa(all_labels, all_preds, multilabel=self.multilabel)
            metrics.update(kappa_metrics)
        except Exception as e:
            logger.warning(f"Could not compute Kappa: {e}")
        
        return metrics

    # -----------------------------
    # Calibration helpers
    # -----------------------------
    def set_temperature(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def set_thresholds(self, thresholds: List[float]) -> None:
        assert self.multilabel, "Thresholds only apply to multi-label classification"
        assert len(thresholds) == self.num_labels, "Thresholds length must equal num_labels"
        self._thresholds = [float(t) for t in thresholds]
    
    def fit_temperature(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        max_iter: int = 50
    ) -> float:
        """
        Fit temperature scaling on validation set to calibrate probabilities.
        
        Args:
            val_logits: Validation logits, shape [N, C]
            val_labels: Validation labels (integers for single-label, multi-hot for multi-label)
            max_iter: Maximum iterations for optimization
            
        Returns:
            Optimal temperature value
        """
        import torch.optim as optim
        import warnings
        
        # Convert to tensors
        logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
        
        if self.multilabel:
            # Multi-label: val_labels should be multi-hot [N, C]
            labels_tensor = torch.tensor(val_labels, dtype=torch.float32)
        else:
            # Single-label: val_labels should be class indices [N]
            if val_labels.ndim == 2:
                # If one-hot, convert to indices
                labels_tensor = torch.tensor(val_labels.argmax(axis=1), dtype=torch.long)
            else:
                labels_tensor = torch.tensor(val_labels, dtype=torch.long)
        
        # Temperature parameter
        temperature = torch.nn.Parameter(torch.ones(1))
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits_tensor / temperature
            
            if self.multilabel:
                loss = F.binary_cross_entropy_with_logits(scaled_logits, labels_tensor)
            else:
                loss = F.cross_entropy(scaled_logits, labels_tensor)
            
            loss.backward()
            return loss
        
        # Suppress LBFGS warning about tensor to scalar conversion
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Converting a tensor.*')
            optimizer.step(eval_loss)
        
        optimal_temp = float(temperature.item())
        self.temperature = optimal_temp
        logger.info(f"Fitted temperature: {optimal_temp:.4f}")
        
        return optimal_temp
    
    # -----------------------------
    # Core inference
    # -----------------------------
    @torch.inference_mode()
    def _forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.model is not None and self.tokenizer is not None, "Call load_model() first"
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        # Temperature scaling
        if self.temperature and self.temperature != 1.0:
            logits = logits / self.temperature
        if self.multilabel:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=-1)
        return logits, probs

    def _postprocess_single(self, logits: torch.Tensor, probs: torch.Tensor) -> PredictOutput:
        if self.multilabel:
            # Multi-label: threshold each class
            thr = torch.tensor(self._thresholds, device=probs.device).view(1, -1)
            pred_vec = (probs >= thr).int().squeeze(0)
            pred_indices = pred_vec.nonzero(as_tuple=False).view(-1).tolist()
            # Entropy: mean Bernoulli entropy over classes
            ent = _binary_entropy(probs).mean(dim=-1).item()
            # Confidence: max class prob among predicted positives; if none, 1 - max prob (uncertain negative)
            if len(pred_indices) > 0:
                conf = probs[0, pred_indices].max().item()
            else:
                conf = 1.0 - probs.max().item()
            return PredictOutput(
                label=pred_indices,
                probabilities=probs.squeeze(0).tolist(),
                entropy=float(ent),
                confidence=float(conf),
                logits=logits.squeeze(0).tolist(),
                thresholded=True,
            )
        else:
            # Single-label
            top = probs.squeeze(0)
            pred = int(torch.argmax(top).item())
            conf = float(top[pred].item())
            ent = float(_categorical_entropy(top.unsqueeze(0)).item())
            return PredictOutput(
                label=pred,
                probabilities=top.tolist(),
                entropy=ent,
                confidence=conf,
                logits=logits.squeeze(0).tolist(),
            )

    # Public APIs --------------------------------------------------------------
    def predict_single(self, text: str) -> Dict[str, Any]:
        logits, probs = self._forward([text])
        out = self._postprocess_single(logits, probs)
        result = {
            "label": out.label,
            "probabilities": out.probabilities,
            "entropy": out.entropy,
            "confidence": out.confidence,
            "logits": out.logits,
        }
        if self.multilabel:
            result["thresholded"] = True
        return result

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            logits, probs = self._forward(chunk)
            for j in range(len(chunk)):
                out = self._postprocess_single(logits[j : j + 1], probs[j : j + 1])
                item = {
                    "label": out.label,
                    "probabilities": out.probabilities,
                    "entropy": out.entropy,
                    "confidence": out.confidence,
                    "logits": out.logits,
                }
                if self.multilabel:
                    item["thresholded"] = True
                results.append(item)
        return results

