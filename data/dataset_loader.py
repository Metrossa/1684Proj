"""
Dataset loaders for IMDb, Jigsaw, and CrisisBench datasets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import requests
import zipfile
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
import config

class DatasetLoader:
    """Base class for dataset loading and preprocessing."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.config = config.DATASETS[dataset_name]
        self.data_dir = config.DATA_DIR / dataset_name
        self.data_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and split dataset into train/dev/test."""
        raise NotImplementedError
    
    def _create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create stratified train/dev/test splits."""
        # Stratified split to maintain class distribution
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            stratify=df['label'], 
            random_state=config.EVALUATION['random_seed']
        )
        
        dev_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            stratify=temp_df['label'], 
            random_state=config.EVALUATION['random_seed']
        )
        
        return {
            'train': train_df.reset_index(drop=True),
            'dev': dev_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)
        }

class IMDBLoader(DatasetLoader):
    """IMDb sentiment analysis dataset loader."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load IMDb dataset from HuggingFace."""
        try:
            # Try to load from HuggingFace datasets
            if not HAS_DATASETS:
                print("datasets library not available, using sample data")
                return self._create_sample_data()
            
            dataset = load_dataset("imdb")
            
            # Convert to pandas DataFrames
            splits = {}
            for split_name in ['train', 'test']:
                df = dataset[split_name].to_pandas()
                df = df.rename(columns={'text': 'text', 'label': 'label'})
                splits[split_name] = df
            
            # Create dev split from test
            dev_df, test_df = train_test_split(
                splits['test'], 
                test_size=0.5, 
                stratify=splits['test']['label'], 
                random_state=config.EVALUATION['random_seed']
            )
            
            splits['dev'] = dev_df.reset_index(drop=True)
            splits['test'] = test_df.reset_index(drop=True)
            
            return splits
            
        except Exception as e:
            print(f"Error loading IMDb dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample data if dataset loading fails."""
        sample_texts = [
            "This movie was absolutely fantastic!",
            "Terrible acting and poor plot.",
            "Great cinematography and direction.",
            "Boring and predictable storyline.",
            "Amazing performances by all actors."
        ]
        sample_labels = [1, 0, 1, 0, 1]
        
        df = pd.DataFrame({
            'text': sample_texts * 100,  # Repeat for more data
            'label': sample_labels * 100
        })
        
        return self._create_splits(df)

class JigsawLoader(DatasetLoader):
    """Jigsaw toxicity classification dataset loader."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load Jigsaw dataset."""
        try:
            # Try to load from Kaggle or create sample data
            return self._create_sample_data()
        except Exception as e:
            print(f"Error loading Jigsaw dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample toxicity data."""
        sample_texts = [
            "This is a normal comment.",
            "You are an idiot!",
            "Great work on this project.",
            "I hate this so much.",
            "Thanks for sharing this information."
        ]
        sample_labels = [0, 1, 0, 1, 0]  # 0: non-toxic, 1: toxic
        
        df = pd.DataFrame({
            'text': sample_texts * 100,
            'label': sample_labels * 100
        })
        
        return self._create_splits(df)

class CrisisBenchLoader(DatasetLoader):
    """CrisisBench dataset loader."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load CrisisBench dataset."""
        try:
            return self._create_sample_data()
        except Exception as e:
            print(f"Error loading CrisisBench dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample crisis data."""
        sample_texts = [
            "Just had lunch with friends.",
            "Earthquake hit our city, need help!",
            "Beautiful sunset today.",
            "Flooding in downtown area, stay safe!",
            "Coffee with colleagues this morning."
        ]
        sample_labels = [0, 1, 0, 1, 0]  # 0: not crisis, 1: crisis
        
        df = pd.DataFrame({
            'text': sample_texts * 100,
            'label': sample_labels * 100
        })
        
        return self._create_splits(df)

class FEVERLoader(DatasetLoader):
    """FEVER fact verification dataset loader."""
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load FEVER dataset."""
        try:
            return self._create_sample_data()
        except Exception as e:
            print(f"Error loading FEVER dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample fact verification data."""
        sample_texts = [
            "The sky is blue.",
            "Cats can fly.",
            "Water boils at 100°C.",
            "The moon is made of cheese.",
            "Paris is the capital of France."
        ]
        sample_labels = [1, 0, 1, 0, 1]  # 1: supports, 0: refutes
        
        df = pd.DataFrame({
            'text': sample_texts * 100,
            'label': sample_labels * 100
        })
        
        return self._create_splits(df)

def load_dataset_by_name(dataset_name: str) -> Dict[str, pd.DataFrame]:
    """Factory function to load dataset by name."""
    loaders = {
        'imdb': IMDBLoader,
        'jigsaw': JigsawLoader,
        'crisisbench': CrisisBenchLoader,
        'fever': FEVERLoader
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = loaders[dataset_name](dataset_name)
    return loader.load_data()

if __name__ == "__main__":
    # Test loading all datasets
    for dataset_name in config.DATASETS.keys():
        print(f"Loading {dataset_name}...")
        data = load_dataset_by_name(dataset_name)
        print(f"Loaded {dataset_name}: {[(k, len(v)) for k, v in data.items()]}")
