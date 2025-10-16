"""
Test script for supervised baseline models and text feature extraction.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_baseline_imports():
    """Test that baseline model components can be imported."""
    print("Testing baseline model imports...")
    
    try:
        from models.baseline_models import SupervisedClassifier, BaselineModelManager, create_baseline_model
        print("[OK] models.baseline_models imports")
        
        from models.text_features import TextFeatureExtractor, create_feature_extractor
        print("[OK] models.text_features imports")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Baseline model import failed: {e}")
        return False


def test_supervised_classifier():
    """Test SupervisedClassifier functionality."""
    print("\nTesting SupervisedClassifier...")
    
    try:
        from models.baseline_models import SupervisedClassifier, create_baseline_model
        
        # Test direct creation
        classifier = SupervisedClassifier("microsoft/deberta-v3-base", "sentiment")
        assert classifier.model_name == "microsoft/deberta-v3-base"
        assert classifier.task_type == "sentiment"
        print("[OK] Direct classifier creation")
        
        # Test factory function
        classifier2 = create_baseline_model("sentiment")
        assert classifier2.task_type == "sentiment"
        print("[OK] Factory function creation")
        
        # Test single prediction (placeholder)
        test_text = "This movie was amazing!"
        result = classifier.predict_single(test_text)
        
        assert isinstance(result, dict)
        assert "label" in result
        assert "probabilities" in result
        assert "entropy" in result
        assert result["text"] == test_text
        print("[OK] Single prediction (placeholder)")
        
        # Test batch prediction (placeholder)
        texts = ["Text 1", "Text 2", "Text 3"]
        results = classifier.predict_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        print("[OK] Batch prediction (placeholder)")
        
        print("[SUCCESS] All SupervisedClassifier tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] SupervisedClassifier test failed: {e}")
        return False


def test_baseline_model_manager():
    """Test BaselineModelManager functionality."""
    print("\nTesting BaselineModelManager...")
    
    try:
        from models.baseline_models import BaselineModelManager
        
        manager = BaselineModelManager()
        
        # Test model retrieval
        model = manager.get_model("sentiment")
        assert isinstance(model, object)  # SupervisedClassifier instance
        print("[OK] Model retrieval")
        
        # Test multiple task types
        for task_type in ["sentiment", "toxicity", "crisis_classification", "fact_verification"]:
            model = manager.get_model(task_type)
            assert model.task_type == task_type
            print(f"    [OK] Retrieved model for {task_type}")
        
        # Test multi-task prediction (placeholder)
        test_text = "This is a test text."
        results = manager.predict_all_tasks(test_text)
        
        assert isinstance(results, dict)
        assert len(results) == 4  # All task types
        print("[OK] Multi-task prediction (placeholder)")
        
        print("[SUCCESS] All BaselineModelManager tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] BaselineModelManager test failed: {e}")
        return False


def test_text_feature_extractor():
    """Test TextFeatureExtractor functionality."""
    print("\nTesting TextFeatureExtractor...")
    
    try:
        from models.text_features import TextFeatureExtractor, create_feature_extractor
        
        # Test direct creation
        extractor = TextFeatureExtractor()
        assert isinstance(extractor.negation_patterns, list)
        assert isinstance(extractor.hedge_patterns, list)
        print("[OK] Direct extractor creation")
        
        # Test factory function
        extractor2 = create_feature_extractor()
        assert isinstance(extractor2, TextFeatureExtractor)
        print("[OK] Factory function creation")
        
        # Test basic feature extraction
        test_text = "This movie was not bad, but it could have been better."
        basic_features = extractor.extract_basic_features(test_text)
        
        assert isinstance(basic_features, dict)
        assert "char_count" in basic_features
        assert "word_count" in basic_features
        assert "sentence_count" in basic_features
        assert basic_features["char_count"] > 0
        assert basic_features["word_count"] > 0
        print("[OK] Basic feature extraction")
        
        # Test negation feature extraction
        negation_features = extractor.extract_negation_features(test_text)
        
        assert isinstance(negation_features, dict)
        assert "negation_count" in negation_features
        assert "negation_density" in negation_features
        assert negation_features["negation_count"] > 0  # "not" should be detected
        print("[OK] Negation feature extraction")
        
        # Test hedge feature extraction
        hedge_features = extractor.extract_hedge_features(test_text)
        
        assert isinstance(hedge_features, dict)
        assert "hedge_count" in hedge_features
        assert "hedge_density" in hedge_features
        assert hedge_features["hedge_count"] > 0  # "could" should be detected
        print("[OK] Hedge feature extraction")
        
        # Test sentiment feature extraction
        sentiment_features = extractor.extract_sentiment_features(test_text)
        
        assert isinstance(sentiment_features, dict)
        assert "sentiment_compound" in sentiment_features
        assert "positive_word_count" in sentiment_features
        assert "negative_word_count" in sentiment_features
        print("[OK] Sentiment feature extraction")
        
        # Test readability feature extraction
        readability_features = extractor.extract_readability_features(test_text)
        
        assert isinstance(readability_features, dict)
        assert "flesch_reading_ease" in readability_features
        assert "avg_sentence_length" in readability_features
        print("[OK] Readability feature extraction")
        
        # Test comprehensive feature extraction
        all_features = extractor.extract_all_features(test_text)
        
        assert isinstance(all_features, dict)
        assert len(all_features) > 10  # Should have many features
        print("[OK] Comprehensive feature extraction")
        
        print("[SUCCESS] All TextFeatureExtractor tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] TextFeatureExtractor test failed: {e}")
        return False


def test_feature_quality():
    """Test that extracted features have reasonable values."""
    print("\nTesting feature quality...")
    
    try:
        from models.text_features import create_feature_extractor
        
        extractor = create_feature_extractor()
        
        # Test with different types of text
        test_cases = [
            "This is a simple sentence.",
            "This movie was not bad, but it could have been better!",
            "I absolutely love this amazing, fantastic, wonderful experience!!!",
            "The quick brown fox jumps over the lazy dog.",
            "I think maybe perhaps it might be somewhat good."
        ]
        
        for i, text in enumerate(test_cases):
            features = extractor.extract_all_features(text)
            
            # Check that features are reasonable
            assert features["char_count"] > 0
            assert features["word_count"] > 0
            assert features["sentence_count"] > 0
            assert 0 <= features["negation_density"] <= 1
            assert 0 <= features["hedge_density"] <= 1
            assert 0 <= features["punctuation_ratio"] <= 1
            assert 0 <= features["caps_ratio"] <= 1
            
            print(f"    [OK] Feature quality test {i+1}")
        
        print("[SUCCESS] All feature quality tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Feature quality test failed: {e}")
        return False


def test_entropy_calculation():
    """Test entropy calculation functionality."""
    print("\nTesting entropy calculation...")
    
    try:
        from models.baseline_models import SupervisedClassifier
        import numpy as np
        
        classifier = SupervisedClassifier("test", "test")
        
        # Test entropy calculation with different probability distributions
        test_cases = [
            ([0.5, 0.5], "balanced"),  # High entropy
            ([0.9, 0.1], "unbalanced"),  # Low entropy
            ([0.33, 0.33, 0.34], "three-class"),  # Medium entropy
        ]
        
        for probs, case_name in test_cases:
            entropy = classifier.calculate_entropy(np.array(probs))
            
            assert isinstance(entropy, float)
            assert entropy >= 0
            assert entropy <= np.log(len(probs))  # Maximum entropy for uniform distribution
            
            print(f"    [OK] Entropy calculation for {case_name}: {entropy:.3f}")
        
        print("[SUCCESS] All entropy calculation tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Entropy calculation test failed: {e}")
        return False


def main():
    """Run all baseline model tests."""
    print("=" * 60)
    print("BASELINE MODELS TEST")
    print("=" * 60)
    
    tests = [
        ("Baseline Model Imports", test_baseline_imports),
        ("SupervisedClassifier", test_supervised_classifier),
        ("BaselineModelManager", test_baseline_model_manager),
        ("TextFeatureExtractor", test_text_feature_extractor),
        ("Feature Quality", test_feature_quality),
        ("Entropy Calculation", test_entropy_calculation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 60)
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("[SUCCESS] All baseline model tests passed!")
    else:
        print("[WARNING] Some baseline model tests failed.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
