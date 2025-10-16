"""
Test script for evaluation module components.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_evaluation_imports():
    """Test that evaluation module components can be imported."""
    print("Testing evaluation module imports...")
    
    try:
        from evaluation.trust_scorer import TrustScoreClassifier, create_trust_scorer
        print("[OK] evaluation.trust_scorer imports")
        
        from evaluation.hard_cases import HardCaseIdentifier, create_hard_case_identifier
        print("[OK] evaluation.hard_cases imports")
        
        from evaluation.metrics import EvaluationMetrics, create_evaluation_metrics
        print("[OK] evaluation.metrics imports")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Evaluation module import failed: {e}")
        return False


def test_trust_score_classifier():
    """Test TrustScoreClassifier functionality."""
    print("\nTesting TrustScoreClassifier...")
    
    try:
        from evaluation.trust_scorer import TrustScoreClassifier, create_trust_scorer
        
        # Test direct creation
        classifier = TrustScoreClassifier("logistic_regression", "sentiment")
        assert classifier.model_type == "logistic_regression"
        assert classifier.task_type == "sentiment"
        assert not classifier.trained
        print("[OK] Direct classifier creation")
        
        # Test factory function
        classifier2 = create_trust_scorer("random_forest", "toxicity")
        assert classifier2.model_type == "random_forest"
        assert classifier2.task_type == "toxicity"
        print("[OK] Factory function creation")
        
        # Test different model types
        model_types = ["logistic_regression", "random_forest", "gradient_boosting"]
        for model_type in model_types:
            classifier3 = create_trust_scorer(model_type, "sentiment")
            assert classifier3.model_type == model_type
            print(f"    [OK] Created {model_type} classifier")
        
        # Test feature extraction (placeholder)
        test_annotations = pd.DataFrame({
            'text': ['Test text 1', 'Test text 2', 'Test text 3'],
            'llm_label': [1, 0, 1],
            'llm_confidence': ['high', 'medium', 'low'],
            'llm_rationale': ['Test rationale 1', 'Test rationale 2', 'Test rationale 3']
        })
        
        features_df = classifier.extract_features(test_annotations)
        assert len(features_df) == len(test_annotations)
        assert len(classifier.feature_columns) > 0
        print("[OK] Feature extraction (placeholder)")
        
        # Test training data preparation (placeholder)
        gold_labels = pd.Series([1, 0, 1])
        X, y = classifier.prepare_training_data(test_annotations, gold_labels)
        assert X.shape[0] == len(test_annotations)
        assert len(y) == len(test_annotations)
        print("[OK] Training data preparation (placeholder)")
        
        # Test training (placeholder)
        metrics = classifier.train(test_annotations, gold_labels)
        assert isinstance(metrics, dict)
        assert 'train_accuracy' in metrics
        assert classifier.trained
        print("[OK] Training (placeholder)")
        
        # Test prediction (placeholder)
        trust_scores = classifier.predict_trust_scores(test_annotations)
        assert len(trust_scores) == len(test_annotations)
        assert all(0 <= score <= 1 for score in trust_scores)
        print("[OK] Trust score prediction (placeholder)")
        
        # Test trust decisions (placeholder)
        trust_scores, trust_decisions = classifier.predict_trust_decisions(test_annotations)
        assert len(trust_decisions) == len(test_annotations)
        assert all(decision in [0, 1] for decision in trust_decisions)
        print("[OK] Trust decision prediction (placeholder)")
        
        # Test evaluation (placeholder)
        eval_metrics = classifier.evaluate(test_annotations, gold_labels)
        assert isinstance(eval_metrics, dict)
        assert 'accuracy' in eval_metrics
        print("[OK] Evaluation (placeholder)")
        
        print("[SUCCESS] All TrustScoreClassifier tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] TrustScoreClassifier test failed: {e}")
        return False


def test_hard_case_identifier():
    """Test HardCaseIdentifier functionality."""
    print("\nTesting HardCaseIdentifier...")
    
    try:
        from evaluation.hard_cases import HardCaseIdentifier, create_hard_case_identifier
        
        # Test direct creation
        identifier = HardCaseIdentifier("sentiment")
        assert identifier.task_type == "sentiment"
        print("[OK] Direct identifier creation")
        
        # Test factory function
        identifier2 = create_hard_case_identifier("toxicity")
        assert identifier2.task_type == "toxicity"
        print("[OK] Factory function creation")
        
        # Test different task types
        task_types = ["sentiment", "toxicity", "crisis_classification", "fact_verification"]
        for task_type in task_types:
            identifier3 = create_hard_case_identifier(task_type)
            assert identifier3.task_type == task_type
            print(f"    [OK] Created identifier for {task_type}")
        
        # Test hard case identification by agreement (placeholder)
        test_annotations = pd.DataFrame({
            'text': ['Test text 1', 'Test text 2', 'Test text 3', 'Test text 4'],
            'llm_label': [1, 0, 1, 0]
        })
        gold_labels = pd.Series([1, 1, 0, 0])  # Some disagreement
        
        result_df = identifier.identify_hard_cases_by_agreement(test_annotations, gold_labels)
        assert len(result_df) == len(test_annotations)
        assert 'agreement_score' in result_df.columns
        assert 'is_hard_case_agreement' in result_df.columns
        print("[OK] Hard case identification by agreement (placeholder)")
        
        # Test hard case identification by entropy (placeholder)
        result_df2 = identifier.identify_hard_cases_by_entropy(test_annotations)
        assert len(result_df2) == len(test_annotations)
        assert 'entropy_score' in result_df2.columns
        assert 'is_hard_case_entropy' in result_df2.columns
        print("[OK] Hard case identification by entropy (placeholder)")
        
        # Test hard case identification by disagreement (placeholder)
        test_annotations_with_supervised = test_annotations.copy()
        test_annotations_with_supervised['supervised_label'] = [0, 1, 1, 0]
        
        result_df3 = identifier.identify_hard_cases_by_disagreement(test_annotations_with_supervised)
        assert len(result_df3) == len(test_annotations)
        assert 'model_disagreement' in result_df3.columns
        assert 'is_hard_case_disagreement' in result_df3.columns
        print("[OK] Hard case identification by disagreement (placeholder)")
        
        # Test comprehensive hard case identification (placeholder)
        result_df4 = identifier.identify_hard_cases_comprehensive(test_annotations, gold_labels)
        assert len(result_df4) == len(test_annotations)
        assert 'difficulty_score' in result_df4.columns
        assert 'is_hard_case_overall' in result_df4.columns
        print("[OK] Comprehensive hard case identification (placeholder)")
        
        # Test pattern analysis (placeholder)
        patterns = identifier.analyze_hard_case_patterns(result_df4)
        assert isinstance(patterns, dict)
        assert 'total_cases' in patterns
        assert 'hard_cases' in patterns
        print("[OK] Pattern analysis (placeholder)")
        
        # Test report generation (placeholder)
        report = identifier.generate_hard_case_report(test_annotations, gold_labels)
        assert isinstance(report, str)
        assert len(report) > 0
        print("[OK] Report generation (placeholder)")
        
        print("[SUCCESS] All HardCaseIdentifier tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] HardCaseIdentifier test failed: {e}")
        return False


def test_evaluation_metrics():
    """Test EvaluationMetrics functionality."""
    print("\nTesting EvaluationMetrics...")
    
    try:
        from evaluation.metrics import EvaluationMetrics, create_evaluation_metrics
        
        # Test direct creation
        metrics_calc = EvaluationMetrics("sentiment")
        assert metrics_calc.task_type == "sentiment"
        print("[OK] Direct metrics calculator creation")
        
        # Test factory function
        metrics_calc2 = create_evaluation_metrics("toxicity")
        assert metrics_calc2.task_type == "toxicity"
        print("[OK] Factory function creation")
        
        # Test different task types
        task_types = ["sentiment", "toxicity", "crisis_classification", "fact_verification"]
        for task_type in task_types:
            metrics_calc3 = create_evaluation_metrics(task_type)
            assert metrics_calc3.task_type == task_type
            print(f"    [OK] Created metrics calculator for {task_type}")
        
        # Create test data
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # Some errors
        y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.4, 0.2, 0.9, 0.1])
        y_true_trust = np.array([1, 1, 1, 0, 0, 1, 1, 1])
        y_pred_trust = np.array([0.8, 0.9, 0.7, 0.3, 0.2, 0.8, 0.9, 0.7])
        
        # Test agreement metrics (placeholder)
        agreement_metrics = metrics_calc.calculate_agreement_metrics(y_true, y_pred)
        assert isinstance(agreement_metrics, dict)
        assert 'accuracy' in agreement_metrics
        assert 'cohen_kappa' in agreement_metrics
        assert 'krippendorff_alpha' in agreement_metrics
        print("[OK] Agreement metrics calculation (placeholder)")
        
        # Test calibration metrics (placeholder)
        calibration_metrics = metrics_calc.calculate_calibration_metrics(y_true, y_prob)
        assert isinstance(calibration_metrics, dict)
        assert 'ece' in calibration_metrics
        assert 'mce' in calibration_metrics
        assert 'brier_score' in calibration_metrics
        print("[OK] Calibration metrics calculation (placeholder)")
        
        # Test fairness metrics (placeholder)
        subgroups = {
            'group1': np.array([True, False, True, False, True, False, True, False]),
            'group2': np.array([False, True, False, True, False, True, False, True])
        }
        fairness_metrics = metrics_calc.calculate_fairness_metrics(y_true, y_pred, subgroups)
        assert isinstance(fairness_metrics, dict)
        assert 'group1' in fairness_metrics
        assert 'group2' in fairness_metrics
        print("[OK] Fairness metrics calculation (placeholder)")
        
        # Test trust prediction metrics (placeholder)
        trust_metrics = metrics_calc.calculate_trust_prediction_metrics(y_true_trust, y_pred_trust)
        assert isinstance(trust_metrics, dict)
        assert 'roc_auc' in trust_metrics
        assert 'pr_auc' in trust_metrics
        print("[OK] Trust prediction metrics calculation (placeholder)")
        
        # Test comprehensive metrics (placeholder)
        comprehensive_metrics = metrics_calc.calculate_comprehensive_metrics(
            y_true, y_pred, y_prob, subgroups, y_true_trust, y_pred_trust
        )
        assert isinstance(comprehensive_metrics, dict)
        assert 'classification' in comprehensive_metrics
        assert 'agreement' in comprehensive_metrics
        assert 'calibration' in comprehensive_metrics
        assert 'fairness' in comprehensive_metrics
        assert 'trust_prediction' in comprehensive_metrics
        print("[OK] Comprehensive metrics calculation (placeholder)")
        
        # Test report generation (placeholder)
        report = metrics_calc.generate_evaluation_report(comprehensive_metrics)
        assert isinstance(report, str)
        assert len(report) > 0
        print("[OK] Report generation (placeholder)")
        
        # Test ECE calculation
        ece = metrics_calc._calculate_ece(y_true, y_prob)
        assert isinstance(ece, float)
        assert 0 <= ece <= 1
        print("[OK] ECE calculation")
        
        print("[SUCCESS] All EvaluationMetrics tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] EvaluationMetrics test failed: {e}")
        return False


def test_metrics_quality():
    """Test that calculated metrics have reasonable values."""
    print("\nTesting metrics quality...")
    
    try:
        from evaluation.metrics import create_evaluation_metrics
        
        metrics_calc = create_evaluation_metrics("sentiment")
        
        # Test with perfect predictions
        y_true_perfect = np.array([1, 0, 1, 0, 1])
        y_pred_perfect = np.array([1, 0, 1, 0, 1])
        
        perfect_metrics = metrics_calc.calculate_agreement_metrics(y_true_perfect, y_pred_perfect)
        assert perfect_metrics['accuracy'] == 1.0
        print("[OK] Perfect prediction metrics")
        
        # Test with random predictions
        y_true_random = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred_random = np.random.randint(0, 2, len(y_true_random))
        
        random_metrics = metrics_calc.calculate_agreement_metrics(y_true_random, y_pred_random)
        assert 0 <= random_metrics['accuracy'] <= 1
        print("[OK] Random prediction metrics")
        
        # Test with calibrated probabilities
        y_true_cal = np.array([1, 0, 1, 0])
        y_prob_cal = np.array([0.9, 0.1, 0.8, 0.2])  # Well-calibrated
        
        cal_metrics = metrics_calc.calculate_calibration_metrics(y_true_cal, y_prob_cal)
        assert 0 <= cal_metrics['ece'] <= 1
        assert 0 <= cal_metrics['brier_score'] <= 1
        print("[OK] Calibrated probability metrics")
        
        print("[SUCCESS] All metrics quality tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Metrics quality test failed: {e}")
        return False


def main():
    """Run all evaluation module tests."""
    print("=" * 60)
    print("EVALUATION MODULE TEST")
    print("=" * 60)
    
    tests = [
        ("Evaluation Module Imports", test_evaluation_imports),
        ("TrustScoreClassifier", test_trust_score_classifier),
        ("HardCaseIdentifier", test_hard_case_identifier),
        ("EvaluationMetrics", test_evaluation_metrics),
        ("Metrics Quality", test_metrics_quality)
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
        print("[SUCCESS] All evaluation module tests passed!")
    else:
        print("[WARNING] Some evaluation module tests failed.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
