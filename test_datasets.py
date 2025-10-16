"""
Simple test script to verify dataset loading functionality.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_dataset_loading():
    """Test dataset loading functionality."""
    print("Testing dataset loading...")
    
    try:
        from data.dataset_loader import load_dataset_by_name
        import config
        
        for dataset_name in config.DATASETS.keys():
            print(f"Loading {dataset_name}...")
            data = load_dataset_by_name(dataset_name)
            print(f"+ {dataset_name} loaded: {[(k, len(v)) for k, v in data.items()]}")
            
            # Show sample data
            print(f"  Sample text: {data['train']['text'].iloc[0]}")
            print(f"  Sample label: {data['train']['label'].iloc[0]}")
            print()
        
        return True
    except Exception as e:
        print(f"- Dataset loading failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        import config
        
        print(f"+ Datasets configured: {list(config.DATASETS.keys())}")
        print(f"+ BERT models configured: {list(config.BERT_MODELS.keys())}")
        print(f"+ LLM models configured: {list(config.LLM_MODELS.keys())}")
        print(f"+ Default LLM: {config.LLM_CONFIG['model_name']}")
        
        return True
    except Exception as e:
        print(f"- Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Dataset Loading Test")
    print("=" * 30)
    
    tests = [test_config, test_dataset_loading]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed!")
    else:
        print("WARNING: Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    main()
