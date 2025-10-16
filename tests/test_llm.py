"""
Test script for LLM annotation module.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_llm_imports():
    """Test that LLM module components can be imported."""
    print("Testing LLM module imports...")
    
    try:
        from llm.prompts import TaskType, PromptTemplate, get_prompt_template
        print("[OK] llm.prompts imports")
        
        from llm.annotator import QwenAnnotator, AnnotationResult, create_annotator
        print("[OK] llm.annotator imports")
        
        return True
    except ImportError as e:
        print(f"[FAIL] LLM import failed: {e}")
        return False


def test_prompt_templates():
    """Test prompt template functionality."""
    print("\nTesting prompt templates...")
    
    try:
        from llm.prompts import TaskType, get_prompt_template, get_json_schema
        
        # Test all task types
        for task_type in TaskType:
            print(f"  Testing {task_type.value}...")
            
            # Test template creation
            template = get_prompt_template(task_type)
            assert isinstance(template, object)  # PromptTemplate instance
            print(f"    [OK] Template created for {task_type.value}")
            
            # Test JSON schema
            schema = get_json_schema(task_type)
            assert isinstance(schema, dict)
            assert "properties" in schema
            print(f"    [OK] Schema created for {task_type.value}")
            
            # Test prompt formatting (placeholder)
            test_text = "This is a test text."
            formatted = template.format_prompt(test_text)
            assert isinstance(formatted, str)
            print(f"    [OK] Prompt formatting for {task_type.value}")
        
        print("[SUCCESS] All prompt template tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Prompt template test failed: {e}")
        return False


def test_annotator_creation():
    """Test annotator creation and basic functionality."""
    print("\nTesting annotator creation...")
    
    try:
        from llm.annotator import QwenAnnotator, create_annotator
        from llm.prompts import TaskType
        
        # Test direct creation
        annotator = QwenAnnotator("qwen-7b", TaskType.SENTIMENT)
        assert annotator.model_name == "qwen-7b"
        assert annotator.task_type == TaskType.SENTIMENT
        print("[OK] Direct annotator creation")
        
        # Test factory function
        annotator2 = create_annotator("sentiment", "qwen-7b")
        assert annotator2.task_type == TaskType.SENTIMENT
        print("[OK] Factory function creation")
        
        # Test all task types
        for task_type in TaskType:
            annotator3 = create_annotator(task_type.value)
            assert annotator3.task_type == task_type
            print(f"    [OK] Created annotator for {task_type.value}")
        
        print("[SUCCESS] All annotator creation tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Annotator creation test failed: {e}")
        return False


def test_annotation_result():
    """Test AnnotationResult class."""
    print("\nTesting AnnotationResult...")
    
    try:
        from llm.annotator import AnnotationResult
        
        # Test result creation
        result = AnnotationResult(
            text="Test text",
            label="positive",
            confidence="high",
            rationale="Test rationale"
        )
        
        assert result.text == "Test text"
        assert result.label == "positive"
        assert result.confidence == "high"
        assert result.is_valid == True
        print("[OK] AnnotationResult creation")
        
        # Test dictionary conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["text"] == "Test text"
        print("[OK] AnnotationResult to_dict")
        
        # Test invalid result
        invalid_result = AnnotationResult(
            text="Test",
            label="",
            confidence="",
            rationale="",
            is_valid=False,
            error="Test error"
        )
        assert invalid_result.is_valid == False
        print("[OK] Invalid AnnotationResult")
        
        print("[SUCCESS] All AnnotationResult tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] AnnotationResult test failed: {e}")
        return False


def test_placeholder_functionality():
    """Test placeholder implementations."""
    print("\nTesting placeholder functionality...")
    
    try:
        from llm.annotator import QwenAnnotator
        from llm.prompts import TaskType
        
        annotator = QwenAnnotator("qwen-7b", TaskType.SENTIMENT)
        
        # Test single annotation (should return placeholder)
        test_text = "This is a test."
        result = annotator.annotate_single(test_text)
        
        assert result.text == test_text
        assert result.label == "TODO"  # Placeholder value
        assert result.is_valid == False  # Placeholder returns invalid
        print("[OK] Placeholder single annotation")
        
        # Test batch annotation
        texts = ["Text 1", "Text 2", "Text 3"]
        results = annotator.annotate_batch(texts)
        
        assert len(results) == 3
        assert all(not r.is_valid for r in results)  # All should be invalid placeholders
        print("[OK] Placeholder batch annotation")
        
        print("[SUCCESS] All placeholder functionality tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Placeholder functionality test failed: {e}")
        return False


def main():
    """Run all LLM module tests."""
    print("=" * 60)
    print("LLM MODULE TEST")
    print("=" * 60)
    
    tests = [
        ("LLM Imports", test_llm_imports),
        ("Prompt Templates", test_prompt_templates),
        ("Annotator Creation", test_annotator_creation),
        ("AnnotationResult", test_annotation_result),
        ("Placeholder Functionality", test_placeholder_functionality)
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
        print("[SUCCESS] All LLM module tests passed!")
    else:
        print("[WARNING] Some LLM module tests failed.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
