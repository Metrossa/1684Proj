"""
Test script to verify directory structure and module imports.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_module_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    modules_to_test = [
        'llm',
        'models',
        'evaluation',
        'dashboard',
        'dashboard.components',
        'pipeline',
        'utils',
        'data.dataset_loader',
        'config'
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
        except ImportError as e:
            print(f"[FAIL] {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n[ERROR] Failed to import {len(failed_imports)} module(s)")
        return False
    else:
        print(f"\n[SUCCESS] All {len(modules_to_test)} modules imported successfully!")
        return True


def test_directory_structure():
    """Test that expected directories exist."""
    import os
    from pathlib import Path
    
    print("\nTesting directory structure...")
    
    project_root = Path(__file__).parent.parent
    
    expected_dirs = [
        'llm',
        'models',
        'evaluation',
        'dashboard',
        'dashboard/components',
        'pipeline',
        'utils',
        'tests',
        'scripts',
        'data',
        'results'
    ]
    
    missing_dirs = []
    
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"[OK] {dir_path}/")
        else:
            print(f"[MISSING] {dir_path}/ (missing)")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n[ERROR] Missing {len(missing_dirs)} director(y/ies)")
        return False
    else:
        print(f"\n[SUCCESS] All {len(expected_dirs)} directories exist!")
        return True


def main():
    """Run all structure tests."""
    print("=" * 60)
    print("DIRECTORY STRUCTURE TEST")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_module_imports)
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
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

