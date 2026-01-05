#!/usr/bin/env python3
"""
Documentation Validation Script

This script validates that all documentation examples and tutorials work correctly.
"""

import os
import sys
import subprocess
import traceback
from typing import List, Dict, Any

def run_script(script_path: str) -> Dict[str, Any]:
    """Run a Python script and return results."""
    result = {
        'script': script_path,
        'success': False,
        'output': '',
        'error': '',
        'return_code': None
    }
    
    try:
        # Run the script
        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        result['return_code'] = process.returncode
        result['output'] = process.stdout
        result['error'] = process.stderr
        result['success'] = process.returncode == 0
        
    except subprocess.TimeoutExpired:
        result['error'] = 'Script timed out after 120 seconds'
    except Exception as e:
        result['error'] = f'Failed to run script: {e}'
    
    return result

def validate_examples() -> List[Dict[str, Any]]:
    """Validate all example scripts."""
    example_scripts = [
        'examples/gsa_basic_example.py',
        'examples/integration_comparison.py',
        'examples/climate_sensitivity.py',
        'docs/examples/financial_risk_analysis.py'
    ]
    
    results = []
    for script in example_scripts:
        if os.path.exists(script):
            print(f"Testing {script}...")
            result = run_script(script)
            results.append(result)
            
            if result['success']:
                print(f"✓ {script}: PASSED")
            else:
                print(f"✗ {script}: FAILED")
                if result['error']:
                    print(f"  Error: {result['error'][:200]}...")
        else:
            print(f"✗ {script}: FILE NOT FOUND")
            results.append({
                'script': script,
                'success': False,
                'error': 'File not found',
                'return_code': -1
            })
    
    return results

def validate_documentation_files() -> List[Dict[str, Any]]:
    """Validate documentation files exist and are readable."""
    doc_files = [
        'README.md',
        'CHANGELOG.md',
        'LICENSE',
        'docs/tutorials/gsa_comprehensive.md'
    ]
    
    results = []
    for doc_file in doc_files:
        result = {
            'file': doc_file,
            'exists': False,
            'readable': False,
            'size': 0
        }
        
        if os.path.exists(doc_file):
            result['exists'] = True
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    result['readable'] = True
                    result['size'] = len(content)
            except Exception as e:
                result['error'] = str(e)
        
        results.append(result)
        
        if result['exists'] and result['readable']:
            print(f"✓ {doc_file}: EXISTS ({result['size']} chars)")
        else:
            print(f"✗ {doc_file}: MISSING OR UNREADABLE")
    
    return results

def main():
    """Run all documentation validation tests."""
    print("MCPost Documentation Validation")
    print("=" * 50)
    
    # Test example scripts
    print("\n1. Testing Example Scripts")
    print("-" * 30)
    example_results = validate_examples()
    
    # Test documentation files
    print("\n2. Validating Documentation Files")
    print("-" * 40)
    doc_results = validate_documentation_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    # Example scripts summary
    example_passed = sum(1 for r in example_results if r['success'])
    example_total = len(example_results)
    print(f"Example Scripts: {example_passed}/{example_total} passed")
    
    # Documentation files summary
    doc_passed = sum(1 for r in doc_results if r['exists'] and r['readable'])
    doc_total = len(doc_results)
    print(f"Documentation Files: {doc_passed}/{doc_total} found and readable")
    
    # Overall result
    all_passed = (example_passed == example_total) and (doc_passed == doc_total)
    
    if all_passed:
        print("\n✓ All documentation validation tests PASSED")
        return 0
    else:
        print("\n✗ Some documentation validation tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())