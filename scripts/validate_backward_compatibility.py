#!/usr/bin/env python3
"""
Backward Compatibility Validation Script

This script validates that the refactored MCPost package maintains backward 
compatibility with the original functionality.
"""

import numpy as np
import pandas as pd
import sys
import os
import traceback
from typing import Dict, Any, Tuple

# Add current directory to path to import original scripts
sys.path.insert(0, os.getcwd())

def test_gsa_compatibility() -> Dict[str, Any]:
    """Test GSA backward compatibility."""
    results = {
        'test_name': 'GSA Backward Compatibility',
        'passed': False,
        'errors': [],
        'details': {}
    }
    
    try:
        # Import both original and refactored versions
        from gsa_pipeline import gsa_for_target as original_gsa
        from mcpost import gsa_for_target as refactored_gsa
        
        # Create test data
        np.random.seed(42)
        n_samples, n_params = 50, 3
        X = np.random.randn(n_samples, n_params)
        Y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * X[:, 2] + 0.1 * np.random.randn(n_samples)
        param_names = ['p1', 'p2', 'p3']
        
        # Test original version
        try:
            original_result = original_gsa(X, Y, param_names=param_names)
            results['details']['original_success'] = True
            results['details']['original_type'] = str(type(original_result))
        except Exception as e:
            results['errors'].append(f"Original GSA failed: {e}")
            results['details']['original_success'] = False
        
        # Test refactored version
        try:
            refactored_result = refactored_gsa(X, Y, param_names=param_names)
            results['details']['refactored_success'] = True
            results['details']['refactored_type'] = str(type(refactored_result))
        except Exception as e:
            results['errors'].append(f"Refactored GSA failed: {e}")
            results['details']['refactored_success'] = False
        
        # Check if both succeeded
        if results['details'].get('original_success') and results['details'].get('refactored_success'):
            results['passed'] = True
            results['details']['compatibility_status'] = 'Both versions work'
        
    except Exception as e:
        results['errors'].append(f"GSA compatibility test failed: {e}")
        results['details']['exception'] = str(e)
    
    return results

def test_integration_compatibility() -> Dict[str, Any]:
    """Test integration backward compatibility."""
    results = {
        'test_name': 'Integration Backward Compatibility',
        'passed': False,
        'errors': [],
        'details': {}
    }
    
    try:
        # Import both original and refactored versions
        from mc_int import monte_carlo_integral as original_mc
        from mcpost import monte_carlo_integral as refactored_mc
        
        # Create test data
        np.random.seed(42)
        n_samples, n_params = 50, 2
        params = np.random.randn(n_samples, n_params)
        data = np.random.randn(n_samples, 1)
        
        def p_target(x):
            return np.exp(-0.5 * np.sum(x**2, axis=1))
        
        # Test original version
        try:
            original_result = original_mc(params, data, p_target)
            results['details']['original_success'] = True
            results['details']['original_type'] = str(type(original_result))
        except Exception as e:
            results['errors'].append(f"Original integration failed: {e}")
            results['details']['original_success'] = False
        
        # Test refactored version
        try:
            refactored_result = refactored_mc(params, data, p_target)
            results['details']['refactored_success'] = True
            results['details']['refactored_type'] = str(type(refactored_result))
        except Exception as e:
            results['errors'].append(f"Refactored integration failed: {e}")
            results['details']['refactored_success'] = False
        
        # Check if both succeeded
        if results['details'].get('original_success') and results['details'].get('refactored_success'):
            results['passed'] = True
            results['details']['compatibility_status'] = 'Both versions work'
        
    except Exception as e:
        results['errors'].append(f"Integration compatibility test failed: {e}")
        results['details']['exception'] = str(e)
    
    return results

def test_api_availability() -> Dict[str, Any]:
    """Test that all expected API functions are available."""
    results = {
        'test_name': 'API Availability',
        'passed': False,
        'errors': [],
        'details': {}
    }
    
    try:
        import mcpost
        
        # Expected core functions
        expected_functions = [
            'gsa_for_target',
            'gsa_pipeline', 
            'monte_carlo_integral',
            'qmc_integral',
            'qmc_integral_auto'
        ]
        
        available_functions = []
        missing_functions = []
        
        for func_name in expected_functions:
            if hasattr(mcpost, func_name):
                available_functions.append(func_name)
            else:
                missing_functions.append(func_name)
        
        results['details']['available_functions'] = available_functions
        results['details']['missing_functions'] = missing_functions
        results['details']['total_expected'] = len(expected_functions)
        results['details']['total_available'] = len(available_functions)
        
        if len(missing_functions) == 0:
            results['passed'] = True
        else:
            results['errors'].append(f"Missing functions: {missing_functions}")
    
    except Exception as e:
        results['errors'].append(f"API availability test failed: {e}")
        results['details']['exception'] = str(e)
    
    return results

def main():
    """Run all backward compatibility tests."""
    print("MCPost Backward Compatibility Validation")
    print("=" * 50)
    
    tests = [
        test_api_availability,
        test_gsa_compatibility,
        test_integration_compatibility
    ]
    
    all_results = []
    total_passed = 0
    
    for test_func in tests:
        print(f"\nRunning {test_func.__name__}...")
        result = test_func()
        all_results.append(result)
        
        if result['passed']:
            print(f"✓ {result['test_name']}: PASSED")
            total_passed += 1
        else:
            print(f"✗ {result['test_name']}: FAILED")
            for error in result['errors']:
                print(f"  Error: {error}")
    
    print("\n" + "=" * 50)
    print(f"Summary: {total_passed}/{len(tests)} tests passed")
    
    if total_passed == len(tests):
        print("✓ All backward compatibility tests PASSED")
        return 0
    else:
        print("✗ Some backward compatibility tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())