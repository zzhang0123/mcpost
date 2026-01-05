"""
Property-based tests for documentation completeness and standards.

**Feature: mcpost-package-improvement, Property 3: Documentation Completeness and Standards**
**Validates: Requirements 3.2, 3.5**
"""

import ast
import inspect
import re
from typing import List, Tuple, Any, Callable
import pytest
from hypothesis import given, strategies as st

import mcpost
import mcpost.gsa
import mcpost.integration
import mcpost.utils


def get_public_functions_and_classes(module) -> List[Tuple[str, Any]]:
    """Get all public functions and classes from a module."""
    public_items = []
    
    # Get items from __all__ if it exists, otherwise use dir()
    if hasattr(module, '__all__'):
        names = module.__all__
    else:
        names = [name for name in dir(module) if not name.startswith('_')]
    
    for name in names:
        try:
            item = getattr(module, name)
            if inspect.isfunction(item) or inspect.isclass(item):
                public_items.append((name, item))
        except AttributeError:
            continue
    
    return public_items


def has_docstring(obj) -> bool:
    """Check if an object has a non-empty docstring."""
    return obj.__doc__ is not None and obj.__doc__.strip() != ""


def is_numpy_style_docstring(docstring: str) -> bool:
    """Check if docstring follows NumPy style conventions."""
    if not docstring:
        return False
    
    # Basic NumPy style patterns
    patterns = [
        r'Parameters\s*\n\s*-+',  # Parameters section
        r'Returns\s*\n\s*-+',     # Returns section
    ]
    
    # Check for at least Parameters or Returns section
    has_sections = any(re.search(pattern, docstring, re.MULTILINE | re.IGNORECASE) 
                      for pattern in patterns)
    
    return has_sections


def extract_parameters_from_signature(func) -> List[str]:
    """Extract parameter names from function signature."""
    try:
        sig = inspect.signature(func)
        return [param.name for param in sig.parameters.values() 
                if param.name not in ('self', 'cls')]
    except (ValueError, TypeError):
        return []


def extract_parameters_from_docstring(docstring: str) -> List[str]:
    """Extract parameter names from NumPy-style docstring."""
    if not docstring:
        return []
    
    # Find Parameters section
    params_match = re.search(r'Parameters\s*\n\s*-+\s*\n(.*?)(?=\n\s*[A-Z][a-z]+\s*\n\s*-+|\n\s*Examples|\n\s*Notes|\Z)', 
                           docstring, re.DOTALL | re.IGNORECASE)
    
    if not params_match:
        return []
    
    params_section = params_match.group(1)
    
    # Extract parameter names (format: param_name : type)
    param_names = []
    for line in params_section.split('\n'):
        line = line.strip()
        if ':' in line and not line.startswith(' '):
            param_name = line.split(':')[0].strip()
            if param_name and not param_name.startswith('-'):
                param_names.append(param_name)
    
    return param_names


class TestDocumentationCompleteness:
    """Test documentation completeness and standards."""
    
    def test_all_public_functions_have_docstrings(self):
        """Test that all public functions have non-empty docstrings."""
        modules_to_test = [
            ('mcpost', mcpost),
            ('mcpost.gsa', mcpost.gsa),
            ('mcpost.integration', mcpost.integration),
            ('mcpost.utils', mcpost.utils),
        ]
        
        missing_docstrings = []
        
        for module_name, module in modules_to_test:
            public_items = get_public_functions_and_classes(module)
            
            for name, item in public_items:
                if inspect.isfunction(item) and not has_docstring(item):
                    missing_docstrings.append(f"{module_name}.{name}")
        
        assert not missing_docstrings, f"Functions missing docstrings: {missing_docstrings}"
    
    def test_docstrings_follow_numpy_style(self):
        """Test that docstrings follow NumPy style conventions."""
        modules_to_test = [
            ('mcpost.gsa', mcpost.gsa),
            ('mcpost.integration', mcpost.integration),
            ('mcpost.utils', mcpost.utils),
        ]
        
        non_numpy_style = []
        
        for module_name, module in modules_to_test:
            public_items = get_public_functions_and_classes(module)
            
            for name, item in public_items:
                if inspect.isfunction(item) and has_docstring(item):
                    if not is_numpy_style_docstring(item.__doc__):
                        non_numpy_style.append(f"{module_name}.{name}")
        
        assert not non_numpy_style, f"Functions not following NumPy style: {non_numpy_style}"
    
    @given(st.sampled_from([
        mcpost.gsa.gsa_pipeline,
        mcpost.gsa.gsa_for_target,
        mcpost.integration.monte_carlo_integral,
        mcpost.integration.qmc_integral,
        mcpost.integration.qmc_integral_auto,
        mcpost.integration.qmc_integral_importance,
        mcpost.utils.validate_inputs,
        mcpost.utils.configure_defaults,
    ]))
    def test_function_parameters_documented(self, func):
        """
        Property test: For any public function, all parameters should be documented.
        
        **Feature: mcpost-package-improvement, Property 3: Documentation Completeness and Standards**
        **Validates: Requirements 3.2, 3.5**
        """
        if not has_docstring(func):
            pytest.skip(f"Function {func.__name__} has no docstring")
        
        signature_params = extract_parameters_from_signature(func)
        docstring_params = extract_parameters_from_docstring(func.__doc__)
        
        # Filter out *args and **kwargs style parameters
        signature_params = [p for p in signature_params if not p.startswith('*')]
        
        # Check that all signature parameters are documented
        missing_params = set(signature_params) - set(docstring_params)
        
        # Allow some common parameters to be undocumented (like *args, **kwargs)
        allowed_missing = {'args', 'kwargs'}
        missing_params = missing_params - allowed_missing
        
        assert not missing_params, (
            f"Function {func.__name__} has undocumented parameters: {missing_params}. "
            f"Signature params: {signature_params}, Docstring params: {docstring_params}"
        )
    
    def test_docstrings_have_examples_for_main_functions(self):
        """Test that main public functions have usage examples."""
        main_functions = [
            mcpost.gsa.gsa_pipeline,
            mcpost.gsa.gsa_for_target,
            mcpost.integration.monte_carlo_integral,
            mcpost.integration.qmc_integral_auto,
        ]
        
        missing_examples = []
        
        for func in main_functions:
            if has_docstring(func):
                docstring = func.__doc__
                # Check for Examples section
                if not re.search(r'Examples\s*\n\s*-+', docstring, re.MULTILINE | re.IGNORECASE):
                    missing_examples.append(func.__name__)
        
        assert not missing_examples, f"Main functions missing examples: {missing_examples}"
    
    def test_docstrings_have_proper_return_documentation(self):
        """Test that functions with return values document them properly."""
        functions_to_test = [
            mcpost.gsa.gsa_pipeline,
            mcpost.gsa.gsa_for_target,
            mcpost.integration.monte_carlo_integral,
            mcpost.integration.qmc_integral,
            mcpost.integration.qmc_integral_auto,
            mcpost.integration.qmc_integral_importance,
        ]
        
        missing_returns = []
        
        for func in functions_to_test:
            if has_docstring(func):
                docstring = func.__doc__
                # Check for Returns section
                if not re.search(r'Returns\s*\n\s*-+', docstring, re.MULTILINE | re.IGNORECASE):
                    missing_returns.append(func.__name__)
        
        assert not missing_returns, f"Functions missing Returns documentation: {missing_returns}"
    
    @given(st.text(min_size=10, max_size=1000))
    def test_docstring_format_validation(self, sample_docstring):
        """
        Property test: Docstring format validation should work correctly.
        
        **Feature: mcpost-package-improvement, Property 3: Documentation Completeness and Standards**
        **Validates: Requirements 3.2, 3.5**
        """
        # Test that our validation functions work correctly
        result = is_numpy_style_docstring(sample_docstring)
        
        # If it claims to be NumPy style, it should have section headers
        if result:
            assert re.search(r'Parameters\s*\n\s*-+|Returns\s*\n\s*-+', 
                           sample_docstring, re.MULTILINE | re.IGNORECASE)
    
    def test_configuration_classes_documented(self):
        """Test that configuration classes have proper documentation."""
        config_classes = [
            mcpost.utils.GSAConfig,
            mcpost.utils.IntegrationConfig,
        ]
        
        missing_docs = []
        
        for cls in config_classes:
            if not has_docstring(cls):
                missing_docs.append(cls.__name__)
            else:
                # Check that class docstring mentions what it configures
                docstring = cls.__doc__.lower()
                if 'config' not in docstring and 'parameter' not in docstring:
                    missing_docs.append(f"{cls.__name__} (unclear purpose)")
        
        assert not missing_docs, f"Configuration classes with poor documentation: {missing_docs}"


class TestDocumentationConsistency:
    """Test documentation consistency across the package."""
    
    def test_consistent_parameter_naming_in_docs(self):
        """Test that parameter names are consistent across similar functions."""
        # Common parameter patterns that should be consistent
        integration_functions = [
            mcpost.integration.monte_carlo_integral,
            mcpost.integration.qmc_integral,
            mcpost.integration.qmc_integral_auto,
            mcpost.integration.qmc_integral_importance,
        ]
        
        # Check that similar parameters have consistent names
        param_patterns = {}
        
        for func in integration_functions:
            if has_docstring(func):
                params = extract_parameters_from_signature(func)
                for param in params:
                    if 'target' in param.lower():
                        param_patterns.setdefault('target_pdf', []).append((func.__name__, param))
                    elif 'sample' in param.lower() and 'n_' not in param.lower():
                        param_patterns.setdefault('sample_pdf', []).append((func.__name__, param))
        
        # Check for consistency (this is more of a style check)
        inconsistencies = []
        for pattern_type, occurrences in param_patterns.items():
            param_names = [param for _, param in occurrences]
            if len(set(param_names)) > 1:
                inconsistencies.append(f"{pattern_type}: {occurrences}")
        
        # This is informational rather than a hard requirement
        if inconsistencies:
            print(f"Parameter naming inconsistencies (informational): {inconsistencies}")
    
    def test_docstring_sections_are_properly_formatted(self):
        """Test that docstring sections follow proper formatting."""
        functions_to_test = [
            mcpost.gsa.gsa_pipeline,
            mcpost.integration.monte_carlo_integral,
            mcpost.integration.qmc_integral_auto,
        ]
        
        formatting_issues = []
        
        for func in functions_to_test:
            if has_docstring(func):
                docstring = func.__doc__
                
                # Check that section headers are followed by dashes
                sections = re.findall(r'^(\w+)\s*\n\s*(-+)', docstring, re.MULTILINE)
                
                for section_name, dashes in sections:
                    # Dashes should be at least as long as section name
                    if len(dashes) < len(section_name):
                        formatting_issues.append(
                            f"{func.__name__}: {section_name} section underline too short"
                        )
        
        assert not formatting_issues, f"Docstring formatting issues: {formatting_issues}"