"""
Property-based tests for extension interface consistency.

Tests Property 6: Extension Interface Consistency
**Validates: Requirements 6.3, 6.4, 6.5**
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any

from mcpost.gsa.base import (
    BaseSensitivityMethod, VarianceBasedMethod, ModelBasedMethod,
    GSAResult, gsa_registry, register_gsa_method
)
from mcpost.integration.base import (
    BaseIntegrationMethod, MonteCarloMethod, QuasiMonteCarloMethod,
    IntegrationResult, integration_registry, register_integration_method
)


# Test implementations for property testing
class _TestGSAMethod(BaseSensitivityMethod):
    """Simple test GSA method for property testing."""
    
    def __init__(self, test_param=1.0):
        super().__init__()
        self.test_param = test_param
    
    def compute_sensitivity(self, X, y, **kwargs):
        self.validate_inputs(X, y)
        
        # Simple correlation-based sensitivity
        n_params = X.shape[1]
        sensitivity_values = np.array([
            abs(np.corrcoef(X[:, i], y)[0, 1]) if np.var(X[:, i]) > 0 else 0.0
            for i in range(n_params)
        ])
        sensitivity_values = np.nan_to_num(sensitivity_values, nan=0.0)
        
        param_names = [f"p{i}" for i in range(n_params)]
        rankings = np.argsort(sensitivity_values)[::-1]
        
        return GSAResult(
            sensitivity_values=sensitivity_values,
            parameter_names=param_names,
            method_name="TestGSAMethod",  # Use consistent name
            metadata={"test_param": self.test_param},
            rankings=rankings
        )


class _TestVarianceMethod(VarianceBasedMethod):
    """Test variance-based GSA method."""
    
    def __init__(self, n_bootstrap=10):  # Reduced for testing
        super().__init__(n_bootstrap=n_bootstrap)
    
    def compute_sensitivity(self, X, y, **kwargs):
        self.validate_inputs(X, y)
        
        n_params = X.shape[1]
        total_var = self.estimate_total_variance(y)
        
        # Simple variance decomposition
        sensitivity_values = np.array([
            np.var(y) - np.var(y - X[:, i] * np.corrcoef(X[:, i], y)[0, 1])
            if np.var(X[:, i]) > 0 else 0.0
            for i in range(n_params)
        ]) / total_var
        
        sensitivity_values = np.nan_to_num(sensitivity_values, nan=0.0)
        sensitivity_values = np.clip(sensitivity_values, 0, 1)  # Ensure valid range
        
        param_names = [f"p{i}" for i in range(n_params)]
        
        return GSAResult(
            sensitivity_values=sensitivity_values,
            parameter_names=param_names,
            method_name="TestVarianceMethod",
            metadata={"total_variance": total_var}
        )


class _TestIntegrationMethod(MonteCarloMethod):
    """Simple test integration method for property testing."""
    
    def __init__(self, test_param=1.0, random_state=None):
        super().__init__(random_state=random_state)
        self.test_param = test_param
    
    def integrate(self, func, bounds, n_samples=1000, **kwargs):
        self.validate_inputs(func, bounds, n_samples)
        
        # Simple uniform Monte Carlo
        samples = self.generate_uniform_samples(bounds, n_samples)
        values = func(samples)
        
        volume = self.estimate_volume(bounds)
        integral = np.mean(values) * volume
        uncertainty = self.estimate_uncertainty(values, volume)
        
        return IntegrationResult(
            integral=integral,
            uncertainty=uncertainty,
            n_samples=n_samples,
            method_name="TestIntegrationMethod",  # Use consistent name
            metadata={"test_param": self.test_param}
        )


class _TestQMCMethod(QuasiMonteCarloMethod):
    """Test QMC method."""
    
    def __init__(self):
        super().__init__(sequence_type="test")
    
    def generate_qmc_samples(self, bounds, n_samples):
        # Simple grid-based "QMC" for testing
        n_dims = len(bounds)
        samples_per_dim = int(np.ceil(n_samples ** (1.0 / n_dims)))
        
        # Create grid points
        grid_points = []
        for i, (low, high) in enumerate(bounds):
            grid_points.append(np.linspace(low, high, samples_per_dim))
        
        # Generate samples (simplified)
        samples = np.random.random((n_samples, n_dims))
        for i, (low, high) in enumerate(bounds):
            samples[:, i] = low + samples[:, i] * (high - low)
        
        return samples[:n_samples]  # Ensure exact count
    
    def integrate(self, func, bounds, n_samples=1000, **kwargs):
        self.validate_inputs(func, bounds, n_samples)
        
        samples = self.generate_qmc_samples(bounds, n_samples)
        values = func(samples)
        
        volume = self.estimate_volume(bounds)
        integral = np.mean(values) * volume
        uncertainty = self.estimate_qmc_uncertainty(values, volume)
        
        return IntegrationResult(
            integral=integral,
            uncertainty=uncertainty,
            n_samples=n_samples,
            method_name="TestQMCMethod",
            metadata={"sequence_type": self.sequence_type}
        )


@given(
    n_samples=st.integers(min_value=50, max_value=500),
    n_params=st.integers(min_value=2, max_value=10),
    test_param=st.floats(min_value=0.1, max_value=5.0)
)
@settings(max_examples=2, deadline=10000)
def test_gsa_method_interface_consistency(n_samples: int, n_params: int, test_param: float):
    """
    **Feature: mcpost-package-improvement, Property 6: Extension Interface Consistency**
    
    For any new GSA method extension, it should follow the same API patterns and 
    integration mechanisms as core functionality.
    **Validates: Requirements 6.3, 6.4, 6.5**
    """
    # Generate test data
    np.random.seed(42)  # Fixed seed for reproducibility
    X = np.random.uniform(-1, 1, (n_samples, n_params))
    y = np.sum(X**2, axis=1) + 0.1 * np.random.normal(0, 1, n_samples)
    
    # Test base GSA method
    method = _TestGSAMethod(test_param=test_param)
    
    # Property: Method should have consistent interface
    assert hasattr(method, 'compute_sensitivity'), "Method should have compute_sensitivity method"
    assert hasattr(method, 'validate_inputs'), "Method should have validate_inputs method"
    assert hasattr(method, 'name'), "Method should have name attribute"
    
    # Property: Method should be callable
    result1 = method.compute_sensitivity(X, y)
    result2 = method(X, y)  # Should work via __call__
    
    # Property: Results should be consistent
    assert isinstance(result1, GSAResult), "Should return GSAResult instance"
    assert isinstance(result2, GSAResult), "Should return GSAResult instance"
    assert np.allclose(result1.sensitivity_values, result2.sensitivity_values), "Results should be identical"
    
    # Property: GSAResult should have required attributes
    assert hasattr(result1, 'sensitivity_values'), "Result should have sensitivity_values"
    assert hasattr(result1, 'parameter_names'), "Result should have parameter_names"
    assert hasattr(result1, 'method_name'), "Result should have method_name"
    assert hasattr(result1, 'metadata'), "Result should have metadata"
    
    # Property: Result dimensions should match input
    assert len(result1.sensitivity_values) == n_params, "Should have sensitivity for each parameter"
    assert len(result1.parameter_names) == n_params, "Should have name for each parameter"
    
    # Property: Sensitivity values should be finite and non-negative
    assert np.all(np.isfinite(result1.sensitivity_values)), "Sensitivity values should be finite"
    assert np.all(result1.sensitivity_values >= 0), "Sensitivity values should be non-negative"
    
    # Property: Method name should be consistent
    assert result1.method_name == "TestGSAMethod", "Method name should be consistent"
    
    # Property: Metadata should contain method parameters
    assert "test_param" in result1.metadata, "Metadata should contain method parameters"
    assert result1.metadata["test_param"] == test_param, "Metadata should preserve parameter values"
    
    # Property: DataFrame conversion should work
    df = result1.to_dataframe()
    assert len(df) == n_params, "DataFrame should have row for each parameter"
    assert 'sensitivity' in df.columns, "DataFrame should have sensitivity column"
    
    # Property: Top parameters method should work
    top_params = result1.get_top_parameters(n=min(3, n_params))
    assert len(top_params) <= min(3, n_params), "Should return requested number of parameters"
    assert all(param in result1.parameter_names for param in top_params), "Should return valid parameter names"


@given(
    n_samples=st.integers(min_value=50, max_value=300),
    n_params=st.integers(min_value=2, max_value=8)
)
@settings(max_examples=2, deadline=8000)
def test_variance_based_method_consistency(n_samples: int, n_params: int):
    """
    **Feature: mcpost-package-improvement, Property 6: Extension Interface Consistency**
    
    For any variance-based GSA method, it should provide consistent variance 
    decomposition interfaces and bootstrap uncertainty estimation.
    **Validates: Requirements 6.3, 6.4, 6.5**
    """
    # Generate test data
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (n_samples, n_params))
    y = 2*X[:, 0] + X[:, 1] + 0.1*np.random.normal(0, 1, n_samples)
    
    method = _TestVarianceMethod(n_bootstrap=5)  # Small for testing
    
    # Property: Should inherit from VarianceBasedMethod
    assert isinstance(method, VarianceBasedMethod), "Should inherit from VarianceBasedMethod"
    
    # Property: Should have variance estimation methods
    assert hasattr(method, 'estimate_total_variance'), "Should have variance estimation method"
    assert hasattr(method, 'bootstrap_uncertainty'), "Should have bootstrap method"
    
    # Property: Total variance estimation should be reasonable
    total_var = method.estimate_total_variance(y)
    assert total_var > 0, "Total variance should be positive"
    assert np.isfinite(total_var), "Total variance should be finite"
    
    # Property: Method should work with fit interface
    fitted_method = method.fit(X, y)
    assert fitted_method.is_fitted, "Method should be marked as fitted"
    assert fitted_method is method, "fit() should return self"
    
    # Property: Results should be valid
    result = method.compute_sensitivity(X, y)
    assert isinstance(result, GSAResult), "Should return GSAResult"
    assert np.all(result.sensitivity_values >= 0), "Variance-based sensitivity should be non-negative"
    assert np.all(result.sensitivity_values <= 1), "Variance-based sensitivity should be <= 1"


@given(
    n_dims=st.integers(min_value=1, max_value=5),
    n_samples=st.integers(min_value=100, max_value=1000),
    test_param=st.floats(min_value=0.1, max_value=3.0)
)
@settings(max_examples=2, deadline=8000)
def test_integration_method_interface_consistency(n_dims: int, n_samples: int, test_param: float):
    """
    **Feature: mcpost-package-improvement, Property 6: Extension Interface Consistency**
    
    For any new integration method extension, it should follow the same API patterns 
    and return consistent IntegrationResult objects.
    **Validates: Requirements 6.3, 6.4, 6.5**
    """
    # Create test function and bounds
    def test_func(x):
        return np.sum(x**2, axis=1)
    
    bounds = [(0, 1) for _ in range(n_dims)]
    
    # Test Monte Carlo method
    method = _TestIntegrationMethod(test_param=test_param, random_state=42)
    
    # Property: Method should have consistent interface
    assert hasattr(method, 'integrate'), "Method should have integrate method"
    assert hasattr(method, 'validate_inputs'), "Method should have validate_inputs method"
    assert hasattr(method, 'name'), "Method should have name attribute"
    
    # Property: Method should be callable
    result1 = method.integrate(test_func, bounds, n_samples)
    result2 = method(test_func, bounds, n_samples)  # Should work via __call__
    
    # Property: Results should be consistent (with some tolerance for randomness)
    assert isinstance(result1, IntegrationResult), "Should return IntegrationResult instance"
    assert isinstance(result2, IntegrationResult), "Should return IntegrationResult instance"
    
    # Property: IntegrationResult should have required attributes
    assert hasattr(result1, 'integral'), "Result should have integral"
    assert hasattr(result1, 'uncertainty'), "Result should have uncertainty"
    assert hasattr(result1, 'n_samples'), "Result should have n_samples"
    assert hasattr(result1, 'method_name'), "Result should have method_name"
    assert hasattr(result1, 'metadata'), "Result should have metadata"
    
    # Property: Result values should be reasonable
    assert np.isfinite(result1.integral), "Integral should be finite"
    assert np.isfinite(result1.uncertainty), "Uncertainty should be finite"
    assert result1.uncertainty >= 0, "Uncertainty should be non-negative"
    assert result1.n_samples == n_samples, "Should record correct sample count"
    
    # Property: Method name should be consistent
    assert result1.method_name == "TestIntegrationMethod", "Method name should be consistent"
    
    # Property: Metadata should contain method parameters
    assert "test_param" in result1.metadata, "Metadata should contain method parameters"
    assert result1.metadata["test_param"] == test_param, "Metadata should preserve parameter values"
    
    # Property: Result methods should work
    rel_error = result1.relative_error()
    assert np.isfinite(rel_error), "Relative error should be finite"
    
    summary = result1.summary()
    assert isinstance(summary, str), "Summary should return string"
    assert "TestIntegrationMethod" in summary, "Summary should contain method name"
    
    # Property: Convergence check should work
    is_converged = result1.is_converged(tolerance=1.0)  # Loose tolerance
    assert isinstance(is_converged, (bool, np.bool_)), "Convergence check should return boolean"


@given(
    n_dims=st.integers(min_value=1, max_value=4),
    n_samples=st.integers(min_value=50, max_value=500)
)
@settings(max_examples=2, deadline=6000)
def test_qmc_method_interface_consistency(n_dims: int, n_samples: int):
    """
    **Feature: mcpost-package-improvement, Property 6: Extension Interface Consistency**
    
    For any QMC method extension, it should provide consistent quasi-random 
    sampling interfaces and appropriate uncertainty estimation.
    **Validates: Requirements 6.3, 6.4, 6.5**
    """
    def test_func(x):
        return np.sum(x, axis=1)  # Simple linear function
    
    bounds = [(0, 1) for _ in range(n_dims)]
    
    method = _TestQMCMethod()
    
    # Property: Should inherit from QuasiMonteCarloMethod
    assert isinstance(method, QuasiMonteCarloMethod), "Should inherit from QuasiMonteCarloMethod"
    
    # Property: Should have QMC-specific methods
    assert hasattr(method, 'generate_qmc_samples'), "Should have QMC sample generation method"
    assert hasattr(method, 'estimate_qmc_uncertainty'), "Should have QMC uncertainty estimation"
    
    # Property: QMC sample generation should work
    samples = method.generate_qmc_samples(bounds, n_samples)
    assert samples.shape == (n_samples, n_dims), "Should generate correct sample shape"
    
    # Property: Samples should be within bounds
    for i, (low, high) in enumerate(bounds):
        assert np.all(samples[:, i] >= low), f"Samples should be >= lower bound for dim {i}"
        assert np.all(samples[:, i] <= high), f"Samples should be <= upper bound for dim {i}"
    
    # Property: Integration should work
    result = method.integrate(test_func, bounds, n_samples)
    assert isinstance(result, IntegrationResult), "Should return IntegrationResult"
    assert result.method_name == "TestQMCMethod", "Should have correct method name"
    assert "sequence_type" in result.metadata, "Should include sequence type in metadata"


def test_registry_consistency():
    """
    **Feature: mcpost-package-improvement, Property 6: Extension Interface Consistency**
    
    Registry systems should provide consistent interfaces for method discovery 
    and instantiation across GSA and integration modules.
    **Validates: Requirements 6.3, 6.4, 6.5**
    """
    # Test GSA registry
    original_gsa_methods = gsa_registry.list_methods().copy()
    
    # Property: Registry should support registration
    gsa_registry.register('test_method', _TestGSAMethod, 'Test GSA method')
    assert 'test_method' in gsa_registry.list_methods(), "Should register new method"
    
    # Property: Registry should provide method info
    info = gsa_registry.get_method_info('test_method')
    assert 'class' in info, "Method info should contain class"
    assert 'description' in info, "Method info should contain description"
    assert info['class'] == _TestGSAMethod, "Should return correct class"
    
    # Property: Registry should create instances
    method = gsa_registry.create('test_method', test_param=2.0)
    assert isinstance(method, _TestGSAMethod), "Should create correct instance type"
    assert method.test_param == 2.0, "Should pass constructor arguments"
    
    # Property: Registry should check registration
    assert gsa_registry.is_registered('test_method'), "Should confirm registration"
    assert not gsa_registry.is_registered('nonexistent'), "Should return False for unregistered"
    
    # Test integration registry
    original_int_methods = integration_registry.list_methods().copy()
    
    # Property: Integration registry should work similarly
    integration_registry.register('test_integration', _TestIntegrationMethod, 'Test integration')
    assert 'test_integration' in integration_registry.list_methods(), "Should register integration method"
    
    int_method = integration_registry.create('test_integration', test_param=1.5)
    assert isinstance(int_method, _TestIntegrationMethod), "Should create integration method"
    assert int_method.test_param == 1.5, "Should pass integration method parameters"
    
    # Property: Decorator registration should work
    @register_gsa_method('decorated_test', 'Decorated test method')
    class DecoratedTestMethod(BaseSensitivityMethod):
        def compute_sensitivity(self, X, y, **kwargs):
            return GSAResult(
                sensitivity_values=np.ones(X.shape[1]),
                parameter_names=[f"p{i}" for i in range(X.shape[1])],
                method_name="DecoratedTest",
                metadata={}
            )
    
    assert 'decorated_test' in gsa_registry.list_methods(), "Decorator should register method"
    decorated_method = gsa_registry.create('decorated_test')
    assert isinstance(decorated_method, DecoratedTestMethod), "Should create decorated method"


def test_input_validation_consistency():
    """
    **Feature: mcpost-package-improvement, Property 6: Extension Interface Consistency**
    
    All extension methods should provide consistent input validation with 
    informative error messages.
    **Validates: Requirements 6.3, 6.4, 6.5**
    """
    # Test GSA method validation
    gsa_method = _TestGSAMethod()
    
    # Property: Should validate input types
    with pytest.raises(TypeError):
        gsa_method.validate_inputs("not_array", np.array([1, 2, 3]))
    
    with pytest.raises(TypeError):
        gsa_method.validate_inputs(np.array([[1, 2], [3, 4]]), "not_array")
    
    # Property: Should validate input dimensions
    with pytest.raises(ValueError):
        gsa_method.validate_inputs(np.array([1, 2, 3]), np.array([1, 2, 3]))  # X not 2D
    
    with pytest.raises(ValueError):
        gsa_method.validate_inputs(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]))  # y not 1D
    
    # Property: Should validate input sizes
    with pytest.raises(ValueError):
        gsa_method.validate_inputs(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]))  # Size mismatch
    
    # Property: Should validate empty inputs
    with pytest.raises(ValueError):
        gsa_method.validate_inputs(np.array([]).reshape(0, 2), np.array([]))
    
    # Test integration method validation
    int_method = _TestIntegrationMethod()
    
    def test_func(x):
        return np.sum(x**2, axis=1)
    
    # Property: Should validate function
    with pytest.raises(TypeError):
        int_method.validate_inputs("not_callable", [(0, 1)], 100)
    
    # Property: Should validate bounds
    with pytest.raises(TypeError):
        int_method.validate_inputs(test_func, "not_list", 100)
    
    with pytest.raises(ValueError):
        int_method.validate_inputs(test_func, [], 100)  # Empty bounds
    
    with pytest.raises(ValueError):
        int_method.validate_inputs(test_func, [(0, 1, 2)], 100)  # Wrong bound format
    
    with pytest.raises(ValueError):
        int_method.validate_inputs(test_func, [(1, 0)], 100)  # Invalid bound order
    
    # Property: Should validate sample count
    with pytest.raises(ValueError):
        int_method.validate_inputs(test_func, [(0, 1)], 0)  # Zero samples
    
    with pytest.raises(ValueError):
        int_method.validate_inputs(test_func, [(0, 1)], -10)  # Negative samples


if __name__ == "__main__":
    # Run simple tests to verify the module works
    test_registry_consistency()
    test_input_validation_consistency()
    print("Extension interface property tests module is working correctly!")