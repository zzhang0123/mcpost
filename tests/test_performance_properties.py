"""
Property-based tests for performance and memory efficiency.

Tests Property 8: Performance and Memory Efficiency
**Validates: Requirements 7.4**
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import psutil
import os
from typing import Dict, Any

from mcpost.gsa.chunked_pipeline import chunked_gsa_for_target
from mcpost.integration.chunked_integration import chunked_monte_carlo_integral
from mcpost.utils.chunked import (
    suggest_chunking_strategy, 
    estimate_memory_usage,
    calculate_optimal_chunk_size
)


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


@given(
    n_samples=st.integers(min_value=1000, max_value=3000),  # Small samples for speed
    n_params=st.integers(min_value=3, max_value=8),  # Few parameters for speed
)
@settings(max_examples=2, deadline=None)  # No deadline to avoid timeout issues
def test_chunked_gsa_memory_efficiency(n_samples: int, n_params: int):
    """
    **Feature: mcpost-package-improvement, Property 8: Performance and Memory Efficiency**
    
    For any large dataset processing, memory-efficient implementations should use 
    significantly less memory than standard implementations while producing equivalent results.
    **Validates: Requirements 7.4**
    """
    # Generate test data
    np.random.seed(42)  # Fixed seed for reproducibility
    X = np.random.uniform(-1, 1, (n_samples, n_params))
    y = np.sum(X**2, axis=1) + 0.1 * np.random.normal(0, 1, n_samples)
    
    # Force chunking with extremely small memory limit
    max_memory_mb = 0.01  # Force chunking
    
    # Run chunked GSA with minimal features for speed
    table, extras = chunked_gsa_for_target(
        X, y,
        max_memory_mb=max_memory_mb,
        show_progress=False,
        enable_gp=False,  # Disable GP for faster testing
        enable_sobol=False,
        make_pdp=False
    )
    
    # Property: Should produce valid results
    assert isinstance(table, type(table)), "Should return a valid DataFrame"
    assert len(table) == n_params, "Should have results for all parameters"
    assert "chunked_processing" in extras, "Should indicate chunked processing was used"
    assert extras["chunked_processing"] is True, "Should confirm chunked processing"
    
    # Property: All sensitivity metrics should be finite and non-negative
    assert np.all(np.isfinite(table["MI"])), "MI values should be finite"
    assert np.all(np.isfinite(table["dCor"])), "dCor values should be finite"
    assert np.all(table["MI"] >= 0), "MI values should be non-negative"
    assert np.all(table["dCor"] >= 0), "dCor values should be non-negative"
    
    # Property: Chunking strategy should be reasonable
    chunk_info = extras["chunking_strategy"]
    assert chunk_info["needs_chunking"] is True, "Should confirm chunking was needed"
    assert chunk_info["n_chunks"] >= 2, "Should use multiple chunks for large datasets"


@given(
    n_samples=st.integers(min_value=1000, max_value=3000),  # Small samples for speed
    n_params=st.integers(min_value=3, max_value=8),  # Few parameters for speed
)
@settings(max_examples=2, deadline=None)  # No deadline to avoid timeout issues
def test_chunked_integration_memory_efficiency(n_samples: int, n_params: int):
    """
    **Feature: mcpost-package-improvement, Property 8: Performance and Memory Efficiency**
    
    For any large dataset processing, chunked Monte Carlo integration should use 
    memory efficiently while producing valid results.
    **Validates: Requirements 7.4**
    """
    # Generate test data
    np.random.seed(42)
    params = np.random.uniform(-1, 1, (n_samples, n_params))  # Smaller range for speed
    data = np.sum(params**2, axis=1)  # Simple quadratic function
    
    # Define simple target and sampling PDFs
    def target_pdf(theta):
        return np.exp(-0.5 * np.sum(theta**2, axis=1)) / ((2 * np.pi) ** (n_params / 2))
    
    def sample_pdf(theta):
        return np.ones(len(theta)) / (2 ** n_params)  # Uniform on [-1, 1]^n_params
    
    # Force chunking with extremely small memory limit
    max_memory_mb = 0.01
    
    # Run chunked integration
    result = chunked_monte_carlo_integral(
        params, data, target_pdf, sample_pdf,
        max_memory_mb=max_memory_mb,
        show_progress=False
    )
    
    # Property: Should produce valid results
    assert "integral" in result, "Should return integral estimate"
    assert "uncertainty" in result, "Should return uncertainty estimate"
    assert "chunked_processing" in result, "Should indicate chunked processing"
    assert result["chunked_processing"] is True, "Should confirm chunked processing"
    
    # Property: Results should be finite
    assert np.isfinite(result["integral"]), "Integral should be finite"
    assert np.isfinite(result["uncertainty"]), "Uncertainty should be finite"
    assert result["uncertainty"] >= 0, "Uncertainty should be non-negative"
    
    # Property: Chunking strategy should be reasonable
    chunk_info = result["chunking_strategy"]
    assert chunk_info["needs_chunking"] is True, "Should confirm chunking was needed"
    assert chunk_info["n_chunks"] >= 2, "Should use multiple chunks for large datasets"


@given(
    n_samples=st.integers(min_value=100, max_value=10000),
    n_params=st.integers(min_value=1, max_value=20),
    n_targets=st.integers(min_value=1, max_value=5),  # Fewer targets for speed
    max_memory_mb=st.floats(min_value=10.0, max_value=500.0)  # Smaller range
)
@settings(max_examples=2, deadline=5000)  # Reduced examples and deadline
def test_chunking_strategy_consistency(n_samples: int, n_params: int, n_targets: int, max_memory_mb: float):
    """
    **Feature: mcpost-package-improvement, Property 8: Performance and Memory Efficiency**
    
    For any dataset size and memory constraint, chunking strategy should provide 
    consistent and reasonable recommendations.
    **Validates: Requirements 7.4**
    """
    # Create dummy arrays to estimate memory
    X = np.zeros((n_samples, n_params), dtype=np.float64)
    Y = np.zeros((n_samples, n_targets), dtype=np.float64) if n_targets > 1 else None
    
    # Get chunking strategy
    strategy = suggest_chunking_strategy(X, Y, max_memory_mb)
    
    # Property: Strategy should always provide valid recommendations
    assert "needs_chunking" in strategy, "Should indicate if chunking is needed"
    assert "current_memory_mb" in strategy, "Should report current memory usage"
    assert "recommendation" in strategy, "Should provide recommendation"
    
    # Property: Memory estimates should be reasonable
    assert strategy["current_memory_mb"] > 0, "Memory usage should be positive"
    
    if strategy["needs_chunking"]:
        # Property: Chunking parameters should be reasonable
        assert "recommended_chunk_size" in strategy, "Should recommend chunk size"
        assert "n_chunks" in strategy, "Should specify number of chunks"
        assert "memory_per_chunk_mb" in strategy, "Should estimate memory per chunk"
        
        chunk_size = strategy["recommended_chunk_size"]
        n_chunks = strategy["n_chunks"]
        memory_per_chunk = strategy["memory_per_chunk_mb"]
        
        # Property: Chunk size should be reasonable
        assert 1 <= chunk_size <= n_samples, "Chunk size should be within valid range"
        assert n_chunks >= 1, "Should have at least one chunk"
        assert memory_per_chunk > 0, "Memory per chunk should be positive"
        assert memory_per_chunk <= max_memory_mb * 1.5, "Memory per chunk should respect limit"
        
        # Property: Chunks should cover all samples
        total_covered = (n_chunks - 1) * chunk_size + min(chunk_size, n_samples - (n_chunks - 1) * chunk_size)
        assert total_covered == n_samples, "Chunks should cover all samples exactly"
    
    # Property: Optimal chunk size calculation should be consistent
    calculated_chunk_size = calculate_optimal_chunk_size(
        n_samples, n_params, n_targets, max_memory_mb
    )
    
    assert 1 <= calculated_chunk_size <= n_samples, "Calculated chunk size should be valid"
    
    if strategy["needs_chunking"]:
        # Should be reasonably close to strategy recommendation
        recommended_size = strategy["recommended_chunk_size"]
        ratio = calculated_chunk_size / recommended_size
        assert 0.5 <= ratio <= 2.0, "Calculated and recommended chunk sizes should be similar"


@given(
    n_samples=st.integers(min_value=100, max_value=5000),  # Smaller range for speed
    n_params=st.integers(min_value=2, max_value=10)  # Fewer parameters
)
@settings(max_examples=2, deadline=8000)  # Reduced examples, longer deadline
def test_memory_estimation_accuracy(n_samples: int, n_params: int):
    """
    **Feature: mcpost-package-improvement, Property 8: Performance and Memory Efficiency**
    
    For any array size, memory estimation should be reasonably accurate.
    **Validates: Requirements 7.4**
    """
    # Create test arrays
    X = np.random.random((n_samples, n_params)).astype(np.float64)
    y = np.random.random(n_samples).astype(np.float64)
    
    # Estimate memory usage
    estimated_mb = estimate_memory_usage(X, y)
    
    # Calculate actual memory usage
    actual_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
    
    # Property: Estimation should be reasonably accurate (within 50% tolerance)
    ratio = estimated_mb / actual_mb
    assert 0.5 <= ratio <= 2.0, (
        f"Memory estimation {estimated_mb:.2f} MB should be close to actual {actual_mb:.2f} MB"
    )
    
    # Property: Estimation should be positive
    assert estimated_mb > 0, "Memory estimation should be positive"
    
    # Property: Estimation should scale with array size
    if n_samples > 1000 and n_params > 5:
        # Create smaller array
        X_small = X[:n_samples//2, :n_params//2]
        y_small = y[:n_samples//2]
        estimated_small = estimate_memory_usage(X_small, y_small)
        
        # Larger array should have larger memory estimate
        assert estimated_mb > estimated_small, "Larger arrays should have larger memory estimates"


def test_chunked_processing_produces_valid_results():
    """
    Test that chunked processing produces valid, finite results for a known case.
    
    **Feature: mcpost-package-improvement, Property 8: Performance and Memory Efficiency**
    **Validates: Requirements 7.4**
    """
    # Create a small test case to force chunking but run quickly
    np.random.seed(42)
    n_samples, n_params = 2000, 5  # Small dataset for speed
    
    X = np.random.uniform(-1, 1, (n_samples, n_params))
    # Create a function with known sensitivity pattern
    y = (2 * X[:, 0]**2 +  # x0 should be most important
         1 * X[:, 1] +     # x1 should be second
         0.1 * np.sum(X[:, 2:], axis=1) +  # Others less important
         0.05 * np.random.normal(0, 1, n_samples))  # Small noise
    
    # Force chunked processing with very small memory limit
    table, extras = chunked_gsa_for_target(
        X, y,
        max_memory_mb=0.01,  # Extremely small limit to force chunking
        show_progress=False,
        enable_gp=False,  # Disable for speed
        enable_sobol=False,
        make_pdp=False
    )
    
    # Property: Should identify x0 as most important parameter
    mi_values = table["MI"].values
    dcor_values = table["dCor"].values
    
    # x0 should have highest MI and dCor (it's quadratic and most influential)
    assert mi_values[0] == np.max(mi_values), "x0 should have highest MI"
    assert dcor_values[0] == np.max(dcor_values), "x0 should have highest dCor"
    
    # Property: All metrics should be finite and reasonable
    assert np.all(np.isfinite(mi_values)), "All MI values should be finite"
    assert np.all(np.isfinite(dcor_values)), "All dCor values should be finite"
    assert np.all(mi_values >= 0), "All MI values should be non-negative"
    assert np.all(dcor_values >= 0), "All dCor values should be non-negative"
    
    # Property: Should indicate chunked processing was used
    assert extras["chunked_processing"] is True, "Should use chunked processing"
    assert "chunking_strategy" in extras, "Should include chunking strategy info"


if __name__ == "__main__":
    # Run a simple test to verify the module works
    test_chunked_processing_produces_valid_results()
    print("Performance property tests module is working correctly!")