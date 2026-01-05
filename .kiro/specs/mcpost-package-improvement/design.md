# Design Document

## Overview

This design outlines the transformation of MCPost from a collection of Python scripts into a professional, distributable Python package following modern packaging standards. The design emphasizes maintainability, extensibility, and user experience while preserving the existing functionality of global sensitivity analysis and Monte Carlo integration.

## Architecture

### Package Structure

The new MCPost package will follow the modern Python packaging standards using `pyproject.toml` configuration and a clean module hierarchy:

```
mcpost/
├── pyproject.toml              # Modern packaging configuration
├── README.md                   # Comprehensive package documentation
├── LICENSE                     # Open source license
├── CHANGELOG.md               # Version history
├── MANIFEST.in                # Additional files for distribution
├── mcpost/                    # Main package directory
│   ├── __init__.py           # Public API exports
│   ├── gsa/                  # Global Sensitivity Analysis module
│   │   ├── __init__.py
│   │   ├── pipeline.py       # Main GSA pipeline (refactored from gsa_pipeline.py)
│   │   ├── metrics.py        # Sensitivity metrics computation
│   │   ├── kernels.py        # GP kernel utilities
│   │   └── plotting.py       # GSA visualization functions
│   ├── integration/          # Monte Carlo Integration module
│   │   ├── __init__.py
│   │   ├── monte_carlo.py    # MC integration (refactored from mc_int.py)
│   │   ├── quasi_monte_carlo.py  # QMC methods
│   │   └── importance.py     # Importance sampling utilities
│   ├── utils/               # Shared utilities
│   │   ├── __init__.py
│   │   ├── data.py          # Data preprocessing utilities
│   │   ├── validation.py    # Input validation functions
│   │   └── config.py        # Configuration management
│   └── _version.py          # Version information
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_gsa/           # GSA tests
│   ├── test_integration/   # Integration tests
│   ├── test_utils/         # Utility tests
│   └── conftest.py         # Pytest configuration
├── docs/                   # Documentation
│   ├── source/
│   ├── examples/           # Example notebooks and scripts
│   └── tutorials/          # Tutorial notebooks
├── benchmarks/             # Performance benchmarks
└── scripts/               # Development and maintenance scripts
```

### Module Design

#### GSA Module (`mcpost.gsa`)
- **pipeline.py**: Main GSA pipeline functions (`gsa_for_target`, `gsa_pipeline`)
- **metrics.py**: Individual sensitivity metrics (MI, dCor, permutation importance, Sobol indices)
- **kernels.py**: Gaussian Process kernel construction and utilities
- **plotting.py**: Visualization functions for sensitivity analysis results

#### Integration Module (`mcpost.integration`)
- **monte_carlo.py**: Standard Monte Carlo integration methods
- **quasi_monte_carlo.py**: Quasi-Monte Carlo methods (Sobol, Halton sequences)
- **importance.py**: Importance sampling techniques

#### Utils Module (`mcpost.utils`)
- **data.py**: Data preprocessing, scaling, constant column detection
- **validation.py**: Input validation and error handling
- **config.py**: Configuration management and default parameters

## Components and Interfaces

### Public API Design

The package will expose a clean, intuitive API through `mcpost.__init__.py`:

```python
# Core GSA functions
from mcpost.gsa import gsa_pipeline, gsa_for_target, plot_sensitivity_metrics

# Core integration functions  
from mcpost.integration import monte_carlo_integral, qmc_integral, qmc_integral_auto

# Utility functions
from mcpost.utils import validate_inputs, configure_defaults
```

### Configuration System

A centralized configuration system will manage default parameters:

```python
# mcpost/utils/config.py
class GSAConfig:
    DEFAULT_SCALER = "minmax"
    DEFAULT_KERNEL = "rbf"
    DEFAULT_N_SOBOL = 4096
    DEFAULT_ENABLE_PERM = True
    DEFAULT_ENABLE_GP = True
    DEFAULT_ENABLE_SOBOL = True

class IntegrationConfig:
    DEFAULT_QMC_METHOD = "sobol"
    DEFAULT_N_SAMPLES = 10000
```

### Backward Compatibility

The refactored code will maintain full backward compatibility with existing function signatures. Users can continue using the same function calls with identical parameters and return values.

## Data Models

### Input Validation Schema

```python
@dataclass
class GSAInputs:
    X: np.ndarray  # Parameters (N_samples, N_params)
    Y: np.ndarray  # Targets (N_samples, N_targets)
    param_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    bounds: Optional[List[List[float]]] = None
    
    def __post_init__(self):
        # Validation logic
        pass

@dataclass
class IntegrationInputs:
    params: np.ndarray
    data: np.ndarray
    p_target: Callable
    q_sample: Optional[Callable] = None
    bounds: Optional[List[Tuple[float, float]]] = None
```

### Result Objects

Structured result objects will replace dictionary returns for better type safety:

```python
@dataclass
class GSAResult:
    table: pd.DataFrame
    models: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def plot_heatmap(self, **kwargs):
        """Generate sensitivity heatmap"""
        pass
    
    def to_latex(self, **kwargs):
        """Export results to LaTeX table"""
        pass

@dataclass
class IntegrationResult:
    integral: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Now I need to use the prework tool to analyze the acceptance criteria before writing the correctness properties:

<function_calls>
<invoke name="prework">
<parameter name="featureName">mcpost-package-improvement

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing the acceptance criteria, several properties can be consolidated to avoid redundancy:
- File existence checks (1.1-1.5, 2.1-2.4, 3.1, 3.3-3.4, 4.1-4.4, 6.1, 8.1-8.4) can be combined into comprehensive structure validation properties
- API compatibility and consistency properties (2.5, 6.5) can be unified
- Documentation standards properties (3.2, 3.5) can be combined
- Configuration and extensibility properties (6.2, 6.3-6.4) share similar testing patterns

### Core Properties

**Property 1: Package Structure Integrity**
*For any* valid MCPost installation, the package directory structure should contain all required files and modules in their expected locations with proper Python package hierarchy
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4**

**Property 2: API Backward Compatibility**
*For any* existing MCPost function call with valid parameters, the refactored package should produce identical results to the original implementation
**Validates: Requirements 2.5**

**Property 3: Documentation Completeness and Standards**
*For any* public function or class in MCPost, it should have complete docstrings following NumPy/Google style standards and be included in API documentation
**Validates: Requirements 3.2, 3.5**

**Property 4: Dependency Management Correctness**
*For any* installation scenario, required dependencies should be automatically installed and optional dependencies should provide clear error messages when missing
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

**Property 5: Configuration Override Consistency**
*For any* configuration parameter, setting it via environment variable or config file should override the default value consistently across all modules
**Validates: Requirements 6.2**

**Property 6: Extension Interface Consistency**
*For any* new GSA or integration method extension, it should follow the same API patterns and integration mechanisms as core functionality
**Validates: Requirements 6.3, 6.4, 6.5**

**Property 7: Numerical Correctness Preservation**
*For any* valid input dataset, GSA and integration computations should produce numerically equivalent results before and after refactoring within acceptable floating-point precision
**Validates: Requirements 4.5, 7.1, 7.2**

**Property 8: Performance and Memory Efficiency**
*For any* large dataset processing, memory-efficient implementations should use significantly less memory than standard implementations while producing equivalent results
**Validates: Requirements 7.4**

## Error Handling

### Input Validation Strategy

The package will implement comprehensive input validation at module boundaries:

```python
def validate_gsa_inputs(X, Y, param_names=None, **kwargs):
    """Validate inputs for GSA functions"""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if len(X) == 0:
        raise ValueError("X cannot be empty")
    # Additional validation...
```

### Dependency Error Handling

Optional dependencies will be handled gracefully with informative error messages:

```python
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def plot_sensitivity_metrics(*args, **kwargs):
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for plotting. "
            "Install with: pip install mcpost[viz]"
        )
    # Plotting implementation...
```

### Configuration Error Handling

Configuration errors will provide clear guidance:

```python
def validate_config(config_dict):
    """Validate configuration parameters"""
    allowed_scalers = ["minmax", "standard", None]
    if config_dict.get("scaler") not in allowed_scalers:
        raise ValueError(
            f"Invalid scaler '{config_dict['scaler']}'. "
            f"Must be one of {allowed_scalers}"
        )
```

## Testing Strategy

### Dual Testing Approach

The testing strategy combines unit tests and property-based tests for comprehensive coverage:

**Unit Tests:**
- Verify specific examples and edge cases
- Test integration points between modules
- Validate error conditions and exception handling
- Test configuration and setup functionality

**Property-Based Tests:**
- Verify numerical correctness across random inputs
- Test API consistency and backward compatibility
- Validate performance characteristics
- Test configuration override behavior

### Testing Framework Configuration

- **Framework**: pytest with minimum 100 iterations per property test
- **Coverage Target**: Minimum 80% code coverage for core functionality
- **Property Test Library**: Hypothesis for Python property-based testing
- **Performance Testing**: pytest-benchmark for performance regression detection

### Test Organization

```
tests/
├── test_gsa/
│   ├── test_pipeline.py          # GSA pipeline unit tests
│   ├── test_metrics.py           # Sensitivity metrics tests
│   ├── test_kernels.py           # GP kernel tests
│   └── test_properties.py        # GSA property-based tests
├── test_integration/
│   ├── test_monte_carlo.py       # MC integration unit tests
│   ├── test_quasi_monte_carlo.py # QMC tests
│   └── test_properties.py        # Integration property-based tests
├── test_utils/
│   ├── test_data.py              # Data utilities tests
│   ├── test_validation.py        # Input validation tests
│   └── test_config.py            # Configuration tests
├── test_api/
│   ├── test_backward_compatibility.py  # API compatibility tests
│   └── test_public_interface.py        # Public API tests
└── integration/
    ├── test_end_to_end.py        # Full workflow tests
    └── test_examples.py          # Example script tests
```

### Property Test Tags

Each property test will be tagged with comments referencing design properties:
- **Feature: mcpost-package-improvement, Property 1**: Package structure integrity tests
- **Feature: mcpost-package-improvement, Property 2**: Backward compatibility tests
- **Feature: mcpost-package-improvement, Property 7**: Numerical correctness tests