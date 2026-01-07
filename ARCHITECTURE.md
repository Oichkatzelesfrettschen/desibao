# Architecture Documentation

## Overview

The `desibao` repository implements a professional, modular architecture for analyzing DESI Y3 BAO data, Supernova systematics, and Dark Matter phenomenology. This document describes the architectural improvements and best practices implemented.

## Project Structure

```
desibao/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── cosmology/               # Cosmology calculations
│   │   ├── __init__.py
│   │   ├── models.py            # Dark energy models (CPL, quintessence, phantom)
│   │   ├── distances.py         # Distance calculations (luminosity, comoving)
│   │   └── power_spectrum.py    # Power spectrum and BAO/DAO models
│   ├── analysis/                # Analysis modules
│   │   ├── __init__.py
│   │   ├── wedge.py             # Canonical quintessence wedge analysis
│   │   ├── dao.py               # Dark Acoustic Oscillation stress tests
│   │   ├── supernova.py         # SN Ia age bias corrections
│   │   ├── dark_matter.py       # DM parameter scans
│   │   └── chain_analysis.py    # MCMC chain analysis utilities
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── data_loader.py       # Data downloading and loading
│   │   ├── plotting.py          # Plotting utilities
│   │   └── logging_utils.py     # Logging configuration
│   ├── desi_y3_comprehensive_analysis.py  # Main analysis script
│   ├── desi_real_chain_analysis.py        # Real chain analysis
│   ├── cpu_dao_sim.py           # CPU-based simulations
│   ├── gpu_dao_sim.py           # GPU-based simulations
│   └── dao_exclusion_plot.py    # DAO exclusion plotting
├── tests/                       # Test suite
│   ├── conftest.py             # Pytest configuration
│   ├── test_cosmology_models.py # Cosmology tests
│   └── test_wedge_analysis.py  # Wedge analysis tests
├── artifacts/                   # Data and results
│   ├── data/                   # Downloaded and generated data
│   └── reports/                # Analysis reports
├── docs/                       # Documentation and plots
├── .github/                    # GitHub Actions CI/CD
│   └── workflows/
│       └── ci.yml              # CI pipeline
├── setup.py                    # Package setup
├── requirements.txt            # Core dependencies
├── requirements-test.txt       # Test dependencies
└── README.md                   # Project documentation
```

## Architectural Principles

### 1. Modularity

**Problem**: Original code had monolithic scripts with duplicate functionality across files.

**Solution**: 
- Separated concerns into distinct modules (cosmology, analysis, utils)
- Each module has a clear, single responsibility
- Shared functionality extracted into reusable libraries
- Clean import structure with `__init__.py` files

**Benefits**:
- Easier testing and maintenance
- Code reuse across different analysis scripts
- Clear dependencies and interfaces

### 2. Type Safety

**Implementation**:
- Type hints using `typing` and `numpy.typing`
- Clear function signatures with return types
- Dataclasses for structured data (e.g., `CosmoParams`)

**Benefits**:
- Better IDE support and autocomplete
- Catches type errors early
- Self-documenting code

### 3. Configuration Management

**Problem**: Hard-coded constants scattered throughout code.

**Solution**:
- Centralized configuration in `utils/config.py`
- YAML-based configuration files
- Environment-specific settings
- Defaults with override capabilities

**Benefits**:
- Easy parameter tuning
- Reproducible analyses
- Clear documentation of analysis choices

### 4. Testing Infrastructure

**Implementation**:
- `pytest` test framework
- Unit tests for critical components
- Coverage reporting
- Continuous integration

**Test Coverage**:
- Cosmology models and calculations
- Wedge analysis algorithms
- Distance computations
- Dark energy parameterizations

### 5. Documentation

**Standards**:
- Google-style docstrings for all functions
- Module-level documentation
- Type hints for parameters
- Usage examples in docstrings

**Files**:
- `ARCHITECTURE.md` - This document
- Module docstrings - Implementation details
- `README.md` - User-facing documentation

## Module Descriptions

### cosmology/

Core cosmological calculations independent of data analysis.

**models.py**: 
- `CosmoParams` - Dataclass for cosmological parameters
- CPL dark energy parameterization: w(a) = w0 + wa(1-a)
- Hubble parameter evolution E(z)
- Classification functions for quintessence vs phantom

**distances.py**:
- Comoving distance calculations
- Luminosity distance for SNe
- Distance modulus
- Angular diameter distance

**power_spectrum.py**:
- Linear matter power spectrum
- BAO wiggle features
- DAO contamination models
- Error estimation for P(k)
- Chi-squared fitting utilities

### analysis/

High-level analysis modules for specific physics questions.

**wedge.py**:
- Canonical quintessence wedge analysis
- Posterior mass calculations
- Bayes factor computation
- Prior volume fraction

**dao.py** (to be implemented):
- DAO stress testing
- Fourier-space P(k) fitting
- Bias estimation on α parameter
- Exclusion limits

**supernova.py** (to be implemented):
- Age bias modeling (Son et al. 2025)
- Distance modulus corrections
- w0 recovery with/without corrections

**dark_matter.py** (to be implemented):
- DM parameter space scans
- Cross-section limits
- Multi-messenger constraints

**chain_analysis.py** (to be implemented):
- MCMC chain loading and processing
- Convergence diagnostics
- Best-fit calculations
- Evidence estimation

### utils/

Common utilities used across the codebase.

**config.py**:
- Configuration dataclass
- YAML serialization/deserialization
- Default values
- Path management

**data_loader.py**:
- DESI chain downloading
- MCMC file parsing
- Caching logic

**plotting.py**:
- Matplotlib style configuration
- Multi-format figure saving
- Consistent plot aesthetics

**logging_utils.py**:
- Structured logging setup
- File and console handlers
- Log level configuration

## Design Patterns

### 1. Dependency Injection

Configuration and dependencies are passed explicitly rather than using global state.

```python
def analyze(data: NDArray, config: Config) -> Result:
    # config is injected, not imported globally
    pass
```

### 2. Dataclasses for Structured Data

```python
@dataclass
class CosmoParams:
    h: float = 0.674
    Om0: float = 0.315
    w0: float = -1.0
    wa: float = 0.0
```

### 3. Factory Pattern for Configuration

```python
config = Config.load(path)  # or Config() for defaults
```

### 4. Strategy Pattern for Distance Calculations

Different distance measures (comoving, luminosity, angular diameter) share common infrastructure but implement specific calculations.

## Code Quality Standards

### Style Guide
- PEP 8 compliance
- Black formatter for consistent formatting
- isort for import organization
- Line length: 100 characters
- Type hints required

### Testing Standards
- Minimum 80% code coverage
- Unit tests for all public functions
- Integration tests for analysis pipelines
- Fixtures for common test data

### Documentation Standards
- Docstrings for all public functions
- Type hints with descriptions
- Example usage in docstrings
- Module-level documentation

## Performance Considerations

### Vectorization
- NumPy operations for array computations
- Avoid Python loops where possible
- Use `scipy.integrate.quad` for integrals

### Caching
- Download once, cache locally
- Memoization for expensive computations
- Lazy loading of large datasets

### GPU Acceleration
- JAX for GPU-accelerated simulations
- Separate CPU/GPU code paths
- Configurable backend selection

## Security Best Practices

### Data Validation
- Input validation in all public functions
- Range checks on physical parameters
- Safe file path handling

### Dependency Management
- Pinned versions in requirements.txt
- Regular security updates
- Minimal dependency footprint

## Future Improvements

### Short Term
1. Complete analysis modules (DAO, SN, DM)
2. Integration tests for full pipeline
3. Performance profiling and optimization
4. Extended documentation with tutorials

### Medium Term
1. Web-based visualization dashboard
2. Automated report generation
3. Parallel processing for chain analysis
4. Docker containerization

### Long Term
1. Interactive Jupyter notebooks
2. API for external access
3. Database backend for results
4. Cloud deployment options

## Migration Guide

For migrating existing scripts to use the new architecture:

1. **Import new modules**:
```python
from src.cosmology.models import CosmoParams
from src.cosmology.distances import dist_mod
from src.analysis.wedge import wedge_analysis
```

2. **Use configuration**:
```python
from src.utils.config import Config
config = Config()
```

3. **Leverage utilities**:
```python
from src.utils.plotting import setup_matplotlib_style, save_figure
setup_matplotlib_style(config.plot_style)
```

4. **Structured data**:
```python
cosmo = CosmoParams(h=0.67, Om0=0.31, w0=-1.0, wa=0.0)
dl = luminosity_distance(z, cosmo)
```

## Conclusion

This architecture provides:
- **Maintainability**: Clear module boundaries and responsibilities
- **Testability**: Isolated components with dependency injection
- **Extensibility**: Easy to add new analyses or models
- **Performance**: Optimized numerical computations
- **Reproducibility**: Configuration management and version control

The modular design enables independent development of components while maintaining a cohesive analysis pipeline.
