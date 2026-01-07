# Development Guide

## Getting Started

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Oichkatzelesfrettschen/desibao.git
cd desibao
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt  # For development
```

4. **Install package in development mode**:
```bash
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_cosmology_models.py

# Run with verbose output
pytest -v
```

### Code Quality

#### Formatting
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/
```

#### Linting
```bash
# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Running Analysis

```bash
# Run main comprehensive analysis
python src/desi_y3_comprehensive_analysis.py

# Run real chain analysis
python src/desi_real_chain_analysis.py

# Generate DAO exclusion plot
python src/dao_exclusion_plot.py
```

## Project Structure

See `ARCHITECTURE.md` for detailed architecture documentation.

## Development Workflow

### 1. Feature Development

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Implement your feature with tests

3. Run tests and linters:
```bash
pytest
black src/ tests/
flake8 src/ tests/
```

4. Commit changes:
```bash
git add .
git commit -m "feat: your feature description"
```

5. Push and create pull request

### 2. Commit Message Convention

Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 3. Code Review Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`black`, `isort`)
- [ ] Linting passes (`flake8`)
- [ ] Type hints added (`mypy`)
- [ ] Docstrings complete
- [ ] Documentation updated
- [ ] No hardcoded paths or values
- [ ] Changelog updated

## Adding New Modules

### 1. Cosmology Module

Example: Adding a new distance calculation

```python
# src/cosmology/distances.py

def your_new_distance(z: Union[float, NDArray], cosmo: CosmoParams) -> Union[float, NDArray]:
    """
    Compute your distance metric.
    
    Args:
        z: Redshift
        cosmo: Cosmological parameters
        
    Returns:
        Distance in appropriate units
    """
    # Implementation
    pass
```

### 2. Analysis Module

Example: Adding a new analysis

```python
# src/analysis/your_analysis.py

def your_analysis_function(data: NDArray, config: Config) -> dict:
    """
    Perform your analysis.
    
    Args:
        data: Input data
        config: Configuration parameters
        
    Returns:
        Dictionary with analysis results
    """
    # Implementation
    return results
```

### 3. Add Tests

```python
# tests/test_your_module.py

import pytest
from src.your_module import your_function

class TestYourFunction:
    def test_basic_case(self):
        """Test basic functionality."""
        result = your_function(input_data)
        assert result == expected
```

### 4. Update Documentation

- Add docstrings to new functions
- Update `ARCHITECTURE.md` if adding major components
- Update `README.md` if user-facing changes

## Configuration

### Creating Custom Config

```python
from src.utils.config import Config
from pathlib import Path

# Create custom configuration
config = Config(
    data_dir=Path("my_data"),
    output_dir=Path("my_output"),
    analysis_params={
        'n_mcmc_samples': 500_000,
        'survey_volume': 15.0,
    }
)

# Save configuration
config.save(Path("my_config.yaml"))

# Load configuration
config = Config.load(Path("my_config.yaml"))
```

## Performance Optimization

### Profiling

```bash
# Profile a script
python -m cProfile -o output.prof script.py

# Analyze profile
python -m pstats output.prof
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory_profiler

# Profile memory
python -m memory_profiler script.py
```

### GPU Acceleration

For JAX-based simulations:

```python
# Force CPU
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Force GPU (if available)
os.environ["JAX_PLATFORM_NAME"] = "gpu"
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Ensure package is installed in development mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Test Failures

```bash
# Run tests with verbose output
pytest -v -s

# Run specific test
pytest tests/test_file.py::TestClass::test_method -v
```

### Dependency Issues

```bash
# Recreate virtual environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## CI/CD

The project uses GitHub Actions for continuous integration:

- Runs on every push and pull request
- Tests multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Checks code formatting and linting
- Generates coverage reports

## Resources

- **NumPy Documentation**: https://numpy.org/doc/
- **SciPy Documentation**: https://docs.scipy.org/
- **pytest Documentation**: https://docs.pytest.org/
- **Black Formatter**: https://black.readthedocs.io/
- **Type Hints**: https://docs.python.org/3/library/typing.html

## Getting Help

- Review `ARCHITECTURE.md` for design details
- Check existing tests for usage examples
- Open an issue on GitHub for bugs
- Consult with team members for design questions
