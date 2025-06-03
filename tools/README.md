# Development Tools

This directory contains development and testing tools for the semantic-clustify project.

## Available Tools

### `dev_helper.py` ⭐ **Recommended**
Main development helper script that provides common development tasks in a Python-native way:

**Usage:**
```bash
# Run all quality checks
python tools/dev_helper.py check

# Run tests with coverage  
python tools/dev_helper.py test

# Format code
python tools/dev_helper.py format

# Lint code
python tools/dev_helper.py lint

# Build package
python tools/dev_helper.py build

# Clean build artifacts
python tools/dev_helper.py clean

# Install in development mode
python tools/dev_helper.py install

# Show version
python tools/dev_helper.py version
```

**Enhanced Clean Function:**
The `clean` command now removes temporary CLI output files generated during testing, including:
- Auto-generated clustering output files (e.g., `*_kmeans_grouped_*.jsonl`)
- Temporary test files (e.g., `tmp*.jsonl`)
- Build artifacts and cache files

### Automatic Test Cleanup
Tests now automatically clean up temporary output files through:
1. **Smart test detection**: CLI uses temp directories when running under pytest
2. **Fixture-based cleanup**: Auto-cleanup of generated files after each test
3. **Enhanced clean patterns**: Comprehensive removal of all temporary file types

### `demo_comprehensive.py`
Comprehensive demonstration script that showcases all major features of semantic-clustify, including:
- Different clustering algorithms (KMeans, DBSCAN, Hierarchical, GMM)
- Parameter optimization
- Quality metrics calculation
- Sample data generation

**Usage:**
```bash
python tools/demo_comprehensive.py
```

### `test_runner.py`
Advanced test runner with various testing options and coverage reports.

**Usage:**
```bash
# Run from project root
python tools/test_runner.py [options]

# Options:
--quick          # Run quick smoke tests only
--coverage       # Run tests with coverage report
--verbose, -v    # Verbose output
--parallel, -p   # Run tests in parallel
--benchmark      # Run performance benchmarks
```

### `dev.py`
Legacy development script (kept for backward compatibility).

## Development Workflow

**Recommended workflow using dev_helper.py:**

1. **Initial Setup**: `python tools/dev_helper.py install`
2. **During Development**: `python tools/dev_helper.py check` (runs format, lint, test)
3. **Before Commit**: `python tools/dev_helper.py format && python tools/dev_helper.py test`
4. **Build Package**: `python tools/dev_helper.py build`
5. **Demonstration**: Run `python tools/demo_comprehensive.py` to see all features in action

## Testing Options

# Run with coverage
python tools/test_runner.py --coverage

# Run specific test types
python tools/test_runner.py --quick
python tools/test_runner.py --integration

### `demo_comprehensive.py`
Comprehensive demonstration of all semantic-clustify features.

```bash
# Run the demo
python tools/demo_comprehensive.py
```

## Usage

All tools should be run from the project root directory for proper path resolution.
