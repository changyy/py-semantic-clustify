#!/usr/bin/env python3
"""
Test runner for semantic-clustify with various testing options.

This script provides a convenient way to run tests with different configurations
and generate coverage reports.

Usage: Run from the project root directory
    python tools/test_runner.py [options]
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path


def ensure_project_root():
    """Ensure we're running from the project root directory."""
    current_dir = Path.cwd()

    # Check if we're in the tools directory
    if current_dir.name == "tools":
        os.chdir(current_dir.parent)
        print(f"Changed directory to project root: {Path.cwd()}")

    # Verify we're in the right place
    if not (Path.cwd() / "pyproject.toml").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Usage: python tools/test_runner.py")
        sys.exit(1)


def run_command(cmd, description=""):
    """Run a command and handle output."""
    if description:
        print(f"\n🔧 {description}")

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"❌ Command failed with exit code {result.returncode}")
            return False
        else:
            print("✅ Command completed successfully")
            return True

    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Test runner for semantic-clustify")

    parser.add_argument(
        "--quick", action="store_true", help="Run quick smoke tests only"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage report"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--marker",
        "-m",
        type=str,
        help="Run tests with specific marker (e.g., 'core', 'kmeans')",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running tests",
    )

    args = parser.parse_args()

    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print(f"🚀 semantic-clustify Test Runner")
    print(f"Working directory: {project_root}")

    # Install dependencies if requested
    if args.install_deps:
        if not run_command(
            ["pip", "install", "-e", ".[dev]"], "Installing development dependencies"
        ):
            return 1

    # Check if pytest is available
    try:
        subprocess.run(["pytest", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ pytest not found. Installing test dependencies...")
        if not run_command(["pip", "install", "pytest", "pytest-cov"]):
            return 1

    # Build pytest command
    pytest_cmd = ["pytest"]

    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")

    if args.quick:
        pytest_cmd.extend(["-m", "quick or smoke"])
        description = "Running quick smoke tests"
    elif args.marker:
        pytest_cmd.extend(["-m", args.marker])
        description = f"Running tests with marker '{args.marker}'"
    else:
        description = "Running all tests"

    if args.coverage:
        pytest_cmd.extend(
            [
                "--cov=semantic_clustify",
                "--cov-report=term-missing",
                "--cov-report=html",
            ]
        )
        description += " with coverage"

    # Add test directory if it exists
    if Path("tests").exists():
        pytest_cmd.append("tests")
    else:
        print("⚠️  No tests directory found, creating basic test structure...")
        create_basic_tests()
        pytest_cmd.append("tests")

    # Run tests
    success = run_command(pytest_cmd, description)

    if args.coverage and success:
        print(f"\n📊 Coverage report generated in htmlcov/")

    # Run additional checks
    if not args.quick:
        print(f"\n🔍 Running additional checks...")

        # Type checking with mypy (if available)
        try:
            subprocess.run(["mypy", "--version"], capture_output=True, check=True)
            run_command(["mypy", "semantic_clustify"], "Type checking with mypy")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️  mypy not available, skipping type checking")

        # Code formatting check with black (if available)
        try:
            subprocess.run(["black", "--version"], capture_output=True, check=True)
            run_command(
                ["black", "--check", "semantic_clustify"],
                "Code format checking with black",
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️  black not available, skipping format checking")

    return 0 if success else 1


def create_basic_tests():
    """Create basic test structure if tests don't exist."""
    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)

    # Create __init__.py
    (tests_dir / "__init__.py").touch()

    # Create basic test file
    basic_test = tests_dir / "test_basic.py"
    if not basic_test.exists():
        basic_test.write_text(
            '''"""
Basic smoke tests for semantic-clustify.
"""

import pytest
from semantic_clustify import SemanticClusterer


@pytest.mark.smoke
def test_import():
    """Test that we can import the main classes."""
    assert SemanticClusterer is not None


@pytest.mark.smoke 
def test_clusterer_creation():
    """Test basic clusterer creation."""
    clusterer = SemanticClusterer(method="kmeans")
    assert clusterer is not None
    assert clusterer.method == "kmeans"


@pytest.mark.quick
def test_supported_methods():
    """Test that all supported methods are available."""
    expected_methods = ["kmeans", "dbscan", "hierarchical", "gmm"]
    
    for method in expected_methods:
        clusterer = SemanticClusterer(method=method)
        assert clusterer.method == method


@pytest.mark.core
def test_basic_clustering():
    """Test basic clustering functionality."""
    # Sample data with vectors
    data = [
        {"title": "Doc 1", "embedding": [0.1, 0.2, 0.3]},
        {"title": "Doc 2", "embedding": [0.15, 0.25, 0.35]},
        {"title": "Doc 3", "embedding": [0.8, 0.1, 0.2]},
        {"title": "Doc 4", "embedding": [0.85, 0.15, 0.25]},
    ]
    
    clusterer = SemanticClusterer(method="kmeans", n_clusters=2)
    clusters = clusterer.fit_predict(data, vector_field="embedding")
    
    assert isinstance(clusters, list)
    assert len(clusters) <= 2  # Should have at most 2 clusters
    assert sum(len(cluster) for cluster in clusters) == len(data)


if __name__ == "__main__":
    pytest.main([__file__])
'''
        )

    print(f"✅ Created basic test structure in {tests_dir}")


if __name__ == "__main__":
    sys.exit(main())
