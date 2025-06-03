#!/usr/bin/env python3
"""
Development helper script for semantic-clustify.

This script provides convenient commands for development tasks.
Usage: python dev.py [command]
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return success status."""
    if description:
        print(f"🔧 {description}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def install():
    """Install package in development mode."""
    return run_command(
        ["pip", "install", "-e", "."], "Installing package in development mode"
    )


def test():
    """Run all tests."""
    return run_command(["python", "-m", "pytest", "tests/", "-v"], "Running all tests")


def test_quick():
    """Run quick smoke tests."""
    return run_command(
        ["python", "-m", "pytest", "tests/", "-v", "-m", "quick or smoke"],
        "Running quick smoke tests",
    )


def test_coverage():
    """Run tests with coverage."""
    return run_command(
        [
            "python",
            "-m",
            "pytest",
            "tests/",
            "--cov=semantic_clustify",
            "--cov-report=html",
            "--cov-report=term",
        ],
        "Running tests with coverage",
    )


def lint():
    """Run linting."""
    success = True
    success &= run_command(
        ["python", "-m", "flake8", "semantic_clustify/", "tests/"],
        "Running flake8 linting",
    )
    return success


def typecheck():
    """Run type checking."""
    return run_command(
        ["python", "-m", "mypy", "semantic_clustify/"], "Running mypy type checking"
    )


def format_code():
    """Format code with black."""
    return run_command(
        ["python", "-m", "black", "semantic_clustify/", "tests/"],
        "Formatting code with black",
    )


def build():
    """Build distribution packages."""
    return run_command(["python", "-m", "build"], "Building distribution packages")


def clean():
    """Clean build artifacts."""
    import shutil

    print("🧹 Cleaning build artifacts...")

    paths_to_clean = [
        "build/",
        "dist/",
        "htmlcov/",
        "semantic_clustify.egg-info/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".coverage",
    ]

    for path in paths_to_clean:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                shutil.rmtree(path_obj)
                print(f"Removed directory: {path}")
            else:
                path_obj.unlink()
                print(f"Removed file: {path}")

    # Clean __pycache__ directories
    for pycache in Path(".").rglob("__pycache__"):
        shutil.rmtree(pycache)
        print(f"Removed: {pycache}")

    # Clean .pyc files
    for pyc in Path(".").rglob("*.pyc"):
        pyc.unlink()
        print(f"Removed: {pyc}")

    return True


def demo():
    """Run comprehensive demo."""
    return run_command(
        ["python", "tools/demo_comprehensive.py"], "Running comprehensive demo"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Development helper for semantic-clustify"
    )

    parser.add_argument(
        "command",
        choices=[
            "install",
            "test",
            "test-quick",
            "test-coverage",
            "lint",
            "typecheck",
            "format",
            "build",
            "clean",
            "demo",
        ],
        help="Development command to run",
    )

    args = parser.parse_args()

    # Ensure we're in project root
    if not Path("pyproject.toml").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    command_map = {
        "install": install,
        "test": test,
        "test-quick": test_quick,
        "test-coverage": test_coverage,
        "lint": lint,
        "typecheck": typecheck,
        "format": format_code,
        "build": build,
        "clean": clean,
        "demo": demo,
    }

    success = command_map[args.command]()

    if success:
        print("✅ Command completed successfully")
    else:
        print("❌ Command failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
