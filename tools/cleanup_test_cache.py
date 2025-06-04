#!/usr/bin/env python3
"""
Test Cache Cleanup Tool

This tool is used to clean up temporary files and cache data generated during testing.
"""

import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import cleanup_test_files, TEST_CACHE_DIR

def main():
    """Main cleanup function"""
    print("🧹 Starting test environment cleanup...")
    
    # Clean test cache files
    if TEST_CACHE_DIR.exists():
        file_count = len(list(TEST_CACHE_DIR.iterdir()))
        cleanup_test_files()
        print(f"✅ Cleaned {file_count} test cache files")
    else:
        print("📁 Test cache directory does not exist")
    
    # Clean pytest cache
    pytest_cache = Path(__file__).parent.parent / ".pytest_cache"
    if pytest_cache.exists():
        import shutil
        shutil.rmtree(pytest_cache)
        print("✅ Cleaned pytest cache")
    
    # Clean __pycache__ directories
    root_dir = Path(__file__).parent.parent
    pycache_dirs = list(root_dir.rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        import shutil
        shutil.rmtree(pycache_dir)
    
    if pycache_dirs:
        print(f"✅ Cleaned {len(pycache_dirs)} __pycache__ directories")
    
    print("🎉 Test environment cleanup completed!")

if __name__ == "__main__":
    main()
