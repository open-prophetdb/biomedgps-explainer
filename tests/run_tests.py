#!/usr/bin/env python3
"""
Test runner script for the biomedgps-explainer project.
Run all tests or specific test modules.
"""

import sys
import os
import unittest
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests(module_name=None, verbose=False):
    """
    Run tests for the specified module or all tests.
    
    Args:
        module_name (str, optional): Name of the test module to run (without .py)
        verbose (bool): Whether to run tests in verbose mode
    """
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    if module_name:
        # Run specific test module
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
        
        print(f"Running tests for module: {module_name}")
        
        # Import and run the specific test module
        try:
            test_module = __import__(f'tests.{module_name}', fromlist=['*'])
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
            
            if result.wasSuccessful():
                print("✅ All tests passed!")
                return True
            else:
                print("❌ Some tests failed!")
                return False
                
        except ImportError as e:
            print(f"❌ Error importing test module '{module_name}': {e}")
            return False
    else:
        # Run all tests
        print("Running all tests...")
        
        # Discover and run all tests
        loader = unittest.TestLoader()
        suite = loader.discover(str(tests_dir), pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            return False

def list_test_modules():
    """List all available test modules."""
    tests_dir = Path(__file__).parent
    test_files = list(tests_dir.glob('test_*.py'))
    
    print("Available test modules:")
    for test_file in test_files:
        module_name = test_file.stem
        print(f"  - {module_name}")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Run tests for biomedgps-explainer')
    parser.add_argument('module', nargs='?', help='Test module to run (without .py extension)')
    parser.add_argument('--list', action='store_true', help='List all available test modules')
    parser.add_argument('--verbose', '-v', action='store_true', help='Run tests in verbose mode')
    
    args = parser.parse_args()
    
    if args.list:
        list_test_modules()
        return
    
    success = run_tests(args.module, args.verbose)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 