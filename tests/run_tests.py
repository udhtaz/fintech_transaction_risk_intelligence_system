#!/usr/bin/env python
"""
Test runner for the Fintech Transaction Risk Intelligence System

This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os

# Add parent directory to sys.path to ensure we can import from project
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def run_tests():
    """Discover and run all tests"""
    # Discover all tests in the current directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__))
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())
