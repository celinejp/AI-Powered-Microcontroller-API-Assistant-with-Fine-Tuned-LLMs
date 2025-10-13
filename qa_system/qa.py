#!/usr/bin/env python3
"""
QA Runner - CI Gatekeeper for Microcontroller API Assistant
==========================================================

A simple runner script that executes the QA test suite and acts as a CI gatekeeper.
This script can be run locally to validate that the system meets quality thresholds
before deployment.

Usage:
    python qa.py [options]

Examples:
    python qa.py                           # Run all tests with default settings
    python qa.py --test-suite basic        # Run only basic tests
    python qa.py --verbose                 # Run with verbose logging
    python qa.py --temperature 0.0         # Run with deterministic settings
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from runner import main

if __name__ == "__main__":
    # Run the main function and exit with its return code
    exit_code = main()
    sys.exit(exit_code)
