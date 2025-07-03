#!/usr/bin/env python3
"""
drugs4disease example

This script demonstrates how to use the drugs4disease package for drug discovery analysis.
Support automatic decompression of ZIP format model files
"""

import os
import sys
import pandas as pd

# add project root directory to Python path, so that the drugs4disease package can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from drugs4disease.full_pipeline import main as run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(disease_id="MONDO:0004979")
