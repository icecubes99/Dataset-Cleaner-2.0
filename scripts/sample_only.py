#!/usr/bin/env python3
"""
Sample only script - runs just the stratified sampling process.

Usage:
    python scripts/sample_only.py [input_file]
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from stratified_sampler import create_annotation_dataset

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed/cleaned_comments.csv"
    create_annotation_dataset(input_file)
