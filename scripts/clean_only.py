#!/usr/bin/env python3
"""
Clean only script - runs just the data cleaning process.

Usage:
    python scripts/clean_only.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_cleaner import clean_for_annotation

if __name__ == "__main__":
    clean_for_annotation()
