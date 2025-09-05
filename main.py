#!/usr/bin/env python3
"""
Main script to run the complete Facebook comment cleaning and annotation pipeline.

This script orchestrates the entire process:
1. Data cleaning and preprocessing
2. Stratified sampling for annotation dataset creation

Usage:
    python main.py [--config CONFIG_FILE]
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset_cleaner import clean_for_annotation
from src.stratified_sampler import create_annotation_dataset

def main():
    """
    Execute the complete pipeline:
    1. Clean the raw data
    2. Create stratified annotation dataset
    """
    print("🎯 FACEBOOK COMMENT PROCESSING PIPELINE")
    print("=" * 50)
    
    # Step 1: Clean the raw data
    print("\n📋 STEP 1: DATA CLEANING")
    print("-" * 30)
    success = clean_for_annotation()
    
    if not success:
        print("❌ Data cleaning failed. Exiting.")
        return False
    
    # Step 2: Create annotation dataset
    print("\n📋 STEP 2: STRATIFIED SAMPLING")
    print("-" * 30)
    success = create_annotation_dataset()
    
    if success:
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\n📁 Output files:")
        print("   📊 Cleaned data: data/processed/cleaned_comments.csv")
        print("   📝 Annotation dataset: data/annotation/annotation_dataset.csv")
        print("   📦 Archive dataset: data/annotation/unlabeled_archive.csv")
        return True
    else:
        print("\n❌ PIPELINE FAILED!")
        return False

if __name__ == "__main__":
    main()
