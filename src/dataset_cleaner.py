import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import emoji
from collections import Counter
import os

class SocialMediaAnnotationCleaner:
    """
    A specialized dataset cleaner for preparing social media comments for sentiment annotation.
    
    Follows exact specifications for Facebook comment cleaning and unique ID generation.
    Input: Excel with columns [ID, TITLE, COMMENT, LIKES, POST URL]
    Output: CSV with columns [unique_comment_id, context_title, comment_to_annotate, LIKES, POST URL]
    """
    
    def __init__(self, input_file_path, output_file_path="cleaned_for_annotation.csv"):
        """
        Initialize the Social Media Annotation Cleaner.
        
        Args:
            input_file_path (str): Path to the input Excel file
            output_file_path (str): Path for the output CSV file (default: cleaned_for_annotation.csv)
        """
        self.input_file_path = Path(input_file_path)
        self.output_file_path = Path(output_file_path)
        self.df = None
        self.original_count = 0
        self.cleaning_log = []
        
    def log_progress(self, message):
        """Log progress messages and store them for final report."""
        print(message)
        self.cleaning_log.append(message)
    
    def load_data(self):
        """
        Phase 1: Initialization and Data Loading
        Load the Excel file and perform initial validation.
        """
        try:
            print("=" * 60)
            print("PHASE 1: INITIALIZATION AND DATA LOADING")
            print("=" * 60)
            
            # Load data with error handling
            self.df = pd.read_excel(self.input_file_path)
            self.original_count = len(self.df)
            
            # Verify required columns exist
            required_columns = ['ID', 'TITLE', 'COMMENT', 'LIKES', 'POST URL']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.log_progress(f"✅ Successfully loaded {self.original_count:,} rows from {self.input_file_path.name}")
            self.log_progress(f"📋 Columns found: {list(self.df.columns)}")
            
            return True
            
        except FileNotFoundError:
            error_msg = f"❌ Error: Input file '{self.input_file_path}' not found."
            print(error_msg)
            print("Please ensure the file exists and try again.")
            return False
            
        except Exception as e:
            error_msg = f"❌ Error loading data: {e}"
            print(error_msg)
            return False
    
    def preprocess_and_generate_ids(self):
        """
        Phase 2: Pre-Processing and Unique ID Generation
        Remove nulls, deduplicate, and generate unique IDs.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: PRE-PROCESSING AND UNIQUE ID GENERATION")
        print("=" * 60)
        
        # Step 1: Sanitize nulls
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['COMMENT'])
        self.df = self.df[self.df['COMMENT'].astype(str).str.strip() != '']
        after_null_removal = len(self.df)
        null_removed = initial_count - after_null_removal
        
        self.log_progress(f"🗑️ Removed {null_removed:,} rows with null/empty comments")
        
        # Step 2: Deduplicate based on COMMENT column
        before_dedup = len(self.df)
        self.df = self.df.drop_duplicates(subset=['COMMENT'], keep='first')
        after_dedup = len(self.df)
        duplicates_removed = before_dedup - after_dedup
        
        self.log_progress(f"🗑️ Removed {duplicates_removed:,} duplicate comments")
        self.log_progress(f"📊 Comments remaining after deduplication: {after_dedup:,}")
        
        # Step 3: Generate unique post ID
        self.df['post_id'] = self.df.groupby('POST URL').ngroup()
        
        # Step 4: Generate unique comment ID
        self.df['unique_comment_id'] = self.df.apply(
            lambda row: f"p{row['post_id']}_c{row['ID']}", axis=1
        )
        
        self.log_progress(f"✅ Successfully created unique IDs for {len(self.df):,} comments")
        self.log_progress(f"📈 Unique posts identified: {self.df['post_id'].nunique():,}")
        
        return True
    
    def clean_text(self, text):
        """
        Single, reusable cleaning function following exact specifications.
        
        Args:
            text: Input string to clean
            
        Returns:
            str: Cleaned text following all specified rules
        """
        # Handle non-string inputs gracefully
        if not isinstance(text, str):
            return ""
        
        if pd.isna(text):
            return ""
        
        # Step 1: Convert emojis to text
        text = emoji.demojize(text, delimiters=(":", ":"))
        
        # Step 2: Remove URLs
        text = re.sub(r'https?://[^\s]+', ' ', text)
        text = re.sub(r'www\.[^\s]+', ' ', text)
        
        # Step 3: Remove user mentions
        text = re.sub(r'@\w+', ' ', text)
        
        # Step 4: Remove HTML entities and tags
        text = re.sub(r'<[^>]+>', ' ', text)  # HTML tags
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # HTML entities like &amp;
        text = re.sub(r'&#\d+;', ' ', text)  # Numeric HTML entities
        
        # Step 5: Normalize hashtags (remove # but keep text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Step 6: Convert to lowercase
        text = text.lower()
        
        # Step 7: Standardize whitespace
        text = re.sub(r'\s+', ' ', text)  # Multiple whitespace to single space
        text = text.strip()  # Remove leading/trailing whitespace
        
        return text
    
    def apply_text_cleaning(self):
        """
        Phase 3: Text Cleaning Logic
        Apply the clean_text function to TITLE and COMMENT columns.
        """
        print("\n" + "=" * 60)
        print("PHASE 3: TEXT CLEANING LOGIC")
        print("=" * 60)
        
        # Apply cleaning function to TITLE column
        self.log_progress("🧽 Cleaning TITLE column...")
        self.df['cleaned_title'] = self.df['TITLE'].apply(self.clean_text)
        
        # Apply cleaning function to COMMENT column
        self.log_progress("🧽 Cleaning COMMENT column...")
        self.df['cleaned_comment'] = self.df['COMMENT'].apply(self.clean_text)
        
        self.log_progress("✅ Text cleaning completed for both TITLE and COMMENT columns")
        
        # Show some examples of cleaning
        print("\n📝 CLEANING EXAMPLES:")
        print("-" * 40)
        sample_data = self.df[['COMMENT', 'cleaned_comment']].head(3)
        for i, (_, row) in enumerate(sample_data.iterrows(), 1):
            original = str(row['COMMENT'])[:100]
            cleaned = str(row['cleaned_comment'])[:100]
            print(f"{i}. Original: {original}...")
            print(f"   Cleaned:  {cleaned}...")
            print()
        
        return True
    
    def final_filtering_and_structuring(self):
        """
        Phase 4: Final Filtering and Structuring
        Remove empty, short, and long comments, handle title-comment duplicates, then structure final output.
        """
        print("\n" + "=" * 60)
        print("PHASE 4: FINAL FILTERING AND STRUCTURING")
        print("=" * 60)
        
        before_filtering = len(self.df)
        
        # Step 1: Filter empty comments
        self.df = self.df[self.df['cleaned_comment'].str.strip() != '']
        after_empty_filter = len(self.df)
        empty_removed = before_filtering - after_empty_filter
        
        self.log_progress(f"🗑️ Removed {empty_removed:,} comments that became empty after cleaning")
        
        # Step 2: Filter short comments (< 3 words)
        self.df = self.df[self.df['cleaned_comment'].str.split().str.len() >= 3]
        after_short_filter = len(self.df)
        short_removed = after_empty_filter - after_short_filter
        
        self.log_progress(f"🗑️ Removed {short_removed:,} comments with fewer than 3 words")
        
        # Step 3: Filter long comments (> 100 words)
        self.df = self.df[self.df['cleaned_comment'].str.split().str.len() <= 100]
        after_long_filter = len(self.df)
        long_removed = after_short_filter - after_long_filter
        
        self.log_progress(f"🗑️ Removed {long_removed:,} comments with more than 100 words")
        
        # Step 4: Handle cases where title equals comment (data quality issue)
        title_equals_comment = self.df['cleaned_title'] == self.df['cleaned_comment']
        problematic_posts_same = self.df[title_equals_comment]['post_id'].nunique()
        
        if problematic_posts_same > 0:
            self.log_progress(f"⚠️ Found {problematic_posts_same:,} posts where title equals comment (data quality issue)")
            
            # For these cases, create a generic title based on post URL
            self.df.loc[title_equals_comment, 'cleaned_title'] = self.df.loc[title_equals_comment, 'POST URL'].apply(
                lambda url: f"facebook post comments - {url.split('/')[-1] if '/' in str(url) else 'unknown post'}"
            )
            
            self.log_progress(f"✅ Fixed {problematic_posts_same:,} posts by generating generic titles")
        
        # Step 5: Handle cases where title is empty/null (missing data issue)
        empty_titles = (self.df['cleaned_title'].str.strip() == '') | (self.df['cleaned_title'].isna())
        problematic_posts_empty = self.df[empty_titles]['post_id'].nunique()
        
        if problematic_posts_empty > 0:
            self.log_progress(f"⚠️ Found {problematic_posts_empty:,} posts with missing/empty titles (data collection issue)")
            
            # For these cases, create a generic title based on post URL
            self.df.loc[empty_titles, 'cleaned_title'] = self.df.loc[empty_titles, 'POST URL'].apply(
                lambda url: f"facebook post comments - {url.split('/')[-1] if '/' in str(url) else 'unknown post'}"
            )
            
            self.log_progress(f"✅ Fixed {problematic_posts_empty:,} posts by generating titles from URLs")
        
        self.log_progress(f"📊 Final comment count after all filters: {after_long_filter:,}")
        
        # Step 6: Select and rename final columns
        final_df = self.df[[
            'unique_comment_id',
            'cleaned_title',
            'cleaned_comment',
            'LIKES',
            'POST URL'
        ]].copy()
        
        # Step 7: Rename columns to final names
        final_df = final_df.rename(columns={
            'cleaned_title': 'context_title',
            'cleaned_comment': 'comment_to_annotate'
        })
        
        self.final_df = final_df
        
        self.log_progress("✅ Final dataset structure completed")
        
        return True
    
    def save_output_file(self):
        """
        Phase 5: File Output
        Save the final cleaned dataset as CSV with UTF-8 encoding and BOM for Excel compatibility.
        """
        print("\n" + "=" * 60)
        print("PHASE 5: FILE OUTPUT")
        print("=" * 60)
        
        try:
            # Save CSV with UTF-8 BOM for better Excel compatibility
            self.final_df.to_csv(
                self.output_file_path,
                index=False,
                encoding='utf-8-sig'  # UTF-8 with BOM for Excel compatibility
            )
            
            self.log_progress(f"✅ CSV file successfully created: {self.output_file_path}")
            self.log_progress(f"📁 File size: {os.path.getsize(self.output_file_path) / 1024:.1f} KB")
            self.log_progress("🔤 Encoding: UTF-8 with BOM (Excel compatible)")
            
            # Also save as Excel file for guaranteed compatibility
            excel_path = self.output_file_path.with_suffix('.xlsx')
            self.final_df.to_excel(excel_path, index=False, engine='openpyxl')
            self.log_progress(f"✅ Excel backup created: {excel_path}")
            
            # Display sample of final output
            print("\n📋 SAMPLE OF FINAL OUTPUT:")
            print("-" * 50)
            print(self.final_df.head().to_string(index=False, max_cols=5))
            
            # Additional encoding note
            print(f"\n💡 ENCODING SOLUTIONS:")
            print(f"📄 CSV file: {self.output_file_path} (UTF-8 with BOM)")
            print(f"📊 Excel file: {excel_path} (No encoding issues)")
            print(f"\nIf you still see encoding issues in CSV:")
            print(f"  • Use the Excel file instead")
            print(f"  • In Excel: Data → Get Data → From Text/CSV → Select UTF-8")
            print(f"  • In VS Code: Bottom right corner → Select encoding → UTF-8")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Error saving output file: {e}"
            print(error_msg)
            return False
    
    def generate_final_report(self):
        """
        Generate comprehensive final report with statistics.
        """
        print("\n" + "=" * 60)
        print("📊 FINAL PROCESSING REPORT")
        print("=" * 60)
        
        retention_rate = (len(self.final_df) / self.original_count) * 100
        
        print(f"📁 Input file: {self.input_file_path.name}")
        print(f"📁 Output file: {self.output_file_path.name}")
        print(f"📅 Processing completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 PROCESSING STATISTICS:")
        print("-" * 30)
        print(f"Original comments: {self.original_count:,}")
        print(f"Final comments: {len(self.final_df):,}")
        print(f"Retention rate: {retention_rate:.1f}%")
        print(f"Unique posts: {self.final_df['context_title'].nunique():,}")
        
        print(f"\n📏 TEXT STATISTICS:")
        print("-" * 20)
        avg_title_length = self.final_df['context_title'].str.len().mean()
        avg_comment_length = self.final_df['comment_to_annotate'].str.len().mean()
        avg_comment_words = self.final_df['comment_to_annotate'].str.split().str.len().mean()
        
        print(f"Average title length: {avg_title_length:.1f} characters")
        print(f"Average comment length: {avg_comment_length:.1f} characters")
        print(f"Average comment words: {avg_comment_words:.1f} words")
        
        print(f"\n🔧 PROCESSING STEPS COMPLETED:")
        print("-" * 35)
        for i, step in enumerate(self.cleaning_log, 1):
            print(f"{i}. {step}")
        
        print(f"\n🎯 READY FOR ANNOTATION!")
        print(f"Your dataset is now cleaned and ready for sentiment annotation.")
        print(f"Output format: CSV with UTF-8 encoding")
        print(f"Final columns: {list(self.final_df.columns)}")


def clean_for_annotation(input_file="data/raw/RawThesisData.xlsx", output_file="data/processed/cleaned_comments.csv"):
    """
    Main function to execute the complete annotation cleaning pipeline.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path for output CSV file
    """
    print("🎯 FACEBOOK COMMENT CLEANING FOR SENTIMENT ANNOTATION (V2)")
    print("📋 Following exact specification requirements + 100-word limit + title fix")
    print("🔄 Processing pipeline: Excel → Cleaned CSV")
    print("=" * 60)
    
    # Initialize cleaner
    cleaner = SocialMediaAnnotationCleaner(input_file, output_file)
    
    # Execute pipeline
    success = True
    
    # Phase 1: Load data
    if not cleaner.load_data():
        return False
    
    # Phase 2: Preprocess and generate IDs
    if success:
        success = cleaner.preprocess_and_generate_ids()
    
    # Phase 3: Apply text cleaning
    if success:
        success = cleaner.apply_text_cleaning()
    
    # Phase 4: Final filtering and structuring
    if success:
        success = cleaner.final_filtering_and_structuring()
    
    # Phase 5: Save output
    if success:
        success = cleaner.save_output_file()
    
    # Generate final report
    if success:
        cleaner.generate_final_report()
        print("\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("\n❌ PROCESSING FAILED!")
        return False


if __name__ == "__main__":
    # Configuration variables (easily modifiable)
    INPUT_FILE = "data/raw/RawThesisData.xlsx"
    OUTPUT_FILE = "data/processed/cleaned_comments.csv"
    
    # Execute cleaning pipeline
    clean_for_annotation(INPUT_FILE, OUTPUT_FILE)
