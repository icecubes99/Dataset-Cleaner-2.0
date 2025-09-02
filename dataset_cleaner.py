import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import emoji
from collections import Counter

import pandas as pd
import numpy as np
from pathlib import Path
import re
import emoji
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
        Remove empty and short comments, then structure final output.
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
        self.log_progress(f"📊 Final comment count: {after_short_filter:,}")
        
        # Step 3: Select and rename final columns
        final_df = self.df[[
            'unique_comment_id',
            'cleaned_title',
            'cleaned_comment',
            'LIKES',
            'POST URL'
        ]].copy()
        
        # Step 4: Rename columns to final names
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


def clean_for_annotation(input_file="RawThesisData.xlsx", output_file="cleaned_for_annotation.csv"):
    """
    Main function to execute the complete annotation cleaning pipeline.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path for output CSV file
    """
    print("🎯 FACEBOOK COMMENT CLEANING FOR SENTIMENT ANNOTATION")
    print("📋 Following exact specification requirements")
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
    INPUT_FILE = "RawThesisData.xlsx"
    OUTPUT_FILE = "cleaned_for_annotation.csv"
    
    # Execute cleaning pipeline
    clean_for_annotation(INPUT_FILE, OUTPUT_FILE)
    
    def log_action(self, action):
        """Log cleaning actions for tracking."""
        self.cleaning_log.append(action)
    
    def get_social_media_overview(self):
        """
        Get a comprehensive overview focused on social media text characteristics.
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("=" * 60)
        print("📱 SOCIAL MEDIA DATASET OVERVIEW")
        print("=" * 60)
        
        # Basic info
        print(f"Total comments: {len(self.df):,}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Text column: '{self.text_column}'")
        
        # Text-specific analysis
        text_data = self.df[self.text_column].astype(str)
        
        print(f"\n📊 TEXT CHARACTERISTICS:")
        print("-" * 40)
        print(f"Average comment length: {text_data.str.len().mean():.1f} characters")
        print(f"Median comment length: {text_data.str.len().median():.1f} characters")
        print(f"Shortest comment: {text_data.str.len().min()} characters")
        print(f"Longest comment: {text_data.str.len().max()} characters")
        
        # Empty/null analysis
        null_count = self.df[self.text_column].isnull().sum()
        empty_count = (text_data.str.strip() == '').sum()
        very_short = (text_data.str.len() < 10).sum()
        
        print(f"\n🔍 DATA QUALITY:")
        print("-" * 40)
        print(f"Null comments: {null_count} ({null_count/len(self.df)*100:.1f}%)")
        print(f"Empty comments: {empty_count} ({empty_count/len(self.df)*100:.1f}%)")
        print(f"Very short (<10 chars): {very_short} ({very_short/len(self.df)*100:.1f}%)")
        
        # Emoji analysis
        emoji_count = text_data.apply(lambda x: len(re.findall(r'[^\w\s,.\-!?]', x))).sum()
        comments_with_emojis = text_data.apply(lambda x: bool(re.search(r'[^\w\s,.\-!?]', x))).sum()
        
        print(f"\n😊 EMOJI ANALYSIS:")
        print("-" * 40)
        print(f"Total emojis/special chars: {emoji_count:,}")
        print(f"Comments with emojis: {comments_with_emojis} ({comments_with_emojis/len(self.df)*100:.1f}%)")
        
        # URL analysis
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        comments_with_urls = text_data.str.contains(url_pattern, regex=True, na=False).sum()
        
        # Mention analysis
        mention_pattern = r'@\w+'
        comments_with_mentions = text_data.str.contains(mention_pattern, regex=True, na=False).sum()
        
        # Hashtag analysis
        hashtag_pattern = r'#\w+'
        comments_with_hashtags = text_data.str.contains(hashtag_pattern, regex=True, na=False).sum()
        
        print(f"\n🔗 SOCIAL MEDIA ELEMENTS:")
        print("-" * 40)
        print(f"Comments with URLs: {comments_with_urls} ({comments_with_urls/len(self.df)*100:.1f}%)")
        print(f"Comments with @mentions: {comments_with_mentions} ({comments_with_mentions/len(self.df)*100:.1f}%)")
        print(f"Comments with #hashtags: {comments_with_hashtags} ({comments_with_hashtags/len(self.df)*100:.1f}%)")
        
        # Language characteristics (basic detection)
        english_words = text_data.str.count(r'\b(?:the|and|to|of|a|in|for|is|on|that|by|this|with|i|you|it|not|or|be|are|from|at|as|your|all|have|new|more|an|was|we|will|home|can|us|about|if|page|my|has|search|free|but|our|one|other|do|no|information|time|they|site|he|up|may|what|which|their|news|out|use|any|there|see|only|so|his|when|contact|here|business|who|web|also|now|help|get|pm|view|online|c|e|first|am|been|would|how|were|me|s|services|some|these|click|its|like|service|x|than|find)\b', flags=re.IGNORECASE)
        tagalog_words = text_data.str.count(r'\b(?:ang|ng|sa|na|ay|mga|at|para|kung|ako|ka|mo|ko|siya|tayo|kayo|sila|ito|iyan|yun|dito|dyan|doon|ano|sino|saan|kailan|bakit|paano|hindi|oo|opo|po|ho|kasi|pero|tapos|yung|lang|naman|din|rin|pala|kaya|siguro|baka|dapat|pwede|gusto|ayaw|mahal|ganda|pangit|mabait|masama)\b', flags=re.IGNORECASE)
        
        print(f"\n🌏 LANGUAGE INDICATORS:")
        print("-" * 40)
        print(f"Avg English words per comment: {english_words.mean():.1f}")
        print(f"Avg Tagalog words per comment: {tagalog_words.mean():.1f}")
        
        # Sample comments
        print(f"\n📝 SAMPLE COMMENTS:")
        print("-" * 40)
        sample_comments = self.df[self.text_column].dropna().head(3)
        for i, comment in enumerate(sample_comments, 1):
            print(f"{i}. {str(comment)[:100]}{'...' if len(str(comment)) > 100 else ''}")
    
    def remove_exact_duplicates(self):
        """
        Remove exact duplicate comments (spam/bot detection).
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        initial_count = len(self.df)
        
        # Remove exact duplicates based on text content
        self.df = self.df.drop_duplicates(subset=[self.text_column], keep='first')
        
        removed_count = initial_count - len(self.df)
        
        self.log_action(f"Removed {removed_count} exact duplicate comments")
        print(f"🗑️ Removed {removed_count} exact duplicate comments ({removed_count/initial_count*100:.1f}%)")
        
        return removed_count
    
    def convert_emojis_to_text(self):
        """
        Convert emojis to text representation to preserve sentiment signals.
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        def emoji_to_text(text):
            """Convert emojis in text to their text description."""
            if pd.isna(text):
                return text
            # Convert emojis to text (e.g., 😂 -> :face_with_tears_of_joy:)
            return emoji.demojize(str(text), delimiters=(":", ":"))
        
        # Count emojis before conversion
        emoji_pattern = r'[^\w\s,.\-!?(){}[\]"\'`~@#$%^&*+=|\\<>/]'
        before_count = self.df[self.text_column].astype(str).str.count(emoji_pattern).sum()
        
        # Apply emoji conversion
        self.df[self.text_column] = self.df[self.text_column].apply(emoji_to_text)
        
        # Count after conversion (should be much less)
        after_count = self.df[self.text_column].astype(str).str.count(emoji_pattern).sum()
        converted_count = before_count - after_count
        
        self.log_action(f"Converted ~{converted_count} emojis to text representation")
        print(f"😊➡️📝 Converted ~{converted_count} emojis to text representation")
        
        return converted_count
    
    def remove_non_informative_elements(self):
        """
        Remove URLs, mentions, and HTML tags while preserving meaningful content.
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        def clean_text(text):
            if pd.isna(text):
                return text
            
            text = str(text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
            
            # Remove user mentions (@username)
            text = re.sub(r'@\w+', ' ', text)
            
            # Remove HTML tags and entities
            text = re.sub(r'<[^>]+>', ' ', text)  # HTML tags
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # HTML entities like &amp;
            text = re.sub(r'&#\d+;', ' ', text)  # Numeric HTML entities
            
            return text
        
        # Count elements before removal
        url_count = self.df[self.text_column].astype(str).str.count(r'http[s]?://').sum()
        mention_count = self.df[self.text_column].astype(str).str.count(r'@\w+').sum()
        html_count = self.df[self.text_column].astype(str).str.count(r'<[^>]+>').sum()
        
        # Apply cleaning
        self.df[self.text_column] = self.df[self.text_column].apply(clean_text)
        
        total_removed = url_count + mention_count + html_count
        self.log_action(f"Removed {url_count} URLs, {mention_count} mentions, {html_count} HTML tags")
        print(f"🗑️ Removed non-informative elements:")
        print(f"   📎 URLs: {url_count}")
        print(f"   👤 Mentions: {mention_count}")
        print(f"   🏷️ HTML tags: {html_count}")
        
        return total_removed
    
    def normalize_text(self):
        """
        Normalize text: lowercase, standardize whitespace, handle hashtags.
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        def normalize_single_text(text):
            if pd.isna(text):
                return text
            
            text = str(text)
            
            # Convert to lowercase (standard for NLP)
            text = text.lower()
            
            # Handle hashtags: remove # but keep the text
            text = re.sub(r'#(\w+)', r'\1', text)
            
            # Standardize whitespace: replace multiple spaces, newlines, tabs with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            return text
        
        # Count hashtags before processing
        hashtag_count = self.df[self.text_column].astype(str).str.count(r'#\w+').sum()
        
        # Apply normalization
        self.df[self.text_column] = self.df[self.text_column].apply(normalize_single_text)
        
        self.log_action(f"Normalized text: lowercase, whitespace standardization, processed {hashtag_count} hashtags")
        print(f"🔄 Text normalization completed:")
        print(f"   🔤 Converted to lowercase")
        print(f"   📏 Standardized whitespace")
        print(f"   #️⃣ Processed {hashtag_count} hashtags (removed # symbols)")
        
        return hashtag_count
    
    def filter_meaningful_comments(self, min_words=3, min_chars=10):
        """
        Remove empty comments and those too short to be meaningful.
        
        Args:
            min_words (int): Minimum number of words required
            min_chars (int): Minimum number of characters required
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        initial_count = len(self.df)
        
        # Remove null/empty comments
        self.df = self.df.dropna(subset=[self.text_column])
        after_null = len(self.df)
        
        # Remove comments that are too short by character count
        self.df = self.df[self.df[self.text_column].astype(str).str.len() >= min_chars]
        after_chars = len(self.df)
        
        # Remove comments that are too short by word count
        self.df = self.df[self.df[self.text_column].astype(str).str.split().str.len() >= min_words]
        final_count = len(self.df)
        
        null_removed = initial_count - after_null
        chars_removed = after_null - after_chars
        words_removed = after_chars - final_count
        total_removed = initial_count - final_count
        
        self.log_action(f"Filtered out {total_removed} non-meaningful comments")
        print(f"🔍 Filtered out non-meaningful comments:")
        print(f"   🚫 Null/empty: {null_removed}")
        print(f"   📏 Too short (<{min_chars} chars): {chars_removed}")
        print(f"   📝 Too few words (<{min_words} words): {words_removed}")
        print(f"   📊 Total removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
        
        return total_removed
    
    def analyze_cleaning_impact(self):
        """
        Analyze the impact of cleaning on text characteristics.
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        print("=" * 60)
        print("📊 CLEANING IMPACT ANALYSIS")
        print("=" * 60)
        
        # Current statistics
        text_data = self.df[self.text_column].astype(str)
        
        print(f"Final dataset size: {len(self.df):,} comments")
        print(f"Retention rate: {len(self.df)/self.original_shape[0]*100:.1f}%")
        
        print(f"\n📏 TEXT LENGTH DISTRIBUTION:")
        print("-" * 40)
        lengths = text_data.str.len()
        print(f"Mean: {lengths.mean():.1f} characters")
        print(f"Median: {lengths.median():.1f} characters")
        print(f"Std Dev: {lengths.std():.1f} characters")
        print(f"Min: {lengths.min()} characters")
        print(f"Max: {lengths.max()} characters")
        
        # Word count distribution
        word_counts = text_data.str.split().str.len()
        print(f"\n📝 WORD COUNT DISTRIBUTION:")
        print("-" * 40)
        print(f"Mean: {word_counts.mean():.1f} words")
        print(f"Median: {word_counts.median():.1f} words")
        print(f"Min: {word_counts.min()} words")
        print(f"Max: {word_counts.max()} words")
        
        # Sample cleaned comments
        print(f"\n✨ SAMPLE CLEANED COMMENTS:")
        print("-" * 40)
        sample_cleaned = text_data.head(5)
        for i, comment in enumerate(sample_cleaned, 1):
            print(f"{i}. {comment[:150]}{'...' if len(comment) > 150 else ''}")
    
    def create_sentiment_cleaning_report(self):
        """
        Create a comprehensive report of all sentiment-preserving cleaning actions.
        """
        print("=" * 60)
        print("📋 SENTIMENT-PRESERVING CLEANING REPORT")
        print("=" * 60)
        
        print(f"🗂️ Dataset: {self.file_path.name}")
        print(f"📅 Processed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 SIZE CHANGES:")
        print("-" * 30)
        print(f"Original: {self.original_shape[0]:,} comments")
        print(f"Final: {len(self.df):,} comments")
        print(f"Retained: {len(self.df)/self.original_shape[0]*100:.1f}%")
        
        print(f"\n🔧 CLEANING ACTIONS PERFORMED:")
        print("-" * 35)
        for i, action in enumerate(self.cleaning_log, 1):
            print(f"{i}. {action}")
        
        print(f"\n✅ WHAT WAS PRESERVED (Following Best Practices):")
        print("-" * 50)
        print("• Slang and informal language (dpt, aq, etc.)")
        print("• Taglish words and code-switching")
        print("• Sentiment-indicating repeated letters (grrrr, hahaha)")
        print("• Stopwords that provide context in Taglish")
        print("• Original word forms (no stemming/lemmatization)")
        print("• Emoji sentiment converted to text descriptions")
        
        print(f"\n🚫 WHAT WAS REMOVED (Noise Only):")
        print("-" * 35)
        print("• Exact duplicate comments (spam/bots)")
        print("• URLs and web links")
        print("• User mentions (@username)")
        print("• HTML tags and entities")
        print("• Excessive whitespace")
        print("• Empty or meaningless comments")
    
    def save_cleaned_data(self, output_path=None, include_stats=True):
        """
        Save the cleaned dataset optimized for sentiment analysis.
        
        Args:
            output_path (str): Path to save the file
            include_stats (bool): Whether to include cleaning statistics
        """
        if self.df is None:
            print("❌ No data to save.")
            return
        
        if output_path is None:
            stem = self.file_path.stem
            output_path = self.file_path.parent / f"{stem}_sentiment_cleaned.xlsx"
        
        try:
            # Save the main cleaned data
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                
                if include_stats:
                    # Create statistics sheet
                    stats_data = {
                        'Metric': [
                            'Original Comments',
                            'Final Comments', 
                            'Retention Rate (%)',
                            'Average Length (chars)',
                            'Average Words',
                            'Text Column'
                        ],
                        'Value': [
                            self.original_shape[0],
                            len(self.df),
                            f"{len(self.df)/self.original_shape[0]*100:.1f}%",
                            f"{self.df[self.text_column].astype(str).str.len().mean():.1f}",
                            f"{self.df[self.text_column].astype(str).str.split().str.len().mean():.1f}",
                            self.text_column
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Cleaning_Stats', index=False)
                    
                    # Create cleaning log sheet
                    log_df = pd.DataFrame({'Cleaning_Actions': self.cleaning_log})
                    log_df.to_excel(writer, sheet_name='Cleaning_Log', index=False)
            
            print(f"✅ Sentiment-ready dataset saved to: {output_path}")
            self.log_action(f"Data saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")


if __name__ == "__main__":
    # Configuration variables (easily modifiable)
    INPUT_FILE = "RawThesisData.xlsx"
    OUTPUT_FILE = "cleaned_for_annotation.csv"
    
    # Execute cleaning pipeline
    clean_for_annotation(INPUT_FILE, OUTPUT_FILE)
