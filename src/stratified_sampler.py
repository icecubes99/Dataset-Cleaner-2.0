import pandas as pd
import numpy as np
from pathlib import Path
import os

class StratifiedCommentSampler:
    """
    Stratified random sampling for creating representative annotation datasets.
    
    Uses post-level stratification to ensure proportional representation
    across all Facebook posts in the final annotation dataset.
    """
    
    def __init__(self, input_file="data/processed/cleaned_comments.csv", 
                 target_annotation_size=10000, random_state=42):
        """
        Initialize the Stratified Comment Sampler.
        
        Args:
            input_file (str): Path to the cleaned comments CSV file
            target_annotation_size (int): Target number of comments for annotation (default: 10,000)
            random_state (int): Random seed for reproducible sampling (default: 42)
        """
        self.input_file = Path(input_file)
        self.target_annotation_size = target_annotation_size
        self.random_state = random_state
        
        # Output file names
        self.annotation_output_file = "data/annotation/annotation_dataset.csv"
        self.archive_output_file = "data/annotation/unlabeled_archive.csv"
        
        # Data containers
        self.df = None
        self.annotation_df = None
        self.archive_df = None
        
        # Statistics
        self.total_cleaned_size = 0
        self.sampling_fraction = 0
        self.processing_log = []
        
    def log_progress(self, message):
        """Log progress messages and store them for final report."""
        print(message)
        self.processing_log.append(message)
    
    def load_and_initialize(self):
        """
        Phase 1: Initialization and Calculation
        Load data and calculate sampling parameters.
        """
        print("=" * 60)
        print("PHASE 1: INITIALIZATION AND CALCULATION")
        print("=" * 60)
        
        try:
            # Load data
            self.df = pd.read_csv(self.input_file, encoding='utf-8')
            self.total_cleaned_size = len(self.df)
            
            # Verify required columns exist
            required_columns = ['unique_comment_id', 'context_title', 'comment_to_annotate', 'LIKES', 'POST URL']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Calculate sampling fraction
            self.sampling_fraction = self.target_annotation_size / self.total_cleaned_size
            
            self.log_progress(f"✅ Successfully loaded {self.total_cleaned_size:,} comments from {self.input_file.name}")
            self.log_progress(f"📊 Target annotation size: {self.target_annotation_size:,} comments")
            self.log_progress(f"📈 Sampling fraction: {self.sampling_fraction:.3f} ({self.sampling_fraction*100:.1f}%)")
            self.log_progress(f"🎲 Random state: {self.random_state} (for reproducibility)")
            
            # Display output file names
            print(f"\n📁 OUTPUT FILES:")
            print(f"   📝 Annotation dataset: {self.annotation_output_file}")
            print(f"   📦 Archive dataset: {self.archive_output_file}")
            
            return True
            
        except FileNotFoundError:
            error_msg = f"❌ Error: Input file '{self.input_file}' not found."
            print(error_msg)
            print("Please ensure the cleaned annotation file exists.")
            return False
            
        except Exception as e:
            error_msg = f"❌ Error loading data: {e}"
            print(error_msg)
            return False
    
    def perform_stratified_sampling(self):
        """
        Phase 2: Stratified Sampling
        Perform stratified random sampling by post.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: STRATIFIED SAMPLING")
        print("=" * 60)
        
        # Step 1: Identify strata (posts)
        self.log_progress("🎯 Identifying strata using POST URL...")
        
        # Generate post_id for easier grouping
        self.df['post_id'] = self.df.groupby('POST URL').ngroup()
        unique_posts = self.df['post_id'].nunique()
        
        self.log_progress(f"📊 Found {unique_posts:,} unique posts in dataset")
        
        # Step 2: Analyze post distribution
        post_sizes = self.df.groupby('post_id').size()
        self.log_progress(f"📈 Post size statistics:")
        self.log_progress(f"   • Average comments per post: {post_sizes.mean():.1f}")
        self.log_progress(f"   • Median comments per post: {post_sizes.median():.1f}")
        self.log_progress(f"   • Largest post: {post_sizes.max():,} comments")
        self.log_progress(f"   • Smallest post: {post_sizes.min():,} comments")
        
        # Step 3: Perform stratified sampling
        self.log_progress(f"\n🎲 Performing stratified random sampling...")
        self.log_progress(f"   • Sampling {self.sampling_fraction:.1%} from each post")
        self.log_progress(f"   • Using random_state={self.random_state} for reproducibility")
        
        # Apply stratified sampling
        def sample_from_group(group):
            """Sample a fraction of rows from each group (post)."""
            n_to_sample = max(1, int(len(group) * self.sampling_fraction))
            return group.sample(n=min(n_to_sample, len(group)), random_state=self.random_state)
        
        # Group by post_id and sample from each group
        self.annotation_df = self.df.groupby('post_id', group_keys=False).apply(sample_from_group).reset_index(drop=True)
        
        actual_annotation_size = len(self.annotation_df)
        self.log_progress(f"✅ Stratified sampling completed!")
        self.log_progress(f"📊 Selected {actual_annotation_size:,} comments for annotation")
        
        # Check how close we got to target
        target_diff = abs(actual_annotation_size - self.target_annotation_size)
        percentage_diff = (target_diff / self.target_annotation_size) * 100
        
        if percentage_diff <= 5:  # Within 5% is good
            self.log_progress(f"🎯 Target achieved! ({target_diff:,} comments difference, {percentage_diff:.1f}%)")
        else:
            self.log_progress(f"⚠️  Target variance: {target_diff:,} comments ({percentage_diff:.1f}% difference)")
        
        # Verify representation
        annotation_posts = self.annotation_df['post_id'].nunique()
        self.log_progress(f"🏛️ Representation: {annotation_posts:,} of {unique_posts:,} posts included ({annotation_posts/unique_posts*100:.1f}%)")
        
        return True
    
    def split_and_create_archive(self):
        """
        Phase 3: Splitting the Data
        Create the archive dataset with remaining comments.
        """
        print("\n" + "=" * 60)
        print("PHASE 3: SPLITTING THE DATA")
        print("=" * 60)
        
        # Get IDs of comments selected for annotation
        annotation_ids = set(self.annotation_df['unique_comment_id'])
        
        # Create archive dataset (comments NOT in annotation set)
        self.archive_df = self.df[~self.df['unique_comment_id'].isin(annotation_ids)].copy()
        
        # Remove the temporary post_id column from both datasets
        self.annotation_df = self.annotation_df.drop('post_id', axis=1)
        self.archive_df = self.archive_df.drop('post_id', axis=1)
        
        self.log_progress(f"📝 Annotation dataset: {len(self.annotation_df):,} comments")
        self.log_progress(f"📦 Archive dataset: {len(self.archive_df):,} comments")
        self.log_progress(f"✅ Total accounted for: {len(self.annotation_df) + len(self.archive_df):,} comments")
        
        return True
    
    def verify_and_save(self):
        """
        Phase 4: Verification and Saving
        Verify the split and save both datasets.
        """
        print("\n" + "=" * 60)
        print("PHASE 4: VERIFICATION AND SAVING")
        print("=" * 60)
        
        # Critical verification
        total_accounted = len(self.annotation_df) + len(self.archive_df)
        
        try:
            assert total_accounted == self.total_cleaned_size, \
                f"Data loss detected! Expected {self.total_cleaned_size}, got {total_accounted}"
            self.log_progress("✅ Data integrity verified: No comments lost during splitting")
        except AssertionError as e:
            print(f"❌ CRITICAL ERROR: {e}")
            return False
        
        # Verify no overlap
        annotation_ids = set(self.annotation_df['unique_comment_id'])
        archive_ids = set(self.archive_df['unique_comment_id'])
        overlap = annotation_ids.intersection(archive_ids)
        
        try:
            assert len(overlap) == 0, f"Overlap detected: {len(overlap)} comments in both datasets"
            self.log_progress("✅ No overlap verified: Each comment is in exactly one dataset")
        except AssertionError as e:
            print(f"❌ CRITICAL ERROR: {e}")
            return False
        
        # Save annotation dataset
        try:
            self.annotation_df.to_csv(self.annotation_output_file, index=False, encoding='utf-8-sig')
            annotation_size_kb = os.path.getsize(self.annotation_output_file) / 1024
            self.log_progress(f"📝 Annotation dataset saved: {self.annotation_output_file}")
            self.log_progress(f"   • Size: {annotation_size_kb:.1f} KB")
            self.log_progress(f"   • Comments: {len(self.annotation_df):,}")
        except Exception as e:
            print(f"❌ Error saving annotation dataset: {e}")
            return False
        
        # Save archive dataset
        try:
            self.archive_df.to_csv(self.archive_output_file, index=False, encoding='utf-8-sig')
            archive_size_kb = os.path.getsize(self.archive_output_file) / 1024
            self.log_progress(f"📦 Archive dataset saved: {self.archive_output_file}")
            self.log_progress(f"   • Size: {archive_size_kb:.1f} KB")
            self.log_progress(f"   • Comments: {len(self.archive_df):,}")
        except Exception as e:
            print(f"❌ Error saving archive dataset: {e}")
            return False
        
        return True
    
    def analyze_representation(self):
        """
        Analyze how well the annotation dataset represents the full dataset.
        """
        print("\n" + "=" * 60)
        print("📊 REPRESENTATION ANALYSIS")
        print("=" * 60)
        
        # Re-add post_id for analysis
        self.df['post_id'] = self.df.groupby('POST URL').ngroup()
        self.annotation_df['post_id'] = self.annotation_df.merge(
            self.df[['unique_comment_id', 'post_id']], 
            on='unique_comment_id', 
            how='left'
        )['post_id']
        
        # Post representation analysis
        original_post_dist = self.df.groupby('post_id').size()
        annotation_post_dist = self.annotation_df.groupby('post_id').size()
        
        # Calculate sampling rates per post
        sampling_rates = annotation_post_dist / original_post_dist
        
        print(f"📈 SAMPLING RATE ANALYSIS:")
        print(f"   • Target sampling rate: {self.sampling_fraction:.1%}")
        print(f"   • Actual average rate: {sampling_rates.mean():.1%}")
        print(f"   • Rate standard deviation: {sampling_rates.std():.1%}")
        print(f"   • Minimum rate: {sampling_rates.min():.1%}")
        print(f"   • Maximum rate: {sampling_rates.max():.1%}")
        
        # Show examples
        print(f"\n📋 SAMPLE POST REPRESENTATION:")
        print("-" * 40)
        sample_posts = original_post_dist.head(5)
        for post_id in sample_posts.index:
            original_count = original_post_dist[post_id]
            annotation_count = annotation_post_dist.get(post_id, 0)
            rate = annotation_count / original_count if original_count > 0 else 0
            print(f"Post {post_id}: {original_count:3d} → {annotation_count:3d} comments ({rate:.1%})")
        
        # Clean up temporary post_id columns
        self.annotation_df = self.annotation_df.drop('post_id', axis=1)
        self.df = self.df.drop('post_id', axis=1)
    
    def generate_final_report(self):
        """
        Generate comprehensive final report.
        """
        print("\n" + "=" * 60)
        print("📋 FINAL STRATIFIED SAMPLING REPORT")
        print("=" * 60)
        
        print(f"🗂️ Input file: {self.input_file.name}")
        print(f"📅 Processing completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎲 Random seed: {self.random_state}")
        
        print(f"\n📊 DATASET STATISTICS:")
        print("-" * 30)
        print(f"Original comments: {self.total_cleaned_size:,}")
        print(f"Target for annotation: {self.target_annotation_size:,}")
        print(f"Actually selected: {len(self.annotation_df):,}")
        print(f"Archived comments: {len(self.archive_df):,}")
        print(f"Sampling fraction: {self.sampling_fraction:.1%}")
        
        print(f"\n📁 OUTPUT FILES:")
        print("-" * 20)
        print(f"📝 {self.annotation_output_file}")
        print(f"📦 {self.archive_output_file}")
        
        print(f"\n✅ QUALITY ASSURANCE:")
        print("-" * 25)
        print("• Stratified sampling ensures fair representation")
        print("• Each post contributes proportionally to annotation set")
        print("• Random seed ensures reproducible results")
        print("• No data loss or overlap between datasets")
        print("• UTF-8 encoding for international character support")
        
        print(f"\n🎯 READY FOR ANNOTATION!")
        print(f"Your {len(self.annotation_df):,}-comment dataset is methodologically sound")
        print(f"and representative of your full {self.total_cleaned_size:,}-comment corpus.")


def create_annotation_dataset(input_file="data/processed/cleaned_comments.csv", 
                            target_size=10000, random_state=42):
    """
    Main function to create stratified annotation dataset.
    
    Args:
        input_file (str): Path to cleaned comments file
        target_size (int): Target number of comments for annotation
        random_state (int): Random seed for reproducibility
    """
    print("🎯 STRATIFIED RANDOM SAMPLING FOR ANNOTATION DATASET")
    print("📊 Creating representative sample using post-level stratification")
    print("🔬 Methodology: Proportional sampling from each Facebook post")
    print("=" * 60)
    
    # Initialize sampler
    sampler = StratifiedCommentSampler(input_file, target_size, random_state)
    
    # Execute pipeline
    success = True
    
    # Phase 1: Load and initialize
    if not sampler.load_and_initialize():
        return False
    
    # Phase 2: Perform stratified sampling
    if success:
        success = sampler.perform_stratified_sampling()
    
    # Phase 3: Split data
    if success:
        success = sampler.split_and_create_archive()
    
    # Phase 4: Verify and save
    if success:
        success = sampler.verify_and_save()
    
    # Additional analysis
    if success:
        sampler.analyze_representation()
        sampler.generate_final_report()
        print("\n🎉 STRATIFIED SAMPLING COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("\n❌ STRATIFIED SAMPLING FAILED!")
        return False


if __name__ == "__main__":
    import sys
    
    # Configuration variables (easily modifiable)
    INPUT_FILE = "data/processed/cleaned_comments.csv"
    TARGET_ANNOTATION_SIZE = 10000
    RANDOM_STATE = 42  # For reproducible results
    
    # Check if input file was provided as command line argument
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
        print(f"📁 Using command line input file: {INPUT_FILE}")
    
    # Execute stratified sampling
    create_annotation_dataset(INPUT_FILE, TARGET_ANNOTATION_SIZE, RANDOM_STATE)
