# Facebook Comment Annotation Cleaner

A specialized Python tool for cleaning Facebook comments and preparing them for sentiment annotation. Follows exact specifications for structured data processing with unique ID generation and CSV output.

## 📋 Input & Output Specification

### Input File:
- **Format**: Excel Spreadsheet (.xlsx)
- **Expected Columns**:
  - `ID`: Non-unique integer (resets for each post)
  - `TITLE`: Facebook post title (string)
  - `COMMENT`: User-submitted comment (string)
  - `LIKES`: Number of likes on comment (integer)
  - `POST URL`: Full URL of source Facebook post (string)

### Output File:
- **Name**: `cleaned_for_annotation.csv` (configurable)
- **Format**: CSV with UTF-8 encoding
- **Final Columns**:
  - `unique_comment_id`: Globally unique identifier
  - `context_title`: Cleaned title text
  - `comment_to_annotate`: Cleaned comment text
  - `LIKES`: Original likes count (preserved)
  - `POST URL`: Original post URL (preserved)

## 🔄 Processing Pipeline

### Phase 1: Initialization and Data Loading
- ✅ Load Excel file with error handling
- ✅ Validate required columns exist
- ✅ Log initial row count

### Phase 2: Pre-Processing and Unique ID Generation
- 🗑️ Remove null/empty comments
- 🗑️ Remove exact duplicate comments
- 🆔 Generate unique post IDs (`post_id`)
- 🆔 Generate unique comment IDs (`p{post_id}_c{ID}`)

### Phase 3: Text Cleaning Logic
Single `clean_text()` function performs (in order):
1. **Convert emojis to text**: � → `:smiling_face:`
2. **Remove URLs**: `http://example.com` → removed
3. **Remove mentions**: `@username` → removed
4. **Remove HTML**: `<b>text</b> &amp;` → `text`
5. **Normalize hashtags**: `#hashtag` → `hashtag`
6. **Convert to lowercase**: `HELLO` → `hello`
7. **Standardize whitespace**: Multiple spaces → single space

### Phase 4: Final Filtering and Structuring
- 🗑️ Remove comments that became empty after cleaning
- 🗑️ Remove comments with fewer than 3 words
- 📊 Select and rename final columns
- 📋 Structure output format

### Phase 5: File Output
- 💾 Save as CSV with UTF-8 encoding
- 📝 Display sample of final output
- ✅ Confirm successful completion

## 🚀 Quick Start

### 1. Setup Environment
```powershell
# Navigate to project directory
cd "d:\School\Cleaner"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Run Cleaning Pipeline
```powershell
# Default: RawThesisData.xlsx → cleaned_for_annotation.csv
python dataset_cleaner.py
```

### 3. Custom Configuration
Edit the configuration variables at the top of `dataset_cleaner.py`:
```python
INPUT_FILE = "your_file.xlsx"
OUTPUT_FILE = "your_output.csv"
```

## 📊 Expected Results

### Data Retention:
- **Typical retention**: 80-95% of original comments
- **Null removal**: ~2-5% of comments
- **Duplicate removal**: ~5-15% depending on spam level
- **Short comment filtering**: ~3-8% of comments

### Processing Output:
```
📊 FINAL PROCESSING REPORT
====================================
� Input file: RawThesisData.xlsx
📁 Output file: cleaned_for_annotation.csv
📅 Processing completed: 2025-09-02 22:30:15

📊 PROCESSING STATISTICS:
Original comments: 15,420
Final comments: 13,891
Retention rate: 90.1%
Unique posts: 1,247

📏 TEXT STATISTICS:
Average title length: 67.3 characters
Average comment length: 89.7 characters
Average comment words: 16.2 words
```

## ✨ Key Features

### � Robust Processing:
- **Error handling**: Graceful failure with informative messages
- **Progress logging**: Step-by-step status updates
- **Validation**: Ensures required columns exist
- **Memory efficient**: Processes large datasets smoothly

### 🆔 Unique ID Generation:
- **Post-level IDs**: Groups comments by Facebook post
- **Comment-level IDs**: Format `p{post_id}_c{original_id}`
- **Global uniqueness**: No duplicate IDs across entire dataset

### 🧹 Intelligent Cleaning:
- **Preserves context**: Keeps meaningful elements for annotation
- **Removes noise**: URLs, mentions, HTML don't help sentiment analysis
- **Standardizes format**: Consistent lowercase, spacing
- **Emoji handling**: Converts to text descriptions for ML processing

## 📁 File Structure
```
d:\School\Cleaner\
├── RawThesisData.xlsx                # Input: Your Facebook comments
├── dataset_cleaner.py               # Main cleaning script
├── cleaned_for_annotation.csv       # Output: Ready for annotation
├── requirements.txt                 # Python dependencies
├── README.md                       # This documentation
└── venv/                          # Virtual environment
```

## 🎯 Perfect For:
- **Sentiment annotation preparation**
- **Facebook comment datasets**
- **Social media research**
- **NLP preprocessing pipelines**
- **Academic research on Philippine social media**

## 🔧 Dependencies
- **pandas**: Data manipulation and analysis
- **openpyxl**: Excel file handling
- **emoji**: Emoji to text conversion
- **regex**: Advanced text processing

## � Sample Output
```csv
unique_comment_id,context_title,comment_to_annotate,LIKES,POST URL
p0_c1,budget proposal for infrastructure development,ang mahal naman ng mga projects na yan,15,https://facebook.com/post1
p0_c2,budget proposal for infrastructure development,sana makakabuti ito sa aming lugar,8,https://facebook.com/post1
p1_c1,new healthcare policy announcement,salamat sa mga doktor natin,23,https://facebook.com/post2
```

## 🎉 Ready for Annotation!
Your cleaned dataset is now optimized for:
- **Manual sentiment labeling**
- **Crowdsourced annotation**
- **Machine learning training**
- **Academic research analysis**
