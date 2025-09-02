import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DatasetCleaner:
    """
    A comprehensive dataset cleaner for Excel files.
    """
    
    def __init__(self, file_path):
        """
        Initialize the DatasetCleaner with the path to the Excel file.
        
        Args:
            file_path (str): Path to the Excel file
        """
        self.file_path = Path(file_path)
        self.df = None
        self.original_shape = None
        self.cleaning_log = []
        
    def load_data(self, sheet_name=0):
        """
        Load data from Excel file.
        
        Args:
            sheet_name: Sheet name or index to load (default: 0 for first sheet)
        """
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            self.original_shape = self.df.shape
            self.log_action(f"Data loaded successfully: {self.original_shape[0]} rows, {self.original_shape[1]} columns")
            print(f"✅ Data loaded: {self.original_shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def log_action(self, action):
        """Log cleaning actions for tracking."""
        self.cleaning_log.append(action)
    
    def get_data_overview(self):
        """
        Get a comprehensive overview of the dataset.
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("=" * 50)
        print("📊 DATASET OVERVIEW")
        print("=" * 50)
        
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📋 COLUMN INFORMATION:")
        print("-" * 30)
        info_df = pd.DataFrame({
            'Column': self.df.columns,
            'Type': self.df.dtypes,
            'Non-Null Count': self.df.count(),
            'Null Count': self.df.isnull().sum(),
            'Null %': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        print(info_df.to_string(index=False))
        
        print("\n📈 NUMERICAL COLUMNS SUMMARY:")
        print("-" * 35)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(self.df[numeric_cols].describe())
        else:
            print("No numerical columns found.")
        
        print("\n📝 CATEGORICAL COLUMNS SUMMARY:")
        print("-" * 37)
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            print(f"{col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"  Values: {list(self.df[col].value_counts().head().index)}")
            print()
    
    def check_duplicates(self):
        """
        Check for duplicate rows in the dataset.
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        duplicates = self.df.duplicated().sum()
        print(f"🔍 Found {duplicates} duplicate rows ({duplicates/len(self.df)*100:.2f}%)")
        
        if duplicates > 0:
            print("\nFirst few duplicate rows:")
            print(self.df[self.df.duplicated()].head())
        
        return duplicates
    
    def remove_duplicates(self, keep='first'):
        """
        Remove duplicate rows from the dataset.
        
        Args:
            keep (str): Which duplicates to keep ('first', 'last', False)
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(keep=keep)
        removed_count = initial_count - len(self.df)
        
        self.log_action(f"Removed {removed_count} duplicate rows")
        print(f"✅ Removed {removed_count} duplicate rows")
    
    def handle_missing_values(self, strategy='auto', columns=None):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): 'auto', 'drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_zero'
            columns (list): Specific columns to handle (None for all)
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            print(f"Handling {missing_count} missing values in '{col}'...")
            
            if strategy == 'auto':
                # Auto strategy based on data type and missing percentage
                missing_pct = missing_count / len(self.df) * 100
                
                if missing_pct > 50:
                    print(f"  ⚠️  Column '{col}' has {missing_pct:.1f}% missing values. Consider dropping this column.")
                    continue
                elif self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"  ✅ Filled with median value")
                else:
                    self.df[col].fillna(self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'Unknown', inplace=True)
                    print(f"  ✅ Filled with mode value")
            
            elif strategy == 'drop':
                self.df.dropna(subset=[col], inplace=True)
                print(f"  ✅ Dropped rows with missing values")
            
            elif strategy == 'fill_mean' and self.df[col].dtype in ['int64', 'float64']:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
                print(f"  ✅ Filled with mean value")
            
            elif strategy == 'fill_median' and self.df[col].dtype in ['int64', 'float64']:
                self.df[col].fillna(self.df[col].median(), inplace=True)
                print(f"  ✅ Filled with median value")
            
            elif strategy == 'fill_mode':
                mode_val = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
                print(f"  ✅ Filled with mode value")
            
            elif strategy == 'fill_zero':
                self.df[col].fillna(0, inplace=True)
                print(f"  ✅ Filled with zero")
        
        self.log_action(f"Handled missing values using strategy: {strategy}")
    
    def detect_outliers(self, columns=None, method='iqr'):
        """
        Detect outliers in numerical columns.
        
        Args:
            columns (list): Columns to check for outliers (None for all numerical)
            method (str): 'iqr' or 'zscore'
        """
        if self.df is None:
            print("❌ No data loaded.")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outliers_info = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = self.df[z_scores > 3]
            
            outliers_count = len(outliers)
            outliers_info[col] = {
                'count': outliers_count,
                'percentage': outliers_count / len(self.df) * 100,
                'indices': outliers.index.tolist()
            }
            
            print(f"🎯 Column '{col}': {outliers_count} outliers ({outliers_count/len(self.df)*100:.2f}%)")
        
        return outliers_info
    
    def create_cleaning_report(self):
        """
        Create a summary report of all cleaning actions performed.
        """
        print("=" * 50)
        print("📋 CLEANING REPORT")
        print("=" * 50)
        
        print(f"Original dataset shape: {self.original_shape}")
        print(f"Current dataset shape: {self.df.shape if self.df is not None else 'No data loaded'}")
        
        if self.original_shape and self.df is not None:
            rows_changed = self.original_shape[0] - self.df.shape[0]
            cols_changed = self.original_shape[1] - self.df.shape[1]
            print(f"Rows removed: {rows_changed}")
            print(f"Columns removed: {cols_changed}")
        
        print(f"\n📝 Actions performed ({len(self.cleaning_log)}):")
        print("-" * 30)
        for i, action in enumerate(self.cleaning_log, 1):
            print(f"{i}. {action}")
    
    def save_cleaned_data(self, output_path=None, format='excel'):
        """
        Save the cleaned dataset.
        
        Args:
            output_path (str): Path to save the file (None for auto-generated name)
            format (str): 'excel' or 'csv'
        """
        if self.df is None:
            print("❌ No data to save.")
            return
        
        if output_path is None:
            stem = self.file_path.stem
            if format == 'excel':
                output_path = self.file_path.parent / f"{stem}_cleaned.xlsx"
            else:
                output_path = self.file_path.parent / f"{stem}_cleaned.csv"
        
        try:
            if format == 'excel':
                self.df.to_excel(output_path, index=False)
            else:
                self.df.to_csv(output_path, index=False)
            
            print(f"✅ Cleaned data saved to: {output_path}")
            self.log_action(f"Data saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")


# Example usage function
def main():
    """
    Main function demonstrating how to use the DatasetCleaner.
    """
    # Initialize the cleaner with your Excel file
    cleaner = DatasetCleaner("RawThesisData.xlsx")
    
    # Load the data
    if cleaner.load_data():
        # Get overview of the dataset
        cleaner.get_data_overview()
        
        # Check for duplicates
        cleaner.check_duplicates()
        
        # Remove duplicates if found
        cleaner.remove_duplicates()
        
        # Handle missing values automatically
        cleaner.handle_missing_values(strategy='auto')
        
        # Detect outliers
        outliers = cleaner.detect_outliers()
        
        # Create cleaning report
        cleaner.create_cleaning_report()
        
        # Save cleaned data
        cleaner.save_cleaned_data()


if __name__ == "__main__":
    main()
