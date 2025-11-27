"""
Enhanced Data Loader - Robust handling of misaligned and malformed data.

Handles common data alignment issues:
- Column name inconsistencies (case, whitespace, special chars)
- Ragged/jagged CSVs (different row lengths)
- Delimiter detection
- Header detection
- Encoding issues
- Column order mismatches
- Data validation and repair
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import warnings
import chardet


class DataLoadingError(Exception):
    """Custom exception for data loading issues."""
    pass


class EnhancedDataLoader:
    """
    Intelligent data loader that handles misaligned and malformed data.

    Features:
    - Automatic delimiter detection
    - Encoding detection
    - Column name normalization
    - Ragged CSV handling
    - Header detection
    - Data validation and repair
    """

    def __init__(self, auto_fix: bool = True, verbose: bool = True):
        """
        Initialize EnhancedDataLoader.

        Args:
            auto_fix: Automatically fix common issues (default: True)
            verbose: Print warnings and fixes (default: True)
        """
        self.auto_fix = auto_fix
        self.verbose = verbose
        self.issues_found = []
        self.fixes_applied = []

    def load(
        self,
        data: Union[str, Path, pd.DataFrame],
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load and validate data with automatic issue detection and fixing.

        Args:
            data: File path or DataFrame
            target_column: Optional target column name

        Returns:
            Tuple of (cleaned_dataframe, metadata_dict)
        """
        self.issues_found = []
        self.fixes_applied = []

        # If already a DataFrame, validate it
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            metadata = {'source': 'dataframe', 'original_shape': df.shape}
        else:
            # Load from file with smart detection
            df, metadata = self._load_from_file(data)

        # Normalize column names
        df, column_mapping = self._normalize_column_names(df)
        metadata['column_mapping'] = column_mapping

        # Update target column if renamed
        if target_column and target_column in column_mapping:
            target_column = column_mapping[target_column]

        # Validate and fix data alignment
        df = self._validate_and_fix_alignment(df)

        # Detect and fix mixed dtypes within columns
        df = self._fix_mixed_dtypes(df)

        # Final validation
        self._final_validation(df)

        metadata['issues_found'] = self.issues_found
        metadata['fixes_applied'] = self.fixes_applied
        metadata['final_shape'] = df.shape

        if self.verbose and self.fixes_applied:
            self._print_summary()

        return df, metadata

    def _load_from_file(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
        """Load data from file with smart detection."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = {
            'source': str(file_path),
            'file_size': file_path.stat().st_size,
            'extension': file_path.suffix.lower()
        }

        ext = file_path.suffix.lower()

        # Handle CSV with smart detection
        if ext == '.csv':
            df = self._load_csv_smart(file_path, metadata)

        # Handle Excel
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)

        # Handle JSON
        elif ext == '.json':
            df = pd.read_json(file_path)

        # Handle Parquet
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)

        # Handle TSV
        elif ext == '.tsv':
            df = pd.read_csv(file_path, sep='\t')

        else:
            raise ValueError(f"Unsupported file format: {ext}")

        metadata['loaded_shape'] = df.shape
        return df, metadata

    def _load_csv_smart(self, file_path: Path, metadata: Dict) -> pd.DataFrame:
        """Load CSV with automatic delimiter, encoding, and header detection."""

        # Step 1: Detect encoding
        encoding = self._detect_encoding(file_path)
        metadata['encoding'] = encoding

        if encoding != 'utf-8':
            self.issues_found.append(f"Non-UTF-8 encoding detected: {encoding}")
            self.fixes_applied.append(f"Using detected encoding: {encoding}")

        # Step 2: Detect delimiter
        delimiter = self._detect_delimiter(file_path, encoding)
        metadata['delimiter'] = delimiter

        if delimiter != ',':
            self.issues_found.append(f"Non-standard delimiter: '{delimiter}'")
            self.fixes_applied.append(f"Using detected delimiter: '{delimiter}'")

        # Step 3: Try loading with error handling for ragged CSVs
        try:
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding,
                on_bad_lines='warn'  # Python 3.9+
            )
        except TypeError:
            # Fallback for older pandas versions
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding,
                error_bad_lines=False,
                warn_bad_lines=True
            )
            self.issues_found.append("Some malformed rows were skipped")
            self.fixes_applied.append("Skipped malformed rows")

        # Step 4: Validate header
        if not self._is_valid_header(df):
            self.issues_found.append("Invalid or missing header detected")

            if self.auto_fix:
                # Regenerate column names
                df.columns = [f'column_{i}' for i in range(len(df.columns))]
                self.fixes_applied.append("Generated column names: column_0, column_1, ...")

        # Step 5: Handle ragged data (rows with different column counts)
        df = self._fix_ragged_data(df)

        return df

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'  # Default fallback

    def _detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """Detect CSV delimiter by analyzing first few lines."""
        delimiters = [',', '\t', ';', '|', ' ']

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_lines = [f.readline() for _ in range(5)]

            # Count occurrences of each delimiter
            delimiter_counts = {}
            for delim in delimiters:
                counts = [line.count(delim) for line in first_lines if line.strip()]
                if counts and all(c == counts[0] for c in counts) and counts[0] > 0:
                    delimiter_counts[delim] = counts[0]

            if delimiter_counts:
                # Return delimiter with most consistent occurrences
                return max(delimiter_counts, key=delimiter_counts.get)

        except Exception:
            pass

        return ','  # Default to comma

    def _is_valid_header(self, df: pd.DataFrame) -> bool:
        """Check if the first row is a valid header."""
        # Header should contain strings, not numbers
        # And shouldn't look like data

        header = df.columns.tolist()

        # Check if column names are auto-generated (Unnamed, etc.)
        if any('Unnamed' in str(col) for col in header):
            return False

        # Check if header looks like numeric data
        numeric_count = sum(1 for col in header if isinstance(col, (int, float)))
        if numeric_count > len(header) * 0.5:
            return False

        return True

    def _fix_ragged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle rows with inconsistent column counts."""
        # Check if there are any completely null columns
        null_columns = df.columns[df.isnull().all()].tolist()

        if null_columns:
            self.issues_found.append(
                f"Found {len(null_columns)} completely empty columns (ragged data)"
            )
            if self.auto_fix:
                df = df.drop(columns=null_columns)
                self.fixes_applied.append(f"Removed {len(null_columns)} empty columns")

        # Check for rows that are mostly NaN (likely ragged)
        null_threshold = 0.8
        mostly_null_rows = (df.isnull().sum(axis=1) / len(df.columns)) > null_threshold

        if mostly_null_rows.sum() > 0:
            self.issues_found.append(
                f"Found {mostly_null_rows.sum()} rows with >{null_threshold*100}% missing values"
            )
            if self.auto_fix:
                df = df[~mostly_null_rows]
                self.fixes_applied.append(
                    f"Removed {mostly_null_rows.sum()} severely incomplete rows"
                )

        return df

    def _normalize_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize column names to handle inconsistencies.

        Fixes:
        - Case inconsistencies (Price vs price vs PRICE)
        - Whitespace (leading/trailing/multiple spaces)
        - Special characters
        - Duplicate names
        """
        original_columns = df.columns.tolist()
        normalized_columns = []
        column_mapping = {}

        for col in original_columns:
            # Convert to string and normalize
            normalized = str(col)

            # Remove leading/trailing whitespace
            normalized = normalized.strip()

            # Replace multiple spaces with single space
            normalized = ' '.join(normalized.split())

            # Convert to lowercase
            normalized = normalized.lower()

            # Replace special characters with underscore
            normalized = normalized.replace(' ', '_')
            normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized)

            # Remove duplicate underscores
            normalized = '_'.join(filter(None, normalized.split('_')))

            # Handle empty names
            if not normalized:
                normalized = f'column_{len(normalized_columns)}'

            # Handle duplicates
            if normalized in normalized_columns:
                suffix = 1
                while f"{normalized}_{suffix}" in normalized_columns:
                    suffix += 1
                normalized = f"{normalized}_{suffix}"

            normalized_columns.append(normalized)
            column_mapping[col] = normalized

        # Check if any changes were made
        if normalized_columns != original_columns:
            changes = sum(1 for old, new in zip(original_columns, normalized_columns)
                         if old != new)
            self.issues_found.append(f"Found {changes} column names needing normalization")
            self.fixes_applied.append(
                f"Normalized {changes} column names (lowercase, no special chars)"
            )
            df.columns = normalized_columns

        return df, column_mapping

    def _validate_and_fix_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data alignment and fix common issues."""

        # Check for index issues
        if df.index.duplicated().any():
            self.issues_found.append("Duplicate index values detected")
            if self.auto_fix:
                df = df.reset_index(drop=True)
                self.fixes_applied.append("Reset index to remove duplicates")

        # Check for single-value columns (no variance)
        single_value_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                single_value_cols.append(col)

        if single_value_cols:
            self.issues_found.append(
                f"Found {len(single_value_cols)} columns with only one unique value"
            )
            # Don't auto-remove these - might be intentional
            # Just warn the user

        return df

    def _fix_mixed_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and fix columns with mixed data types."""

        for col in df.columns:
            # Skip if column is empty
            if df[col].isnull().all():
                continue

            # Get non-null values
            non_null = df[col].dropna()

            if len(non_null) == 0:
                continue

            # Check if column has mixed types
            types = non_null.apply(type).unique()

            if len(types) > 1:
                self.issues_found.append(
                    f"Column '{col}' has mixed data types: {[t.__name__ for t in types]}"
                )

                if self.auto_fix:
                    # Try to coerce to most appropriate type
                    # Priority: numeric > datetime > string

                    # Try numeric
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        self.fixes_applied.append(f"Coerced column '{col}' to numeric")
                        continue
                    except:
                        pass

                    # Try datetime
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        self.fixes_applied.append(f"Coerced column '{col}' to datetime")
                        continue
                    except:
                        pass

                    # Default to string
                    df[col] = df[col].astype(str)
                    self.fixes_applied.append(f"Coerced column '{col}' to string")

        return df

    def _final_validation(self, df: pd.DataFrame):
        """Perform final validation checks."""

        # Check if DataFrame is empty
        if len(df) == 0:
            raise DataLoadingError("DataFrame is empty after processing")

        if len(df.columns) == 0:
            raise DataLoadingError("DataFrame has no columns after processing")

        # Check for excessive missing data
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100

        if missing_pct > 90:
            warnings.warn(
                f"Dataset has {missing_pct:.1f}% missing values. "
                "Results may be unreliable."
            )

    def _print_summary(self):
        """Print summary of issues and fixes."""
        print("\n" + "="*60)
        print("DATA LOADING SUMMARY")
        print("="*60)

        if self.issues_found:
            print("\n⚠️  Issues Detected:")
            for issue in self.issues_found:
                print(f"   - {issue}")

        if self.fixes_applied:
            print("\n✅ Fixes Applied:")
            for fix in self.fixes_applied:
                print(f"   - {fix}")

        print("\n" + "="*60 + "\n")

    def get_report(self) -> Dict[str, Any]:
        """Get detailed report of loading process."""
        return {
            'issues_found': self.issues_found,
            'fixes_applied': self.fixes_applied,
            'num_issues': len(self.issues_found),
            'num_fixes': len(self.fixes_applied)
        }
