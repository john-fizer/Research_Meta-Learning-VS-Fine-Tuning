"""
Demonstration: How Plug-and-Play ML Handles Misaligned Data

This script demonstrates the enhanced data loader's ability to handle
various data alignment issues automatically.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.plug_and_play import PlugAndPlayML


def create_messy_csv_files():
    """Create sample CSV files with various alignment issues."""

    print("Creating sample files with misaligned data...\n")

    # 1. Column name inconsistencies
    data1 = pd.DataFrame({
        '  Price  ': [100, 200, 300, 400],  # Leading/trailing spaces
        'QUANTITY': [1, 2, 3, 4],  # ALL CAPS
        'product Name': [' A', 'B ', 'C', 'D'],  # Mixed case + spaces
        'is_available?': [1, 0, 1, 1]  # Special characters
    })
    data1.to_csv('messy_columns.csv', index=False)
    print("✓ Created 'messy_columns.csv' - Column name inconsistencies")

    # 2. Mixed data types in columns
    data2_dict = {
        'age': ['25', 30, '35', 40, '45'],  # Mixed str/int
        'income': [50000, '60000', 70000, 'unknown', 90000],  # Mixed with text
        'score': [85.5, 90, '95.5', 88, 92.0],  # Mixed formats
        'approved': [1, 0, 1, 'yes', 0]  # Mixed boolean types
    }
    df2 = pd.DataFrame(data2_dict)
    df2.to_csv('mixed_types.csv', index=False)
    print("✓ Created 'mixed_types.csv' - Mixed data types")

    # 3. Different delimiter (semicolon)
    data3 = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'C', 'D', 'E']
    })
    data3.to_csv('semicolon_delim.csv', sep=';', index=False)
    print("✓ Created 'semicolon_delim.csv' - Semicolon delimiter")

    # 4. Duplicate column names
    with open('duplicate_columns.csv', 'w') as f:
        f.write('value,value,score,score\n')
        f.write('1,2,3,4\n')
        f.write('5,6,7,8\n')
        f.write('9,10,11,12\n')
    print("✓ Created 'duplicate_columns.csv' - Duplicate column names")

    # 5. Ragged CSV (different row lengths)
    with open('ragged_data.csv', 'w') as f:
        f.write('col1,col2,col3,col4\n')
        f.write('1,2,3,4\n')
        f.write('5,6,7\n')  # Missing one value
        f.write('8,9,10,11\n')
        f.write('12,13\n')  # Missing two values
        f.write('14,15,16,17\n')
    print("✓ Created 'ragged_data.csv' - Inconsistent row lengths")

    # 6. Tab-delimited file
    data6 = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40],
        'target': [0, 1, 0, 1]
    })
    data6.to_csv('tab_delimited.tsv', sep='\t', index=False)
    print("✓ Created 'tab_delimited.tsv' - Tab delimiter\n")


def demo_1_column_name_normalization():
    """Demo 1: Handling column name inconsistencies."""
    print("\n" + "="*70)
    print("DEMO 1: Column Name Normalization")
    print("="*70)

    print("\n📂 Loading file with messy column names...")
    print("   Columns: '  Price  ', 'QUANTITY', 'product Name', 'is_available?'")

    model = PlugAndPlayML(verbose=True)
    results = model.run('messy_columns.csv')

    print("\n✅ Result: All column names automatically normalized!")
    print(f"   Final columns: {results['preprocessed_data']['X'].columns.tolist()}")


def demo_2_mixed_data_types():
    """Demo 2: Handling mixed data types."""
    print("\n" + "="*70)
    print("DEMO 2: Mixed Data Types")
    print("="*70)

    print("\n📂 Loading file with mixed types in columns...")
    print("   - 'age': mix of strings and integers")
    print("   - 'income': includes 'unknown' text")
    print("   - 'approved': mix of 1/0 and 'yes'/'no'")

    model = PlugAndPlayML(verbose=True)
    results = model.run('mixed_types.csv', target_column='approved')

    print("\n✅ Result: Data types automatically fixed!")
    if model.loading_metadata:
        print(f"\n   Issues found: {len(model.loading_metadata['issues_found'])}")
        print(f"   Fixes applied: {len(model.loading_metadata['fixes_applied'])}")


def demo_3_delimiter_detection():
    """Demo 3: Automatic delimiter detection."""
    print("\n" + "="*70)
    print("DEMO 3: Automatic Delimiter Detection")
    print("="*70)

    print("\n📂 Loading semicolon-delimited file...")

    model = PlugAndPlayML(verbose=True)
    results = model.run('semicolon_delim.csv')

    print("\n✅ Result: Delimiter automatically detected!")
    if model.loading_metadata:
        print(f"   Detected delimiter: '{model.loading_metadata.get('delimiter', ',')}'")


def demo_4_duplicate_columns():
    """Demo 4: Handling duplicate column names."""
    print("\n" + "="*70)
    print("DEMO 4: Duplicate Column Names")
    print("="*70)

    print("\n📂 Loading file with duplicate column names...")
    print("   Columns: value, value, score, score")

    model = PlugAndPlayML(verbose=True)
    results = model.run('duplicate_columns.csv')

    print("\n✅ Result: Duplicate columns automatically renamed!")
    print(f"   Final columns: {results['preprocessed_data']['X'].columns.tolist()}")


def demo_5_ragged_data():
    """Demo 5: Handling ragged/jagged CSV."""
    print("\n" + "="*70)
    print("DEMO 5: Ragged Data (Inconsistent Row Lengths)")
    print("="*70)

    print("\n📂 Loading CSV with different row lengths...")
    print("   Row 1: 4 columns")
    print("   Row 2: 3 columns (missing 1)")
    print("   Row 3: 4 columns")
    print("   Row 4: 2 columns (missing 2)")

    model = PlugAndPlayML(verbose=True)
    results = model.run('ragged_data.csv')

    print("\n✅ Result: Ragged rows automatically handled!")
    print(f"   Original shape: varies")
    print(f"   Final shape: {results['preprocessed_data']['X'].shape}")


def demo_6_tab_delimited():
    """Demo 6: Tab-delimited files."""
    print("\n" + "="*70)
    print("DEMO 6: Tab-Delimited File (TSV)")
    print("="*70)

    print("\n📂 Loading .tsv file...")

    model = PlugAndPlayML(verbose=True)
    results = model.run('tab_delimited.tsv')

    print("\n✅ Result: TSV format automatically handled!")


def demo_7_comprehensive_test():
    """Demo 7: Real-world messy data."""
    print("\n" + "="*70)
    print("DEMO 7: Comprehensive Real-World Messy Data")
    print("="*70)

    # Create a really messy dataset
    with open('real_messy_data.csv', 'w') as f:
        f.write('  Customer ID  ;  Customer Name  ;PURCHASE_AMOUNT;purchase_date;  Approved?  \n')
        f.write('1001;  John Doe  ;500.50;2024-01-15;Yes\n')
        f.write('1002;Jane Smith;750;invalid-date;1\n')
        f.write('1003;  Bob Johnson  ;unknown;2024-02-20;true\n')
        f.write('1004;Alice;250.75;2024-03-01;0\n')

    print("\n📂 Loading extremely messy real-world data...")
    print("   Issues:")
    print("   - Mixed delimiters (commas and semicolons)")
    print("   - Inconsistent column names (spaces, case, underscores)")
    print("   - Mixed data types")
    print("   - Invalid dates")
    print("   - Inconsistent boolean values")
    print("   - Leading/trailing spaces in values")

    model = PlugAndPlayML(verbose=True)
    results = model.run('real_messy_data.csv', target_column='Approved?')

    print("\n✅ Result: All issues automatically fixed!")
    print(f"\n   Problem detected: {results['problem_definition'].problem_name}")
    print(f"   Recommended model: {results['pipeline_recommendation'].models[0].model_name}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("MISALIGNED DATA HANDLING DEMONSTRATIONS")
    print("="*70)
    print("\nThis demo shows how the Plug-and-Play framework automatically")
    print("handles various data alignment issues without any manual intervention.\n")

    # Create sample files
    create_messy_csv_files()

    try:
        # Run demos
        input("Press Enter to start Demo 1 (Column Name Normalization)...")
        demo_1_column_name_normalization()

        input("\n\nPress Enter to continue to Demo 2 (Mixed Data Types)...")
        demo_2_mixed_data_types()

        input("\n\nPress Enter to continue to Demo 3 (Delimiter Detection)...")
        demo_3_delimiter_detection()

        input("\n\nPress Enter to continue to Demo 4 (Duplicate Columns)...")
        demo_4_duplicate_columns()

        input("\n\nPress Enter to continue to Demo 5 (Ragged Data)...")
        demo_5_ragged_data()

        input("\n\nPress Enter to continue to Demo 6 (Tab-Delimited)...")
        demo_6_tab_delimited()

        input("\n\nPress Enter to continue to Demo 7 (Comprehensive Test)...")
        demo_7_comprehensive_test()

        print("\n\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED!")
        print("="*70)
        print("\n🎉 The system automatically handled ALL data alignment issues!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Column name normalization (case, spaces, special chars)")
        print("  ✓ Mixed data type handling")
        print("  ✓ Automatic delimiter detection")
        print("  ✓ Duplicate column name resolution")
        print("  ✓ Ragged/jagged CSV handling")
        print("  ✓ Multiple file format support")
        print("  ✓ Encoding detection")
        print("\nYour messy data is automatically cleaned and ready for modeling! 🚀\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        import os
        cleanup_files = [
            'messy_columns.csv', 'mixed_types.csv', 'semicolon_delim.csv',
            'duplicate_columns.csv', 'ragged_data.csv', 'tab_delimited.tsv',
            'real_messy_data.csv'
        ]
        print("\nCleaning up demo files...")
        for f in cleanup_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass


if __name__ == "__main__":
    main()
