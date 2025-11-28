"""
Demonstration: Automated Feature Engineering

This script demonstrates how the Plug-and-Play framework automatically
creates powerful engineered features without manual intervention.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.plug_and_play import PlugAndPlayML


def create_sample_dataset():
    """Create a realistic dataset for demonstration."""
    np.random.seed(42)

    n_samples = 1000

    data = {
        # Numerical features
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt': np.random.randint(0, 100000, n_samples),
        'years_employed': np.random.randint(0, 40, n_samples),

        # Categorical features
        'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n_samples),
        'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Sales', 'Other'], n_samples),
        'city': np.random.choice(['NYC', 'SF', 'LA', 'Chicago', 'Boston'], n_samples),

        # Text feature
        'customer_notes': [
            f"Customer {i} - notes about their account" for i in range(n_samples)
        ],

        # Date feature
        'account_opened': pd.date_range('2020-01-01', periods=n_samples, freq='D')
    }

    df = pd.DataFrame(data)

    # Create target based on complex interaction
    df['approved'] = (
        (df['credit_score'] > 650) &
        (df['income'] / (df['debt'] + 1) > 2) &
        (df['years_employed'] > 2)
    ).astype(int)

    return df


def demo_basic_vs_engineered():
    """Compare results with and without feature engineering."""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING DEMONSTRATION")
    print("="*70)

    # Create dataset
    print("\n📊 Creating sample dataset...")
    df = create_sample_dataset()
    df.to_csv('demo_loan_data.csv', index=False)

    print(f"\nOriginal dataset:")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns) - 1}")  # Exclude target
    print(f"  - Target: 'approved'")

    print("\n" + "-"*70)
    print("SCENARIO 1: WITHOUT Feature Engineering")
    print("-"*70)

    model_no_fe = PlugAndPlayML(verbose=False)
    model_no_fe.preprocessor = model_no_fe.preprocessor.__class__(
        enable_feature_engineering=False
    )

    print("\n⚙️  Running analysis (feature engineering disabled)...")
    results_no_fe = model_no_fe.run('demo_loan_data.csv', target_column='approved')

    print(f"\n✅ Results:")
    print(f"  - Final features: {results_no_fe['preprocessed_data']['X'].shape[1]}")

    print("\n" + "-"*70)
    print("SCENARIO 2: WITH Automated Feature Engineering")
    print("-"*70)

    model_with_fe = PlugAndPlayML(verbose=True)

    print("\n⚙️  Running analysis (feature engineering enabled)...")
    results_with_fe = model_with_fe.run('demo_loan_data.csv', target_column='approved')

    print(f"\n✅ Results:")
    print(f"  - Final features: {results_with_fe['preprocessed_data']['X'].shape[1]}")

    # Show feature engineering report
    if model_with_fe.preprocessor.feature_engineer:
        print("\n" + "="*70)
        print("FEATURE ENGINEERING REPORT")
        print("="*70)
        print(model_with_fe.preprocessor.feature_engineer.get_summary())

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    features_without = results_no_fe['preprocessed_data']['X'].shape[1]
    features_with = results_with_fe['preprocessed_data']['X'].shape[1]
    improvement = ((features_with - features_without) / features_without) * 100

    print(f"\nWithout Feature Engineering: {features_without} features")
    print(f"With Feature Engineering:    {features_with} features")
    print(f"Improvement:                 {improvement:.1f}% more features!")

    print("\n💡 Key Benefits:")
    print("  ✓ Interaction features capture relationships between variables")
    print("  ✓ Polynomial features capture non-linear patterns")
    print("  ✓ Ratio features (e.g., income/debt) add domain knowledge")
    print("  ✓ Statistical features aggregate row-wise information")
    print("  ✓ Domain-specific features leverage naming patterns")
    print("  ✓ All created automatically - no manual work!")


def demo_feature_types():
    """Demonstrate different types of features created."""
    print("\n\n" + "="*70)
    print("TYPES OF ENGINEERED FEATURES")
    print("="*70)

    df = create_sample_dataset()
    df.to_csv('demo_features.csv', index=False)

    model = PlugAndPlayML(verbose=False)
    results = model.run('demo_features.csv', target_column='approved')

    X = results['preprocessed_data']['X']
    all_features = X.columns.tolist()

    # Categorize features
    interaction_features = [f for f in all_features if '_x_' in f or '_plus_' in f or '_minus_' in f]
    ratio_features = [f for f in all_features if '_div_' in f]
    poly_features = [f for f in all_features if 'poly_' in f]
    stat_features = [f for f in all_features if any(s in f for s in ['row_mean', 'row_std', 'row_min', 'row_max'])]
    binned_features = [f for f in all_features if '_binned' in f]
    text_features = [f for f in all_features if any(t in f for t in ['num_chars', 'num_words', 'lexical'])]
    time_features = [f for f in all_features if any(t in f for t in ['_year', '_month', '_day', '_hour', '_weekend', '_business'])]

    print("\n1️⃣  INTERACTION FEATURES:")
    print(f"   Count: {len(interaction_features)}")
    if interaction_features:
        print("   Examples:")
        for feat in interaction_features[:3]:
            print(f"     - {feat}")

    print("\n2️⃣  RATIO/DIVISION FEATURES:")
    print(f"   Count: {len(ratio_features)}")
    if ratio_features:
        print("   Examples:")
        for feat in ratio_features[:3]:
            print(f"     - {feat}")
        print("   💡 Great for capturing proportions (e.g., debt-to-income ratio)")

    print("\n3️⃣  POLYNOMIAL FEATURES:")
    print(f"   Count: {len(poly_features)}")
    if poly_features:
        print("   Examples:")
        for feat in poly_features[:3]:
            print(f"     - {feat}")
        print("   💡 Captures non-linear relationships")

    print("\n4️⃣  STATISTICAL FEATURES:")
    print(f"   Count: {len(stat_features)}")
    if stat_features:
        print("   Examples:")
        for feat in stat_features[:3]:
            print(f"     - {feat}")
        print("   💡 Row-wise aggregations across numerical features")

    print("\n5️⃣  BINNED/DISCRETIZED FEATURES:")
    print(f"   Count: {len(binned_features)}")
    if binned_features:
        print("   Examples:")
        for feat in binned_features[:3]:
            print(f"     - {feat}")
        print("   💡 Converts continuous to categorical")

    print("\n6️⃣  TEXT-DERIVED FEATURES:")
    print(f"   Count: {len(text_features)}")
    if text_features:
        print("   Examples:")
        for feat in text_features[:3]:
            print(f"     - {feat}")
        print("   💡 Extracts numeric features from text")

    print("\n7️⃣  TIME-BASED FEATURES:")
    print(f"   Count: {len(time_features)}")
    if time_features:
        print("   Examples:")
        for feat in time_features[:3]:
            print(f"     - {feat}")
        print("   💡 Extracts temporal patterns")


def demo_domain_features():
    """Demonstrate domain-specific feature engineering."""
    print("\n\n" + "="*70)
    print("DOMAIN-SPECIFIC FEATURE ENGINEERING")
    print("="*70)

    print("\n💡 The system recognizes common column name patterns and creates")
    print("   domain-specific features automatically!\n")

    # Example 1: Financial data
    print("📊 Example 1: Financial Data")
    financial_data = pd.DataFrame({
        'price': [100, 200, 150, 300],
        'cost': [80, 150, 120, 250],
        'shipping_fee': [10, 15, 12, 20],
        'target': [0, 1, 0, 1]
    })
    financial_data.to_csv('financial_demo.csv', index=False)

    model = PlugAndPlayML(verbose=False)
    results = model.run('financial_demo.csv', target_column='target')

    total_price_exists = 'total_price' in results['preprocessed_data']['X'].columns
    print(f"   ✓ Detected price-related columns → Created 'total_price': {total_price_exists}")

    # Example 2: Age data
    print("\n📊 Example 2: Age-Based Data")
    age_data = pd.DataFrame({
        'customer_age': [25, 45, 65, 35, 55],
        'income': [50000, 80000, 60000, 70000, 90000],
        'target': [0, 1, 1, 0, 1]
    })
    age_data.to_csv('age_demo.csv', index=False)

    model = PlugAndPlayML(verbose=False)
    results = model.run('age_demo.csv', target_column='target')

    age_group_exists = any('age_group' in col for col in results['preprocessed_data']['X'].columns)
    print(f"   ✓ Detected 'age' column → Created age groups: {age_group_exists}")

    print("\n💡 The system is smart enough to recognize:")
    print("   • Price/Cost/Amount columns → Creates totals, averages")
    print("   • Age columns → Creates age groups (young, adult, middle, senior)")
    print("   • Location columns (lat/lon) → Creates distance features")
    print("   • And more patterns automatically!")


def main():
    """Run all feature engineering demonstrations."""
    print("\n" + "="*70)
    print("AUTOMATED FEATURE ENGINEERING DEMONSTRATION")
    print("="*70)
    print("\nThis demonstrates how the Plug-and-Play framework automatically")
    print("creates powerful engineered features without any manual work!\n")

    try:
        # Demo 1: Basic vs Engineered
        input("Press Enter to start Demo 1 (Basic vs Engineered Features)...")
        demo_basic_vs_engineered()

        # Demo 2: Feature Types
        input("\n\nPress Enter to continue to Demo 2 (Types of Features)...")
        demo_feature_types()

        # Demo 3: Domain Features
        input("\n\nPress Enter to continue to Demo 3 (Domain-Specific Features)...")
        demo_domain_features()

        print("\n\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED!")
        print("="*70)

        print("\n🎉 KEY TAKEAWAYS:")
        print("\n1. AUTOMATIC FEATURE CREATION:")
        print("   • Interaction features (multiplication, addition, subtraction)")
        print("   • Polynomial features (2nd, 3rd degree)")
        print("   • Ratio and proportion features")
        print("   • Statistical aggregations (mean, std, min, max)")
        print("   • Binning and discretization")
        print("   • Text-derived numeric features")
        print("   • Time-based features (cyclical encoding)")

        print("\n2. DOMAIN INTELLIGENCE:")
        print("   • Recognizes common naming patterns (price, age, location)")
        print("   • Creates domain-specific features automatically")
        print("   • No manual feature engineering needed!")

        print("\n3. SMART FEATURE SELECTION:")
        print("   • Automatically ranks features by importance")
        print("   • Keeps only the most valuable features")
        print("   • Prevents feature explosion")

        print("\n4. COMPLETELY AUTOMATED:")
        print("   • Just load your CSV")
        print("   • System handles everything")
        print("   • No data science expertise required!")

        print("\n🚀 The Plug-and-Play framework does the work of a skilled")
        print("   data scientist - automatically creating features that")
        print("   improve model performance!\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        import os
        cleanup_files = [
            'demo_loan_data.csv', 'demo_features.csv',
            'financial_demo.csv', 'age_demo.csv'
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
