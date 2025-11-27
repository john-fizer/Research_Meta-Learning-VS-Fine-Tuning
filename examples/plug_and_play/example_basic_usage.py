"""
Plug-and-Play ML/DL Framework - Basic Usage Example

This example demonstrates the simplest possible usage:
Just load your CSV and let the system do everything automatically!
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.plug_and_play import PlugAndPlayML


def create_sample_classification_data():
    """Create a sample classification dataset for demonstration."""
    np.random.seed(42)

    n_samples = 1000

    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['NYC', 'SF', 'LA', 'Chicago', 'Boston'], n_samples),
        'has_mortgage': np.random.choice([True, False], n_samples),
        'num_dependents': np.random.randint(0, 5, n_samples),
    }

    # Create target based on some logic
    df = pd.DataFrame(data)
    df['approved'] = (
        (df['credit_score'] > 650) &
        (df['income'] > 50000) |
        (df['employment_length'] > 5)
    ).astype(int)

    return df


def create_sample_regression_data():
    """Create a sample regression dataset."""
    np.random.seed(42)

    n_samples = 800

    data = {
        'square_feet': np.random.randint(500, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age_years': np.random.randint(0, 100, n_samples),
        'neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples),
        'has_garage': np.random.choice([True, False], n_samples),
        'has_pool': np.random.choice([True, False], n_samples),
    }

    df = pd.DataFrame(data)

    # Create target (house price) based on features
    df['price'] = (
        df['square_feet'] * 200 +
        df['bedrooms'] * 50000 +
        df['bathrooms'] * 30000 -
        df['age_years'] * 1000 +
        np.random.normal(0, 50000, n_samples)
    )

    return df


def create_sample_text_data():
    """Create a sample text classification dataset."""
    texts = [
        "This product is amazing! Best purchase ever!",
        "Terrible quality, waste of money",
        "It's okay, nothing special",
        "Absolutely love it! Highly recommend",
        "Disappointed with the quality",
        "Great value for the price",
        "Not worth it at all",
        "Exceeded my expectations!",
        "Poor customer service",
        "Will definitely buy again",
    ] * 50  # Repeat to get 500 samples

    sentiments = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1] * 50

    df = pd.DataFrame({
        'review_text': texts,
        'sentiment': sentiments
    })

    return df


def example_1_classification():
    """Example 1: Binary Classification - Loan Approval Prediction"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Binary Classification - Loan Approval Prediction")
    print("="*70 + "\n")

    # Create sample data
    df = create_sample_classification_data()
    print(f"Sample data created: {len(df)} rows × {len(df.columns)} columns\n")
    print(df.head())

    # Save to CSV
    csv_path = "sample_loan_data.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n💾 Data saved to: {csv_path}\n")

    # THE MAGIC HAPPENS HERE - Just one line!
    print("🚀 Running PlugAndPlayML...\n")

    model = PlugAndPlayML(verbose=True)
    results = model.run(csv_path, target_column="approved")

    # View results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print("\n📊 Data Characteristics:")
    print(f"  - Problem Type: {results['problem_definition'].problem_name}")
    print(f"  - Complexity: {results['problem_definition'].complexity.value}")
    print(f"  - Confidence: {results['problem_definition'].confidence:.2%}")

    print("\n🎯 Recommended Models:")
    for i, model_rec in enumerate(results['pipeline_recommendation'].models[:3], 1):
        print(f"  {i}. {model_rec.model_name} ({model_rec.framework})")
        print(f"     - Expected Performance: {model_rec.expected_performance}")
        print(f"     - Training Time: {model_rec.training_time}")

    print("\n💡 Recommendations:")
    print(model.get_recommendations())


def example_2_regression():
    """Example 2: Regression - House Price Prediction"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Regression - House Price Prediction")
    print("="*70 + "\n")

    # Create sample data
    df = create_sample_regression_data()
    print(f"Sample data created: {len(df)} rows × {len(df.columns)} columns\n")

    # Save to CSV
    csv_path = "sample_house_prices.csv"
    df.to_csv(csv_path, index=False)

    # Run PlugAndPlayML
    print("🚀 Running PlugAndPlayML...\n")

    model = PlugAndPlayML(verbose=True)
    results = model.run(csv_path, target_column="price")

    print("\n✅ Analysis Complete!")
    print(f"Problem Type: {results['problem_definition'].problem_name}")
    print(f"Recommended Framework: {results['pipeline_recommendation'].models[0].framework}")


def example_3_text_classification():
    """Example 3: Text Classification - Sentiment Analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Text Classification - Sentiment Analysis")
    print("="*70 + "\n")

    # Create sample data
    df = create_sample_text_data()
    print(f"Sample data created: {len(df)} rows × {len(df.columns)} columns\n")

    # Save to CSV
    csv_path = "sample_reviews.csv"
    df.to_csv(csv_path, index=False)

    # Run PlugAndPlayML
    print("🚀 Running PlugAndPlayML...\n")

    model = PlugAndPlayML(verbose=True)
    results = model.run(csv_path, target_column="sentiment")

    print("\n✅ Text analysis complete!")
    print(f"Detected NLP Task: {results['problem_definition'].problem_name}")


def example_4_quick_analysis():
    """Example 4: Quick Analysis Without Training"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Quick Data Exploration (Analysis Only)")
    print("="*70 + "\n")

    # Create sample data
    df = create_sample_classification_data()
    csv_path = "sample_data.csv"
    df.to_csv(csv_path, index=False)

    # Quick analysis without training
    model = PlugAndPlayML(verbose=False)
    analysis = model.analyze_only(csv_path)

    print("📊 Quick Analysis Results:\n")
    print(analysis['summary'])


def example_5_convenience_function():
    """Example 5: Using the Convenience Function"""
    print("\n" + "="*70)
    print("EXAMPLE 5: One-Liner Convenience Function")
    print("="*70 + "\n")

    from app.plug_and_play import auto_ml

    # Create sample data
    df = create_sample_regression_data()
    csv_path = "sample_quick.csv"
    df.to_csv(csv_path, index=False)

    # One-liner!
    print("🚀 Running with auto_ml() convenience function...\n")

    results = auto_ml(csv_path, target_column="price", verbose=True)

    print("\n✅ Done!")
    print(f"Problem: {results['problem_definition'].problem_name}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PLUG-AND-PLAY ML/DL FRAMEWORK - EXAMPLES")
    print("="*70)

    try:
        # Run examples
        example_1_classification()
        input("\n\nPress Enter to continue to Example 2...")

        example_2_regression()
        input("\n\nPress Enter to continue to Example 3...")

        example_3_text_classification()
        input("\n\nPress Enter to continue to Example 4...")

        example_4_quick_analysis()
        input("\n\nPress Enter to continue to Example 5...")

        example_5_convenience_function()

        print("\n\n" + "="*70)
        print("ALL EXAMPLES COMPLETED!")
        print("="*70)
        print("\n🎉 The Plug-and-Play ML/DL Framework is ready to use!")
        print("\nNext steps:")
        print("  1. Try with your own CSV files")
        print("  2. Experiment with different datasets")
        print("  3. Explore the auto-generated recommendations")
        print("\nHappy AutoML-ing! 🚀\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
