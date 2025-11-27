"""
Simplest Possible Demo - Plug-and-Play ML/DL Framework

This is the absolute simplest way to use the system.
Perfect for getting started!
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.plug_and_play import PlugAndPlayML


# Create a simple dataset
print("Creating sample dataset...")

np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.choice(['A', 'B', 'C'], 100),
    'target': np.random.choice([0, 1], 100)
})

# Save to CSV
data.to_csv('demo_data.csv', index=False)
print("✅ Sample data saved to demo_data.csv\n")

# THE MAGIC - Just 3 lines!
print("🚀 Running Plug-and-Play ML...\n")

model = PlugAndPlayML()
results = model.run('demo_data.csv')

print("\n" + "="*60)
print("✅ DONE!")
print("="*60)
print(f"\nProblem Detected: {results['problem_definition'].problem_name}")
print(f"Recommended Model: {results['pipeline_recommendation'].models[0].model_name}")
print(f"Framework: {results['pipeline_recommendation'].models[0].framework}")

print("\n🎯 That's it - the system analyzed your data automatically!")
print("Check the output above for detailed analysis.\n")
