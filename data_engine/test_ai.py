import os
import pandas as pd
from ai_engine import suggest_model_config
from dotenv import load_dotenv

load_dotenv()

# Create dummy dataframe
df = pd.DataFrame({
    'sqft': [1000, 1500, 2000],
    'bedrooms': [2, 3, 4],
    'price': [200000, 300000, 400000]
})

goal = "Predict house prices"

print(f"Goal: {goal}")
print("Calling AI...")
try:
    result = suggest_model_config(df, goal)
    print("Result:")
    print(result)
except Exception as e:
    print(f"Error: {e}")
