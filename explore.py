import pandas as pd

# Load the dataset
df = pd.read_csv("Combined Data.csv")

# See the first 5 rows
print("=== FIRST 5 ROWS ===")
print(df.head())

# See how many rows and columns
print("\n=== SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# See all column names
print("\n=== COLUMNS ===")
print(df.columns.tolist())

# See what categories we're predicting
print("\n=== UNIQUE LABELS ===")
print(df['status'].value_counts())

# Check for missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())