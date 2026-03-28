# Step 1: Import the tools we need
import pandas as pd      # for reading and handling data (like Excel in Python)
import numpy as np       # for math operations on arrays of numbers

# Step 2: Load the data files
train = pd.read_csv('train.csv')   # songs WITH popularity scores
test  = pd.read_csv('test.csv')    # songs WITHOUT popularity scores (we predict these)

# Step 3: Explore what we're working with
print("Training set shape:", train.shape)   # (rows, columns)
print("Test set shape:", test.shape)

print("\nFirst 3 rows of training data:")
print(train.head(3))

print("\nColumn names:")
print(list(train.columns))

print("\nPopularity score statistics:")
print(train['target'].describe())