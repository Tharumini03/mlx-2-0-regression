import pandas as pd      
import numpy as np      
from sklearn.preprocessing import LabelEncoder


# DATA CLEANING
def clean_data(df):
    df = df.copy()

    # too many unique values to be useful
    columns_to_drop = [
        'track_identifier',     
        'creator_collective',   
        'composition_label_0',  
        'composition_label_1',  
        'composition_label_2'   
    ]
    df.drop(columns=columns_to_drop, inplace=True)
    print("✓ Dropped unhelpful text columns") 

    # Extract the year and month as separate number columns
    if 'publication_timestamp' in df.columns:
        df['publication_timestamp'] = pd.to_datetime(
            df['publication_timestamp'], errors='coerce'
        )
        
        df['release_year']  = df['publication_timestamp'].dt.year   
        df['release_month'] = df['publication_timestamp'].dt.month 

        df.drop(columns=['publication_timestamp'], inplace=True)
        print("✓ Converted dates → release_year and release_month")
    
    # Convert text categories to numbers using Label Encoding
    categorical_columns = ['weekday_of_release', 'season_of_release', 'lunar_phase']
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    print("✓ Converted text categories to numbers")

    # Fill missing values with the MEDIAN of that column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )
    print("✓ Filled missing values with column medians")

    return df



# Load the data files
train = pd.read_csv('train.csv')   
test  = pd.read_csv('test.csv')   

print("Training set shape:", train.shape)   
print("Test set shape:", test.shape)

print("\nFirst 3 rows of training data:")
print(train.head(3))

# Apply the cleaning function to both datasets
train_clean = clean_data(train)
test_clean  = clean_data(test)

print("\nTrain shape after cleaning:", train_clean.shape)
print("Test shape after cleaning:", test_clean.shape)

# Verify no missing values remain in numeric columns

remaining_missing = train_clean.isnull().sum().sum()
print("Total missing values remaining:", remaining_missing)
print(train_clean[['release_year', 'release_month', 'weekday_of_release', 'season_of_release', 'lunar_phase']].head())

