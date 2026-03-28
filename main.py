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

def feature_engineering(df):

    df = df.copy()

    # ---------------------------------------------------------------
    # For each audio feature that exists for tracks 0, 1, and 2,
    # we create two new summary columns:
    #
    #   _mean → the average value across the 3 tracks
    #   _std  → how much the value varies across the 3 tracks
    #           (std = standard deviation — 0 means all tracks are
    #            identical, high means they vary a lot)
    #
    # Example:
    #   rhythmic_cohesion_0 = 0.8
    #   rhythmic_cohesion_1 = 0.6
    #   rhythmic_cohesion_2 = 0.7
    #   → rhythmic_cohesion_mean = 0.70
    #   → rhythmic_cohesion_std  = 0.08
    # ---------------------------------------------------------------

    # List of all the base feature names (without the _0, _1, _2 suffix)
    base_features = [
        'rhythmic_cohesion',       # danceability
        'intensity_index',         # energy
        'organic_texture',         # acousticness
        'beat_frequency',          # tempo in BPM
        'emotional_charge',        # valence × energy
        'groove_efficiency',       # energy / danceability ratio
        'organic_immersion',       # acousticness × duration
        'emotional_resonance',     # another derived emotion metric
        'performance_authenticity',# live performance feel
        'vocal_presence',          # how prominent vocals are
        'instrumental_density',    # how many instruments
        'duration_ms',             # track length in milliseconds
        'harmonic_scale',          # musical key (0-11)
        'tonal_mode',              # major (1) or minor (0)
        'time_signature',          # meter e.g. 3/4 or 4/4
    ]

    for base in base_features:
        # Collect the 3 track columns if they exist in the dataframe
        # e.g. ['rhythmic_cohesion_0', 'rhythmic_cohesion_1', 'rhythmic_cohesion_2']
        track_cols = [
            f'{base}_0',
            f'{base}_1',
            f'{base}_2'
        ]
        # Only keep the ones that actually exist in this dataframe
        existing_cols = [c for c in track_cols if c in df.columns]

        if existing_cols:
            # axis=1 means "calculate across columns" (row by row)
            df[f'{base}_mean'] = df[existing_cols].mean(axis=1)
            df[f'{base}_std']  = df[existing_cols].std(axis=1)

    print(f"✓ Created mean and std features for {len(base_features)} audio feature groups")

    # ---------------------------------------------------------------
    # Fill any new NaN values that appeared from std calculation
    # (if only 1 track exists for a row, std will be NaN)
    # We fill with 0 because "no variation" is a reasonable default
    # ---------------------------------------------------------------
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )
    print("✓ Filled any new missing values created during engineering")

    return df


# Apply feature engineering to both cleaned datasets
train_fe = feature_engineering(train_clean)
test_fe  = feature_engineering(test_clean)

print("\nTrain shape before feature engineering:", train_clean.shape)
print("Train shape after feature engineering: ", train_fe.shape)

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

