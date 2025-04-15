import pandas as pd 
import os
from bs4 import BeautifulSoup
from io import StringIO
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler, PolynomialFeatures
from imblearn.over_sampling import RandomOverSampler


# Dictionary of dataset paths
data_dirs = {
    "spotify_songs": "./input/spotify-dataset",  # Path to the Spotify songs dataset
    "spotify_user_behavior": "./input/spotify-user-behavior-analysis",  # Path to the Spotify user behavior dataset
}


# Select feature columns for clustering
# used for clustering songs into mood categories
features_for_clustering = [
        'valence', 'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'
    ]


# Used to extract tables from HTML files (used for the user behavior dataset)
def extract_tables_from_html(html_path):
    try:
        # Open the HTML file in read mode and specify the encoding format
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')  # Parse the HTML file using BeautifulSoup

        # Use StringIO to avoid FutureWarning and extract all tables from the HTML
        tables = pd.read_html(StringIO(str(soup)))
        return tables  # Return the extracted tables
    except Exception as e:
        print(f"Error reading HTML file: {e}")  # Print error information if an error occurs
        return None  # Return None to indicate no data


# function to get all csv files in a directory and its subdirectories
def list_files(directory):
    # Return a list of all files in the directory, including files in subfolders
    file_list = []
    for root, dirs, files in os.walk(directory):  # Traverse subdirectories
        for file in files:
            if file.endswith(".csv"):  # Select only CSV files
                file_list.append(os.path.join(root, file))  # Add the file path to the list
    return file_list
    

# function to load specific datasets from defined paths
def load_specific_datasets(data_dirs):
    datasets = {}  # Dictionary for storing each dataset

    # Load the Spotify user behavior dataset (HTML format)
    spotify_html_data = extract_tables_from_html(os.path.join(data_dirs["spotify_user_behavior"], "__results__.html"))
    datasets["spotify_user_behavior"] = spotify_html_data[0] if spotify_html_data else None  # Extract HTML table data
    
    # Load the Spotify dataset
    spotify_files = list_files(data_dirs["spotify_songs"])  # Get a list of all CSV files
    if spotify_files:  # Ensure files are obtained
        spotify_data = {file.replace(".csv", ""): pd.read_csv(file) for file in spotify_files}  # Read files
        datasets["spotify_songs"] = spotify_data  # Save data
    else:
        datasets["spotify_songs"] = None  # Set to None if no files are found

    return datasets  # Return a dictionary containing all datasets


# Used to normalize feature columns in a DataFrame
def normalize_features(df, columns=None):
    if df is None:
        print("Error: DataFrame is None.")  # Print error information if the DataFrame is empty
        return None  # Return None

    scaler = MinMaxScaler()  # Initialize MinMaxScaler for data normalization
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original data

    # If no columns to be normalized are specified, select all numerical (float64 and int64) columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Filter out columns that exist in the DataFrame
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        # print(f"No matching columns found for normalization. DataFrame columns: {df.columns.tolist()}")  # Print a prompt if no matching columns are found
        return df  # Return the unprocessed DataFrame

    # Normalize the selected columns
    df[available_columns] = scaler.fit_transform(df[available_columns])
    return df  # Return the normalized DataFrame


# function to encode categorical features to numerical values
def encode_categorical_features(df):
    if df is None:
        print("Error: DataFrame is None.")  # Print error information if the DataFrame is empty
        return None, {}  # Return None and an empty dictionary

    df = df.copy()  # Create a copy of the DataFrame
    # Detect categorical columns (columns of object type or category type)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Remove the 'music_Influencial_mood' column
    categorical_columns = [col for col in categorical_columns if col != 'music_Influencial_mood']
    # print(f"Detected categorical columns for encoding: {categorical_columns}")  # Print the detected categorical columns

    label_encoders = {}  # Dictionary for storing LabelEncoders
    # Encode each column of categorical data
    for col in categorical_columns:
        le = LabelEncoder()  # Initialize LabelEncoder
        df[col] = le.fit_transform(df[col].astype(str))  # Convert the categorical column to a numerical code
        label_encoders[col] = le  # Store the encoder in the dictionary
        # print(f"Encoded column: {col}")  # Print the encoded column

    return df, label_encoders  # Return the encoded DataFrame and the LabelEncoders dictionary


# function to prepare the user behavior dataset for mood prediction modeling
def load_user_behavior_data():
    # Load data
    all_datasets = load_specific_datasets(data_dirs)
    # Normalize features of the Spotify user behavior dataset
    spotify_user_behavior_raw = all_datasets.get("spotify_user_behavior")
    if spotify_user_behavior_raw is not None:
        spotify_user_behavior = normalize_features(spotify_user_behavior_raw)

    # Encode features of the Spotify user behavior dataset
    if spotify_user_behavior is not None:
        # print("Encoding Spotify User Behavior DataFrame:")
        spotify_user_behavior, spotify_user_behavior_encoders = encode_categorical_features(spotify_user_behavior)

    # Use the DataFrame extracted from HTML
        df_cleaned = spotify_user_behavior.copy()  # Copy the original DataFrame to prevent modifying the original data

    # List of categorical features to be encoded
    categorical_features = ['Gender', 'spotify_subscription_plan', 'preferred_listening_content', 
                            'fav_music_genre', 'music_time_slot', 'music_expl_method']  

    label_encoders = {}  # Dictionary for storing LabelEncoder objects
    for col in categorical_features:
        le = LabelEncoder()  # Initialize LabelEncoder
        df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))  # Convert categorical features to numerical values
        label_encoders[col] = le  # Store the encoder

    # Encode the target variable (user mood)
    label_encoder_mood = LabelEncoder()  # Initialize LabelEncoder for the target variable
    df_cleaned['Encoded_Mood'] = label_encoder_mood.fit_transform(df_cleaned['music_Influencial_mood'].astype(str))  # Encode the 'music_Influencial_mood' column

    # Select features and target variables
    features = ['Gender', 'spotify_subscription_plan', 'preferred_listening_content', 
                'fav_music_genre', 'music_time_slot', 'music_recc_rating', 'music_expl_method']  # Define the feature list
    X = df_cleaned[features]  # Extract features
    y = df_cleaned['Encoded_Mood']  # Extract the target variable

    # Feature scaling
    scaler = RobustScaler()  # Initialize RobustScaler (insensitive to outliers)
    X_scaled = scaler.fit_transform(X)  # Scale the features

    # Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)  # Create quadratic polynomial features (consider only interaction terms)
    X_poly = poly.fit_transform(X_scaled)  # Create polynomial features

    # Use RandomOverSampler for resampling
    ros = RandomOverSampler(random_state=42)  # Initialize RandomOverSampler to handle imbalanced data
    X_resampled, y_resampled = ros.fit_resample(X_poly, y)  # Resample the data

    return X_resampled, y_resampled, label_encoder_mood    