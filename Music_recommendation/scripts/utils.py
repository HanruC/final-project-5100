import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score

from scripts.dataset import load_specific_datasets


# function to evaluate the model performance and deliver evaluation visualizations
def evaluate_model(y_true, y_pred, model_name, label_encoder_mood):
    accuracy = accuracy_score(y_true, y_pred)  # Accuracy
    f1 = f1_score(y_true, y_pred, average='weighted')  # F1 score
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)  # Precision
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)  # Recall

    print(f"{model_name} Performance:")  # Print model performance
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    unique_classes = np.unique(y_true)  # Get all classes of the target variable
    target_names = label_encoder_mood.inverse_transform(unique_classes)  # Inverse-encode class labels to original labels

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=target_names, zero_division=1))  # Print classification report

    if model_name == 'User Behavior Prediction Mood Model':
        out_dir = './output/user_behavior_runs/'
    else:
        out_dir = './output/music_runs/'

    # Create the folder if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)  # Calculate the confusion matrix
    plt.figure(figsize=(8, 6))

    # Use numeric labels instead of long string labels
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)

    plt.title(f'{model_name} - Confusion Matrix')  # Plot the confusion matrix heatmap
    plt.xlabel('Predicted Mood (Numeric)')  # X-axis title
    plt.ylabel('Actual Mood (Numeric)')  # Y-axis title

    # Save the confusion matrix image
    plt.savefig(out_dir + 'confusion_matrix.png')
    plt.close()  # Close the figure to release memory

    # Misclassification analysis plot
    error_mask = y_true != y_pred
    mis_counts = np.bincount(np.array(y_true)[error_mask], minlength=len(label_encoder_mood.classes_))
    plt.figure(figsize=(8, 5))
    sns.barplot(x=np.arange(len(mis_counts)), y=mis_counts)
    plt.title(f'{model_name} - Misclassification Count')
    plt.xlabel('Actual Class')
    plt.ylabel('Misclassified Samples')
    plt.savefig(out_dir + 'misclassification_bar.png')
    plt.close()


# function to creates correlation heatmap and pairplot, and distribution plots for the audio features
def visualize_spotify_statistics(data_dirs):
    # Suppress FutureWarning warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Create a folder to save the images
    output_dir = './output/spotify_statistics_visualization'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and merge all datasets
    all_datasets = load_specific_datasets(data_dirs)
    combined_data = all_datasets.get("spotify_songs")  # Only take the spotify_songs dataset
    print(combined_data)
    if isinstance(combined_data, dict):
        combined_data = pd.concat(combined_data.values(), ignore_index=True)  # Concatenate all DataFrames

    # Define statistical features
    music_features = ['duration_ms', 'key', 'tempo', 'valence', 'liveness', 'energy', 
                      'loudness', 'acousticness' , 'danceability', 'speechiness']

    # Filter out the selected features from the combined data
    available_music_features = [feature for feature in music_features if feature in combined_data.columns]
    music_data = combined_data[available_music_features]

    # Convert non-numeric columns to numeric types, use .loc to avoid SettingWithCopyWarning
    for col in music_data.columns:
        music_data.loc[:, col] = pd.to_numeric(music_data[col], errors='coerce')  # Replace with NaN if unable to convert to numeric

    # Suppress RuntimeWarning warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Calculate the correlation matrix
    correlation_matrix = music_data.corr()  # Calculate the correlation matrix between features

    # Visualization: Correlation heatmap
    plt.figure(figsize=(12, 10))  # Set the canvas size
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')  # Plot the correlation heatmap, showing each correlation coefficient
    plt.title('Music Feature Correlation Heatmap')  # Chart title
    # Adjust the chart position, move the image up
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))  # Save the heatmap
    plt.close()  # Close the current chart

    # Plot a Pairplot for the main numerical features (select only the first 7 features for readability)
    key_features = music_data.select_dtypes(include=['float64', 'int64']).columns[:7]  # Select the first 7 numerical features
    sns.pairplot(music_data[key_features])  # Plot a scatter plot matrix between features
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))  # Save the Pairplot
    plt.close()  # Close the current chart

    # Plot the distribution of representative features
    for feature in key_features:
        plt.figure(figsize=(8, 4))  # Set the canvas size
        sns.histplot(music_data[feature], bins=30, kde=True)  # Plot the histogram and KDE density curve of the feature
        plt.title(f'Distribution of {feature}')  # Set the title
        plt.xlabel(feature)  # Set the X-axis label
        plt.ylabel('Frequency')  # Set the Y-axis label
        plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))  # Save the distribution plot
        plt.close()  # Close the current chart    