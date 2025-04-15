import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from scripts.dataset import load_user_behavior_data, features_for_clustering, data_dirs
from scripts.model import UserMoodPredictionModel
from scripts.utils import evaluate_model, visualize_spotify_statistics

# train a model to predict user mood based on user behavior data
# loads user behavior data, trains logistic regression or random forest model, and evaluates the model
def user_behavior_prediction_mood_model_train(save_model_dir, model_type='random_forest'):
    # Load user behavior data
    X_resampled, y_resampled, label_encoder_mood = load_user_behavior_data()

    # Split the dataset (training set and test set)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)  # Split the data into training and test sets

    # Select the model to train
    print('Start to train user behavior prediction mood model!')
    if model_type == 'logistic_regression_model':
        logistic_model = UserMoodPredictionModel()  # Use the logistic regression model
        logistic_model.fit(X_train, y_train)  # Train the model
        joblib.dump(logistic_model, save_model_dir)  # Save the model
    elif model_type == 'random_forest':
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Use the random forest classifier
        rf_model.fit(X_train, y_train)  # Train the model
        joblib.dump(rf_model, save_model_dir)  # Save the model
    else:
        raise ValueError("Model type must be 'logistic_regression_model' or 'random_forest_model'")

    # Prediction on the test set
    if model_type == 'logistic_regression_model':
        logistic_pred = logistic_model.predict(X_test)  # Logistic regression prediction
    elif model_type == 'random_forest':
        rf_pred = rf_model.predict(X_test)  # Random forest prediction

    # Model evaluation
    if model_type == 'logistic_regression_model':
        evaluate_model(y_test, logistic_pred, "User Behavior Prediction Mood Model", label_encoder_mood)  # Evaluate the UserMoodPredictionModel
    elif model_type == 'random_forest':
        evaluate_model(y_test, rf_pred, "User Behavior Prediction Mood Model", label_encoder_mood)  # Evaluate the RandomForestClassifier
    print('User behavior prediction mood model training ends!')


# Function to train the model to predict song mood based on audio features
# load clustered song data with mood labels, train a random forest classifier, and evaluate the model
def music_prediction_mood_model_train(input_file, model_file, label_encoder_file):
    # Read the clustered data
    df = pd.read_csv(input_file)

    # Select the feature columns for training
    features_for_clustering = [
        'valence', 'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'
    ]

    # Extract features and labels
    X = df[features_for_clustering]  # Music audio features
    y = df['mood_label']             # Cluster labels as mood "pseudo-labels"

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)

    # Initialize and train the model
    print('\nStart train music prediction mood model!')
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Prediction on the test set
    pred = clf.predict(X_test)  # Random forest prediction

    # Model evaluation
    evaluate_model(y_test, pred, "Music Prediction Mood Model", label_encoder)  # Evaluate the UserMoodPredictionModel

    # Save the trained model
    joblib.dump(clf, model_file)
    joblib.dump(label_encoder, label_encoder_file)  # Save the label encoder for use during prediction
    print('Music prediction mood model training ends!')

    return clf, label_encoder


# assign a mood label to a cluster based on the centroid's audio features
def label_cluster_by_centroid(centroid, thresholds):
    # If both the valence and energy of the cluster center are higher than the thresholds, the label is 'happy'
    if centroid['valence'] > thresholds['valence_high'] and centroid['energy'] > thresholds['energy_high']:
        return 'happy'
    # If the valence of the cluster center is low and the acousticness is high, the label is 'melancholy'
    elif centroid['valence'] < thresholds['valence_low'] and centroid['acousticness'] > thresholds['acoustic_high']:
        return 'melancholy'
    # If both the energy and danceability of the cluster center are high, the label is 'party'
    elif centroid['energy'] > thresholds['energy_high'] and centroid['danceability'] > thresholds['dance_high']:
        return 'party'
    # If both the acousticness and instrumentalness of the cluster center are high, the label is 'relax'
    elif centroid['acousticness'] > thresholds['acoustic_high'] and centroid['instrumentalness'] > thresholds['instr_high']:
        return 'relax'
    # If the speechiness of the cluster center is high, the label is 'talkative'
    elif centroid['speechiness'] > thresholds['speech_high']:
        return 'talkative'
    # If the liveness of the cluster center is high, the label is 'live'
    elif centroid['liveness'] > thresholds['liveness_high']:
        return 'live'
    # If both the valence and tempo of the cluster center are high, the label is 'uplifting'
    elif centroid['valence'] > thresholds['valence_high'] and centroid['tempo'] > thresholds['tempo_high']:
        return 'uplifting'
    # If both the valence and energy of the cluster center are low, the label is 'calm'
    elif centroid['valence'] < thresholds['valence_low'] and centroid['energy'] < thresholds['energy_low']:
        return 'calm'
    # If the instrumentalness of the cluster center is high and the valence is low, the label is 'deep'
    elif centroid['instrumentalness'] > thresholds['instr_high'] and centroid['valence'] < thresholds['valence_low']:
        return 'deep'
    # If both the tempo and energy of the cluster center are high, the label is 'motivating'
    elif centroid['tempo'] > thresholds['tempo_high'] and centroid['energy'] > thresholds['energy_high']:
        return 'motivating'
    # If both the danceability and valence of the cluster center are high, the label is 'fun'
    elif centroid['danceability'] > thresholds['dance_high'] and centroid['valence'] > thresholds['valence_high']:
        return 'fun'
    # If none of the above conditions are met, return 'other'
    else:
        return 'other'


def main():
    # Train the user behavior prediction mood model
    user_behavior_prediction_mood_model_train(model_type='random_forest', save_model_dir='./output/user_behavior_prediction_mood_model.pkl')

    # KMeans clustering
    # Read the data
    df = pd.read_csv("./input/spotify-dataset/data/data.csv")

    # Visualize data statistics
    visualize_spotify_statistics(data_dirs)

    # Keep useful columns and remove missing values
    df_cleaned = df[features_for_clustering + ['name', 'artists']].dropna()

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cleaned[features_for_clustering])

    # Perform KMeans clustering into 12 classes
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    df_cleaned['mood_cluster'] = kmeans.fit_predict(scaled_features)

    # Inverse standardize the cluster centers for labeling
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features_for_clustering
    )

    # Define the thresholds for each mood label (based on quartiles)
    # Calculate the high and low thresholds for each feature here to determine the labels of the cluster centers later
    thresholds = {
        'valence_high': centroids['valence'].quantile(0.75),  # High threshold for valence, taking the 75th percentile
        'valence_low': centroids['valence'].quantile(0.25),  # Low threshold for valence, taking the 25th percentile
        'energy_high': centroids['energy'].quantile(0.75),  # High threshold for energy, taking the 75th percentile
        'energy_low': centroids['energy'].quantile(0.25),  # Low threshold for energy, taking the 25th percentile
        'acoustic_high': centroids['acousticness'].quantile(0.75),  # High threshold for acousticness, taking the 75th percentile
        'dance_high': centroids['danceability'].quantile(0.75),  # High threshold for danceability, taking the 75th percentile
        'instr_high': centroids['instrumentalness'].quantile(0.75),  # High threshold for instrumentalness, taking the 75th percentile
        'speech_high': centroids['speechiness'].quantile(0.75),  # High threshold for speechiness, taking the 75th percentile
        'liveness_high': centroids['liveness'].quantile(0.75),  # High threshold for liveness, taking the 75th percentile
        'tempo_high': centroids['tempo'].quantile(0.75),  # High threshold for tempo, taking the 75th percentile
    }

    # Assign labels
    cluster_labels = {
        i: label_cluster_by_centroid(centroids.iloc[i], thresholds)
        for i in range(12)
    }
    df_cleaned['mood_label'] = df_cleaned['mood_cluster'].map(cluster_labels)

    # Assume 'df_cleaned' contains music features and 'kmeans.labels_' stores the cluster labels
    df_cleaned["mood_cluster"] = kmeans.labels_  # Add the cluster labels to the data frame as a new column "mood_cluster"

    # Save the data with cluster labels to a CSV file
    df_cleaned.to_csv("./output/clustered_songs.csv", index=False)  # Save the data frame as a CSV file without the index

    # PCA dimensionality reduction and scatter plot drawing
    # PCA (Principal Component Analysis) is used for dimensionality reduction and visualization
    pca = PCA(n_components=2)  # Initialize PCA, retaining 2 principal components
    pca_result = pca.fit_transform(scaled_features)  # Perform PCA dimensionality reduction on the standardized features
    df_cleaned['PCA1'] = pca_result[:, 0]  # Add the first principal component to the data frame
    df_cleaned['PCA2'] = pca_result[:, 1]  # Add the second principal component to the data frame

    # Map the cluster labels to the labels of each cluster
    cluster_label_map = df_cleaned.groupby("mood_cluster")["mood_label"].first().to_dict()

    # Draw a scatter plot of the clusters
    plt.figure(figsize=(8, 6))  # Set the size of the plot
    scatter = plt.scatter(df_cleaned['PCA1'], df_cleaned['PCA2'], c=df_cleaned['mood_cluster'], cmap='viridis', s=10)  # Draw a scatter plot of PCA1 and PCA2, with the color of the points set according to the cluster labels

    # Draw the cluster centers
    cluster_centers = pca.transform(kmeans.cluster_centers_)  # Perform PCA dimensionality reduction on the cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')  # Draw the cluster centers (marked with red X)

    # Add labels to the cluster centers
    for i, (x, y) in enumerate(cluster_centers):
        cluster_name = cluster_label_map.get(i, f"Cluster {i}")  # Get the cluster label name, if there is no label, display "Cluster i" by default
        plt.text(x, y, cluster_name, fontsize=12, fontweight='bold', ha='center', color='black',
                 bbox=dict(facecolor='white', alpha=0.6))  # Add label text near the cluster centers

    # Finally, adjust the chart details
    plt.title('Clusters Visualization (PCA)')  # Set the chart title
    plt.xlabel('PCA 1')  # Set the X-axis label
    plt.ylabel('PCA 2')  # Set the Y-axis label
    plt.colorbar(scatter, label='Cluster')  # Add a color bar to the scatter plot to show the cluster information
    plt.legend()  # Add a legend
    # Save the scatter plot
    plt.savefig('./output/kmeans_cluster_scatter.png')

    # Train the music prediction mood model
    music_prediction_mood_model_train(input_file='./output/clustered_songs.csv', model_file='./output/music_prediction_mood_model.pkl', label_encoder_file='./output/music_label_encoder.pkl')


if __name__ == "__main__":
    main()    