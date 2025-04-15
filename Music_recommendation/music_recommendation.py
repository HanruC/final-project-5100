import re
import joblib
import pandas as pd

from scripts.dataset import load_user_behavior_data, features_for_clustering


# predict emotions for users based on their behavior data using pre-trained model
def predict_emotion_for_single_instance(model_file):
    # Load the trained model
    clf = joblib.load(model_file)

    # Load user behavior data
    X_resampled, y_resampled, label_encoder_mood = load_user_behavior_data()

    # Use the model for prediction
    prediction = clf.predict(X_resampled)
    
    # Inverse decode to get emotion labels
    predicted_mood = label_encoder_mood.inverse_transform(prediction)
    
    return predicted_mood


def main():
    # Predict Emotion Based on User Behavior
    # Predict user behavior data
    predicted_mood = predict_emotion_for_single_instance(model_file='./output/user_behavior_prediction_mood_model.pkl')
    print(f"The predicted mood is: {predicted_mood}")
    print("\n")

    # Music Recommendation 
    # Use the predicted emotion of the first user as input
    input_mood = predicted_mood[0]

    # Load the trained model and label encoder
    clf = joblib.load('./output/music_prediction_mood_model.pkl')
    label_encoder = joblib.load('./output/music_label_encoder.pkl')
    
    # Mapping input mood tags to corresponding music recommendation labels
    mood_label_mapping = {
        'relax': ['relax', 'calm', 'deep'],
        'sad': ['melancholy'],
        'party': ['party', 'fun'],
        'motivation': ['motivating', 'uplifting'],
        'happy': ['happy', 'fun', 'uplifting'],
        'talk': ['talkative'],
        'live': ['live'],
        'other': ['other']
    }

    # Assume input_mood is a string containing multiple emotion tags (e.g., "Relaxation and stress relief, Uplifting and mo...")
    # Split the compound mood tags (separated by commas and 'and')
    input_mood_tags = [tag.strip() for tag in re.split(r',|\band\b', input_mood)]

    # Get the tags that match the label mapping
    matched_labels = []
    for input_tag in input_mood_tags:
        matched = False  # Flag to indicate if a matching label is found
        # Traverse the mapping table to find the mood containing the label, case-insensitive match
        for key, labels in mood_label_mapping.items():
            if any(tag.lower() in input_tag.lower() for tag in labels):  # Partial match, case-insensitive
                matched_labels.extend(labels)
                matched = True
                break
        if not matched:
            matched_labels.append('other')

    # Read the clustered data
    df = pd.read_csv('./output/clustered_songs.csv')

    # Filter the songs matching the emotion labels
    subset = df[df['mood_label'].isin(matched_labels)]

    # Feature data
    X_subset = subset[features_for_clustering]

    # Predict emotion scores
    proba = clf.predict_proba(X_subset)

    # Get the index of the emotion score for input_mood, matched_labels[0] is the main label
    main_target_label = matched_labels[0]
    target_class_idx = list(label_encoder.classes_).index(main_target_label)
    score_col_name = f'{main_target_label}_score'
    
    out_subset = subset.copy()
    out_subset.loc[:, score_col_name] = proba[:, target_class_idx]

    # Sort and select the Top 10
    top10_songs = out_subset.sort_values(by=score_col_name, ascending=False).head(10)

    # Output the results
    print(f"The mood is: {input_mood}")
    print("Music recommendation:")
    print(top10_songs[['name', 'artists', 'mood_label', score_col_name]])


if __name__ == "__main__":
    main()
