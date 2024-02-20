from joblib import load


def load_model(model_path):
    return load(model_path)


def get_features(df, selected_features):
    return df[selected_features]


def predict(model, df_selected):
    return model.predict(df_selected)


def get_high_prob_employee_codes(model, df, predictions):
    # Get the predicted probabilities for each class
    predicted_probabilities = model.predict_proba(df)

    # Get the index of class B
    class_b_index = list(model.classes_).index('B')

    # Extract the probabilities for class B
    class_b_probabilities = predicted_probabilities[:, class_b_index]

    # Assuming the confidence level is the probability of class B
    confidence_level_b = class_b_probabilities

    # Filter predictions with probability > 0.9
    high_confidence_predictions = [(predicted, probability) for predicted, probability in
                                   zip(predictions, confidence_level_b) if probability > 0.9]

    # Display filtered predictions
    for predicted, probability in high_confidence_predictions:
        print(f"Predicted: {predicted}, Probability: {probability}")

    # Filter predictions with probability > 0.9 and predicted as category B
    high_confidence_category_b = [df.iloc[i]['Encoded Code'] for i, (_, predicted, probability) in
                                  enumerate(zip(predictions, confidence_level_b)) if
                                  predicted == 'B' and probability > 0.9]

    return high_confidence_category_b
