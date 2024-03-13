import pickle
import pandas as pd


def onehot_encode(df, column, prefix):
    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)

    return df


def rf_load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def get_features(df, selected_features):
    return df[selected_features]


def predict(model, df_selected):
    return model.predict(df_selected)


def get_high_prob_employee_info(model, df, predictions):
    # Get the predicted probabilities for each class
    predicted_probabilities = model.predict_proba(df)

    # Get the index of class B
    class_b_index = list(model.classes_).index('B')

    # Extract the probabilities for class B
    class_b_probabilities = predicted_probabilities[:, class_b_index]

    high_confidence_employee_info = {}

    for i, (predicted, probability) in enumerate(zip(predictions, class_b_probabilities)):
        if predicted == 'B' and probability > 0.75 and (df.iloc[i]['Encoded Department'] == 2 or df.iloc[i]['Encoded Department'] == 0):
            encoded_code = df.iloc[i]['Encoded Code']
            if encoded_code not in high_confidence_employee_info:
                encoded_department = df.iloc[i]['Encoded Department']
                high_confidence_employee_info[encoded_code] = {'department': encoded_department, 'probability': probability}

    # Separate the dictionary into lists for employee codes, departments, and probabilities
    high_confidence_employee_codes = list(high_confidence_employee_info.keys())
    high_confidence_departments = [info['department'] for info in high_confidence_employee_info.values()]
    high_confidence_probabilities = [info['probability'] for info in high_confidence_employee_info.values()]

    return high_confidence_employee_codes, high_confidence_departments, high_confidence_probabilities




