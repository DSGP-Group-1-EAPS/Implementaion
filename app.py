from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__, template_folder='Templates')

# Load the trained random forest model
rf_model = load('C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/rf_model.joblib')

# Selected features for prediction
selected_features = ['DaysWorked', 'DayOfWeek', 'Encoded Code', 'LeaveMonth', 'Encoded Status',
                     'Encoded Absenteeism Type', 'Encoded Shift', 'LeaveYear', 'NumOfLeaveDays',
                     'Reason_0', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Reason_5',
                     'Reason_6', 'Reason_7', 'Reason_8', 'Reason_9', 'Reason_10', 'Reason_11',
                     'Reason_12', 'Reason_13', 'Reason_14', 'Reason_15', 'Reason_16', 'Reason_17',
                     'Reason_18']

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('EAPSPage.html')

    if request.method == 'POST':
        # Check if the file is present in the request
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return 'No file selected', 400

        # Read the Excel file into a pandas DataFrame
        try:
            df = pd.read_excel(file)
        except Exception as e:
            return f'Error reading Excel file: {e}', 400

        # Select only the relevant features for prediction
        df_selected = df[selected_features]

        # Make predictions using the trained model
        predictions = rf_model.predict(df_selected)

        # Get the indices of predicted values greater than 4
        predicted_indices_greater_than_4 = df_selected.index[predictions > 4]

        # Get the corresponding 'Encoded Code' values at those indices
        encoded_codes_greater_than_4 = df_selected.loc[predicted_indices_greater_than_4, 'Encoded Code']

        # Convert predictions to a list
        predictions_list = predictions.tolist()

        # Convert 'Encoded Code' values to a list
        encoded_codes_list = encoded_codes_greater_than_4.tolist()

        # Prepare JSON response
        response = {
            'predictions': predictions_list,
            'encoded_codes_greater_than_4': encoded_codes_list
        }

        # Return JSON response
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
