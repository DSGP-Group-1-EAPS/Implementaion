from joblib import load
import pandas as pd

class Preprocessing:
    @staticmethod
    def select_features(df, selected_features):
        return df[selected_features]

class RandomForestModel:
    def __init__(self, model_path, selected_features):
        self.model = load(model_path)
        self.selected_features = selected_features

    def predict(self, df_selected):
        return self.model.predict(df_selected)

class TimeSeriesModel:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_time_series_data(self):
        leave_days_by_month = self.dataset.groupby(['LeaveYear', 'LeaveMonth']).size().reset_index(name='TotalLeaveDays')
        return leave_days_by_month['TotalLeaveDays']

    def get_time_series_forecast(self, model, steps):
        forecast = model.forecast(steps=steps)
        return forecast


from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='Templates')

rf_selected_features = ['DaysWorked', 'DayOfWeek', 'Encoded Code', 'LeaveMonth', 'Encoded Status',
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
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']

        if file.filename == '':
            return 'No file selected', 400

        try:
            df = pd.read_excel(file)
        except Exception as e:
            return f'Error reading Excel file: {e}', 400

        df_selected = Preprocessing.select_features(df, rf_selected_features)

        rf_model = RandomForestModel('C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/rf_model.joblib', rf_selected_features)

        predictions = rf_model.predict(df_selected)

        predicted_indices_greater_than_4 = df_selected.index[predictions > 4]
        encoded_codes_greater_than_4 = df_selected.loc[predicted_indices_greater_than_4, 'Encoded Code']
        predictions_list = predictions.tolist()
        encoded_codes_list = encoded_codes_greater_than_4.tolist()

        response = {
            'predictions': predictions_list,
            'encoded_codes_greater_than_4': encoded_codes_list
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
