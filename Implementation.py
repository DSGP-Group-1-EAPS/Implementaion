from joblib import load
import pandas as pd
import RandomForestClassificationModel
import TimeSeriesModel

from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='Templates')

rf_selected_features = ['DaysWorked', 'DayOfWeek', 'Encoded Code', 'LeaveMonth', 'Encoded Status',
                        'Encoded Absenteeism Type', 'Encoded Shift', 'LeaveYear', 'NumOfLeaveDays', 'forecast',
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

        ts_model = TimeSeriesModel.load_model('C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/ts_model.joblib')
        forecast = TimeSeriesModel.get_time_series_forecast(ts_model, 1)
        updated_df = TimeSeriesModel.add_to_dataset(df, forecast)

        df_selected = RandomForestClassificationModel.get_features(updated_df, rf_selected_features)
        rf_model = RandomForestClassificationModel.load_model(
            'C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/rf_model.joblib')
        predictions = RandomForestClassificationModel.predict(rf_model, df_selected)

        employee_codes = RandomForestClassificationModel.get_high_prob_employee_codes(rf_model, df_selected,
                                                                                      predictions)
        predictions_list = employee_codes.tolist()
        response = {
            'predictions': predictions_list
        }

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
