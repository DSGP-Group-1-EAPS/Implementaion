import pandas as pd
import RandomForestClassificationModel
import TimeSeriesModel

from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='Templates')

rf_selected_features = ['DaysWorked', 'DayOfWeek', 'Encoded Code', 'LeaveMonth', 'Encoded Status',
                        'Encoded Absenteeism Type', 'Encoded Shift', 'LeaveYear', 'NumOfLeaveDays', 'MonthlyTotal',
                        'Reason_0', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Reason_5',
                        'Reason_6', 'Reason_7', 'Reason_8', 'Reason_9', 'Reason_10', 'Reason_11',
                        'Reason_12', 'Reason_13', 'Reason_14', 'Reason_15', 'Reason_16', 'Reason_17',
                        'Reason_18']


@app.route('/', methods=['GET', 'POST'])
def main():
    global updated_df
    month = 10
    year = 2023
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

        ts_model = TimeSeriesModel.load_model(
            'C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/arima_model.pkl')
        forecast = TimeSeriesModel.get_time_series_forecast(ts_model, 3)
        print(forecast[22])
        if df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 9:
            updated_df = TimeSeriesModel.add_to_dataset(df, forecast[22])
        elif df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 10:
            updated_df = TimeSeriesModel.add_to_dataset(df, forecast[23])
        elif df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 11:
            updated_df = TimeSeriesModel.add_to_dataset(df, forecast[24])

        print("ARIMA Forecast done")
        df_selected = RandomForestClassificationModel.get_features(updated_df, rf_selected_features)
        rf_model = RandomForestClassificationModel.load_model(
            'C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/rf_model.pkl')
        predictions = RandomForestClassificationModel.predict(rf_model, df_selected)

        employee_codes = RandomForestClassificationModel.get_high_prob_employee_codes(rf_model, df_selected,
                                                                                      predictions)
        predictions_list = list(employee_codes)
        response = {
            'predictions': predictions_list
        }

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
