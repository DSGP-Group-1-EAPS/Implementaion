from flask import Flask, render_template, request
import pandas as pd
from RandomForestClassificationModel import rf_load_model, get_features, predict, get_high_prob_employee_codes
from SARIMA_Model import ts_load_model, get_time_series_forecast, add_to_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from S3Connection import access_iam_role, get_resource, get_bucket

app = Flask(__name__, template_folder='Templates')

rf_selected_features = ['DaysWorked', 'DayOfWeek', 'Encoded Code', 'LeaveMonth', 'Encoded Status',
                        'Encoded Absenteeism Type', 'Encoded Shift', 'LeaveYear', 'NumOfLeaveDays', 'MonthlyTotal',
                        'Reason_0', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Reason_5',
                        'Reason_6', 'Reason_7', 'Reason_8', 'Reason_9', 'Reason_10', 'Reason_11',
                        'Reason_12', 'Reason_13', 'Reason_14', 'Reason_15', 'Reason_16', 'Reason_17',
                        'Reason_18']


@app.route('/')
def index():
    return render_template('Main.html')


@app.route('/EAPSPage', methods=['GET', 'POST'])
def main():
    global updated_df
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

        # Add IAm Role Credentials to the session
        s3_iam_role= access_iam_role('AKIAZQ3DTKYSPHFJXFEG', 'zhysfEeBye5EF36jGFKLDdz22QcaVqgEasfBKzbn',
                                'ap-south-1')
        s3 = get_resource(s3_iam_role, 's3')

        s3_bucket = get_bucket(s3, 'eapss3')
        s3_bucket.upload_fileobj(file, f"{df['LeaveYear'][0]}_{df['LeaveMonth'][0]}_data.xlsx")

        sewing_model = ts_load_model(
            'C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/sewing_sarima_model.pkl')
        sewing_forecast = get_time_series_forecast(sewing_model, 3)

        mat_model = ts_load_model(
            'C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/mat_sarima_model.pkl')
        mat_forecast = get_time_series_forecast(mat_model, 3)

        jumper_model = ts_load_model(
            'C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/jumper_sarima_model.pkl')
        jumper_forecast = get_time_series_forecast(jumper_model, 3)

        print("SARIMA Forecast done")

        if df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 9:
            updated_df = add_to_dataset(df, sewing_forecast[22], mat_forecast[22], jumper_forecast[22])
        elif df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 10:
            updated_df = add_to_dataset(df, sewing_forecast[23], mat_forecast[23], jumper_forecast[23])
        elif df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 11:
            updated_df = add_to_dataset(df, sewing_forecast[24], mat_forecast[24], jumper_forecast[24])

        print("ARIMA Forecast done")
        df_selected = get_features(updated_df, rf_selected_features)
        rf_model = rf_load_model(
            'C:/Ranidu/University/2nd Year/2nd Year/Semester 1/DSGP/Model/rf_model.pkl')
        predictions = predict(rf_model, df_selected)

        employee_codes = get_high_prob_employee_codes(rf_model, df_selected,
                                                      predictions)
        predictions_list = list(employee_codes)
        print(len(predictions_list))
        leave_reason_counts = df['Reason'].value_counts()

        # Create bar plot for leave reasons
        plt.figure(figsize=(10, 6))
        sns.barplot(x=leave_reason_counts.index, y=leave_reason_counts.values)
        plt.title('Leave Reasons')
        plt.xlabel('Reason')
        plt.ylabel('Number of Leaves')
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust layout to prevent overlapping labels
        leave_reason_plot_path = 'Images/leave_reason_plot.jpeg'
        plt.savefig(leave_reason_plot_path)

        return render_template('EAPSPage.html', predictions=predictions_list,
                               leave_reason_plot_path=leave_reason_plot_path)


if __name__ == '__main__':
    app.run(debug=True)
