from flask import Flask, render_template, request
import pandas as pd
from RandomForestClassificationModel import rf_load_model, get_features, predict, get_high_prob_employee_info
from SARIMA_Model import get_time_series_forecast, add_to_dataset, ts_load_model
import matplotlib.pyplot as plt
import seaborn as sns
from S3Connection import access_iam_role, get_resource, get_bucket, get_model

app = Flask(__name__, template_folder='Templates')

rf_selected_features = ['Encoded Code', 'Encoded Department', 'YearsWorked', 'DayOfWeek',
                        'LeaveMonth', 'LeaveYear', 'Encoded Reason', 'Encoded Status',
                        'Encoded Absenteeism Type', 'Encoded Shift', 'MonthlyDeptTotal']


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

        # Add IAM Role Credentials to the session
        s3_iam_role = access_iam_role('AKIAZQ3DTKYSPHFJXFEG', 'zhysfEeBye5EF36jGFKLDdz22QcaVqgEasfBKzbn',
                                      'ap-south-1')
        s3 = get_resource(s3_iam_role, 's3')
        s3_bucket = get_bucket(s3, 'eapss3')
        s3_bucket.upload_fileobj(file, f"Datasets/{df['LeaveYear'][0]}_{df['LeaveMonth'][0]}_data.xlsx")

        get_model('eapss3', 'Models/Sewing_sarimax.pkl', 'Model/Sewing_sarimax.pkl')
        get_model('eapss3', 'Models/Mat_sarimax.pkl', 'Model/Mat_sarimax.pkl')
        get_model('eapss3', 'Models/Jumper_sarimax.pkl', 'Model/Jumper_sarimax.pkl')

        sewing_model = ts_load_model('Model/Sewing_sarimax.pkl')
        mat_model = ts_load_model('Model/Mat_sarimax.pkl')
        jumper_model = ts_load_model('Model/Jumper_sarimax.pkl')

        sewing_forecast = get_time_series_forecast(sewing_model, 3)
        mat_forecast = get_time_series_forecast(mat_model, 3)
        jumper_forecast = get_time_series_forecast(jumper_model, 3)

        if df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 9:
            updated_df = add_to_dataset(df, sewing_forecast[22], mat_forecast[22], jumper_forecast[22])
        elif df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 10:
            updated_df = add_to_dataset(df, sewing_forecast[23], mat_forecast[23], jumper_forecast[23])
        elif df['LeaveYear'][0] == 2023 and df['LeaveMonth'][0] == 11:
            updated_df = add_to_dataset(df, sewing_forecast[24], mat_forecast[24], jumper_forecast[24])

        get_model('eapss3', 'Models/rf_model_ts_original.pkl', 'Model/rf_model_ts_original.pkl')

        df_selected = get_features(updated_df, rf_selected_features)
        rf_model = rf_load_model('Model/rf_model_ts_original.pkl')
        predictions = predict(rf_model, df_selected)

        employee_codes, departments, probabilities = get_high_prob_employee_info(rf_model, df_selected,
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
        plt.tight_layout()
        leave_reason_plot_path = 'Images/leave_reason_plot.jpeg'
        plt.savefig(leave_reason_plot_path)

        return render_template('EAPSPage.html', predictions=predictions_list, probabilities=probabilities,
                               departments=departments, leave_reason_plot_path=leave_reason_plot_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
