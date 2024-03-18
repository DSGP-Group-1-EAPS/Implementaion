from io import BytesIO

from flask import Flask, render_template, request
import pandas as pd
from RandomForestClassificationModel import rf_load_model, get_features, predict, get_high_prob_employee_info
from SARIMA_Model import get_time_series_forecast, add_to_dataset, ts_load_model
from flask import jsonify
from S3Connection import access_iam_role, get_resource, get_bucket, get_model, download_dataset
from Preprocessing import feature_engineering, remove_features, get_last_month

app = Flask(__name__, template_folder='Templates')

rf_selected_features = ['Encoded Code', 'Encoded Department', 'YearsWorked', 'DayOfWeek',
                        'LeaveMonth', 'LeaveYear', 'Encoded Reason', 'Encoded Status',
                        'Encoded Absenteeism Type', 'Encoded Shift', 'MonthlyDeptTotal']
updated_df = pd.DataFrame()


@app.route('/')
def index():
    return render_template('Main.html')


@app.route('/EAPSPage', methods=['GET', 'POST'])
def main():
    global updated_df, X_retrain, Y_retrain
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

        # Download training dataset from S3(s3://eapss3/Datasets/Training Dataset/training_dataset_original.xlsx)
        # (If training dataset's last month is not == to provided dataset's month - 1), then
        #   download prev_month_data from S3 and update that prev_month_data using the provided months data and retrain
        #   the model for the prev_month_data and then get predictions
        # (else)
        #   save the current months data as the prev_month_data and get predictions as usual

        # Add IAM Role Credentials to the session
        s3_iam_role = access_iam_role('AKIAZQ3DTKYSPHFJXFEG', 'zhysfEeBye5EF36jGFKLDdz22QcaVqgEasfBKzbn',
                                      'ap-south-1')
        s3 = get_resource(s3_iam_role, 's3')
        s3_bucket = get_bucket(s3, 'eapss3')
        MonthltDeptTotal = pd.read_excel('Datasets/cleaned_Monthly_Dept_Total.xlsx')
        training_df = download_dataset('Datasets/Training Dataset/training_dataset_original.xlsx')
        if get_last_month(training_df) != df['LeaveMonth'][0] - 1:
            prev_month_data = download_dataset('Datasets/Training Dataset/prev_monthly_data.xlsx')
            updated_training_df = remove_features(training_df)
            combined_df = pd.concat([updated_training_df, prev_month_data, df])
            print("Datasets combined")
            preprocessed_retraining_df = feature_engineering(combined_df, MonthltDeptTotal)
            preprocessed_retraining_df = preprocessed_retraining_df[preprocessed_retraining_df['Date'] < f'2023-{get_last_month(preprocessed_retraining_df)}-01']
            print(preprocessed_retraining_df.shape)
            print("Dataset preprocessed and seperated")
            X_retrain = preprocessed_retraining_df[rf_selected_features]
            Y_retrain = preprocessed_retraining_df['TargetCategory']

        get_model('eapss3', 'Models/rf_model_original.pkl', 'Model/rf_model_original.pkl')
        get_model('eapss3', 'Models/Catboost_model_original.pkl', 'Model/Catboost_model_original.pkl')
        get_model('eapss3', 'Models/LightGBM_model_original.pkl', 'Model/LightGBM_model_original.pkl')

        rf_model = rf_load_model('Model/rf_model_original.pkl')
        print("rf model loaded")
        cb_model = rf_load_model('Model/Catboost_model_original.pkl')
        print("cb model loaded")
        lgbm_model = rf_load_model('Model/LightGBM_model_original.pkl')
        print("lgbm model loaded")

        rf_model.fit(X_retrain, Y_retrain)
        print("RF model retrained")
        cb_model.fit(X_retrain, Y_retrain)
        print("CatBoost model retrained")
        lgbm_model.fit(X_retrain, Y_retrain)
        print("LGBM model retrained")

        # Upload the user-provided file to S3
        s3_bucket.upload_fileobj(BytesIO(file.read()), f"Datasets/{df['LeaveYear'][0]}_{df['LeaveMonth'][0]}_data.xlsx")
        print("User-provided file uploaded to S3")

        # Upload the updated previous month's data to S3
        # s3_bucket.upload_fileobj(BytesIO(file.read()),
        #                          'Datasets/Training Dataset/prev_monthly_data.xlsx')
        # print("Updated previous month's data uploaded to S3")
        print("File uploaded to S3")

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

        df_selected = get_features(updated_df, rf_selected_features)
        print(df_selected.shape)
        rf_pred = predict(rf_model, df_selected)
        cb_pred = predict(cb_model, df_selected)
        lgbm_pred = predict(lgbm_model, df_selected)

        rf_pred_proba = rf_model.predict_proba(df_selected)
        cb_pred_proba = cb_model.predict_proba(df_selected)
        lgbm_pred_proba = lgbm_model.predict_proba(df_selected)

        # Create a DataFrame with predictions from each model and their probabilities
        predictions_df = pd.DataFrame({
            'Employee Code': df_selected['Encoded Code'],
            'Department': df_selected['Encoded Department'],
            'RF_Pred': rf_pred,
            'RF_Proba': rf_pred_proba.max(axis=1),  # Taking the maximum probability across all classes
            'CatBoost_Pred': cb_pred,
            'CatBoost_Proba': cb_pred_proba.max(axis=1),  # Taking the maximum probability across all classes
            'LGBM_Pred': lgbm_pred,
            'LGBM_Proba': lgbm_pred_proba.max(axis=1)  # Taking the maximum probability across all classes
        })

        # Determine the majority vote for each row
        predictions_df['Majority_Vote'] = predictions_df.mode(axis=1)[0]

        # Calculate the mean probability for each majority vote
        mean_proba = []
        for index, row in predictions_df.iterrows():
            proba_sum = 0
            count = 0
            for model in ['RF', 'CatBoost', 'LGBM']:
                if row[model + '_Pred'] == row['Majority_Vote']:
                    proba_sum += row[model + '_Proba']
                    count += 1
            mean_proba.append(proba_sum / count if count > 0 else 0)

        # Add the mean probability column to the predictions DataFrame
        predictions_df['Mean_Proba'] = mean_proba
        # employee_codes, departments, probabilities = get_high_prob_employee_info(rf_model, df_selected, predictions)

        filtered_df = predictions_df[(predictions_df['Majority_Vote'] == 'B') &
                                     (predictions_df['Mean_Proba'] > 0.70) &
                                     ((predictions_df['Department'] == 2) |
                                      (predictions_df['Department'] == 0))]

        # Drop duplicate rows based on the 'Employee_Code' column to keep only unique employee codes
        filtered_df_unique = filtered_df.drop_duplicates(subset=['Employee Code'])
        print("Number of rows in filtered DataFrame:", filtered_df_unique.shape[0])

        # Create a dictionary with the data
        data = {
            'employee_codes': filtered_df_unique['Employee Code'].tolist(),
            'departments': filtered_df_unique['Department'].tolist(),
            'mean probabilities': filtered_df_unique['Mean_Proba'].tolist(),
            'majority votes': filtered_df_unique['Majority_Vote'].tolist()
        }

        # Return the data as JSON
        return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
