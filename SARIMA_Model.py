from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle


def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def get_time_series_forecast(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast


def add_to_dataset(df, sewing_forecast, mat_forecast, jumper_forecast):
    for index, row in df.iterrows():
        sub_dept = row['Department']
        if sub_dept.startswith('Sewing Team'):
            df.at[index, 'MonthlyTotal'] = sewing_forecast
        elif sub_dept.startswith('Maternity'):
            df.at[index, 'MonthlyTotal'] = mat_forecast
        elif sub_dept.startswith('Jumper'):
            df.at[index, 'MonthlyTotal'] = jumper_forecast
    return df


