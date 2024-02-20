from joblib import load


def load_model(model_path):
    return load(model_path)


def get_time_series_forecast(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast


def add_to_dataset(df, forecast):
    df['Forecast'] = forecast
    return df
