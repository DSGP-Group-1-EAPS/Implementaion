import pickle


def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def get_time_series_forecast(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast


def add_to_dataset(df, forecast):
    print(forecast)
    df['MonthlyTotal'] = forecast
    return df
