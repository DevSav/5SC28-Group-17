import pandas as pd


def load_csv(path):
    """Load one CSV file."""
    return pd.read_csv(path)


def save_csv(data, path):
    """Save data to a CSV file without the index column."""
    data.to_csv(path, index=False)


def split_train_validation(data, validation_fraction=0.2):
    """Split data into a train part and a validation part."""
    split_index = int(len(data) * (1.0 - validation_fraction))
    train_data = data.iloc[:split_index].copy()
    validation_data = data.iloc[split_index:].copy()
    return train_data, validation_data

