import pandas as pd
import hashlib


def load_local_data(file_path, data_format="dense"):
    """
    Load local dataset from a file.
    Supports dense and sparse data formats.
    :param file_path: Path to the dataset file.
    :param data_format: Format of the data ("dense", "sparse_tag", "sparse_tag_value").
    :return: Loaded data as a Pandas DataFrame or list.
    """
    if data_format == "dense":
        # Load dense data as a Pandas DataFrame
        return pd.read_csv(file_path)
    elif data_format == "sparse_tag":
        # Parse sparse tag format (e.g., feature1 feature2 ...)
        with open(file_path, 'r') as f:
            data = [line.strip().split() for line in f.readlines()]
        return data
    elif data_format == "sparse_tag_value":
        # Parse sparse tag:value format (e.g., feature1:1 feature2:0.5 ...)
        with open(file_path, 'r') as f:
            data = []
            for line in f:
                row = {k: float(v) for k, v in [pair.split(':') for pair in line.strip().split()]}
                data.append(row)
        return pd.DataFrame(data).fillna(0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def save_data(data, file_path):
    """
    Save processed data to a file.
    :param data: Data to save (Pandas DataFrame or list).
    :param file_path: Path to save the file.
    """
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, index=False)
    elif isinstance(data, list):  # Assuming sparse data
        with open(file_path, 'w') as f:
            for row in data:
                f.write(" ".join(row) + "\n")
    else:
        raise ValueError("Unsupported data type for saving!")


def private_set_intersection(set_a, set_b):
    """
    Compute the intersection of two private datasets using hashing (PSI).
    :param set_a: List of elements from party A.
    :param set_b: List of elements from party B.
    :return: List of elements in the intersection.
    """
    hash_set_a = {hashlib.sha256(x.encode()).hexdigest() for x in set_a}
    hash_set_b = {hashlib.sha256(x.encode()).hexdigest() for x in set_b}
    intersection = hash_set_a.intersection(hash_set_b)
    return [x for x in set_a if hashlib.sha256(x.encode()).hexdigest() in intersection]


def preprocess_data(data, target_column=None):
    """
    Preprocess the dataset by handling missing values and splitting features/target.
    :param data: Pandas DataFrame containing the dataset.
    :param target_column: Name of the target column (if any).
    :return: Tuple (X, y) where X is the feature matrix and y is the target vector.
    """
    # Handle missing values by filling them with the column mean
    data = data.fillna(data.mean())

    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y
    return data, None


def normalize_data(data):
    """
    Normalize numerical features in the dataset to range [0, 1].
    :param data: Pandas DataFrame containing the dataset.
    :return: Normalized dataset as Pandas DataFrame.
    """
    return (data - data.min()) / (data.max() - data.min())


def standardize_data(data):
    """
    Standardize numerical features to have zero mean and unit variance.
    :param data: Pandas DataFrame containing the dataset.
    :return: Standardized dataset as Pandas DataFrame.
    """
    return (data - data.mean()) / data.std()
