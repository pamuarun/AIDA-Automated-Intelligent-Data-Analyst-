# utils/file_utils.py

import os
import pandas as pd

def ensure_folder(path: str):
    """
    Creates folder if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def save_plot(plot_bytes: bytes, filename: str, folder="outputs/visualizations"):
    """
    Save a plot PNG bytes to disk.
    """
    ensure_folder(folder)

    file_path = os.path.join(folder, filename)
    with open(file_path, "wb") as f:
        f.write(plot_bytes)

    return file_path


def save_cleaned_dataset(df: pd.DataFrame, filename="cleaned_dataset.csv", folder="outputs"):
    """
    Save cleaned dataset as CSV.
    """
    ensure_folder(folder)

    file_path = os.path.join(folder, filename)
    df.to_csv(file_path, index=False)

    return file_path


def load_file(uploaded_file):
    """
    Loads a file from Streamlit uploader into a DataFrame.
    Supports CSV and Excel.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")
