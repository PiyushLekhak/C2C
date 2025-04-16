import pandas as pd
import os
import warnings


def load_data(
    file_paths, column_names=None, missing_values=["?"], header=None, skiprows=None
):
    """
    Load dataset(s) from one or more CSV files into a single pandas DataFrame.

    Args:
        file_paths (Union[str, List[str]]): Path or list of paths to CSV file(s) to be loaded.
        column_names (Optional[List[str]], optional): List of column names to assign if the CSV files do not have headers. Ignored if 'header' is specified. Defaults to None.
        missing_values (Optional[List[str]], optional): List of string values to interpret as missing/NaN. Defaults to ["?"].
        header (Optional[int], optional): Row number to use as column headers (0-indexed). Set to None if there is no header row in the file. Defaults to None.
        skiprows (Optional[int], optional): Number of lines to skip at the beginning of the file. Defaults to None.

    Raises:
        FileNotFoundError: If any specified file path does not exist.
        ValueError: If any specified file is empty.
        ValueError: If an error occurs while reading a file (e.g., parsing errors, encoding issues).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data. If multiple files are provided, they are concatenated into a single DataFrame.
    """
    if header is not None and column_names is not None:
        warnings.warn(
            "Both 'header' and 'column_names' provided. 'column_names' will be ignored."
        )

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    dataframes = []

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"File is empty: {path}")

        try:
            df = pd.read_csv(
                path,
                header=header,
                names=column_names if header is None else None,
                na_values=missing_values,
                skipinitialspace=True,
                skiprows=skiprows,
            )
            dataframes.append(df)
        except Exception as e:
            raise ValueError(f"Error loading {path}: {str(e)}")

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pd.concat(dataframes, ignore_index=True)
