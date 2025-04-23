import pandas as pd
import os
import warnings


def load_data(
    file_paths, column_names=None, missing_values=["?"], header=None, skiprows=None
):
    """
    Loads one or more files (CSV, Excel, or JSON) into a single pandas DataFrame.

    Args:
        file_paths (str or list of str): Path or list of paths to file(s).
        column_names (list, optional): Column names to assign if the file has no header.
            Ignored if 'header' is specified. Defaults to None.
        missing_values (list, optional): Strings to interpret as missing/NaN. Defaults to ["?"].
        header (int or None, optional): Row number to use as column headers (0-indexed).
            Set to None if the file has no header. Defaults to None.
        skiprows (int or None, optional): Number of lines to skip at the start of the file. Defaults to None.

    Raises:
        FileNotFoundError: If any specified file path does not exist.
        ValueError: If any specified file is empty or cannot be read.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
            If multiple files are provided, they are concatenated into a single DataFrame.
    """
    if header is not None and column_names is not None:
        warnings.warn(
            "Both 'header' and 'column_names' provided. 'column_names' will be ignored."
        )

    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    dataframes = []

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"File is empty, cannot proceed with loading.: {path}")

        # Get the file extension to decide the loading function
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".csv":
                df = pd.read_csv(
                    path,
                    header=header,
                    names=column_names if header is None else None,
                    na_values=missing_values,
                    skipinitialspace=True,
                    skiprows=skiprows,
                )
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(
                    path,
                    header=header,
                    names=column_names if header is None else None,
                    na_values=missing_values,
                    skiprows=skiprows,
                )
            elif ext == ".json":
                df = pd.read_json(
                    path,
                    encoding="utf-8",
                )
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            dataframes.append(df)
        except Exception as e:
            raise ValueError(f"Error loading {path}: {str(e)}")

    if len(dataframes) == 1:
        return dataframes[0]
    else:
        return pd.concat(dataframes, ignore_index=True)
