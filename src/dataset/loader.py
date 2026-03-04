"""Dataset loading utilities for MIMIC-IV discharge notes."""

import logging
import os

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

def load_discharge_dataset(
        file_path: str | None = None,
        n: int | None = None
    ) -> pl.DataFrame:
    """
    Loads the discharge dataset from a CSV file into a Polars DataFrame.
    This function attempts to load data from a specified CSV file. If no file path is provided,
    it defaults to looking for 'discharge.csv' in the 'data' directory relative to the project root.
    It supports loading a subset of rows for previewing or testing purposes.
    Parameters
    ----------
    file_path : str | None, optional
        The absolute or relative path to the CSV file. If None, the function constructs
        a default path based on the project structure (project_root/data/discharge.csv).
        Defaults to None.
    n : int | None, optional
        The number of rows to read from the CSV file. If None, the entire dataset is loaded.
        Defaults to None.
    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the loaded data.
    Raises
    ------
    FileNotFoundError
        If the file at the specified or default `file_path` does not exist.
    Exception
        If any error occurs during the file reading process (e.g., parsing errors, permission issues).
    Examples
    --------
    >>> # Load the full default dataset
    >>> df = load_discharge_dataset()
    >>> # Load only the first 100 rows from a custom path
    >>> df = load_discharge_dataset(file_path="path/to/my_data.csv", n=100)
    """

    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        file_path = os.path.join(project_root, "data", "discharge.csv")

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found {file_path}")

    logger.debug("Loading dataset from: %s ...", file_path)
    
    try:
        # Polars is optimised for fast reading of large files
        if n is not None:
            df = pl.read_csv(file_path, n_rows=n)
        else:
            df = pl.read_csv(file_path)
        logger.info("Dataset loaded successfully! Dimensions: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error loading the dataset: %s", e)
        raise e
