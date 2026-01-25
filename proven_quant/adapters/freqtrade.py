import pandas as pd


def merge_labels_into_dataframe(
    dataframe: pd.DataFrame,
    labels: pd.Series,
    label_col_name: str = 'label',
) -> pd.DataFrame:
    """Merge labels into the original dataframe.

    Args:
        dataframe (pd.DataFrame): Original DataFrame.
        labels (pd.Series): Series containing labels with datetime index.
        label_col_name (str): Name of the label column to be added. Defaults to 'label'.

    Returns:
        pd.DataFrame: DataFrame with labels merged.
    """
    dataframe = dataframe.copy()
    dataframe[label_col_name] = labels
    return dataframe