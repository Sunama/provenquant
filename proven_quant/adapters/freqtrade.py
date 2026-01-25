import pandas as pd

def merge_dataframe_into_labels(
    dataframe: pd.DataFrame,
    labels: pd.Series,
    datetime_col: str = 'index',
) -> pd.DataFrame:
    """Merge original dataframe features into labels based on datetime index.

    Args:
        dataframe (pd.DataFrame): Original DataFrame.
        labels (pd.Series): Series containing labels with datetime index.
        datetime_col (str): Name of the datetime column in dataframe.
                            Defaults to 'index'.

    Returns:
        pd.DataFrame: DataFrame with features merged into labels.
    """
    
    if datetime_col != 'index':
        merged_df = pd.merge(
            labels.to_frame(name='label'),
            dataframe,
            left_index=True,
            right_on=datetime_col,
            how='left'
        )
    else:
        merged_df = pd.merge(
            labels.to_frame(name='label'),
            dataframe,
            left_index=True,
            right_index=True,
            how='left'
        )
    
    merged_df.reset_index(drop=True, inplace=True)
    if 'index' in merged_df.columns:
        merged_df.drop(columns=['index'], inplace=True)
    
    return merged_df

def merge_labels_into_dataframe(
    dataframe: pd.DataFrame,
    labels: pd.Series,
    datetime_col: str = 'index',
    label_col_name: str = 'label',
) -> pd.DataFrame:
    """Merge labels into the original dataframe.

    Args:
        dataframe (pd.DataFrame): Original DataFrame.
        labels (pd.Series): Series containing labels with datetime index.
        datetime_col (str): Name of the datetime column in dataframe.
                            Defaults to 'index'.
        label_col_name (str): Name of the label column to be added.
                              Defaults to 'label'.

    Returns:
        pd.DataFrame: DataFrame with labels merged.
    """
    
    dataframe = dataframe.copy()
    
    if datetime_col != 'index':
        dataframe = pd.merge(
            dataframe,
            labels.rename(label_col_name),
            left_on=datetime_col,
            right_index=True,
            how='left'
        )
    else:
        dataframe = pd.merge(
            dataframe,
            labels.rename(label_col_name),
            left_index=True,
            right_index=True,
            how='left'
        )
    
    dataframe.fillna({label_col_name: 0}, inplace=True)
    
    return dataframe
