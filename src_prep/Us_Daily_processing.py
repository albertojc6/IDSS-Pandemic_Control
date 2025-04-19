#%%
from operator import index

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import seaborn as sns

# READ DATA
states_daily = pd.read_csv('../data/states_daily.csv')

# UTILITARY FUNCTIONS
def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = {'totalTestResultsSource', 'lastUpdateEt', 'dateModified', 'checkTimeEt', 'dateChecked', 'hash',
                        'fips'}
    low_unique_value_columns = {'grade', 'score', 'positiveScore', 'negativeScore', 'negativeRegularScore',
                                'commercialScore', 'dataQualityGrade'}

    all_to_remove = metadata_columns.union(low_unique_value_columns)

    df = df[[column for column in list(df.columns) if column not in all_to_remove]]

    return df


def keep_with_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_shared_all_states = [
        'date',
        'state',
        'positive',
        'totalTestResults',
        'death',
        'fips',
        'positiveIncrease',
        'negativeIncrease',
        'total',
        'totalTestResultsIncrease',
        'posNeg',
        'deathIncrease',
        'hospitalizedIncrease']

    df = df[[column for column in list(df.columns) if column in cols_shared_all_states]]

    return df


def change_data_format_and_order(df):
    df_tmp = df.copy()
    df_tmp['date'] = pd.to_datetime(df_tmp['date'], format='%Y%m%d')

    df_sorted = df_tmp.sort_values(by='date', ascending=True)

    return df_sorted


def deal_with_MAR(df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.copy()

    # Set multi-index and sort by date within each state
    df_tmp = (
        df_tmp.set_index(['date', 'state'])
        .sort_index(level=['state', 'date'])
    )

    # Group by state and interpolate within each group
    df_tmp = (
        df_tmp.groupby('state', group_keys=False)
        .apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'))
    )

    # Reset index to restore original structure
    df_tmp = df_tmp.reset_index()

    return df_tmp


def deal_with_MNAR(df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.copy()

    # Set multi-index
    df_tmp = df_tmp.fillna(0)

    return df_tmp


def truncate_numeric_columns(df):

    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    df_copy[numeric_cols] = df_copy[numeric_cols].map(np.trunc)
    return df_copy


def pipeline_preprocessing(df):
    df = filter_columns(df)
    df = keep_with_common_columns(df)
    df = change_data_format_and_order(df)
    df = deal_with_MAR(df)
    df = deal_with_MNAR(df)
    df = truncate_numeric_columns(df)

    return df

#APPLY PIPELINE TO DATA
df = pipeline_preprocessing(states_daily)

#SAVE DATA
df.to_csv('../preprocessed_data/states_daily_cleaned.csv',index=False)