import pandas as pd
import numpy as np

def clean_genres(df):
    '''Removes characters – []'
    Expects a df with a column called "genres".
    Returns the df with a new column named "clean_genres"
    '''
    if 'clean_genres' in df.columns:
        df['clean_genres'] = df['genres'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace("'", '')
        )
        return df
    else:
        return '"clean_genres" not found in columns'


