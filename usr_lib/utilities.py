#!/home/sqpr14_/anaconda3/envs/dss python
# -*- coding: utf-8 -*-


"""utilities.py: This script contains some useful utilities for data exploration, 
data cleaning and exploration"""

import pandas as pd



def rename_variables(df, columns,prefix):
    """
    This method returns a Pandas DataFrame with renamed columns according to the passed by value parameters:
    df -> Pandas DataFrame
    columns -> Group of selected columns that this function will renaname, thgis param must be a list
    prefix -> selected prefix according to variable types:
    ------------------------------------------------------
    c_ | numerical variables
    v_ | categorical variables
    d_ | date related variables
    t_ | text variables
    g_ | geographic variables
    ------------------------------------------------------
    """
    valid = ['c_', 'v_', 'd_', 't_', 'g_']
    if prefix in valid:
        n_feats = [prefix + x.lower() for x in columns]
        df = df.rename(columns=dict(zip(columns, n_feats)))
        return df
    else:
        print("invalid prefix")
        return df


def text_cleaning(**args):
    pass

