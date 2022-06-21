#bibliotecas
import sys
import warnings
import pathlib
from termcolor import colored
#bibliotecas para manejo de datos
import pandas as pd
import numpy as np
from scipy import stats
import re
import unicodedata
#import nltk
import unicodedata
from random import sample
#from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
#from category_encoders import TargetEncoder
from unidecode import unidecode
#from nltk.corpus import stopwords
#from nltk import FreqDist
from statsmodels.stats.outliers_influence import variance_inflation_factor
#bibliotecas para graficar
import plotly
import plotly.graph_objects as go
import plotly.express as px
import cufflinks as cf
#import stylecloud
#from PIL import Image
from plotly.offline import plot,iplot
pd.options.plotting.backend = "plotly"
cf.go_offline()
pd.set_option("display.max_columns",200)

####   funciones para manejo de datos ####


def label_columns(df,feats,prefix):
    """labels columns' names with the given prefix.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame whose columns will be labeled.
    feats : list of strings
        list with column names to label.
    prefix : string
        string prefix to add at the begining of coumn names.

    Returns
    -------
    pandas.DataFrame
        Returns the same dataframe from the input with column names labeled.
    """
    feats_new=[prefix+x for x in feats]
    df=df.rename(columns=dict(zip(feats,feats_new)))
    return df


