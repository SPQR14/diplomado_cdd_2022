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
def completitud(df):
    """Checks percentage of non missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Data

    Returns
    -------
    pandas.DataFrame
        dataframe with the columns of:
            columna: column
            total: total number of missings
            completitud: percentage of non missing values
    """
    comp=pd.DataFrame(df.isnull().sum())
    comp.reset_index(inplace=True)
    comp=comp.rename(columns={"index":"columna",0:"total"})
    comp["completitud"]=(1-comp["total"]/df.shape[0])*100
    comp=comp.sort_values(by="completitud",ascending=True)
    comp.reset_index(drop=True,inplace=True)
    return comp
def clean_text(text, pattern="[^a-zA-Z0-9 ]",replace=" "):
    """Removes special characters from strings.

    Parameters
    ----------
    text : string
        text to be cleaned.
    pattern : str, optional
        regex of character to be replaced, by default "[^a-zA-Z0-9 ]"
    replace : str, optional
        string to replace the coincidences, by default " "

    Returns
    -------
    string
        Returns cleaned text.
    """
    cleaned_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    cleaned_text = re.sub(pattern, replace, cleaned_text.decode("utf-8"), flags=re.UNICODE)
    cleaned_text = u' '.join(cleaned_text.strip().lower().split())
    return cleaned_text
def transform_salary(text):
    """ matches and transfroms string number inputs into integer type.

    Parameters
    ----------
    text : string
        string containing the salary

    Returns
    -------
    int
        returns the salary in integer.
    """
    number_search = int(re.search('(\d{1,3})', text).group(1))
    return float(number_search)*1000

def OUTLIERS(data,cols):
    """searches for outliers with three different methods and returns dataframe with information of matched outliers.

    Parameters
    ----------
    data : pandas.DataFrame
        data frame to be analyzed for outliers.
    cols : list of strings
        columns to analyze for outliers

    Returns
    -------
    pandas.DataFrame
        Data frame with outlier information.
    """
    df=data.copy()
    results=pd.DataFrame()
    data_iqr=df.copy()
    data_per=df.copy()
    total=[]
    total_per=[]
    total_z=[]
    indices_=[]

    for col in cols:
        #IQR
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3-Q1
        INF=Q1-1.5*(IQR)
        SUP=Q3+1.5*(IQR)
    
        
        n_outliers=df[(df[col] < INF) | (df[col] > SUP)].shape[0]
        total.append(n_outliers)
        indices_iqr=list(df[(df[col] < INF) | (df[col] > SUP)].index)
        #data_iqr=data_iqr[~(data_iqr[col] < INF) | (data_iqr[col] > SUP)].reset_index(drop=True)
        
        #Percentiles
        INF_pe=np.percentile(df[col].dropna(),5)
    
        SUP_pe=np.percentile(df[col].dropna(),95)
        n_outliers_per=df[(df[col] < INF_pe) | (df[col] > SUP_pe)].shape[0]
        total_per.append(n_outliers_per)
        indices_per=list(df[(df[col] < INF_pe) | (df[col] > SUP_pe)].index)
        #data_per=data_per[~(data_per[col] < INF_pe) | (data_per[col] > SUP_pe)].reset_index(drop=True)
        
        #Z-Score
        
        z=np.abs(stats.zscore(df[col],nan_policy='omit'))
        #df[f"zscore_{col}"]=abs((df[col] - df[col].mean())/df[col].std(ddof=0))
        total_z.append(df[[col]][(z>=3)].shape[0])
        indices_z=list(df[[col]][(z>=3)].index)
        
        indices_.append(aux_outliers(indices_iqr,indices_per,indices_z))
        
    results["features"]=cols
    results["n_outliers_IQR"]=total
    results["n_outliers_Percentil"]=total_per
    results["n_outliers_Z_Score"]=total_z
    results["n_outliers_IQR_%"]=round((results["n_outliers_IQR"]/df.shape[0])*100,2)
    results["n_outliers_Percentil_%"]=round((results["n_outliers_Percentil"]/df.shape[0])*100,2)
    results["n_outliers_Z_Score_%"]=round((results["n_outliers_Z_Score"]/df.shape[0])*100,2)
    results["indices"]=indices_
    results["total_outliers"]=results["indices"].map(lambda x:len(x))
    results["%_outliers"]=results["indices"].map(lambda x:round(((len(x)/df.shape[0])*100),2))
    results=results[['features', 'n_outliers_IQR', 'n_outliers_Percentil',
       'n_outliers_Z_Score', 'n_outliers_IQR_%', 'n_outliers_Percentil_%',
       'n_outliers_Z_Score_%',  'total_outliers', '%_outliers','indices']]
    return results
def aux_outliers(a,b,c):
    """auxiliary function for the OUTLIERS function. It gets the conjunction of index sets obtained from different methods used in the OUTLIERS function.

    Parameters
    ----------
    a : int list
        list of indexes
    b : int list
        list of indexes
    c : int list
        list of indexes

    Returns
    -------
    list
        returns list with unique indexes
    """
    a=set(a)
    b=set(b)
    c=set(c)
    
    a_=a.intersection(b)

    b_=b.intersection(c)

    c_=a.intersection(c)

    outliers_index=list(set(list(a_)+list(b_)+list(c_)))
    return outliers_index
def calc_vif(X):
    """calculates VIF to check for multilinearity among data frame.

    Parameters
    ----------
    X : pandas.DataFrame
        Data frame with columns:
            columna: column
            VIF: VIF value of the column
    """
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
def imputar_moda(df,col,X_train,X_test):
    """imputates mode if the new distribution passes chi square test.

    Parameters
    ----------
    df : pandas.DataFrame
        Original data frame
    col : string
        column to be imputated
    X_train : pandas.DataFrame
        train data frame set
    X_test : pandas.DataFrame
        test data frame test

    Returns
    -------
    pandas.DataFrame , pandas.DataFrame
        Returns train and test data frames with imputated mode if chi square passed.
        Returns original train and test dataframes if chi squared failed.
    """
    valor_miss = X_train[col].mode()[0]
    
    x_i=df[col].fillna(valor_miss).value_counts()
    k=x_i.sum()
    p_i=df[col].dropna().value_counts(1)
    m_i=k*p_i
    chi=stats.chisquare(f_obs=x_i,f_exp=m_i)
    p_val=chi.pvalue
    alpha=0.05
    if p_val<alpha:
        print(colored("Rechazamos HO(La porporción de categorias es la misma que la general)",'red'))
        return (X_train[col],X_test[col])
    else:
        print(colored("Aceptamos HO(La porporción de categorias es la misma que la general)",'green'))
        print("Se reemplazan los valores ausentes.")
        return (X_train[col].fillna(valor_miss),X_test[col].fillna(valor_miss))
def imputar_continua(df,col):
    """ prints KS values for mean, median and mode values of the data frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame to be analyzed
    col : string
        column to be analyzed with the KS.
    """
    aux = df[col].dropna()
    estadisticos = dict(mean=aux.mean(),median=aux.median(),mode=aux.mode())
    originales=list(df[col].dropna().values)
    for key,value in estadisticos.items():
        imputados=list(df[col].fillna(value).values)
        print(f'{key}\n{stats.ks_2samp(originales,imputados)}')
def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)



####   funciones para graficar   ####


def my_histogram(df,col,bins,title="",x_title="",y_title="conteo"):
    """generates plotly histogram

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    col : [string
        column from data frame to plot
    bins : int
        number of bins for histogram
    title : str, optional
        title of the plot, by default ""
    x_title : str, optional
        x axis title, by default ""
    y_title : str, optional
        y axis title, by default "conteo"

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(font_family="Courier New, monospace",
        font_color="black",title_text=title,title_font_size=20,
        xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size":12,"color": '#002e4d'}}},
        yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},               
        title_font_family="Arial",title_font_color="#002020",
        template="plotly_white", plot_bgcolor="rgb(168,168,168)")
    fig=df[[col]].iplot(kind='histogram',x=col,bins=bins,title=title,asFigure=True,layout=layout,sortbars=True,linecolor='#2b2b2b')
    fig.update_traces(marker_color='#045C8C',opacity=0.7)
    return fig

def my_bar_count(df,x,title="",x_title="",y_title=""):
    """ counts categories in the variable and generates plotly bar plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    col : [string
        column from data frame to plot
    title : str, optional
        title of the plot, by default ""
    x_title : str, optional
        x axis title, by default ""
    y_title : str, optional
        y axis title, by default ""

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(font_family="Courier New, monospace",
        font_color="black",title_text=title,title_font_size=20,
        xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        title_font_family="Arial",title_font_color="#003030",
        template="plotly_white", plot_bgcolor="rgb(168,168,168)")
    aux=pd.DataFrame(df[x].value_counts()).reset_index().rename(columns={"index":"conteo"})
    fig=aux.iplot(kind='bar',x="conteo",y=x,title=title,asFigure=True,barmode="overlay",sortbars=True,color='#2b2b2b',layout=layout,width=5)
    fig.update_layout(width=800)
    fig.update_traces(marker_color='#045C8C',opacity=0.7)
    return fig

def my_bar(df,x,y,title="",x_title="",y_title=""):
    """ generates plotly bar plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    x : string
        column that defines independent values (x axis) of the plot
    y : string
        column that defines dependent values (y axis) of the plot
    title : str, optional
        title of the plot, by default ""
    x_title : str, optional
        x axis title, by default ""
    y_title : str, optional
        y axis title, by default ""

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(font_family="Courier New, monospace",
        font_color="black",title_text=title,title_font_size=20,
        xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        title_font_family="Arial",title_font_color="#002020",
        template="plotly_white", plot_bgcolor="rgb(168,168,168)")
    fig=df.iplot(kind='bar',x=x,y=y,title=title,asFigure=True,barmode="overlay",sortbars=True,color='#2b2b2b',layout=layout,width=5)
    fig.update_layout(width=800)
    fig.update_traces(marker_color='#045C8C',opacity=0.7)
    return fig

def my_pie_count(df,col,title=""):
    """ counts categories in the variable and generates plotly pie plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    col : string
        column from data frame to plot
    title : str, optional
        title of the plot, by default ""

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(template="plotly_white")
    colors=['#4676d0','#95b0e4','#19293c','#6faa9f','#ccceb1','#344647','#02160f','#779a7c','#070919','#2b2b2b','#121212']
    aux=pd.DataFrame(df[col].value_counts()).reset_index().rename(columns={"index":"conteo"})
    fig=aux.iplot(kind='pie',labels='conteo',values=col,title=title,asFigure=True,theme="white")
    fig.update_traces(textfont_size=10,opacity=0.65,
                  marker=dict(colors=colors))
    fig.update_layout(font_family="Courier New, monospace",
    font_color="black",title_text=title,title_font_size=20,title_font_family="Arial",title_font_color="#002020",template="plotly_white")
    return fig

def my_pie(df,labels,values,title=""):
    """ generates plotly pie plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    labels : string
        column that defines independent values (categories) of the plot
    values  : strings
        column that defines dependent values (quantity of categories) of the plot
    title : str, optional
        title of the plot, by default ""

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(template="plotly_white")
    colors=['#4676d0','#95b0e4','#19293c','#6faa9f','#ccceb1','#344647','#02160f','#779a7c','#070919','#2b2b2b','#121212']*2
    fig=df.iplot(kind='pie',labels=labels,values=values,title=title,asFigure=True,theme="white")
    fig.update_traces(textfont_size=10,opacity=0.65,
                  marker=dict(colors=colors))
    fig.update_layout(font_family="Courier New, monospace",
    font_color="black",title_text=title,title_font_size=20,title_font_family="Arial",title_font_color="#002020",template="plotly_white")
    return fig

def my_box(df,columns,values,title="",x_title="",y_title=""):
    """ generates plotly box plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    columns : string
        column that defines independent values (categories) of the plot
    values  : strings
        column that defines dependent values (values' distribution) of the plot
    title : str, optional
        title of the plot, by default ""
    x_title : str, optional
        x axis title, by default ""
    y_title : str, optional
        y axis title, by default ""

    Returns
    -------
    plotly figure
    """
    colors=['#4676d0','#19293c','#6faa9f','#ccceb1','#344647','#02160f','#779a7c','#070919','#2b2b2b','#121212']
    layout = go.Layout(font_family="Courier New, monospace",
        font_color="black",title_text=title,title_font_size=20,
        xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size": 12,"color": '#002e4d'}}},
        yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        title_font_family="Arial",title_font_color="#002020",
        template="plotly_white", plot_bgcolor="rgb(208,208,208)")
    fig=df.pivot(columns=columns,values=values).iplot(kind='box',title=title,asFigure=True,theme="white",layout=layout,color=colors)
    return fig