#!/home/sqpr14_/anaconda3/envs/dss python
# -*- coding: utf-8 -*-


"""utilities.py: This script contains some useful utilities for data exploration, 
data cleaning and exploration"""

import unicodedata
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import re
import numpy as np

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


def text_cleaning(argv):
    pass

def find_outliers(df, cols=None):
    """
    This method finds the outliers in a given Pandas DataFrame
    """
    results=pd.DataFrame()
    data_iqr=df.copy()
    data_per=df.copy()
    total=[]
    total_per=[]
    total_z=[]
    indices_=[]

    for col in cols:
        # IQR
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3-Q1
        INF=Q1-1.5*(IQR)
        SUP=Q3+1.5*(IQR)
    
        
        n_outliers=df[(df[col] < INF) | (df[col] > SUP)].shape[0]
        total.append(n_outliers)
        indices_iqr=list(df[(df[col] < INF) | (df[col] > SUP)].index)
        # data_iqr=data_iqr[~(data_iqr[col] < INF) | (data_iqr[col] > SUP)].reset_index(drop=True)
        
        # Percentiles
        INF_pe=np.percentile(df[col].dropna(),5)
    
        SUP_pe=np.percentile(df[col].dropna(),95)
        n_outliers_per=df[(df[col] < INF_pe) | (df[col] > SUP_pe)].shape[0]
        total_per.append(n_outliers_per)
        indices_per=list(df[(df[col] < INF_pe) | (df[col] > SUP_pe)].index)
        # data_per=data_per[~(data_per[col] < INF_pe) | (data_per[col] > SUP_pe)].reset_index(drop=True)
        
        # MEAN CHANGE
        
        # Obtenemos todos los percentiles además del máximo
        perc_100 = [x / 100 for x in range(100)]
        dist = df[col].describe(perc_100).iloc[4:]
        # Obtenemos el cambio entre percentiles
        change_dist = df[col].describe(perc_100).iloc[4:].diff()
        # Obtenemos el cambio promedio entre percentiles
        mean_change = df[col].describe(
            perc_100).iloc[4:].diff().mean()
        # Si el cambio entre el percentil 99 y el maximo es mayor a el cambio promedio entonces:
        if change_dist["max"] > mean_change:
            #La banda superior será el máximo menos el cambio promedio
            ub = dist["max"] - mean_change
            #si la banda superior es más pequeña que el percentil 99 , modificamos la banda para que tome el percentil 99
            if ub < dist["99%"]:
                ub = dist["99%"]
        else:
        # Si el cambio entre el percentil 99 y el maximo es menor o igual a el cambio promedio entonces se toma el percentil 99
            ub = dist["max"]

        if change_dist["1%"] > mean_change:
            lb = dist["0%"] + mean_change
            if lb > dist["1%"]:
                lb = dist["1%"]
        else:
            lb = dist["0%"]
        n_total_z=df[(df[col] < lb) | (df[col] > ub)].shape[0]
        total_z.append(n_total_z)
        indices_z=list(df[(df[col] < lb) | (df[col] > ub)].index)
        
        indices_.append(outliers_index(indices_iqr,indices_per,indices_z))
        
    results["features"]=cols
    results["n_outliers_IQR"]=total
    results["n_outliers_Percentil"]=total_per
    results["n_outliers_Mean_Change"]=total_z
    results["n_outliers_IQR_%"]=round((results["n_outliers_IQR"]/df.shape[0])*100,2)
    results["n_outliers_Percentil_%"]=round((results["n_outliers_Percentil"]/df.shape[0])*100,2)
    results["n_outliers_Mean_Change_%"]=round((results["n_outliers_Mean_Change"]/df.shape[0])*100,2)
    results["indices"]=indices_
    results["total_outliers"]=results["indices"].map(lambda x:len(x))
    results["%_outliers"]=results["indices"].map(lambda x:round(((len(x)/df.shape[0])*100),2))
    results=results[['features', 'n_outliers_IQR', 'n_outliers_Percentil',
       'n_outliers_Mean_Change', 'n_outliers_IQR_%', 'n_outliers_Percentil_%',
       'n_outliers_Mean_Change_%',  'total_outliers', '%_outliers','indices']]
    return results


def outliers_index(a,b,c):
    """
    This method returns a series of outliers index 
    """
    a = set(a)
    b = set(b)
    c = set(c)
    
    a_ = a.intersection(b)

    b_ = b.intersection(c)

    c_ = a.intersection(c)

    outliers_index=list(set(list(a_)+list(b_)+list(c_)))
    return outliers_index

#Completitud de las variables
def completitud(df):
    completitud_df=pd.DataFrame(df.isnull().sum())
    completitud_df.reset_index(inplace=True)
    completitud_df=completitud_df.rename(columns={"index":"columna",0:"missings"})
    completitud_df["completitud (%)"]=(1-completitud_df["missings"]/df.shape[0])*100
    completitud_df=completitud_df.sort_values(by="completitud (%)",ascending=True)
    completitud_df.reset_index(drop=True,inplace=True)
    return completitud_df

#Funcion para limpiar caracteres especiales
def clean_text(text, pattern="[^a-zA-Z0-9 ]",replace=" "):
    try:
        cleaned_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
        cleaned_text = re.sub(pattern, replace, cleaned_text.decode("utf-8"), flags=re.UNICODE)
        cleaned_text = u' '.join(cleaned_text.strip().lower().split())
    except:
        return text
    return cleaned_text

#Dataframe con la frecuencia de valores en una columna de un dataset
def conteo_df(df, col, x_label = " ", y_label="conteo"):
    if x_label.isspace():
        x_label = col
    return pd.DataFrame(df[col].value_counts()).reset_index().rename(columns= {"index": x_label, col: y_label})


def tree_cut(X_tr, X_te, feature, new_category, tgt, max_depth, min_samples_leaf):
    X_train = X_tr.copy()
    X_test = X_te.copy()

    # Árbol de decisión
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=1,random_state=0)
    X_feat = X_train.loc[X_train[feature].notnull(), [feature]]
    y_feat = X_train.loc[X_train[feature].notnull(), tgt]
    dt.fit(X_feat, y_feat)
    X_train.loc[X_train[feature].notnull(), new_category] = dt.apply(X_train.loc[X_train[feature].notnull(), [feature]])
    X_test.loc[X_test[feature].notnull(), new_category] = dt.apply(X_test.loc[X_test[feature].notnull(), [feature]])
    
    #intervales
    aux_train = X_train[[new_category, feature]].groupby([new_category]).agg(["min", "max"])

    aux_train.columns = aux_train.columns.droplevel(0)
    
    aux_train[f"Interval_{feature}"]= aux_train.apply(lambda x:[x['min'],x['max']],axis=1)
    inter_list_train=aux_train[f"Interval_{feature}"].tolist()
    inter_list_train[0][0]=-np.Inf
    inter_list_train[-1][-1]=np.Inf
    
    inter_list_train=pd.IntervalIndex.from_tuples(list(map(tuple,inter_list_train)),closed="both")
    
    X_train[new_category]=pd.cut(X_train[feature].astype(float),bins=inter_list_train)
    X_test[new_category]=pd.cut(X_test[feature].astype(float),bins=inter_list_train)

    #Revisamos si hay nulos
    if (X_train[new_category].isnull().sum() > 0):
        X_train[new_category] = X_train[new_category].cat.add_categories('missings')
        X_train[new_category].fillna('missings', inplace =True)

        X_test[new_category] = X_test[new_category].cat.add_categories('missings')
        X_test[new_category].fillna('missings', inplace =True)
            
    return X_train,X_test
    

def WOE(X_aux,feature,tgt):
    aux = X_aux[[feature, tgt]].groupby(feature).agg(["count", "sum"])
    aux["evento"] = aux[tgt, "sum"]
    aux["no_evento"] = aux[tgt, "count"] - aux[tgt, "sum"]
    aux["%evento"] = aux["evento"] / aux["evento"].sum()
    aux["%no_evento"] = aux["no_evento"] / aux["no_evento"].sum()
    aux["WOE"] = np.log(aux["%evento"] / aux["%no_evento"])
    IV=((aux["%evento"] - aux["%no_evento"])*aux["WOE"]).sum()
    aux.columns = aux.columns.droplevel(1)
    aux = aux[["WOE"]].reset_index().rename(columns={"WOE": f"W_{feature}"})
    return aux,IV

