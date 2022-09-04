#!/home/sqpr14_/anaconda3/envs/DSS2 python
# -*- coding: utf-8 -*-

"""practica1_m3.py: Diplomado ed Ciencia de datos Módulo 3 - Pŕactica 1"""

__author__ = "Alberto Isaac Pico Lara"
__date__ = "saturday 15/08/2022"

import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.manifold import MDS


class DimensionalReduction:

    def __init__(self, df, id='') -> None:
        self.df = df
        self.id = id
    
    def apply_pca(self, components=4, threshold=None, column_names=None, verbose=False):
        """
        This methos is intented for dimensional reduction of a DataFrade via Principal Component Analysis
        Args:
            components (int, optional): Dedired number of components. Defaults to 4.
            threshold (int, optional): Desired variabce threshold. Defaults to None.
            column_names (list, optional): Desired columns from a DataFrame. Defaults to None.
            verbose (boolean, optional): If the user wants to display info about the processs. Defaults to False.
        Return:
            DataFrame formed from the principal components of the orifinal DataFrame
            fitted PCA object
        """
        sc = StandardScaler()
        scaled_df = None
        redim_df = None

        if column_names:
            scaled_df = pd.DataFrame(sc.fit_transform(self.df[column_names]), columns=self.df[column_names].columns)
        else:
            column_names = list(self.df.select_dtypes(include=[np.number]))
            scaled_df = pd.DataFrame(sc.fit_transform(self.df[column_names]), columns=self.df[column_names].columns)

        if threshold:
            components = 2
            variance = 0
            while variance < threshold :
                pca = PCA(n_components = components)
                pca.fit(scaled_df)
                variance = sum(pca.explained_variance_ratio_)
                components += 1
            redim_df = pd.DataFrame(pca.transform(scaled_df), columns=[f'p_{i}' for i in range(components - 1)])
        else:
            pca = PCA(n_components=components)
            pca.fit(scaled_df)
            redim_df = pd.DataFrame(pca.transform(scaled_df), columns=[f'p_{i}' for i in range(components)])
        if verbose:
            print("INFO:")
            for i, j in zip(list(pca.explained_variance_ratio_), range(1, len(pca.explained_variance_ratio_) + 1)):
                print(f"Component {j}: {i}")
            print(f"Explained Variance: {sum(pca.explained_variance_ratio_)}")

        return(redim_df, pca)

    def apply_mds(self, dimensions=3, column_names=None, samples=None , verbose=False, tar=None):
        """
        This method is intented for dimensional reduction via Multi Dimensional Scaler
        Args:
            dimensions (int, optional): Desired number of dimensions. Defaults to 3.
            column_names (list, optional): A list of columns to be taken for transform the DataFrame. Defaults to None.
            samples (int, optional): Size of the desired sample that would be taken from the DataFrame. Defaults to None.
            verbose (boolean, optional): True if the user wants to see info about the process. Defaults to False.
        Return:
            scaled DataFrame
            fitted MDS object
        """
        mm = MinMaxScaler()
        scaled_df = None
        redim_df = None
        mds = MDS(dimensions)

        if column_names:
            scaled_df = pd.DataFrame(mm.fit_transform(self.df[column_names]), columns=self.df[column_names].columns)
        else:
            column_names = list(self.df.select_dtypes(include=[np.number]))
            scaled_df = pd.DataFrame(mm.fit_transform(self.df[column_names]), columns=self.df[column_names].columns)
        
        if samples:
            scaled_df[tar] = self.df[tar].copy()
            sampled_df = scaled_df.sample(samples)
            target = sampled_df[tar].copy()
            sampled_df.drop(columns=[tar], axis=1, inplace=True)
            start = time.process_time()
            redim_df = pd.DataFrame(mds.fit_transform(sampled_df), columns=[f'd_{i}' for i in range(1, dimensions + 1)], index=sampled_df.index)
            redim_df[tar] = target.copy()

        else:
            start = time.process_time()
            redim_df = pd.DataFrame(mds.fit_transform(scaled_df), columns=[f'd_{i}' for i in range(1, dimensions + 1)], index=scaled_df.index)

        if verbose:
            print(f"Process took: {time.process_time() - start} time units.")

        return(redim_df, mds)
        pass

    