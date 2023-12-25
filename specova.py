
# _*_ coding: utf-8 _*_

# Package
# Import the required python packages including 
# the custom Chemometric Model objects
import numpy as np
from sklearn import preprocessing
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_classification
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import scale
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go




__author__ = "aeiwz"

class specova():
    """
    A class used to perform ANOVA analysis on a dataset

    ...

    Attributes
    ----------
    data : pandas.DataFrame
        A pandas dataframe containing the data to be analysed
    x : list
        A list containing the names of the columns to be used as the independent variables
    y : str
        A string containing the name of the column to be used as the dependent variable
    alpha : float
        A float containing the alpha value to be used for the analysis
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        A statsmodels object containing the results of the ANOVA analysis
    pvalues : list
        A list containing the p-values for each independent variable
    pvalues_adj : list
        A list containing the adjusted p-values for each independent variable
    pvalues_adj_sig : list
        A list containing the adjusted p-values for each independent variable that are significant
    pvalues_sig : list
        A list containing the p-values for each independent variable that are significant
    pvalues_sig_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj : list
        A list containing the adjusted p-values for each independent variable that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant
    pvalues_sig_adj_names : list
        A list containing the names of the independent variables that are significant

    Methods
    -------
    fit()
        Fits the model to the data
    summary()
        Prints a summary of the model
    predict()

    """
    def __init__(self, data, x, y, alpha=0.05):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            A pandas dataframe containing the data to be analysed
        x : list
            A list containing the names of the columns to be used as the independent variables
        y : str
            A string containing the name of the column to be used as the dependent variable
        alpha : float
            A float containing the alpha value to be used for the analysis
        """
        self.data = data
        self.x = x
        self.y = y
        self.alpha = alpha
        self.model = None
        self.pvalues = None
        self.pvalues_adj = None
        self.pvalues_adj_sig = None
        self.pvalues_sig = None
        self.pvalues_sig_names = None
        self.pvalues_sig_adj = None
        self.pvalues_sig_adj_names = None

    def fit(self):
        """
        Fits the model to the data
        """
        # Create the model
        self.model = smf.ols(formula=self.y + ' ~ ' + ' + '.join(self.x), data=self.data).fit()
        # Get the p-values
        self.pvalues = self.model.pvalues
        # Get the adjusted p-values
        self.pvalues_adj = multipletests(self.pvalues, alpha=self.alpha, method='fdr_bh')[1]
        # Get the adjusted p-values that are significant
        self.pvalues_adj_sig = self.pvalues_adj[self.pvalues_adj < self.alpha]
        # Get the p-values that are significant
        self.pvalues_sig = self.pvalues[self.pvalues < self.alpha]
        # Get the names of the independent variables that are significant
        self.pvalues_sig_names = list(self.pvalues_sig.index)
        # Get the adjusted p-values that are significant
        self.pvalues_sig_adj = self.pvalues_adj[self.pvalues_adj < self.alpha]
        # Get the names of the independent variables that are significant
        self.pvalues_sig_adj_names = list(self.pvalues_sig_adj.index)

    def summary(self):
        """
        Prints a summary of the model
        """
        print(self.model.summary())

    def predict(self, data):
        """
        Predicts the dependent variable values for the given data

        Parameters
        ----------
        data : pandas.DataFrame
            A pandas dataframe containing the data to be predicted
        """
        return self.model.predict(data)

    def plot(self):
        """
        Plots the results of the model
        """
        # Create a dataframe of the p-values
        df = pd.DataFrame({'pvalues': self.pvalues, 'pvalues_adj': self.pvalues_adj})
        # Create a figure
        fig = go.Figure()
        # Add a bar chart of the p-values
        fig.add_trace(go.Bar(x=df.index, y=df['pvalues'], name='p-values'))
        # Add a bar chart of the adjusted p-values
        fig.add_trace(go.Bar(x=df.index, y=df['pvalues_adj'], name='adjusted p-values'))
        # Add a line for the alpha value
        fig.add_trace(go.Scatter(x=df.index, y=[self.alpha for i in range(len(df.index))], name='alpha'))
        # Set the title
        fig.update_layout(title='ANOVA Results')
        # Show the figure
        fig.show()

    def plot_sig(self):
        """
        Plots the results of the model
        """
        # Create a dataframe of the p-values
        df = pd.DataFrame({'pvalues': self.pvalues_sig, 'pvalues_adj': self.pvalues_sig_adj})
        # Create a figure
        fig = go.Figure()
        # Add a bar chart of the p-values
        fig.add_trace(go.Bar(x=df.index, y=df['pvalues'], name='p-values'))
        # Add a bar chart of the adjusted p-values
        fig.add_trace(go.Bar(x=df.index, y=df['pvalues_adj'], name='adjusted p-values'))
        # Add a line for the alpha value
        fig.add_trace(go.Scatter(x=df.index, y=[self.alpha for i in range(len(df.index))], name='alpha'))
        # Set the title
        fig.update_layout(title='ANOVA Results')
        # Show the figure
        fig.show()

    def plot_sig_adj(self):
        """
        Plots the results of the model
        """
        # Create a dataframe of the p-values
        df = pd.DataFrame({'pvalues': self.pvalues_sig_adj})
        # Create a figure
        fig = go.Figure()
        # Add a bar chart of the p-values
        fig.add_trace(go.Bar(x=df.index, y=df['pvalues'], name='p-values'))
        # Add a line for the alpha value
        fig.add_trace(go.Scatter(x=df.index, y=[self.alpha for i in range(len(df.index))], name='alpha'))
        # Set the title
        fig.update_layout(title='ANOVA Results')
        # Show the figure
        fig.show()

        
