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


__auther__ = "aeiwz"


class lin_regression:
    
    
    """
    This function attempts to fit a model of  metabolic profiles by using linear regression
    
    for examples:
        #set parameter
        x = Metabolic profiles dataset (dataset X)
        taget = Metadata of two group (reccomend to sub-class of group label) (dataset Y)
        Feature_name = name of features of dataset X in this case define to columns name of dataset X

        X = spectra_X
        target = meta['Class']
        ppm = spectra_X.columns.astype(float) # columns name of example data is ppm of spectra

        test = lin_regression(X, target, ppm)

        #Create dataset to do linear regression model

        dataset = test.create_dataset()
        test.show_dataset() # "show_dataset()" function will be return dataset to creat

        default methode is "fdr_bh"
        test.fit_model(dataset, method = "fdr_bh") # fit model with linear regression

        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)


        # get report can be use .report() function
        test.report() # "report()" function will be return report dataset as p-value, Beta, R_square, and p-value of F-test in one dataframe

        # or u can return each value can be use .p_value(), .beta_value, .r_square, or .f_test()

    """

   

    def __init__(self, x, target, label, features_name):
        
    
        self.x = x
        self.target = target
        self.y = target
        self.tag = target
        self.Features_name = features_name
        self.label = label
    


    def create_dataset(self):


        # Create new dataset

        y = pd.Categorical(self.y).codes
        y = pd.DataFrame(y, index=self.tag.index)
        x = self.x
        tag = self.label
        dataset = pd.concat([tag, y, x], axis=1)

        # Prepare the dataframe for use with model fitting
        dataset.columns = dataset.columns.astype(str)
        varnames = []
        for i in tqdm(range(len(dataset.columns[2::])), desc="Creating data frame"):
            varnames.append("ppm_{}".format(i))
        newnames = ['Label', 'Target']
        newnames = newnames + varnames
        dataset.columns = newnames
        
        self.dataset = dataset
        return dataset

    def show_dataset(self):
        return self.dataset

    def fit_model(self, dataset, method=None):

        
        self.dataset = dataset
        self.method = method
        
        if method == None:
            method = "fdr_bh"
        else:
            method = method

        # Lists to store the information
        # p-value for the genotype effect
        self.pval = list()
        # regression coefficient for the genotype effect
        self.beta = list()
        # P-value for the F-test 
        self.fpval = list()
        # r2 for the regression model
        self.r2 = list()

        # Fit each column with a spectral variable
        for curr_variable in tqdm(dataset.iloc[:, 2:], desc="Features processed"):
            # Formula for current variable 
            fm = curr_variable + ' ~ C(Target)'
            mod = smf.ols(formula = fm, data=dataset)
            res = mod.fit()
            self.pval.append(res.pvalues[1])
            self.beta.append(res.params[1])
            self.fpval.append(res.f_pvalue)
            self.r2.append(res.rsquared)


        # Adjusting the first analysis without age
        by_res_Target = multipletests(self.pval, alpha=0.05, method=self.method)
        p_byadj = by_res_Target[1]
        
        self.p_byadj_df = pd.DataFrame(p_byadj, index=self.Features_name, columns=["q_value"])
        self.pval_df = pd.DataFrame(self.pval, index=self.Features_name, columns=["P-value"])
        self.beta_df = pd.DataFrame(self.beta, index=self.Features_name, columns=["Beta"])
        self.fpval_df = pd.DataFrame(self.fpval, index=self.Features_name, columns=["pval_F-test"])
        self.r2_df = pd.DataFrame(self.r2, index=self.Features_name, columns=["R2"])


        return("Done")


    def p_value(self):
        pval = self.pval_df
        return pval

    def beta_value(self):
        beta = self.beta_df
        return beta

    def f_test(self):
        fpval = self.fpval_df
        return fpval

    def r_square(self):
        r2 = self.r2_df
        return r2

    def q_value(self):
        qval = self.p_byadj_df
        return qval
    

    def report(self):

        # Creat report dataframe
        pval = self.pval_df
        beta = self.beta_df
        fpval = self.fpval_df
        r2 = self.r2_df
        qval = self.p_byadj_df

        
        stats_table = pd.concat([pval, beta, qval, r2, fpval], axis=1)

        return stats_table


    def spec_uniplot(self, pval_position=0, sample=None, label_a=None, label_b=None):

        self.sample_type = sample
        pval = self.pval_df
        dataset = self.dataset
        ppm = self.Features_name
        self.pval_position = pval_position
        self.label_a = label_a
        self.label_b = label_b

        
        meta = list(dataset.iloc[:, 0])
        if self.label_a == None:
            label_a = meta[0]
        else:
            label_a = label_a
        if self.label_b == None:
            label_b = meta[-1]
        else:
            label_b = label_b
            
        self.label_a = label_a
        self.label_b = label_b
        
        
       
        
            
        spectra = pd.DataFrame(dataset.iloc[:, 2:])
        idx_a = dataset.loc[dataset["Target"] == 0].index
        idx_b = dataset.loc[dataset["Target"] == 1].index
        
        

        #set dataset for plot
        df_a = pd.DataFrame(spectra.loc[idx_a])
        df_b = pd.DataFrame(spectra.loc[idx_b])

  

        meanX_a = pd.DataFrame(list(np.mean(df_a)), index=ppm, columns=[label_a])
        meanX_b = pd.DataFrame(list(np.mean(df_b)), index=ppm, columns=[label_b])
        
        # set p-value position region at -0.8 in y axis
        pval_pos = []
        for i in range(len(ppm)):
            pval_pos.append(self.pval_position)
        pval_pos = pd.DataFrame(pval_pos, index=ppm, columns=["pval_position"])

        plot_df = pd.concat([pval, pval_pos, meanX_a, meanX_b], axis=1)

        # Create a scatter plot with the spectra data and the p-values
        fig = px.scatter(plot_df, x=plot_df.index, y="pval_position", color="P-value", 
                            color_continuous_scale='Inferno', range_color=[0.0000, 1.0000])

        # Add a line plot on top of the scatter plot to represent the line data
        #fig.add_trace(px.line(plot_df, x=plot_df.index, y="Mean_X").data[0])
        fig.add_trace(px.line(plot_df, x=ppm, y=label_a, color_discrete_sequence=["#5C8EFA"]).data[0])
        fig.add_trace(px.line(plot_df, x=ppm, y=label_b, color_discrete_sequence=["#FF6F55"]).data[0])
        

        # Customize the layout and add any necessary labels or titles
        fig.update_layout(title="Spectra and Line with P-Value Color Scale ({} [Red] vs {} [Blue])".format(label_a, label_b),
                        xaxis_title="ppm",
                        yaxis_title="Intensity (AU)")

        fig.update_layout(xaxis = dict(autorange='reversed'))
        # Show the plot
        self.fig = fig

        return fig.show()
    
    def html_plot(self, path_save=None):
        self.path_save = path_save
        fig = self.fig
        return fig.write_html("{}/{}_p_value_plot_{}vs{}.html".format(path_save, self.sample_type, self.label_a, self.label_b))
    def png_plot(self, path_save=None):
        self.path_save = path_save
        fig = self.fig
        return fig.write_image("{}/{}_p_value_plot_{}vs{}.png".format(path_save, self.sample_type, self.label_a, self.label_b))
    
        
        