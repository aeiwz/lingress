
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
        self.features_name = features_name
        self.label = label
    
    def create_dataset(self):

        '''# Create dataset to do linear regression model
        dataset = test.create_dataset()
        print(test.show_dataset()) # Show the created dataset
        '''

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

    def fit_model(self, datasets=None, adj_method=None):

        '''# Fit model with linear regression
        # The method could be one of the following:
        # 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'
        # 'fdr_bh' is Benjamini/Hochberg method for p-value adjustment
        test.fit_model(dataset, method='fdr_bh') 
        '''

        dataset = self.dataset
        self.adj_method = adj_method

        if adj_method == "bonferroni":
            adj_name = "one-step correction"
        elif adj_method == "sidak":
            adj_name = "one-step correction"
        elif adj_method == "holm-sidak":
            adj_name = "step down method using Sidak adjustments"
        elif adj_method == "holm":
            adj_name = "step-down method using Bonferroni adjustments"
        elif adj_method == "simes-hochberg":
            adj_name = "step-up method  (independent)"
        elif adj_method == "hommel":
            adj_name = "closed method based on Simes tests (non-negative)"
        elif adj_method == "fdr_bh":
            adj_name = "Benjamini/Hochberg (non-negative)"
        elif adj_method == "fdr_by":
            adj_name = "BBenjamini/Yekutieli (negative)"
        elif adj_method == "fdr_tsbh":
            adj_name = "two stage fdr correction (non-negative)"
        elif adj_method == "fdr_tsbky":
            adj_name = "two stage fdr correction (non-negative)"

        
        a = dataset.loc[dataset["Target"] == 0].iloc[:, 2:].mean()
        b = dataset.loc[dataset["Target"] == 1].iloc[:, 2:].mean()

        l2fc = np.log2(np.nan_to_num(np.divide(a, b), nan=0))
        df = pd.DataFrame(l2fc, columns=["Log2 Fold change"], index=self.features_name)

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


        a = dataset.loc[dataset["Target"] == 0].iloc[:, 2:].mean()
        b = dataset.loc[dataset["Target"] == 1].iloc[:, 2:].mean()

        l2fc = np.log2(np.nan_to_num(np.divide(a, b), nan=0))
        l2_df = pd.DataFrame(l2fc, columns=["Log2 Fold change"], index=self.features_name)
        self.l2_df2 = l2_df.fillna(0)          

        
        self.pval_df = pd.DataFrame(self.pval, index=self.features_name, columns=["P-value"])
        self.beta_df = pd.DataFrame(self.beta, index=self.features_name, columns=["Beta"])
        self.fpval_df = pd.DataFrame(self.fpval, index=self.features_name, columns=["pval_F-test"])
        self.r2_df = pd.DataFrame(self.r2, index=self.features_name, columns=["R2"])

        if adj_method == None:
            
            No_adj = list()
            for i in range(len(self.features_name)):
                No_adj.append("No q-value")
            self.qval_df = pd.DataFrame(No_adj, index=self.features_name, columns=["q_value"])
            self.fqval_df = pd.DataFrame(No_adj, index=self.features_name, columns=["q_value (F-test)"])

        else:
            adj_method = self.adj_method
            p_est = multipletests(self.pval, alpha=0.05, method=adj_method)
            qval = p_est[1]
            pf_est = multipletests(self.fpval, alpha=0.05, method=adj_method)
            fqval = pf_est[1]
            self.qval_df = pd.DataFrame(qval, index=self.features_name, columns=["q_value"])
            self.fqval_df = pd.DataFrame(fqval, index=self.features_name, columns=["q_value (F-test)"])

        
        if adj_method == None:
            return print("No adjustment p-value Done")
        else:
            return print("adjustment p-value with {} Done".format(adj_name))

    def resampling(self, dataset=None, n_jobs=8, verbose=5, n_boots=50, adj_method=None):

        '''

        # For bootstrap resampling:
        # The parameters are:
        # dataset: the dataset to bootstrap resample
        # n_jobs: the number of CPU cores to use (-1 for all cores)
        # verbose: the verbosity level of the joblib library
        # n_boots: the number of bootstrap iterations
        # adj_method: the p-value adjustment method (same as above)
        test.resampling(dataset=dataset, n_jobs=8, verbose=5, n_boots=50, adj_method='fdr_bh')
        '''

        self.n_jobs=n_jobs
        self.n_boot=n_boots
        self.verbose = verbose
        dataset = self.dataset

        self.adj_method = adj_method

        if adj_method == "bonferroni":
            adj_name = "one-step correction"
        elif adj_method == "sidak":
            adj_name = "one-step correction"
        elif adj_method == "holm-sidak":
            adj_name = "step down method using Sidak adjustments"
        elif adj_method == "holm":
            adj_name = "step-down method using Bonferroni adjustments"
        elif adj_method == "simes-hochberg":
            adj_name = "step-up method  (independent)"
        elif adj_method == "hommel":
            adj_name = "closed method based on Simes tests (non-negative)"
        elif adj_method == "fdr_bh":
            adj_name = "Benjamini/Hochberg (non-negative)"
        elif adj_method == "fdr_by":
            adj_name = "Benjamini/Yekutieli (negative)"
        elif adj_method == "fdr_tsbh":
            adj_name = "two stage fdr correction (non-negative)"
        elif adj_method == "fdr_tsbky":
            adj_name = "two stage fdr correction (non-negative)"


        #Model resampling - bootstrapping
        # Define function that can be called by each worker:
        def bootstrap_model(variable, n_boot, dataset):

            boot_stats = np.zeros((n_boot, 5))
            
            for boot_iter in range(n_boot):
                boot_sample = np.random.choice(dataset.shape[0], dataset.shape[0], replace=True)
                fm = dataset.columns[variable] + ' ~ C(Target)'
                mod = smf.ols(formula = fm, data=dataset.iloc[boot_sample, :])
                res = mod.fit()
                boot_stats[boot_iter, 0] = res.pvalues[0] # p-value
                boot_stats[boot_iter, 1] = res.params[0]   # Beta coeficient
                boot_stats[boot_iter, 2] = res.f_pvalue # p-value of F-test
                boot_stats[boot_iter, 3] = res.rsquared # R^2
                boot_stats[boot_iter, 4] = res.rsquared_adj # R^2 adjustment
                
                
            return boot_stats
        import joblib
        results = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch='1.5*n_jobs')(joblib.delayed(bootstrap_model)(i, n_boots, dataset) for i in range(2, dataset.shape[1]))
    
        self.results = results

        mean_p = np.array([x[:, 0].mean() for x in results])
        std_p = np.array([x[:, 0].std() for x in results])
        mean_beta = np.array([x[:, 1].mean() for x in results])
        std_beta = np.array([x[:, 1].std() for x in results])
        mean_pf = np.array([x[:, 2].mean() for x in results])
        std_pf = np.array([x[:, 2].std() for x in results])
        mean_r2 = np.array([x[:, 3].mean() for x in results])
        std_r2 = np.array([x[:, 3].std() for x in results])
        mean_r2_adj = np.array([x[:, 4].mean() for x in results])
        std_r2_adj = np.array([x[:, 4].std() for x in results])

        results_2 = np.array(results)

        self.p_val_boost = pd.DataFrame(results_2[:, :, 0], index=self.features_name)
        self.beta_boost = pd.DataFrame(results_2[:, :, 1], index=self.features_name)
        self.f_pval_boost = pd.DataFrame(results_2[:, :, 2], index=self.features_name)
        self.r2_boost = pd.DataFrame(results_2[:, :, 3], index=self.features_name)
        self.adj_r2_boost = pd.DataFrame(results_2[:, :, 4], index=self.features_name)

        if adj_method == None:
            
            No_adj = list()
            for i in range(len(self.features_name)):
                No_adj.append("No q-value")
            self.mean_qval_df = pd.DataFrame(No_adj, index=self.features_name, columns=["q_value"])
            

        else:
            adj_method = self.adj_method
            p_est = multipletests(mean_p, alpha=0.05, method=adj_method)
            qval = p_est[1]
            pf_est = multipletests(mean_pf, alpha=0.05, method=adj_method)
            fqval = pf_est[1]
            self.mean_qval_df = pd.DataFrame(qval, index=self.features_name, columns=["q_value"])

        self.mean_p_df = pd.DataFrame(mean_p, index=self.features_name, columns=["Mean P-value"])
        self.std_p_df = pd.DataFrame(std_p, index=self.features_name, columns=["std P-value"])
        self.mean_beta_df = pd.DataFrame(mean_beta, index=self.features_name, columns=["Mean Beta"])
        self.std_beta_df = pd.DataFrame(std_beta, index=self.features_name, columns=["std Beta"])
        self.mean_pf_df = pd.DataFrame(mean_pf, index=self.features_name, columns=["Mean P-value (F-test)"])
        self.std_pf_df = pd.DataFrame(std_pf, index=self.features_name, columns=["std P-value (F-test)"])
        self.mean_r2_df = pd.DataFrame(mean_r2, index=self.features_name, columns=["Mean R-square"])
        self.std_r2_df = pd.DataFrame(std_r2, index=self.features_name, columns=["std R-square"])
        self.mean_r2_adj_df = pd.DataFrame(mean_r2_adj, index=self.features_name, columns=["Mean R-square Adjustment"])
        self.std_r2_adj_df = pd.DataFrame(std_r2_adj, index=self.features_name, columns=["std R-square Adjustment"])

        self.results = results

        return results

    def resampling_df(self, values=None):

        '''
        # To get the resampling results:
        resampling_df = test.resampling_df(values='all')
        print(resampling_df)

        '''

        pval = pd.concat([self.mean_p_df, self.std_p_df], axis=1)
        beta = pd.concat([self.mean_beta_df, self.std_beta_df], axis=1)
        fp_val = pd.concat([self.mean_pf_df, self.std_pf_df], axis=1)
        R2 = pd.concat([self.mean_r2_df, self.std_r2_df], axis=1)
        R2_adj = pd.concat([self.mean_r2_adj_df, self.std_r2_adj_df], axis=1)
        resampling_results_df = pd.concat([pval, beta, fp_val, R2, R2_adj, self.mean_qval_df], axis=1)
        self.resampling_results_df = resampling_results_df

        pval_boost = self.p_val_boost
        beta_boost = self.beta_boost
        f_pval_boost = self.f_pval_boost
        r2_boost = self.r2_boost
        adj_r2_boost = self.adj_r2_boost


        if values == "mean_P-value":
            return pval
        elif values == "mean_Beta":
            return beta
        elif values == "mean_P_F-test":
            return fp_val
        elif values == "mean_R2":
            return R2
        elif values == "mean_R2 adj":
            return R2_adj
        elif values == "mean_q-value":
            return self.mean_qval_df
        elif values == "mean_q-value F-test":
            return self.mean_fqval_df
        elif values == 'P-value':
            return pval_boost
        elif values == 'Beta':
            return beta_boost
        elif values == 'P-value Ftest':
            return f_pval_boost
        elif values == 'R2':
            return r2_boost
        elif values == 'Adjust R2':
            return adj_r2_boost
        else:
            return resampling_results_df
 
    def save_boostrap(self, path_save, sample_type="No type"):
        self.path_save = path_save
        self.sample_type = sample_type
        results = self.results
        return np.save('{}/{}_bootstrap_results_univariate[{}_{}].npy'.format(path_save, sample_type, self.label_a, self.label_b), results)

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
        p_adj = self.qval_df
        return p_adj
    
    def log2_fc(self):
        log2_fc_df = self.l2_df2
        return log2_fc_df
    
    def report(self):

        '''
        # Get report
        # This will return a report dataset as p-value, Beta, R_square, and p-value of F-test in one dataframe
        report = test.report()
        print(report)
        '''

        # Creat report dataframe
        pval = self.pval_df
        beta = self.beta_df
        fpval = self.fpval_df
        r2 = self.r2_df
        p_adj = self.qval_df
        log_2fc = self.l2_df2
 
        stats_table = pd.concat([pval, beta, p_adj, r2, fpval, log_2fc], axis=1)
        self.statstable = stats_table
        return stats_table


    def metname_df(self, x_position, met_names):
        self.met_names = met_names
        self.x_position = x_position
        y = self.dataset
        ppm = self.features_name


        y_df = pd.DataFrame(list(y.iloc[:, 2:].max()), columns=["position_y"], index=ppm)

        x_position = list(np.ravel(x_position))

        y_pos = list()
        for i in range(len(list(np.ravel(x_position)))):
            y_pos.append(y_df.at[x_position[i], "position_y"])
        y_pos = pd.DataFrame(y_pos, columns=["position_y"], index=x_position)

        x_pos = list(np.ravel(x_position))
        x_pos = pd.DataFrame(x_pos, columns=["position_x"], index=x_position)

        met_name = list(np.ravel(met_names))
        met_name = pd.DataFrame(met_names, columns=["metabolite"], index=x_position)

        met_label_df = pd.concat([x_pos, y_pos, met_name], axis=1)

        self.met_label_df = met_label_df

        return met_label_df

    def spec_uniplot(self ,pval_position=0, sample=None, label_a=None, label_b=None, met_label=False, p_value=None):

        self.sample_type = sample
        
        dataset = self.dataset
        ppm = self.features_name
        self.pval_position = pval_position
        self.label_a = label_a
        self.label_b = label_b
        self.p_value = p_value

        

        if p_value == "p-value":
            pval = self.pval_df
        if p_value == "q-value":
            pval = self.qval_df
        elif p_value == "boostrap p-value":
            pval = self.mean_p_df
        elif p_value == "boostrap q-value":
            pval = self.mean_qval_df
        #pval.columns=["P-value"]

        meta = dataset[["Label", "Target"]]
   
        spectra = pd.DataFrame(dataset.iloc[:, 2:])
        idx_a = dataset.loc[dataset["Target"] == 0].index
        idx_b = dataset.loc[dataset["Target"] == 1].index


        #set dataset for plot
        df_a = spectra.loc[idx_a]
        df_b = spectra.loc[idx_b]

        meta_a = meta.loc[idx_a]
        meta_b = meta.loc[idx_b]


        if self.label_a == None:
            label_a = meta_a.iat[0,0]
        else:
            label_a = label_a
        if self.label_b == None:
            label_b = meta_b.iat[0,0]
        else:
            label_b = label_b
            
        self.label_a = label_a
        self.label_b = label_b

        code_a = meta_a.iat[0,1]
        code_b = meta_b.iat[0,1]
    

        meanX_a = pd.DataFrame(list(df_a.mean(axis=0)), index=ppm, columns=[label_a])
        std_a = pd.DataFrame(list(df_a.std(axis=0)), index=ppm, columns=["std_a"])
        meanX_b = pd.DataFrame(list(df_b.mean(axis=0)), index=ppm, columns=[label_b])
        std_b = pd.DataFrame(list(df_b.std(axis=0)), index=ppm, columns=["std_b"])
        max_label_y = spectra.max()

        # set p-value position region at -0.8 in y axis
        pval_pos = []
        for i in range(len(ppm)):
            pval_pos.append(self.pval_position)
        pval_pos = pd.DataFrame(pval_pos, index=ppm, columns=["pval_position"])

        plot_df = pd.concat([pval, pval_pos, meanX_a, std_a, meanX_b, std_b], axis=1)

        # Create a scatter plot with the spectra data and the p-values
        fig = px.scatter(plot_df, x=plot_df.index, y="pval_position", 
                        color=plot_df.iloc[:, 0], 
                        color_continuous_scale="Inferno", 
                        range_color=[0.0001, 0.0800], height=900, width=1600)
    
       
        fig.update_layout(coloraxis_colorbar=dict(
                           title="<i>{}</i>".format(p_value),
                           tickvals=[0.0001, 0.0050, 0.0100, 0.0500],
                           ticktext=["0.0001","0.0050", "0.0100", "0.0500"]
                           ))

        # Add a line plot on top of the scatter plot to represent the line data

        fig.add_trace(px.line(plot_df, x=ppm, y="{}".format(label_a), 
                        color_discrete_sequence=["#5C8EFA"], 
                        labels=label_a).data[0])


        fig.add_trace(px.line(plot_df, x=ppm, y="{}".format(label_b),
                        color_discrete_sequence=["#FF6F55"], 
                        labels=label_b).data[0])

        fig.update_traces(textposition='top center')

        fig.update_traces(textposition='top center').data[0]

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(
                    title={
        'text': "<b><i>{}</i> of mean spectra</b>".format(p_value),
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        font=dict(size=18))

        fig.add_annotation(dict(font=dict(color="#5C8EFA",size=14),
                                x=0.98,
                                y=1.05,
                                showarrow=False,
                                text="<b>[{}] {}</b>".format(code_a, label_a),
                                textangle=0,
                                xref="paper",
                                yref="paper"))

        fig.add_annotation(dict(font=dict(color="#FF6F55",size=14),
                                x=0.98,
                                y=1.03,
                                showarrow=False,
                                text="<b>[{}] {}</b>".format(code_b, label_b),
                                textangle=0,
                                xref="paper",
                                yref="paper"))

        if met_label == False:
            fig
        else:
            fig.add_trace(px.scatter(self.met_label_df, 
                                    x="position_x", 
                                    y="position_y",
                                    text="metabolite"
                                    ).data[0])

        fig.update_traces(textposition='top center').data[0]
        

        fig.update_layout(
            xaxis_title="𝛿<sub>H</sub> in ppm",
            yaxis_title="Intensity (AU)",
            font=dict(
                size=14
            )
        )

        fig.update_layout(xaxis = dict(autorange='reversed'))
        # Show the plot
        self.fig = fig


        return fig.show()

    def manhattan_plot(self, plot_title=None, alpha=None):

        
        self.plot_title = plot_title
        x = self.features_name
        pval = self.mean_p_df
        beta = self.mean_beta_df
        R2 = self.mean_r2_adj_df
        q_val = self.mean_qval_df

        pval.columns=["P-value"]
        beta.columns=["Beta coefficient"]
        R2.columns=["R2"]
        log10_pval = pd.DataFrame(-np.log10(np.ravel(pval)), index=self.features_name, columns=["-log10 P-value"])
        y = pd.DataFrame(beta["Beta coefficient"]*(-np.log10(q_val["q_value"])), index = self.features_name, columns=["beta x -log10 q-value"])
        plot_df = pd.concat([pval, q_val, beta, y, R2, log10_pval], axis=1)

        fig = px.scatter(plot_df, x=x, y="beta x -log10 q-value", text="q_value",
                            color="Beta coefficient", range_color=[-1, 1],
                            color_continuous_scale="RdBu",
                            
                            labels={"beta x -log10 q-value": "β × (-log<sub>10</sub> <i>q-value</i>)",
                                    "x": "𝛿<sub>H</sub> in ppm",
                                    "Beta coefficient": "β coefficient",
                                    "text": "<i>q-value</i>"})

        if alpha == None:
            fig
        else:
            fig.add_shape(type='line', x0=min(x), y0=alpha, x1=max(x), y1=alpha,
                line=dict(color='red', width=2, dash='dot'))

            fig.add_shape(type='line', x0=min(x), y0=-alpha, x1=max(x), y1=-alpha,
                line=dict(color='red', width=2, dash='dot'))



        if plot_title == None:
        
            fig.update_layout(title={
                    'text': "<b>Manhattan plot</b>",
                    'y':0.98,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        
        else:
            fig.update_layout(title={
                    'text': "<b>Manhattan plot  {}</b>".format(plot_title),
                    'y':0.98,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})



        fig.update_layout(xaxis = dict(autorange='reversed'))

        self.fig = fig

        return fig.show()
    
    def html_plot(self, plot_name, path_save=None):
        self.path_save = path_save
        self.plot_name = plot_name
        fig = self.fig
        return fig.write_html("{}/{}{}{}_vs_{}.html".format(path_save, self.sample_type,plot_name, self.label_a, self.label_b))
    def png_plot(self, path_save=None):
        self.path_save = path_save
        fig = self.fig
        return fig.write_image("{}/{}{}{}_vs_{}.png".format(path_save, self.sample_type,plot_name, self.label_a, self.label_b))


    def find_pval(self, ppm):
        stats_table = self.stats_table
        self.ppm = ppm
        
        idx = np.abs(stats_table.index.values.astype(float) - ppm).argmin()
        pos_y = stats_table.iloc[idx, 0]
        print("<i>p-value</i>: {pos_y.f}")

    def volcano_plot(self, plot_title=None):

        log2_fc = self.l2_df2
        pval = self.pval_df
        beta = self.beta_df

        log10_p = -np.log10(pval)
        log10_p.columns=["-Log10 P-value"]


        df_vol = pd.concat([log10_p, log2_fc, beta], axis=1)
        df_vol.columns=["-Log10 P-value", "Log2 FC", "Beta"]


        # x and y given as DataFrame columns


        fig = px.scatter(df_vol, x="Log2 FC", y="-Log10 P-value", text=df_vol.index,
                        color="Beta", range_color=[-1, 1],
                        color_continuous_scale="RdBu",
                        labels={"-Log10 P-value": "-log<sub>10</sub> (<i>p-value</i>)",
                                "Log2 FC": "Log<sub>2</sub> (<i>Fold change</i>)",
                                "Beta": "β coefficient"})
        fig.update_layout(
                            title={
                'text': "<b>Volcano plot {}</b>".format(plot_title),
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        fig.add_shape(type='line', x0=-10, y0=2, x1=10, y1=2,
              line=dict(color='red', width=2, dash='dot'))

        fig.add_shape(type='line', x0=-1, y0=0, x1=-1, y1=10,
              line=dict(color='red', width=2, dash='dot'))
              
        fig.add_shape(type='line', x0=1, y0=0, x1=1, y1=10,
              line=dict(color='red', width=2, dash='dot'))

        self.fig = fig


        return fig.show()

