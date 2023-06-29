# lingress
The lingress project is to develop the pipeline to analyse the Nuclear magnetic resonance (NMR) dataset using a univariate linear regression model. This package contains the analysis with linear regression (OLS) and visualises the interpretation of the results with a p-value of all NMR peaks.


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
