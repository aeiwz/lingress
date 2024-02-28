# lingress
The Lingress project is an initiative aimed at developing a streamlined pipeline for the analysis of Nuclear Magnetic Resonance (NMR) datasets, utilizing a univariate linear regression model. This package encompasses the execution of linear regression analysis via the Ordinary Least Squares (OLS) method and provides visual interpretations of the resultant data. Notably, it includes the p-values of all NMR peaks in its analytical scope.

Functionally, this program strives to fit a model of metabolic profiles through the application of linear regression. Its design and capabilities present a robust tool for in-depth and nuanced data analysis in the realm of metabolic studies.

## **How to install**

```bash
pip install lingress
```

## **Example code**


    for examples:
        #set parameter
        x = Metabolic profiles dataset (dataset X)
        taget = Metadata of two group (reccomend to sub-class of group label) (dataset Y)
        Feature_name = name of features of dataset X in this case define to columns name of dataset X

        X = spectra_X
        target = meta['Class']
        ppm = spectra_X.columns.astype(float) # columns name of example data is ppm of spectra

        test = lin_regression(X, target=target, label=target, features_name=ppm)

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

