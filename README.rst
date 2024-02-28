lingress
========

The Lingress project is an initiative aimed at developing a streamlined
pipeline for the analysis of Nuclear Magnetic Resonance (NMR) datasets,
utilizing a univariate linear regression model. This package encompasses
the execution of linear regression analysis via the Ordinary Least
Squares (OLS) method and provides visual interpretations of the
resultant data. Notably, it includes the p-values of all NMR peaks in
its analytical scope.

Functionally, this program strives to fit a model of metabolic profiles
through the application of linear regression. Its design and
capabilities present a robust tool for in-depth and nuanced data
analysis in the realm of metabolic studies.

**How to install**
------------------

.. code:: bash

   pip install lingress

**Example code**
----------------

You can use function unipair to prepare the data for linear regression
model. The function will return a pair-wise dataframe if you have more
than 2 groups to observe.

.. code:: python

   from lingress import unipair
   import pandas as pd

   # Create a unipair object
   test = unipair(dataset = df, column_name='Class')

   test.get_dataset() # Get list of dataset of all pairs

.. code:: python

   from lingress import lin_regression
   import pandas as pd


   # Create a lin_regression object
   test = lin_regression(spectra_X, target=meta['Class'], label=meta['Class'], features = spectra_X.columns)

   # Create dataset to do linear regression model
   dataset = test.create_dataset()

   # Fit model with linear regression
   test.fit_model(dataset, method = "fdr_bh")

   # Get report
   test.report()

Note:

::

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
