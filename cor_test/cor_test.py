import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import statsmodels.api as sm


## differential correlation test adapted in python from Lea, Subramaniam et al
#https://github.com/AmandaJLea/differential_correlation

def center_scale(vec: pd.Series, covariate: npt.NDArray) -> pd.Series:
    """
    Center and scale the vector of gene expression across all patients in both groups separately (mean 0 and variance 1)
    
    Inputs:
    vec: patients in rows and genes in columns. 
    covariate: only allowed values are 0 and 1 for distinguishing the two groups of patients

    Outputs:
    vec_scaled
    """
    vec_scaled = np.zeros(vec.shape[0])
    vec_reshaped = vec.values.reshape(-1, 1)
    vec_scaled[covariate == 0] = StandardScaler().fit(vec_reshaped[covariate == 0]).transform(vec_reshaped[covariate == 0]).reshape(-1)
    vec_scaled[covariate == 1] = StandardScaler().fit(vec_reshaped[covariate == 1]).transform(vec_reshaped[covariate == 1]).reshape(-1)

    return vec_scaled

def cor_test(vec: pd.Series, ct_prop: npt.NDArray, covariate: npt.NDArray) -> pd.Series:
    """
    Inputs: vec: gene vector across all patients, covariate: patient groups of interest
    Linear regression correlation test in one gene. Output beta, standard error and pvalue
    Output: pandas Series with stats for LR test.
    """

    vec_scaled = center_scale(vec, covariate)
    prod = np.multiply(vec_scaled, ct_prop)

    x = sm.add_constant(covariate.tolist())

    mod = sm.OLS(prod, x)
    fii = mod.fit()
    return fii.summary2().tables[1].iloc[1, :]


def cor_test_all(mat: pd.DataFrame, ct_prop: npt.NDArray, covariate: npt.NDArray) -> pd.DataFrame:
    """
    Run correlation test on all genes
    """
    res = mat.apply(cor_test, ct_prop = ct_prop, covariate = covariate, axis = 0).T
    res.index = mat.columns
    return res
    

### example 
mat = pd.DataFrame(np.random.random((20, 13)), index = [f"p{i}" for i in range(20)], columns = [f"g{i}" for i in range(13)])
covariate = np.random.binomial(1, 0.5, 20)
ct_prop = np.random.random(20)
#result = cor_test(mat.iloc[:, 0], covariate)
test =cor_test_all(mat, ct_prop = ct_prop, covariate = covariate)
