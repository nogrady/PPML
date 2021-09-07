
#---------------------------------------------------
# anonymize.py
# Privacy-Preserving Machine Learning
# Source code for the examples used in the Chapter 8
# Author: Dumindu Samaraweera 
#---------------------------------------------------

import pandas as pd
import numpy as np
import scipy.stats
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder

# get rid of warnings
import warnings
warnings.filterwarnings("ignore")

# get more than one output per Jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# for functions we implement later
from utils import best_fit_distribution
from utils import plot_result

# Section 8.4.1	Implementing Data Sanitization Operations in Python

df = pd.read_csv('./Data/all.data.csv')
df.shape
df.head()

df.drop(columns=["fnlwgt", "relationship"], inplace=True)

encoders = [(["sex"], LabelEncoder()), (["race"], LabelEncoder())]
mapper = DataFrameMapper(encoders, df_out=True)
new_cols = mapper.fit_transform(df.copy())
df = pd.concat([df.drop(columns=["sex", "race"]), new_cols], axis="columns")
df.head()

df.nunique()

#---------------------------------------------------------------------------
categorical = []
continuous = []

#lest assume everything less than 20 is categorical (e.g. education)
#everythinng greater than 20 is continous (e.g. age)

# Listing 1 Perturbation
categorical = ['race']
continuous = ['age']
unchanged = []

for col in list(df):
    if (col not in categorical) and (col not in continuous):
        unchanged.append(col)

for col in categorical:        
        counts = df[col].value_counts()        
        np.random.choice(list(counts.index), p=(counts/len(df)).values, size=5)

best_distributions = []
for col in continuous:
    data = df[col]
    best_dist_name, best_dist_params = best_fit_distribution(data, 50)
    best_distributions.append((best_fit_name, best_fit_params))    
    
print("Best distribution found: ", best_dist_name)
plot_result(df, continuous, best_distributions)

# Listing 2 Perturbation for both numerical and categorical values
def perturb_data(df, unchanged_cols, categorical_cols, continuous_cols, best_distributions, n, seed=0):
    np.random.seed(seed)
    data = {}

    for col in categorical_cols:
        counts = df[col].value_counts()
        data[col] = np.random.choice(list(counts.index), p=(counts/len(df)).values, size=n)

    for col, bd in zip(continuous_cols, best_distributions):
        dist = getattr(scipy.stats, bd[0])        
        data[col] = np.round(dist.rvs(size=n, *bd[1]))
        
        
    for col in unchanged_cols:
        data[col] = df[col]
   
    return pd.DataFrame(data, columns=unchanged_cols+categorical_cols+continuous_cols)    

gendf = perturb_data(df, unchanged, categorical, continuous, best_distributions, n=48842)

gendf.shape
gendf.head()

