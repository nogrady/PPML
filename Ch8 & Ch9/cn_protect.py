
#---------------------------------------------------
# cn_protect.py
# Privacy-Preserving Machine Learning
# Source code for the examples used in the Chapter 8
# Author: Dumindu Samaraweera 
#---------------------------------------------------


# Section 8.4.3 Implementing k-anonymity in Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cn.protect import Protect
sns.set(style="darkgrid")

df = pd.read_csv('./Data/all.data.csv')

df.shape
df.head()

prot = Protect(df)

prot.itypes

prot.itypes.age = 'quasi'
prot.itypes.sex = 'quasi'
prot.itypes


prot.privacy_model

prot.privacy_model.k = 5

prot.privacy_model

#------------------------------------------------------
prot_df = prot.protect()
prot_df

