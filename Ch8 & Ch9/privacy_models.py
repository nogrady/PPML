
#---------------------------------------------------
# privacy_models.py
# Privacy-Preserving Machine Learning
# Source code for the examples used in the Chapter 9
# Author: Dumindu Samaraweera 
#---------------------------------------------------

# Listing 1 Preparing the dataset
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.patches as patches

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

names = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 
         'marital-status', 'occupation', 'relationship', 
         'race', 'sex', 'capital-gain', 'capital-loss', 
         'hours-per-week', 'native-country', 'income',)

categorical = set(('workclass', 'education', 'marital-status', 
                   'occupation', 'relationship', 'sex',
                   'native-country', 'race', 'income',))

df = pd.read_csv("./Data/adult.all.txt", 
                 sep=", ", header=None, names=names, 
                 index_col=False, engine='python')

df.head()
df.nunique()

#make these as categorical
for name in categorical:
    df[name] = df[name].astype('category')    

#----------------------------------------------------------------
# Listing 2 Finding the value span
def get_spans(df, partition, scale=None):
    
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
        print("Column:", column, "Span:", span)
    return spans
  
full_spans = get_spans(df, df.index)

#----------------------------------------------------------------
# Listing 3 Partitiong the dataset
def split(df, partition, column):
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:        
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)

#check whether its k-anonymous (k=3)
def is_k_anonymous(df, partition, sensitive_column, k=3):
    if len(partition) < k:
        return False
    return True

def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

feature_columns = ['age', 'education-num']
sensitive_column = 'income'
finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)

print(len(finished_partitions))

#----------------------------------------------------------------
# Listing 4 Building the anonymized dataset
def agg_categorical_column(series):
    return [','.join(set(series))]

def agg_numerical_column(series):
    return [series.mean()]

def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        values = grouped_columns.iloc[0].to_dict()
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count' : count,

            })
            rows.append(values.copy())
    return pd.DataFrame(rows)

dfn = build_anonymized_dataset(df, finished_partitions, feature_columns, sensitive_column)

dfn.head()

#----------------------------------------------------------------
# Listing 5 Anonymizing the dataset with l-diversity
def diversity(df, partition, column):
    return len(df[column][partition].unique())

#check whether its l-diverse (l=2)
def is_l_diverse(df, partition, sensitive_column, l=2):
    return diversity(df, partition, sensitive_column) >= l

finished_l_diverse_partitions = partition_dataset(df, feature_columns, 
                                                  sensitive_column, 
                                                  full_spans, 
                                                  lambda *args: is_k_anonymous(*args) 
                                                  and is_l_diverse(*args))

print(len(finished_l_diverse_partitions))


column_x, column_y = feature_columns[:2]
dfl = build_anonymized_dataset(df, finished_l_diverse_partitions, feature_columns, sensitive_column)

print(dfl.sort_values([column_x, column_y, sensitive_column]))
dfl.head()

#----------------------------------------------------------------
# Listing 6 Checking the frequencies
global_freqs = {}
total_count = float(len(df))
group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')
for value, count in group_counts.to_dict().items():
    p = count/total_count
    global_freqs[value] = p

print(global_freqs)

#----------------------------------------------------------------
# Listing 7 Anonymizing the dataset with t-closeness
def t_closeness(df, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count/total_count
        d = abs(p-global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max

def is_t_close(df, partition, sensitive_column, global_freqs, p=0.2):
    if not sensitive_column in categorical:
        raise ValueError("this method only works for categorical values")
    return t_closeness(df, partition, sensitive_column, global_freqs) <= p

finished_t_close_partitions = partition_dataset(df, feature_columns, sensitive_column, 
                                                full_spans, 
                                                lambda *args: is_k_anonymous(*args) 
                                                and is_t_close(*args, global_freqs))

print(len(finished_t_close_partitions))



dft = build_anonymized_dataset(df, finished_t_close_partitions, feature_columns, sensitive_column)

print(dft.sort_values([column_x, column_y, sensitive_column]))

dft.head()

