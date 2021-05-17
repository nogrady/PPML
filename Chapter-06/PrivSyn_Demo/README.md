# PrivSyn: Private Synthetic Data Release via Feature-level Microaggregation
## Synthetic data generation tool
This document introduce the features and instruction of our synthetic data generation tool. 

## Features
- Generate the synthetic datasets
- Testing synthetic dataset on 3 machine learning tasks

## Instruction
Import the *PrivSyn_Demo* into Eclipse as a maven project.Import *Testing_LinearRegression.py* and *Testing_KMEANS.py* into any python IDE. We provide 3 toy dataset to run the demo, 
**Step 1**
Split your dataset using *PrivSyn_Demo.SynDataGeneration.DataSetSplit.java*. It will split the dataset by label and generate the seed datasets for next step.
**Step 2**
Generate the synthetic dataset using *PrivSyn_Demo.SynDataGeneration.generativeModel.java*.
**Step 3**
Test the synthetic dataset on support vector machine (*libsvm.crossvalidation.training_and_testing_multithread.java*), linear regression (*Testing_LinearRegression.py*) and k-means (*Testing_KMEANS.py*). 
Notice, before you run the testing on linear regression and k-means, you need to change the format of the synthetic dataset, please refer to(*PrivSyn_Demo.SynDataGeneration.LibSVMtoCSV.java*). If you would like to calculate the average result of SVM, using *libsvm.crossvalidation.getAVGResults.java*
