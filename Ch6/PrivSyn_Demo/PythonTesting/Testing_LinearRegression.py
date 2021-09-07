import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

## input here
dataSetName = "breast-cancer"   # diabetes  breast-cancer  australian
numofEXP = 10
scenarioList = ["SynTR_OrgTE"]
methodList = ["PrivSyn"]
k_list = [25, 50, 75, 100]
eplison_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for scenario in scenarioList:
    for method in methodList:
        f = open("result_" + dataSetName + "_" + str(scenario) + "_" + str(method), "w")
        for clustersize in k_list:
            for eplison in eplison_list:
                f.write(str(clustersize) + "," + str(eplison) + ",")
                avgmse = 0
                for exp in range(numofEXP):
                    path = "./ExpData/" + dataSetName + "/" + scenario + "/"
                    if scenario == "OrgTR_SynTE":
                        dataset_train = np.loadtxt(path + "NotSeed" + dataSetName + "_csv", delimiter=",")

                        dataset_test = np.loadtxt(
                            path + method + "/" + dataSetName + "_syn_" + str(clustersize) + "_" + str(
                                eplison) + "_" + str(exp) + "_csv", delimiter=",")

                        dataset_train_split = np.split(dataset_train, [1], axis=1)
                        dataset_test_split = np.split(dataset_test, [1], axis=1)

                        dataset_X_train = dataset_train_split[1]
                        dataset_X_test = dataset_test_split[1]
                        dataset_y_train = dataset_train_split[0]
                        dataset_y_test = dataset_test_split[0]

                        # Create linear regression object
                        regr = linear_model.LinearRegression()

                        # Train the model using the training sets
                        regr.fit(dataset_X_train, dataset_y_train)

                        # Make predictions using the testing set
                        dataset_y_pred = regr.predict(dataset_X_test)
                        print(dataSetName + str(scenario) + "," + str(method) + "," + str(eplison) + "," + str(
                            clustersize) + "," + str(exp))

                        # The mean squared error
                        print("Mean squared error: %.2f"
                              % mean_squared_error(dataset_y_test, dataset_y_pred))
                        avgmse = avgmse + mean_squared_error(dataset_y_test, dataset_y_pred)
                    else:

                        dataset_train = np.loadtxt(
                            path + method + "/" + dataSetName + "_syn_" + str(clustersize) + "_" + str(
                                eplison) + "_" + str(exp) + "_csv", delimiter=",")



                        dataset_test = np.loadtxt(path + "NotSeed" + dataSetName + "_csv", delimiter=",")

                        dataset_train_split = np.split(dataset_train, [1], axis=1)
                        dataset_test_split = np.split(dataset_test, [1], axis=1)

                        dataset_X_train = dataset_train_split[1]
                        dataset_X_test = dataset_test_split[1]
                        dataset_y_train = dataset_train_split[0]
                        dataset_y_test = dataset_test_split[0]

                        # Create linear regression object
                        regr = linear_model.LinearRegression()

                        # Train the model using the training sets
                        regr.fit(dataset_X_train, dataset_y_train)

                        # Make predictions using the testing set
                        dataset_y_pred = regr.predict(dataset_X_test)
                        print(dataSetName + str(scenario) + "," + str(method) + "," + str(eplison) + "," + str(
                            clustersize) + "," + str(exp))
                        # The mean squared error
                        print("Mean squared error: %.2f"
                              % mean_squared_error(dataset_y_test, dataset_y_pred))
                        avgmse = avgmse + mean_squared_error(dataset_y_test, dataset_y_pred)

                avgmse = avgmse / numofEXP
                f.write(str(avgmse))
                f.write("\n")


