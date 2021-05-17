import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cluster import SpectralClustering
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

## input here
dataSetName = "diabetes"
numofEXP = 10
scenarioList =["SynTR_OrgTE"]      # K means testing only have one scenario
methodList = ["PrivSyn"]
k_list = [25, 50, 75, 100]
eplison_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for scenario in scenarioList:
    for method in methodList:
        f = open("Kmeansresult_" + dataSetName + "_" + str(scenario) + "_" + str(method), "w")
        for clustersize in k_list:
            for eplison in eplison_list:
                f.write(str(clustersize) + "," + str(eplison) + ",")
                avgnmi = 0
                for exp in range(numofEXP):
                    path = "/home/mingchenli/eclipse-workspace/PrivSyn_Demo/ExpData/" + dataSetName + "/" + scenario + "/"
                    if scenario == "SynTR_OrgTE":
                        orgdataset = np.loadtxt(path + "Seed" + dataSetName + "_csv", delimiter=",")

                        syndataset = np.loadtxt(
                            path + method + "/" + dataSetName + "_syn_" + str(clustersize) + "_" + str(
                                eplison) + "_" + str(exp) + "_csv", delimiter=",")
                        orgdataset_split = np.split(orgdataset, [1], axis=1)
                        syndataset_split = np.split(syndataset, [1], axis=1)

                        orgdataset_X = orgdataset_split[1]
                        syndataset_X = syndataset_split[1]


                        kmeans = KMeans(n_clusters=2, n_init=50, random_state=0).fit(orgdataset_X)
                        org_y = kmeans.labels_

                        kmeans = KMeans(n_clusters=2, n_init=50, random_state=0).fit(syndataset_X)
                        syn_y = kmeans.labels_
 
                        nmi_score = normalized_mutual_info_score(org_y, syn_y)

                        print(dataSetName + str(scenario) + "," + str(method) + "," + str(eplison) + "," + str(
                            clustersize) + "," + str(exp))

                        # NMI
                        print("NMI: %.2f"% nmi_score )

                        avgnmi = avgnmi + nmi_score
                    else:
                        pass


                avgnmi = avgnmi/numofEXP
                f.write(str(avgnmi))
                f.write("\n")


