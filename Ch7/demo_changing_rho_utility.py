
import sys
sys.path.append('..');
import numpy as np
from random import randrange
from discriminant_analysis import DCA, PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import *

height = 64
width = 64

data_dir = './CompPrivacy/DataSet/Yale_Faces/';
X = np.loadtxt(data_dir+'Xyale.txt');
y = np.loadtxt(data_dir+'Yyale.txt');

rho = 10; 
rho_p = [-0.05,-0.01,-0.001,0.0,0.001,0.01,0.05]
ntests = 2;

#dims = [2,5,8,10,14,39,1000,2000,3000,4096];
dims = [5,14,50,160];
       
svm_tuned_params = [{'kernel': ['linear'], 'C': [0.1,1,10,100,1000]},{'kernel':
    ['rbf'], 'gamma': [0.00001, 0.0001, 0.001, 0.01], 'C': [0.1,1,10,100,1000]}];
           
utilAcc_dca = np.zeros((len(rho_p),ntests,len(dims)));
reconErr_dca = np.zeros((len(rho_p),ntests,len(dims)));

clf = GridSearchCV(SVC(max_iter=1e5),svm_tuned_params,cv=3);

for k in range(len(rho_p)):
    mydca = DCA(rho,rho_p[k]);

    for i in range(ntests):
        print('DCA rho_p: ' + str(rho_p[k]) + ' - Experiment %d:' %(i+1));
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.1,stratify=y);
        mydca.fit(Xtr,ytr);

        # Pre-compute all the components   
        Dtr_dca = mydca.transform(Xtr);
        Dte_dca = mydca.transform(Xte);   
        print('  Discriminant components were extracted.')  

        for j in range(len(dims)):        
            # Test accuracy of DCA
            clf.fit(Dtr_dca[:,:dims[j]],ytr);
            utilAcc_dca[k,i,j] = clf.score(Dte_dca[:,:dims[j]],yte);
            print('  Utility accuracy of %d-dimensional DCA: %f' 
                  %(dims[j],utilAcc_dca[k,i,j]));

            # Test reconstruction error of DCA
            D = np.r_[Dtr_dca[:,:dims[j]],Dte_dca[:,:dims[j]]];
            Xrec = mydca.inverse_transform(D,dim=dims[j]);
            reconErr_dca[k,i,j] = sum(np.linalg.norm(X-Xrec,2,axis=1))/len(X);
            eigV_dca = np.reshape(Xrec,(len(X),64,64))
            print('  Average reconstruction error of %d-dimensional DCA: %f' 
                  %(dims[j],reconErr_dca[k,i,j]));    
              
        


np.savetxt('utilAcc_dca_new.out', utilAcc_dca, delimiter=',')
np.savetxt('reconErr_dca_new.out', reconErr_dca, delimiter=',')






