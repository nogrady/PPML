
import sys
sys.path.append('..');
import numpy as np
from discriminant_analysis import DCA, PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

rho = 10; 
rho_p = -0.05;

ntests = 30;
dims = [2,5,8,10,14,39,1000,2000,3000,4096];
       
svm_tuned_params = [{'kernel': ['linear'], 'C': [0.1,1,10,100,1000]},{'kernel':
    ['rbf'], 'gamma': [0.00001, 0.0001, 0.001, 0.01], 'C': [0.1,1,10,100,1000]}];

data_dir = './CompPrivacy/DataSet/Olivetti/';
X = np.loadtxt(data_dir+'olivettiX.dat').T;
y = np.loadtxt(data_dir+'olivettiLabel.dat');

utilAcc_pca = np.zeros((ntests,len(dims)));
utilAcc_dca = np.zeros((ntests,len(dims)));
reconErr_pca = np.zeros((ntests,len(dims)));
reconErr_dca = np.zeros((ntests,len(dims)));

mydca = DCA(rho,rho_p);
mypca = PCA();
clf = GridSearchCV(SVC(max_iter=1e5),svm_tuned_params,cv=3);
for i in range(ntests):
    print('Experiment %d:' %(i+1));
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.1,stratify=y);
    mypca.fit(Xtr);
    mydca.fit(Xtr,ytr);
    # Pre-compute all the components
    Dtr_pca = mypca.transform(Xtr);
    Dte_pca = mypca.transform(Xte);
    Dtr_dca = mydca.transform(Xtr);
    Dte_dca = mydca.transform(Xte);   
    print('Principal and discriminant components were extracted.')
    
    for j in range(len(dims)):
        # Test accuracy of PCA
        clf.fit(Dtr_pca[:,:dims[j]],ytr);
        utilAcc_pca[i,j] = clf.score(Dte_pca[:,:dims[j]],yte);
        print('Utility accuracy of %d-dimensional PCA: %f' 
              %(dims[j],utilAcc_pca[i,j]));
        
        # Test reconstruction error of PCA (on train+test)
        D = np.r_[Dtr_pca[:,:dims[j]],Dte_pca[:,:dims[j]]];
        Xrec = np.dot(D,mypca.components[:dims[j],:]);
        reconErr_pca[i,j] = sum(np.linalg.norm(X-Xrec,2,axis=1))/len(X);
        print('Average reconstruction error of %d-dimensional PCA: %f' 
              %(dims[j],reconErr_pca[i,j]));
             
         # Test accuracy of DCA
        clf.fit(Dtr_dca[:,:dims[j]],ytr);
        utilAcc_dca[i,j] = clf.score(Dte_dca[:,:dims[j]],yte);
        print('Utility accuracy of %d-dimensional DCA: %f' 
              %(dims[j],utilAcc_dca[i,j]));
        
        # Test reconstruction error of DCA (on train+test)
        D = np.r_[Dtr_dca[:,:dims[j]],Dte_dca[:,:dims[j]]];
        Xrec = mydca.inverse_transform(D,dim=dims[j]);
        reconErr_dca[i,j] = sum(np.linalg.norm(X-Xrec,2,axis=1))/len(X);
        print('Average reconstruction error of %d-dimensional DCA: %f' 
              %(dims[j],reconErr_dca[i,j]));






