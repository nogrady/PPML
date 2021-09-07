
import sys
sys.path.append('..');
import numpy as np
from discriminant_analysis import DCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

rho = None; 
rho_p = None;

ntests = 50; # 1000 in the paper
# Tries a single sample for each experiment
       
svm_tuned_params = [{'kernel': ['linear'], 'C': [0.1,1,10,100,1000]},{'kernel':
    ['rbf'], 'gamma': [0.00001, 0.0001, 0.001, 0.01], 'C': [0.1,1,10,100,1000]}];

data_dir = './CompPrivacy/DataSet/Glasses/';
X = np.loadtxt(data_dir+'xGlasses2.dat');
y = np.loadtxt(data_dir+'yGlasses2.dat');
p = np.loadtxt(data_dir+'subjectGlasses2.dat');
      
utilAcc_dca = np.zeros((ntests));     
utilAcc_rdca = np.zeros((ntests));
privAcc_dca = np.zeros((ntests));     
privAcc_rdca = np.zeros((ntests));
                       
mydca = DCA(rho,rho_p);
clf = GridSearchCV(SVC(max_iter=1e5),svm_tuned_params,cv=3);
for i in range(ntests):
    print('Experiment %d:' %(i+1));
    Xtr, Xte, ytr, yte, ptr, pte = train_test_split(X,y,p,test_size=0.1);                                         
    
    mydca.fit(Xtr,ytr);
    # Compute DCA components
    Dtr_dca = mydca.transform(Xtr);
    Dte_dca = mydca.transform(Xte);   
    print(' Discriminant components were extracted.')

    # Test accuracy of DCA
    clf.fit(Dtr_dca,ytr);
    utilAcc_dca[i] = clf.score(Dte_dca[-1].reshape(1,-1),[yte[-1]]);
    print(' Utility accuracy of DCA: %f'%utilAcc_dca[i]);
         
    clf.fit(Dtr_dca,ptr);
    privAcc_dca[i] = clf.score(Dte_dca[-1].reshape(1,-1),[pte[-1]]);
    print('Privacy accuracy of DCA: %f'%privAcc_dca[i]);
    
    mydca.fit(Xtr,ptr); 
    n_comps = len(np.unique(ptr))-1;
    # Compute RDCA components
    Dtr_dca = mydca.transform(Xtr,dim=Xtr.shape[1])[:,n_comps:];
    Dte_dca = mydca.transform(Xte,dim=Xte.shape[1])[:,n_comps:];   
    print(' Desensitized components were extracted.')

    # Test accuracy of RDCA
    clf.fit(Dtr_dca,ytr);
    utilAcc_rdca[i] = clf.score(Dte_dca[-1].reshape(1,-1),[yte[-1]]);
    print(' Utility accuracy of RDCA: %f'%utilAcc_rdca[i]);
         
    clf.fit(Dtr_dca,ptr);
    privAcc_rdca[i] = clf.score(Dte_dca[-1].reshape(1,-1),[pte[-1]]);
    print(' Privacy accuracy of RDCA: %f'%privAcc_rdca[i]);
      

