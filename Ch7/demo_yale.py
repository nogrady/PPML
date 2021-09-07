
import sys
sys.path.append('..');
import numpy as np
from random import randrange
from discriminant_analysis import DCA, PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import *

def displayImage(vImage,height,width):
    mImage = np.reshape(vImage, (height,width)).T
    imshow(mImage, cmap='gray')
    axis('off')

height = 64
width = 64

data_dir = './CompPrivacy/DataSet/Yale_Faces/';
X = np.loadtxt(data_dir+'Xyale.txt');
y = np.loadtxt(data_dir+'Yyale.txt');

print('Shape of the dataset: %s' %(X.shape,))

for i in range(4):    
    subplot(1,4,i+1)
    displayImage(X[randrange(165)], height, width)
show()


rho = 10; 
rho_p = -0.05;
ntests = 10;
#dims = [2,5,8,10,14,39,1000,2000,3000,4096];
dims = [5,14,50,160];

mydca = DCA(rho,rho_p);
mypca = PCA();
       
svm_tuned_params = [{'kernel': ['linear'], 'C': [0.1,1,10,100,1000]},{'kernel':
    ['rbf'], 'gamma': [0.00001, 0.0001, 0.001, 0.01], 'C': [0.1,1,10,100,1000]}];
           
utilAcc_pca = np.zeros((ntests,len(dims)));
utilAcc_dca = np.zeros((ntests,len(dims)));
reconErr_pca = np.zeros((ntests,len(dims)));
reconErr_dca = np.zeros((ntests,len(dims)));

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
    
    subplot(2,5,1)
    title('Original',{'fontsize':8})
    displayImage(Xtr[i],height,width)
    
    subplot(2,5,6)
    title('Original',{'fontsize':8})
    displayImage(Xtr[i],height,width)    
    
    for j in range(len(dims)):
        # Test accuracy of PCA
        clf.fit(Dtr_pca[:,:dims[j]],ytr);
        utilAcc_pca[i,j] = clf.score(Dte_pca[:,:dims[j]],yte);
        print('Utility accuracy of %d-dimensional PCA: %f' 
              %(dims[j],utilAcc_pca[i,j]));
        
        # Test reconstruction error of PCA
        D = np.r_[Dtr_pca[:,:dims[j]],Dte_pca[:,:dims[j]]];
        Xrec = np.dot(D,mypca.components[:dims[j],:]);
        reconErr_pca[i,j] = sum(np.linalg.norm(X-Xrec,2,axis=1))/len(X);
        eigV_pca = np.reshape(Xrec,(len(X),64,64))
        print('Average reconstruction error of %d-dimensional PCA: %f' 
              %(dims[j],reconErr_pca[i,j]));
             
        # Test accuracy of DCA
        clf.fit(Dtr_dca[:,:dims[j]],ytr);
        utilAcc_dca[i,j] = clf.score(Dte_dca[:,:dims[j]],yte);
        print('Utility accuracy of %d-dimensional DCA: %f' 
              %(dims[j],utilAcc_dca[i,j]));
        
        # Test reconstruction error of DCA
        D = np.r_[Dtr_dca[:,:dims[j]],Dte_dca[:,:dims[j]]];
        Xrec = mydca.inverse_transform(D,dim=dims[j]);
        reconErr_dca[i,j] = sum(np.linalg.norm(X-Xrec,2,axis=1))/len(X);
        eigV_dca = np.reshape(Xrec,(len(X),64,64))
        print('Average reconstruction error of %d-dimensional DCA: %f' 
              %(dims[j],reconErr_dca[i,j]));    
              
        subplot(2,5,j+2)
        title('DCA dim: ' + str(dims[j]),{'fontsize':8})
        displayImage(eigV_dca[i],height,width) 
        
        subplot(2,5,j+7)
        title('PCA dim: ' + str(dims[j]),{'fontsize':8})
        displayImage(eigV_pca[i],height,width)        
        
    show()
        





