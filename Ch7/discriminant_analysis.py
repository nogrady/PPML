
import numpy as np
import scipy
from sklearn.metrics import pairwise
from sklearn import preprocessing

class DCA:
    def __init__(self, rho=None, rho_p=None, n_components=None):
        self.n_components = n_components
        self.rho = rho
        self.rho_p = rho_p

    def fit(self, X, y):
        (self._Sw, self._Sb) = self._get_Smatrices(X,y)

        if self.rho == None:
            s0 = np.linalg.eigvalsh(self._Sw)
            self.rho = 0.02*np.max(s0)
        if self.rho_p == None:
            self.rho_p = 0.1*self.rho

        pSw = self._Sw + self.rho*np.eye(self._Sw.shape[0])
        pSbar = self._Sb + self._Sw + (self.rho_p+self.rho)*np.eye(self._Sw.shape[0])
        (s1,vr) = scipy.linalg.eigh(pSbar,pSw,overwrite_a=True,overwrite_b=True)
        s1 = s1[::-1] #re-order from large to small
        Wdca = vr.T[::-1]
        self.eigVal = s1
        self.allComponents = Wdca
        if self.n_components:
            self.components = Wdca[0:self.n_components]
        else:
            self.components = Wdca


    def transform(self, X, dim=None):
        if dim == None:
            X_trans = np.inner(self.components,X)
        else:
            X_trans = np.inner(self.allComponents[0:dim],X)
        return X_trans.T

    def inverse_transform(self, Xreduced, projMatrix=None, dim=None):
        if projMatrix is None:
            if dim is None:
                W = self.components
            else:
                W = self.allComponents[0:dim]
        else:
            W = projMatrix
        #W = PxM where P<M
        foo = np.inner(W,W)
        bar = np.linalg.solve(foo.T,W)
        Xhat = np.inner(Xreduced,bar.T)
        return Xhat

    def _get_Smatrices(self, X,y):
        Sb = np.zeros((X.shape[1],X.shape[1]))
        
        S = np.inner(X.T,X.T)
        N = len(X)
        mu = np.mean(X,axis=0)
        classLabels = np.unique(y)
        for label in classLabels:
            classIdx = np.argwhere(y==label).T[0]
            Nl = len(classIdx)
            xL = X[classIdx]
            muL = np.mean(xL,axis=0)
            muLbar = muL - mu
            Sb = Sb + Nl*np.outer(muLbar,muLbar)

        Sbar = S - N*np.outer(mu,mu)
        Sw = Sbar - Sb
        self.mean_ = mu

        return (Sw,Sb)

class PCA:
    def __init__(self,n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        Xbar = X - np.mean(X,axis=0);
        Sbar = np.dot(Xbar.T,Xbar);

        (s1,vr) = scipy.linalg.eigh(Sbar,overwrite_a=True)
        s1 = s1[::-1] #re-order from large to small
        Wpca = vr.T[::-1]
        self.eigVal = s1
        self.allComponents = Wpca
        if self.n_components:
            self.components = Wpca[0:self.n_components]
        else:
            self.components = Wpca

    def transform(self, X, dim=None):
        if dim == None:
            X_trans = np.inner(self.components,X)
        else:
            X_trans = np.inner(self.allComponents[0:dim],X)
        return X_trans.T

    def inverse_transform(self, Xreduced, projMatrix=None, dim=None):
        if projMatrix is None:
            if dim is None:
                W = self.components
            else:
                W = self.allComponents[0:dim]
        else:
            W = projMatrix
            #W = PxM where P<M
        Xhat = np.inner(Xreduced,W.T)
        return Xhat


class KDCA:
    def __init__(self, rho=None, rho_p=None, n_components=None, kernel='rbf',gamma = 1, degree = 3, coef0=1):
        self.n_components = n_components
        self.rho = rho
        self.rho_p = rho_p
        self._kernel = kernel
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0

    def fit(self, X, y):
        self._X = X
        (self._K, self._Kbar, self._Kbar2, self._Kw, self._Kb) = self._get_Kmatrices(X,y)
    
        if self.rho == None:
            s0 = np.linalg.eigvalsh(self._Kbar2)
            self.rho = 0.02*np.max(s0)
        if self.rho_p == None:
            self.rho_p = 0.1*self.rho

        pKb = self._Kb + self.rho_p*self._Kbar
        pKbar2 = self._Kbar2 + self.rho*self._Kbar
        (u,s,vT) = scipy.linalg.svd(pKbar2)
        s2 = np.diag(1.0/np.sqrt(s))
        pKbar2_nhalf = np.inner(u,s2) 

        pKb_cvs = np.inner(np.inner(pKbar2_nhalf.T,pKb),pKbar2_nhalf.T)
        (s3,vr) = scipy.linalg.eigh(pKb_cvs,overwrite_a=True)
        ## cols of u/ rows of vT are eigvect
        self.eigVal = s3[::-1]

        ## backward mapping
        alphaEvs = np.inner(vr.T,pKbar2_nhalf)
        Akdca = alphaEvs[::-1]


# (s1,vr) = scipy.linalg.eigh(pKb,pKbar2,overwrite_a=True,overwrite_b=True)
# s1 = s1[::-1] #re-order from large to small
# Akdca = vr.T[::-1]
# self.eigVal = s1


        self.allComponents = Akdca
        if self.n_components:
            self.components = Akdca[0:self.n_components]
        else:
            self.components = Akdca
        self._alphaBar = Akdca - np.outer(np.sum(Akdca,axis=1),np.ones(len(self._X)))/len(self._X)	


    def transform(self, x, dim=None):
        kx = self._get_kernel_matrix(self._X,x) #kx = [k(x1) k(x2) ...] where x = [x1 x2 ...].T
        if dim == None:
            alphaBar = self._alphaBar[0:self.n_components]
        else:
            alphaBar = self._alphaBar[0:dim]

        X_trans = np.inner(alphaBar,kx.T)

        return X_trans.T


    def _get_Kmatrices(self, X,y):
        K = self._get_kernel_matrix(X,X)
        N = len(X)
        Kw = np.zeros((N,N))
        classLabels = np.unique(y)
        for label in classLabels:
            classIdx = np.argwhere(y==label).T[0]
            Nl = len(classIdx)
            xL = X[classIdx]
            Kl = self._get_kernel_matrix(X,xL)
            Kmul = np.sum(Kl,axis=1)/Nl #vector
            Kmul = np.outer(Kmul,np.ones(Nl)) # matrix
            Klbar = Kl - Kmul
            Kw = Kw + np.inner(Klbar,Klbar)

        #centering
        KwCenterer = preprocessing.KernelCenterer()
        KwCenterer.fit(Kw)
        Kw = KwCenterer.transform(Kw)
        KCenterer = preprocessing.KernelCenterer()
        KCenterer.fit(K)
        Kbar = KCenterer.transform(K)
        Kbar2 = np.inner(Kbar,Kbar.T)
        Kb = Kbar2 - Kw
    
        return (K, Kbar, Kbar2, Kw, Kb)

    def _get_kernel_matrix(self,X1,X2):
        # K is len(X1)-by-len(X2) matrix
        if self._kernel == 'rbf':
            K = pairwise.rbf_kernel(X1,X2,gamma = self._gamma)
        elif self._kernel == 'poly':
            K = pairwise.polynomial_kernel(X1,X2,degree=self._degree, gamma = self._gamma, coef0 = self._coef0)
        elif self._kernel == 'linear':
            K = pairwise.linear_kernel(X1,X2)
        elif self._kernel == 'laplacian':
            K = pairwise.laplacian_kernel(X1,X2,gamma=self._gamma)
        elif self._kernel == 'chi2':
            K = pairwise.chi2_kernel(X1,X2,gamma=self._gamma)
        elif self._kernel == 'additive_chi2':
            K = pairwise.additive_chi2_kernel(X1,X2)
        elif self._kernel == 'sigmoid':
            K = pairwise.sigmoid_kernel(X1,X2,gamma=self._gamma,coef0=self._coef0)
        else:
            print ('[Error] Unknown kernel')
            K = None

        return K


class KPCA:
    def __init__(self, n_components=None, kernel='rbf',gamma = 1, degree = 3, coef0=1):
        self.n_components = n_components
        self._kernel = kernel
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0

    def fit(self, X):
        self._X = X
        (self._K, self._Kbar) = self._get_Kmatrices(X)
        (u,s,vT)= scipy.linalg.svd(self._Kbar,lapack_driver='gesvd')
        self.eigVal = s
        ## scale to meet the constraint
        sqrtSinv = np.diag(1.0/np.sqrt(s))
        alphaEvs = np.inner(sqrtSinv,u)
        Akdca = alphaEvs

        self.allComponents = Akdca
        if self.n_components:
            self.components = Akdca[0:self.n_components]
        else:
            self.components = Akdca
        self._alphaBar = Akdca - np.outer(np.sum(Akdca,axis=1),np.ones(len(self._X)))/len(self._X)	


    def transform(self, x, dim=None):
        kx = _get_kernel_matrix(self._X,x) #kx = [k(x1) k(x2) ...] where x = [x1 x2 ...].T
        if dim is None:
            alphaBar = self._alphaBar[0:self.n_components]
        else:
            alphaBar = self._alphaBar[0:dim]
        X_trans = np.inner(alphaBar,kx.T)
        return X_trans.T


    def _get_Kmatrices(self, X):
        K = _get_kernel_matrix(X, X)
        KCenterer = preprocessing.KernelCenterer()
        KCenterer.fit(K)
        Kbar = KCenterer.transform(K)
        return (K, Kbar)

    def _get_kernel_matrix(self,X1,X2):
        # K is len(X1)-by-len(X2) matrix
        if self._kernel == 'rbf':
            K = pairwise.rbf_kernel(X1,X2,gamma = self._gamma)
        elif self._kernel == 'poly':
            K = pairwise.polynomial_kernel(X1,X2,degree=self._degree, gamma = self._gamma, coef0 = self._coef0)
        elif self._kernel == 'linear':
            K = pairwise.linear_kernel(X1,X2)
        elif self._kernel == 'laplacian':
            K = pairwise.laplacian_kernel(X1,X2,gamma=self._gamma)
        elif self._kernel == 'chi2':
            K = pairwise.chi2_kernel(X1,X2,gamma=self._gamma)
        elif self._kernel == 'additive_chi2':
            K = pairwise.additive_chi2_kernel(X1,X2)
        elif self._kernel == 'sigmoid':
            K = pairwise.sigmoid_kernel(X1,X2,gamma=self._gamma,coef0=self._coef0)
        else:
            print ('[Error] Unknown kernel')
            K = None

        return K






