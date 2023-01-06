
import sys
sys.path.append('..');
import numpy as np
from discriminant_analysis import DCA
from matplotlib.pyplot import *
from random import randrange

def displayImage(vImage,height,width):
    mImage = np.reshape(vImage, (height,width)).T
    imshow(mImage, cmap='gray')
    axis('off')
    
height = 64
width = 64

rho = 10; 
#rho_p = -0.05;

rho_p = [-0.05,-0.01,-0.001,0.0,0.001,0.01,0.05]
#dims = [2,5,8,10,14,39,1000,2000,3000,4096];
selected_dim = 160
image_id = randrange(165)
       
data_dir = './CompPrivacy/DataSet/Yale_Faces/';
X = np.loadtxt(data_dir+'Xyale.txt');
y = np.loadtxt(data_dir+'Yyale.txt');
           
subplot(2,4,1)
title('Original',{'fontsize':8})
displayImage(X[image_id],height,width)

for j in range(len(rho_p)):
    
    mydca = DCA(rho,rho_p[j]);   
    mydca.fit(X,y);
    D_dca = mydca.transform(X); 
    print('Discriminant components were extracted for rho_p: '+str(rho_p[j]))
    
    Xrec = mydca.inverse_transform(D_dca[:,:selected_dim],dim=selected_dim);
    eigV_dca = np.reshape(Xrec,(len(X),64,64))
    
    subplot(2,4,j+2)
    title('rho_p: ' + str(rho_p[j]),{'fontsize':8})
    displayImage(eigV_dca[image_id],height,width)
    
show()
        
    




