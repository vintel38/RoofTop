# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:26:58 2021

@author: Vintel38
"""

import cv2
# import numpy as np
import os
# import matplotlib.pyplot as plt 
import math
from tqdm import tqdm
#import sys

repo = r'C:\Users\VArri\Documents\Rooftop\dataset\dataset\dataset'
train_repo = os.path.join(repo, 'train', 'images')
test_repo = os.path.join(repo, 'test', 'images')

# crée les dossiers nécessaires de preprocess
os.chdir(repo)
if not os.path.isdir('preprocessing'):
    os.mkdir('preprocessing')
    os.mkdir('preprocessing/train')
    os.mkdir('preprocessing/test')

image_dir = os.listdir(train_repo)

def preprocess(repo):
    
    repo_dir = os.path.join(repo, 'images')
    image_dir = os.listdir(repo_dir)
    
    # print(repo_dir)
    # print(image_dir)
    # sys.exit()
    for i in tqdm(range(len(image_dir))):
        
        img_dir = image_dir[i]
        name = img_dir.split('.')[0]
        
        # chaque image INRIA de resolution 0.3m est lue puis resize selon les 
        # deux dimensions par un facteur 2 pour atteindre une résolution de 
        # 0.6 m sur les deux dimensions 
        # print(img.shape)
        img = cv2.imread(os.path.join(repo_dir, img_dir))
        dsize=(int(img.shape[0]/2), int(img.shape[1]/2))
        img_scl = cv2.resize(img, dsize)
        
        
        # print(image_dir[i])
        for j in range(math.ceil(img_scl.shape[0]/1024)):
            for k in range(math.ceil(img_scl.shape[1]/1024)):
                
                # pour chaque cliché, on le découpe en carré de 1024x1024px en
                # gardant le dernier carré non complet. Quand on arrive sur un
                # carré pas totalement dans l'image, on dessine le carré à l'envers
                # en partant de la bordure. Ne fonctionne pas étrangement pour 
                # le carré en bas à droite de l'image. 
                if (j+1)*1024>img_scl.shape[0] and (k+1)*1024>img_scl.shape[1]:
                    continue
                elif (j+1)*1024>img_scl.shape[0]:
                    img_prt = img_scl[-1025:-1,k*1024:(k+1)*1024,:]
                elif (k+1)*1024>img_scl.shape[1]:
                    img_prt = img_scl[j*1024:(j+1)*1024,-1025:-1,:]
                else:
                    img_prt = img_scl[j*1024:(j+1)*1024,k*1024:(k+1)*1024,:]
                # print(img_prt.shape)
                
                # enfin cv2 imwrite est capable d'écrire des fichiers tif dans
                # un emplacement particulier mais sans garder les infos spatiales
                cv2.imwrite('preprocessing/'+repo+'/'+name+str(j)+str(k)+'.png', img_prt)
    
preprocess('train')
preprocess('test')