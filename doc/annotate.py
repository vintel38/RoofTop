# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:45:11 2021

@author: Vintel38
"""

#import numpy as np
#import PIL 
#import matplotlib.pyplot as plt 
import cv2
import os
import sys
# import json

# NE FONCTIONNE PAS POUR LES IMAGES CNIR TIFF !!!!

print(os.getcwd())

repo = r'C:\Users\VArri\Documents\Rooftop\dataset\dataset\dataset'
train_repo = os.path.join(repo, 'preprocessing', 'train')
test_repo = os.path.join(repo, 'preprocessing', 'test')

# train_repo=r'C:\Users\VArri\Google Drive\gee-data'
image_dir = os.listdir(train_repo)

# pour chaque photo satellite en 1024x1024 px resized en 0.6 m de resolution 
# préparation du fichier .JSON d'annotations dans lequel seront stockées les données d'annotations. 
size_l = os.path.getsize(os.path.join(train_repo, image_dir[0]))
size = str(size_l)

def init_json(image_dir, size):
    string = '"'+image_dir+size+'":{"fileref":"","size":'+size+',"filename":"'+image_dir+'","base64_img_data":"","file_attributes":{},"regions":{'
    return string

string = init_json(image_dir[0], str(size_l))
# print(string)
# sys.exit()

def draw_roi(event, x, y, flags, param):
    img2 = img.copy()
 
    if event == cv2.EVENT_LBUTTONDOWN: # Left click, select point
        pts.append((x, y))  
 
    if event == cv2.EVENT_RBUTTONDOWN: # Right click to cancel the last selected point
        pts.pop()  
 
    if len(pts) > 0:
                 # Draw the last point in pts
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
 
    if len(pts) > 1:
                 # 
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1) # x ,y is the coordinates of the mouse click place
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
 
    cv2.imshow('image', img2)
    
print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: determine the ROI area")
print("[INFO] Press ‘S’ to determine the selection area and save it")
print("[INFO] Press ESC to quit")

i=0 # index de l'image dans le répertoire 
j=0 # index de la forme dessinée dans l'image 
while True:
    
    img = cv2.imread(os.path.join(train_repo, image_dir[i]))
    
    pts = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit()
            
            
        if key == ord("s"):
            # sauvegarde les points donnés par l'application dans le str string
            saved_data = pts
            if len(saved_data)<3:
                print('<3 points ne peuvent être traités comme une forme')
                break
            all_pts_x = [saved_data[i][0] for i in range(len(saved_data))]
            all_pts_y = [saved_data[i][1] for i in range(len(saved_data))]
            string = string + '"'+str(j)+'":{"shape_attributes":{"name":"polygon","all_points_x":'+ str(all_pts_x)+',"all_points_y":'+str(all_pts_y)+'},"region_attributes":{"nature":"dark_roof"}},'
            j=j+1
            print("Les {} points ont été ajoutés au fichier .JSON".format(len(saved_data)))
            break
        
        
        if key == ord("n"):
            # sauvegarder les annotations dans un fichier JSON et 
            # change of image en passant à la suivante 
            string = string[:-1] + '}}'
            jsonFile = open( os.path.splitext(image_dir[i])[0]+".json", "w")
            jsonFile.write(string)
            jsonFile.close()
            print(os.getcwd())
            print('Le système a écrit le fichier JSON et passe à l image suivante')
            i=i+1
            j=0
            size_l = os.path.getsize(os.path.join(train_repo, image_dir[i]))
            string = init_json(image_dir[i], size)
            cv2.destroyAllWindows()
            break

