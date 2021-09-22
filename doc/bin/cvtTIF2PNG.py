# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:50:09 2021

@author: Vintel38

based on Programmer Sought webpage 
https://www.programmersought.com/article/48811569657/

Le sens des slashs semble être important dans le bon fonctionnement du système. 

"""


from osgeo import gdal
import os

def converter(file_path, output, extension):
    # file_path="D:/work/python/Tif_to_png/ a_image.tif"
    ds=gdal.Open(file_path)
    print(file_path)
    driver=gdal.GetDriverByName(extension)
    # dst_ds = driver.CreateCopy('D:/work/python/Tif_to_png/example.png', ds)
    dst_ds = driver.CreateCopy(output, ds)
    dst_ds = None
    src_ds = None
    
dir_path = "C:/Users/VArri/Documents/Rooftop/dataset/dataset/dataset/colab/val"
# output_path = "C:/Users/VArri/Documents/Rooftop/dataset/dataset/dataset/colab/austin1.png"
extension = 'PNG'
# converter(file_path, output_path, extension)
    
# if __name__ == '__main__':
    # import argparse
    
    # Parse command line arguments
    # parser = argparse.ArgumentParser(description='Convert TIF files to PNG format')
    # parser.add_argument('--dataset', required=True,
                        # metavar="/path/to/dataset/",
                        # help='Directory of your dataset')
    # parser.add_argument('--output_path', required=True,
                        # metavar="/path/to/output/",
                        # help='Directory of output')
    # parser.add_argument('--extension', 
                        # metavar="type",
                        # help='Extension of the output')
    # args = parser.parse_args()
    
    # path = "D:/DATASET/SpaceNet/Train/AOI_2_Vegas_Train/RGB-PanSharpen/"
    
files = os.listdir(dir_path)
for file in files:
    a, b = os.path.splitext(file) 
    print((a,b))
    if b != '.tif':
        continue
    print('ok')
    resimPath = dir_path + '/' + file
    dstPath   = dir_path+ '/' + a + '.png'
    converter(resimPath,dstPath, extension)