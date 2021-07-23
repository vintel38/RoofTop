# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:50:09 2020
Modified on Mon Jul 19 15:16:23 2021

@author: MUSTAFAAKTAS, Vintel38
"""

from osgeo import gdal
import numpy as np
import os
import subprocess

def converter(inputRaster, outputRaster, outputPixType='Byte', outputFormat='png', percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit
    Source: Medium.com, 'Creating Training Datasets for the SpaceNet Road Detection and Routing Challenge' by Adam Van Etten and Jake Shermeyer
    '''
    print(inputRaster)
    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of', 
           outputFormat]
    
    # iterate through bands
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band = srcRaster.GetRasterBand(bandId)

        bmin = band.GetMinimum()        
        bmax = band.GetMaximum()
        # if not exist minimum and maximum values
        if bmin is None or bmax is None:
            (bmin, bmax) = band.ComputeRasterMinMax(1)
        # else, rescale
        band_arr_tmp = band.ReadAsArray()
        bmin = np.percentile(band_arr_tmp.flatten(), 
                             percentiles[0])
        bmax= np.percentile(band_arr_tmp.flatten(), 
                            percentiles[1])

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))
    cmd.append(inputRaster)
    cmd.append(outputRaster)
    print("Conversin command:", cmd)
    subprocess.call(cmd)
    
if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert TIF files to PNG format')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of your dataset')
    parser.add_argument('--output_path', required=True,
                        metavar="/path/to/output/",
                        help='Directory of output')
    args = parser.parse_args()
    
    # path = "D:/DATASET/SpaceNet/Train/AOI_2_Vegas_Train/RGB-PanSharpen/"
    files = os.listdir(args.dataset)

    for file in files:
        resimPath = args.dataset+'/'+file
        # dstPath   = "D:/DATASET/SpaceNet/Train/AOI_2_Vegas_Train/RGB-PanSharpen-NEW/"
        dstPath   = args.output_path+'/'+file
        converter(resimPath,dstPath)