# script to post process the data emitted from MaskRCNN for the detections
# takes as input the r variables stored in elt file 

# import 
import os 
import numpy as np
from osgeo import gdal,ogr,osr


def GetExtent(ds):
    """ Return list of corner coordinates from a gdal Dataset """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    return xmin, xmax, ymin, ymax, xpixel, ypixel
    
    
def pointing(coord, img_coord, res):
    """ajoute aux img_coord métriques, la distance de chaque détection coord en nombre de pixels"""
    xmin, xmax, ymin, ymax = img_coord
    x_coord, y_coord = coord
    x_c = xmin + res[0]*x_coord
    y_c = ymax - res[1]*y_coord
    return x_c, y_c 
    

def ReprojectCoords(coords,src_srs,tgt_srs):
    """ Reproject a list of x,y coordinates. """
    # https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords


def post_process(img_path, elt):
    """chargement et extraction des variables bbox masks class et scores du fichier elt """
    assert elt.endswith('.npy')
    data = np.load(elt, allow_pickle=True)
    
    bbox=np.array(dict(data.item(0))['rois'])
    masks=np.array(dict(data.item(0))['masks'])
    classes=np.array(dict(data.item(0))['class_ids'])
    scores=np.array(dict(data.item(0))['scores'])
    
    raster = gdal.Open(img_path)
    ext = GetExtent(raster)
    
    src_srs=osr.SpatialReference()
    src_srs.ImportFromWkt(raster.GetProjection())
    #tgt_srs=osr.SpatialReference()
    #tgt_srs.ImportFromEPSG(4326)
    tgt_srs = src_srs.CloneGeogCS()
    
    pt=[]
    for i in range(masks.shape[-1]):
        y_center, x_center = np.argwhere(masks[:,:,i]==1).sum(0)/(masks[:,:,i] == 1).sum()
        point = pointing((x_center, y_center), ext[0:4], ext[4:6])
        print(point)
        pt.append(point)
        
    reproj = ReprojectCoords(pt, src_srs, tgt_srs)
    return reproj