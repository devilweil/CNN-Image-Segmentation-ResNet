# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:15:49 2020
用于预测结果的重新拼接
@author: LW
"""
import imageio
from osgeo import gdal,gdal_array,osr, ogr
import numpy as np
import glob
import os,sys 
from skimage import io
import cv2
from numpy import nan as NaN
import pandas as pd 

lable_path=r'F:\工作文件\论文发表\葡萄主产区优势对比\样本数据\标签数据\标签Raster\grape-lable-new.tif'
file_path=r'F:/工作文件/论文发表/葡萄主产区优势对比/测试数据/20190423_S2A.tif'#+vi+'_2019.tif' 20190423_S2A.tif
non_lable_ds_ref_pngpath=r'F:/工作文件/论文发表/葡萄主产区优势对比/测试数据/20190423_S2A.jpg'

def combineImage(imgList):
    ####获取非标签原始影像的属性信息
    non_lable_ds=gdal.Open(file_path)
    ###获取放射变换信息
    non_lable_transform = non_lable_ds.GetGeoTransform()
#    non_lable_xOrigin = non_lable_transform[0]
#    non_lable_yOrigin = non_lable_transform[3]
    non_lable_pixelWidth = non_lable_transform[1]
    non_lable_pixelHeight = non_lable_transform[5]
    non_lable_cols=non_lable_ds.RasterXSize
    non_lable_rows=non_lable_ds.RasterYSize
    outimg=np.zeros((non_lable_rows,non_lable_cols))
    outimg[outimg==0]=NaN
    
#    outimg[0:2,0:1]=1
    del non_lable_ds
    jpgwidth=64   ###224
    
    ###循环赋值
    for img in imgList:
        r,c,predType=img[0].split('-')[0],img[0].split('-')[1],img[1]
        print('r,c,predType are ',r,c,predType)
        non_lable_xOffset = int(c)*jpgwidth
        non_lable_yOffset = int(r)*jpgwidth
        outimg[non_lable_yOffset:(non_lable_yOffset+jpgwidth),non_lable_xOffset:(non_lable_xOffset+jpgwidth)]=int(predType)  
    
    #导出到TIF中
    savePath=r'H:\gansu\wuwei\预测结果\20190423-predict.tif'
    write_imgArray(savePath,non_lable_cols,non_lable_rows,non_lable_transform,outimg)

        
def write_imgArray(filename,im_width,im_height,im_geotrans,im_data):
    # 生成影像
    dataset = gdal.GetDriverByName('GTiff').Create(filename, xsize=im_width, ysize=im_height, bands=1,
                                                     eType=gdal.GDT_Float32)#gdal.GDT_Float32   GDT_CInt16

    proj = osr.SpatialReference()
    proj.SetWellKnownGeogCS("WGS84"); 
    dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
    dataset.SetProjection(proj.ExportToWkt())         #写入投影
    dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
   
    del dataset       

if __name__ == "__main__": 
    
    ###定义工作空间
#    os.chdir(r'H:\gansu\wuwei\S2-moasic\Clip')
    ###预测结果CSV结果
    predCSVPath=r'H:\gansu\wuwei\预测结果\predict-20190423.csv'
    predCSV = pd.read_csv(predCSVPath, names=['PngName', 'PredType']) 
    print(len(predCSV))
    
    combineImage(predCSV.values)
    
    print('complete!')


