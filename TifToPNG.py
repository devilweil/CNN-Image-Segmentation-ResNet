# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 23:01:26 2020

@author: LW
"""
import imageio
from osgeo import gdal,gdal_array,osr, ogr
import numpy as np
import glob
import os,sys 
from skimage import io
import cv2


vi='NDVI_MAX'
lable_path=r'F:\工作文件\论文发表\葡萄主产区优势对比\样本数据\标签数据\标签Raster\grape-lable.tif'
file_path=r'F:/工作文件/论文发表/葡萄主产区优势对比/测试数据/'+vi+'_2019.tif'

def getLableImageAndImage():
#    lable=[]
#    image=[]
    
    ####获取非标签原始影像的属性信息
    non_lable_ds=gdal.Open(file_path)
    ###获取放射变换信息
    non_lable_transform = non_lable_ds.GetGeoTransform()
    non_lable_xOrigin = non_lable_transform[0]
    non_lable_yOrigin = non_lable_transform[3]
    non_lable_pixelWidth = non_lable_transform[1]
    non_lable_pixelHeight = non_lable_transform[5]
#    non_lable_cols=non_lable_ds.RasterXSize
#    non_lable_rows=non_lable_ds.RasterYSize
    non_lable_ds_ref=io.imread(file_path)
    non_lable_Max=non_lable_ds_ref.max()
    non_lable_Min=non_lable_ds_ref.min()
    del non_lable_ds_ref
    ####获取标签影像的属性信息
    
    lable_ds=gdal.Open(lable_path)
    ###获取放射变换信息
    lable_transform = lable_ds.GetGeoTransform()
    lable_xOrigin = lable_transform[0]
    lable_yOrigin = lable_transform[3]
    lable_pixelWidth = lable_transform[1]
    lable_pixelHeight = lable_transform[5]
    lable_cols=lable_ds.RasterXSize
    lable_rows=lable_ds.RasterYSize
    
#    lableimgs=imageio.imread(lable_path)
    
    
    
    ###循环标签TIF的行列进行寻找256*256的标签PNG
    labe_counts=0
    for r in range(lable_rows-255):
        for c in range(lable_cols-255):
            ###定义分割图像间隔
            if r%25!=0:
                continue
            
            if c%25!=0:
                continue
            
            print('导出第',r,'行',c,'列')
            ###获得标签的256*256的数组
            lableArray=lable_ds.GetRasterBand(1).ReadAsArray(c,r,256,256)
#                print(lableArray)
            
            count=np.sum(lableArray == 1)
            scale=count/(256*256)
#                print(count/(256*256))
            if scale<0.05:
                continue
            ###获取当前点坐标
            print('导出第',labe_counts,'个标签')
            
            lableArray[lableArray!=1]=0
            lable_x=c*lable_pixelWidth+lable_xOrigin
            lable_y=r*lable_pixelHeight+lable_yOrigin
            ###获取非标签位置
            non_lable_xOffset = int((lable_x-non_lable_xOrigin)/non_lable_pixelWidth)
            non_lable_yOffset = int((lable_y-non_lable_yOrigin)/non_lable_pixelHeight)
                
            non_lableArray=non_lable_ds.GetRasterBand(1).ReadAsArray(non_lable_xOffset,non_lable_yOffset,256,256)
            non_lableArray = (non_lableArray-non_lable_Min)*255/(non_lable_Max-non_lable_Min) # (矩阵元素-最小值)/(最大值-最小值)    
            ###保存tif
#            transform=(lable_x,non_lable_transform[1],non_lable_transform[2],lable_y,non_lable_transform[4],non_lable_transform[5])
#            ###保存标签图像
#            save_lable=r'H:/gansu/wuwei/DataSet/lable1/'+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'lable.tif'
#            ##存为tif
#            write_imgArray(save_lable,256,256,transform,lableArray)
#            ###保存非标签图像
#            save_non_lable=r'H:/gansu/wuwei/DataSet/lable1/'+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'_non_lable.tif'
#            ##存为tif
#            write_imgArray(save_non_lable,256,256,transform,non_lableArray)

            ###存为PNG格式
            ###保存标签图像
            save_lable=r'H:/gansu/wuwei/DataSet/lable/'+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'_lable.png'
            save_non_lable=r'H:/gansu/wuwei/DataSet/image/'+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'.png'#_non_lable
            ###按照1：4进行样本划分，
            if labe_counts%5==0:
                save_lable=r'H:/gansu/wuwei/DataSet/test_lable/'+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'_lable.png'
                save_non_lable=r'H:/gansu/wuwei/DataSet/test_image/'+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'.png'
            
            ##存为tif
            cv2.imwrite(save_lable, lableArray)
            ###保存非标签图像
            
            cv2.imwrite(save_non_lable, non_lableArray)
            
            labe_counts+=1
#            if lableimgs[r][c]==1:
#                print('导出第',labe_counts,'个标签')
#                ###获得标签的256*256的数组
#                lableArray=lable_ds.GetRasterBand(1).ReadAsArray(c,r,256,256)
##                print(lableArray)
#                lableArray[lableArray!=1]=0
#                count=np.sum(lableArray == 1)
#                scale=count/(256*256)
##                print(count/(256*256))
#                if scale<0.05:
#                    continue
#                ###获取当前点坐标
#                lable_x=c*lable_pixelWidth+lable_xOrigin
#                lable_y=r*lable_pixelHeight+lable_yOrigin
#                ###获取非标签位置
#                non_lable_xOffset = int((lable_x-non_lable_xOrigin)/non_lable_pixelWidth)
#                non_lable_yOffset = int((lable_y-non_lable_yOrigin)/non_lable_pixelHeight)
#                
#                non_lableArray=non_lable_ds.GetRasterBand(1).ReadAsArray(non_lable_xOffset,non_lable_yOffset,256,256)
#                
#                ###保存tif
#                transform=(lable_x,non_lable_transform[1],non_lable_transform[2],lable_y,non_lable_transform[4],non_lable_transform[5])
#                
#                
#                ###保存标签图像
#                save_lable=r'H:/gansu/wuwei/DataSet/lable/'+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'lable.tif'
#                write_imgArray(save_lable,256,256,transform,lableArray)
#                ###保存非标签图像
#                save_non_lable=r'H:/gansu/wuwei/DataSet/lable/'+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'_non_lable.tif'
#                write_imgArray(save_non_lable,256,256,transform,non_lableArray)
#                
#                labe_counts+=1
    
    
#    return lable,image

def write_imgArray(filename,im_width,im_height,im_geotrans,im_data):
    # 生成影像
    dataset = gdal.GetDriverByName('GTiff').Create(filename, xsize=im_width, ysize=im_height, bands=1,
                                                     eType=gdal.GDT_CInt16)#gdal.GDT_Float32

    proj = osr.SpatialReference()
    proj.SetWellKnownGeogCS("WGS84"); 
    dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
    dataset.SetProjection(proj.ExportToWkt())         #写入投影
    dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
   
    del dataset

if __name__ == "__main__": 
    
    ###定义工作空间
    os.chdir(r'H:\gansu\wuwei\DataSet')
    
    getLableImageAndImage()
    
    print('complete!')
