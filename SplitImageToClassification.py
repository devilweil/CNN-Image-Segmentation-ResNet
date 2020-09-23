# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:25:41 2020

@author: LW
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:02:32 2020

@author: LW
"""

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
from numpy import nan as NaN

vi='NDVI_MAX'
lable_path=r'F:\工作文件\论文发表\葡萄主产区优势对比\样本数据\标签数据\标签Raster\grape-lable-new.tif'
file_path=r'F:/工作文件/论文发表/葡萄主产区优势对比/测试数据/20190423_S2A.tif'#+vi+'_2019.tif'
non_lable_ds_ref_pngpath=r'F:/工作文件/论文发表/葡萄主产区优势对比/测试数据/20190423_S2A.jpg'

def getSplitImageAndImageByMutilBands():
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
    non_lable_cols=non_lable_ds.RasterXSize
    non_lable_rows=non_lable_ds.RasterYSize
    
    non_lable_ds_ref=io.imread(file_path)
    print(non_lable_ds_ref.shape)
#    non_lable_ds_ref_B234=np.array(non_lable_ds_ref[:,:,0:3],dtype=int)
#    imageio.imwrite(non_lable_ds_ref_pngpath, non_lable_ds_ref_B234)
#    cv2.imwrite('20190423_S2A.jpg', non_lable_ds_ref_B234)

    non_lable_ds_ref_B2=np.array(non_lable_ds_ref[:,:,0],dtype=float)
    non_lable_ds_ref_B2[non_lable_ds_ref_B2==65536]=NaN
#    non_lable_ds_ref_B2[non_lable_ds_ref_B2==1]=NaN
    non_lable_Max_B2=non_lable_ds_ref_B2[~np.isnan(non_lable_ds_ref_B2)].max()
    non_lable_Min_B2=non_lable_ds_ref_B2[~np.isnan(non_lable_ds_ref_B2)].min()
    
    non_lable_ds_ref_B3=np.array(non_lable_ds_ref[:,:,1],dtype=float)
    non_lable_ds_ref_B3[non_lable_ds_ref_B3==65536]=NaN
    non_lable_Max_B3=non_lable_ds_ref_B3[~np.isnan(non_lable_ds_ref_B3)].max()
    non_lable_Min_B3=non_lable_ds_ref_B3[~np.isnan(non_lable_ds_ref_B3)].min()
    
    non_lable_ds_ref_B4=np.array(non_lable_ds_ref[:,:,2],dtype=float)
    non_lable_ds_ref_B4[non_lable_ds_ref_B4==65536]=NaN
    non_lable_Max_B4=non_lable_ds_ref_B4[~np.isnan(non_lable_ds_ref_B4)].max()
    non_lable_Min_B4=non_lable_ds_ref_B4[~np.isnan(non_lable_ds_ref_B4)].min()
    
    non_lable_ds_ref_B8=np.array(non_lable_ds_ref[:,:,3],dtype=float)
    non_lable_ds_ref_B8[non_lable_ds_ref_B8==65536]=NaN
    non_lable_Max_B8=non_lable_ds_ref_B8[~np.isnan(non_lable_ds_ref_B8)].max()
    non_lable_Min_B8=non_lable_ds_ref_B8[~np.isnan(non_lable_ds_ref_B8)].min()
    
    
    del non_lable_ds_ref, non_lable_ds_ref_B2,non_lable_ds_ref_B3,non_lable_ds_ref_B4
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
    non_grape_traincountes=0
    grape_traincounts=0
    non_grapecunts=0  #对非葡萄地块的相片进行计数，10选1
    jpgwidth=224
    
    
    rowImageCount=int(non_lable_rows/jpgwidth)
    colImageCount=int(non_lable_cols/jpgwidth)
    print('行照片数',rowImageCount,'列照片数',colImageCount)
    
    for r in range(rowImageCount):
        for c in range(colImageCount):
            print('导出第',r,'行',c,'列')
            ###获取非标签位置
            non_lable_xOffset = c*jpgwidth
            non_lable_yOffset = r*jpgwidth
            
            
            non_lableArray_B2=non_lable_ds.GetRasterBand(1).ReadAsArray(non_lable_xOffset,non_lable_yOffset,jpgwidth,jpgwidth)
            non_lableArray_B2 = (non_lableArray_B2-non_lable_Min_B2)*255/(non_lable_Max_B2-non_lable_Min_B2) # (矩阵元素-最小值)/(最大值-最小值) 
            non_lableArray_B2[non_lableArray_B2>255]=255
            
            non_lableArray_B3=non_lable_ds.GetRasterBand(2).ReadAsArray(non_lable_xOffset,non_lable_yOffset,jpgwidth,jpgwidth)
            non_lableArray_B3 = (non_lableArray_B3-non_lable_Min_B3)*255/(non_lable_Max_B3-non_lable_Min_B3) # (矩阵元素-最小值)/(最大值-最小值) 
            non_lableArray_B3[non_lableArray_B3>255]=255
            
            non_lableArray_B4=non_lable_ds.GetRasterBand(3).ReadAsArray(non_lable_xOffset,non_lable_yOffset,jpgwidth,jpgwidth)
            non_lableArray_B4 = (non_lableArray_B4-non_lable_Min_B4)*255/(non_lable_Max_B4-non_lable_Min_B4) # (矩阵元素-最小值)/(最大值-最小值) 
            non_lableArray_B4[non_lableArray_B4>255]=255
            
            non_lableArray_B8=non_lable_ds.GetRasterBand(3).ReadAsArray(non_lable_xOffset,non_lable_yOffset,jpgwidth,jpgwidth)
            non_lableArray_B8 = (non_lableArray_B8-non_lable_Min_B8)*255/(non_lable_Max_B8-non_lable_Min_B8) # (矩阵元素-最小值)/(最大值-最小值) 
            non_lableArray_B8[non_lableArray_B8>255]=255
#            non_lableArray_B8=non_lable_ds.GetRasterBand(4).ReadAsArray(non_lable_xOffset,non_lable_yOffset,512,512)
#            non_lableArray = (non_lableArray-non_lable_Min)*255/(non_lable_Max-non_lable_Min) # (矩阵元素-最小值)/(最大值-最小值) 
#            
            non_lableArray = np.zeros((jpgwidth,jpgwidth,3))
#            ####导入R，G，B三个波段
#            non_lableArray[:,:,0]=non_lableArray_B4
#            non_lableArray[:,:,1]=non_lableArray_B3
#            non_lableArray[:,:,2]=non_lableArray_B2
            
#            ####导入NIR,R，G三个波段
#            non_lableArray[:,:,0]=non_lableArray_B8
#            non_lableArray[:,:,1]=non_lableArray_B4
#            non_lableArray[:,:,2]=non_lableArray_B3
            
            ####导入NIR，G,B三个波段
            non_lableArray[:,:,0]=non_lableArray_B4
            non_lableArray[:,:,1]=non_lableArray_B3
            non_lableArray[:,:,2]=non_lableArray_B2
            
            non_lableArray =np.array(non_lableArray,dtype=int)
            print(non_lableArray.shape)
            
            ###获得标签的256*256的数组
            lable_x=non_lable_xOffset*non_lable_pixelWidth+non_lable_xOrigin
            lable_y=non_lable_yOffset*non_lable_pixelHeight+non_lable_yOrigin
            
            lable_xOffset = int((lable_x-lable_xOrigin)/lable_pixelWidth)
            lable_yOffset = int((lable_y-lable_yOrigin)/lable_pixelHeight)
            
            lableArray=lable_ds.GetRasterBand(1).ReadAsArray(lable_xOffset,lable_yOffset,jpgwidth,jpgwidth)
#                print(lableArray)
            
            count=np.sum(lableArray == 1)
            scale=count/(jpgwidth*jpgwidth)
            
            if scale>0:
                save_non_grape=r'H:/gansu\wuwei/DataSet/WaitClassificationImage/'+str(r)+'-'+str(c)+'-'+'1'+'.png'
                cv2.imwrite(save_non_grape, non_lableArray)
                save_label_grape=r'H:/gansu\wuwei/DataSet/WaitClassificationImage/mask/'+str(r)+'-'+str(c)+'-'+'1'+'-mask.png'
                cv2.imwrite(save_label_grape, lableArray)
            else:
                save_non_grape=r'H:/gansu\wuwei/DataSet/WaitClassificationImage/'+str(r)+'-'+str(c)+'-'+'0'+'.png'
                cv2.imwrite(save_non_grape, non_lableArray)
    
                




if __name__ == "__main__": 
    
    ###定义工作空间
    os.chdir(r'H:\gansu\wuwei\DataSet')
    
    getSplitImageAndImageByMutilBands()
    
    print('complete!')

