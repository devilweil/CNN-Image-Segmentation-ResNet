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

def getLableImageAndImageByMutilBands():
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
    for r in range(lable_rows-jpgwidth):
        for c in range(lable_cols-jpgwidth):
            ###定义分割图像间隔
            if r%10!=0:
                continue
            
            if c%10!=0:
                continue
            
            print('导出第',r,'行',c,'列')
            ###获得标签的256*256的数组
            lableArray=lable_ds.GetRasterBand(1).ReadAsArray(c,r,jpgwidth,jpgwidth)
#                print(lableArray)
            
            count=np.sum(lableArray == 1)
            scale=count/(jpgwidth*jpgwidth)
#                print(count/(256*256))
            
            if scale<0.05 and scale>0:
                continue
            ###获取当前点坐标
            print('导出第',labe_counts,'个标签')
            
            lableArray[lableArray!=1]=0
            lableArray[lableArray!=0]=255
            
#            lable_RGBArray = np.zeros((jpgwidth,jpgwidth,3))
#            lable_RGBArray[:,:,0]=lableArray
            
            lable_x=c*lable_pixelWidth+lable_xOrigin
            lable_y=r*lable_pixelHeight+lable_yOrigin
            ###获取非标签位置
            non_lable_xOffset = int((lable_x-non_lable_xOrigin)/non_lable_pixelWidth)
            non_lable_yOffset = int((lable_y-non_lable_yOrigin)/non_lable_pixelHeight)
                
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
#            non_lableArray=non_lableArray.reshape((512, 512,3))
#            print(non_lableArray.shape)   
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
            
            if scale==0:
#                save_non_grape=r'H:/gansu/wuwei/DataSet/Non_grape/224/'+str(non_grapecunts)+'.png'
#                cv2.imwrite(save_non_grape, non_lableArray)
#                non_grapecunts=non_grapecunts+1
#                continue  #不然后面的语句执行
                if non_grapecunts%50==0:
                    
                    if non_grape_traincountes%5!=0:###train数据集
                        save_non_grape=r'H:/gansu/wuwei/DataSet/Non_grape/train/'+str(non_grape_traincountes)+'.png'
                        cv2.imwrite(save_non_grape, non_lableArray)
                        non_grapecunts=non_grapecunts+1
                        non_grape_traincountes=non_grape_traincountes+1
                    else:####val数据集
                        save_non_grape=r'H:/gansu/wuwei/DataSet/Non_grape/val/'+str(non_grape_traincountes)+'.png'
                        cv2.imwrite(save_non_grape, non_lableArray)
                        non_grapecunts=non_grapecunts+1
                        non_grape_traincountes=non_grape_traincountes+1
                    continue  #不然后面的语句执行
                else:
                    non_grapecunts=non_grapecunts+1
                    continue  #不然后面的语句执行
            
            ###按照1：4进行样本划分
            if grape_traincounts%5!=0:###train数据集
                save_non_lable=r'H:/gansu/wuwei/DataSet/Grape/train/mask/'+str(grape_traincounts)+'.png'
                save_grape_image=r'H:/gansu/wuwei/DataSet/Grape/train/image/'+str(grape_traincounts)+'.png'#str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'.png'#_non_lable
                ##存为tif
                cv2.imwrite(save_non_lable, lableArray)
                cv2.imwrite(save_grape_image, non_lableArray)
                grape_traincounts=grape_traincounts+1
            else:
                save_non_lable=r'H:/gansu/wuwei/DataSet/Grape/val/mask/'+str(grape_traincounts)+'.png'
                save_grape_image=r'H:/gansu/wuwei/DataSet/Grape/val/image/'+str(grape_traincounts)+'.png'#str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'.png'#_non_lable
                ##存为tif
                cv2.imwrite(save_non_lable, lableArray)
                cv2.imwrite(save_grape_image, non_lableArray)
                grape_traincounts=grape_traincounts+1
#            if labe_counts%5!=0:
#                save_lable=r'H:/gansu/wuwei/DataSet/512-512/train/'+str(traincounts)+'_mask.png'#str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'_lable.png'
#                save_non_lable=r'H:/gansu/wuwei/DataSet/512-512/train/'+str(traincounts)+'.png'#str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'.png'#_non_lable
#                ##存为tif
#                cv2.imwrite(save_lable, lableArray)
#                ###保存非标签图像
#            
#                cv2.imwrite(save_non_lable, non_lableArray)
#            
#                traincounts+=1
#            else:
#                
#                save_lable=r'H:/gansu/wuwei/DataSet/512-512/val/'+str(countes)+'_mask.png'#+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'_lable.png'
#                save_non_lable=r'H:/gansu/wuwei/DataSet/512-512/val/'+str(countes)+'.png'#+str(non_lable_yOffset)+'_'+str(non_lable_xOffset)+'_'+vi+'.png'
#                ##存为tif
#                cv2.imwrite(save_lable, lableArray)
#                ###保存非标签图像
#            
#                cv2.imwrite(save_non_lable, non_lableArray)
#                countes+=1
                
            labe_counts+=1
                


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
    
    getLableImageAndImageByMutilBands()
    
    print('complete!')
