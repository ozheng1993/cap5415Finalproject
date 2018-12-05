# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

bbox=pd.read_csv('BBox.csv')



# =============================================================================
# def IOU(Reframe,GTframe):
#     """
#     自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
#     """
#     x1 = Reframe[0]
#     y1 = Reframe[1]
#     width1 = Reframe[2]-Reframe[0]
#     height1 = Reframe[3]-Reframe[1]
# 
#     x2 = GTframe[0]
#     y2 = GTframe[1]
#     width2 = GTframe[2]-GTframe[0]
#     height2 = GTframe[3]-GTframe[1]
# 
#     endx = max(x1+width1,x2+width2)
#     startx = min(x1,x2)
#     width = width1+width2-(endx-startx)
# 
#     endy = max(y1+height1,y2+height2)
#     starty = min(y1,y2)
#     height = height1+height2-(endy-starty)
# 
#     if width <=0 or height <= 0:
#         ratio = 0 # 重叠率为 0 
#     else:
#         Area = width*height # 两矩形相交面积
#         Area1 = width1*height1
#         Area2 = width2*height2
#         ratio = Area*1./(Area1+Area2-Area)
#     # return IOU
#     return ratio,Reframe,GTframe
# =============================================================================





def IOU(data):
    """
    
    """
    x_true = data[1]
    y_true = data[2]
    width_true = data[3]
    height_true = data[4]

    x_goturn = data[5]
    y_goturn = data[6]
    width_goturn = data[7]
    height_goturn = data[8]
    

    x_new=data[9]
    y_new=data[10]
    width_new=data[11]
    height_new = data[12]


    endx_goturn = max(x_true+width_true,x_goturn+width_goturn)
    startx_goturn = min(x_true,x_goturn)
    width_goturn = width_true+width_goturn-(endx_goturn-startx_goturn)

    endy_goturn = max(y_true+height_true,y_goturn+height_goturn)
    starty_goturn = min(y_true,y_goturn)
    height_goturn = height_true+height_goturn-(endy_goturn-starty_goturn)
    
    
    endx_new = max(x_true+width_true,x_new+width_new)
    startx_new = min(x_true,x_new)
    width_new = width_true+width_new-(endx_new-startx_new)

    endy_new = max(y_true+height_true,y_new+height_new)
    starty_new = min(y_true,y_new)
    height_new = height_true+height_new-(endy_new-starty_new)
         

    if width_goturn <=0 or height_goturn <= 0:
        ratio = 0 # 重叠率为 0 
    else:
        Area = width_goturn*height_goturn # 两矩形相交面积
        Area_true = width_true*height_true
        Area_goturn = width_goturn*height_goturn
        ratio = Area*1./(Area_true+Area_goturn-Area)
        
        
    if width_new <=0 or height_new <= 0:
        ratio1 = 0 # 重叠率为 0 
    else:
        Area1 = width_new*height_new # 两矩形相交面积
        Area_true = width_true*height_true
        Area_new = width_new*height_new
        ratio1 = Area1*1./(Area_true+Area_new-Area1)        
    # return IOU
    return ratio,ratio1


ratio_goturn=[]
ratio_new=[]
for i in range(0,len(bbox)):
    data=bbox.iloc[i]
    ratio0,ratio1=IOU(data)
    ratio_goturn.append(ratio0)
    ratio_new.append(ratio1)   

print(np.mean(ratio_goturn))
print(np.mean(ratio_new))
