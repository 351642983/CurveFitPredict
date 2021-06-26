#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/25 18:55
# @Author: hdq
# @File  : myfunctool.py

# 将会有更精准的拟合效果,但是载入速度更慢

from socket_tools import funcmodel as func1
from socket_tools import funcmodel2  as func2
from socket_tools import basefilehandle as fh
from sklearn import linear_model
from itertools import combinations
import joblib  #保存模型
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
import math

#selectmode:1 func1,2 fucn2,3 func1 or func2
def find_logical(y, inpercent=100, limitfunc=lambda x: x ** 3,selectmode=3):
    fit_coef, alist, x, score=None,None,None,None
    fit_coef2, x2, func, score2=None,None,None,None
    if selectmode==3:
        fit_coef, alist, x, score = func1.auto_find_logical(y, inpercent)
        fit_coef2, x2, func, score2 = func2.find_logical(y, limitfunc)
    elif selectmode==2:
        fit_coef2, x2, func, score2 = func2.find_logical(y, limitfunc)
    elif selectmode==1:
        fit_coef, alist, x, score = func1.auto_find_logical(y, inpercent)

    if (score==None or math.isnan(score)):
        score = -1
    else:
        score = abs(score)
    if (score2==None or math.isnan(score2)):
        score2 = -1
    else:
        score2 = abs(score2)

    if selectmode==1 or (selectmode == 3 and score > score2):
        return 1, list(fit_coef), alist,score,y, list(x)
    else:
        return 2, list(fit_coef2), func,score2,y, list(x2)


def func_general(info, x):
    if(info[-3]==-1):
        return ['无法分析结果']
    if info[0] == 1:
        return func1.func_general(info[1], x)
    else:
        return eval("func2."+info[2])(x, *info[1])


def auto_func(x, y, inpercent=100, limitfunc=lambda x: x ** 3,selectmode=3):
    fit_coef, alist, score = None, None, None
    fit_coef2, func, score2 = None, None, None
    if selectmode == 3:
        fit_coef, alist, score = func1.auto_ax_bfit(x, y, inpercent)
        fit_coef2, func, score2 = func2.polyfit(x, y, limitfunc)
    elif selectmode == 2:
        fit_coef2, func, score2 = func2.polyfit(x, y, limitfunc)
    elif selectmode == 1:
        fit_coef, alist, score = func1.auto_ax_bfit(x, y, inpercent)

    if (score == None or math.isnan(score)):
        score = -1
    else:
        score=abs(score)
    if (score2 == None or math.isnan(score2)):
        score2 = -1
    else:
        score2 = abs(score2)
    if selectmode==1 or (selectmode == 3 and score > score2):
        return 1, list(fit_coef), alist,score, y, list(x)
    else:
        # print(fit_coef2,x)
        return 2, list(fit_coef2), func,score2, y, list(x)


import os
# 展示函数图像
def show_func(info, x_min, x_max, exactvalue=201,savePic_Path=None):
    if(info[-3]==-1):
        return
    show=True
    if savePic_Path:
        show=False
    if info[0] == 1:
        plt=func1.show_func(info[1], x_min, x_max, exactvalue,info,show=show)
    else:
        plt=func2.show_func(info[1], info[2], x_min, x_max, exactvalue,info,show=show)
    if(savePic_Path):
        if( not os.path.exists(fh.get_path_file_subpath(savePic_Path))):
            fh.createDir(fh.get_path_file_subpath(savePic_Path))
        plt.savefig(savePic_Path)
        plt.clf()

# print(fh.get_path_file_subpath(r"H:\pycharm project\YearBooks\tools\myfunctool.py"))


# 得到函数公式
def get_func(info):

    # print(info)
    result = "F(x)="
    if info[0] == 1:
        lennum = len(info[1])
        for i in range(lennum):
            if i == 0:
                result += "("+str(info[1][i])+")"
            else:
                result += "+("+str(info[1][i])+")"
            if (lennum - (i+1)) > 1:
                result += "X^%d" % (lennum - (i+1))
            elif i == (lennum - 2):
                result += "X"
    else:
        try:
            if info[2] == "func_exxaxx":
                result += "(%e)*e^(%e)X+(%e)X^(%e)+(%e)X^(%e)X+(%e)"% tuple(info[1])
            elif info[2] == "func_exxa":
                result += "(%e)*e^(%e)X+(%e)X^(%e)+(%e)" % tuple(info[1])
            elif info[2] == "func_exxx":
                result += "(%e)*e^(%e)X+(%e)X^(%e)X+(%e)"% tuple(info[1])
            elif info[2] == "func_xaxx":
                result += "(%e)X^(%e)+(%e)X^(%e)X+(%e)" % tuple(info[1])
            elif info[2] == "func_ex":
                result += "(%e)*e^(%e)X+(%e)"% tuple(info[1])
            elif info[2] == "func_xa":
                result += "(%e)X^(%e)+(%e)" % tuple(info[1])
            elif info[2] == "func_xx":
                result += "(%e)X^(%e)X+(%e)"% tuple(info[1])
            elif info[2] == "func_sinx":
                result += "(%e)*sin((%e)*x+(%e))+(%e)*x+(%e)" % (info[1][0], info[1][1]*func2.math.pi, info[1][2], info[1][3], info[1][4])
            elif info[2] == "func_gauss":
                result += "(%e)*e^(-(x-(%e))**2/(2*(%e)**2))+(%e)" % tuple(info[1])
            elif info[2] == "func_fourier":
                result += "傅里叶级数参数:"+str(tuple(info[1]))
        except:
            result="分析无任何拟合结果"

    return result

#多参数预测模型(数集小)
def auto_func_list(xlist,ylist):
    model = linear_model.LinearRegression()
    model.fit(xlist, ylist)
    return model

#取列表中最接近的数
def get_near(numlist,nearnum):
    mlist=[]
    for one in numlist:
        mlist.append(abs(one-nearnum))
    minnum=min(mlist)
    for one in numlist:
        if(abs(one-nearnum)==minnum):
            return one
    return 0

#多参数模型预测数据(数集小)
def func_list_predict(model,xlist,inxlist=False,roundnum=False):
    numlist = []
    for i, one in enumerate(model.predict(xlist)):
        result = []
        try:
            for two in one:
                if inxlist:
                    info = get_near(xlist[i], two)
                else:
                    info = two
                if roundnum:
                    info = round(info)
                result.append(info)
            numlist.append(result)
        except:
            numlist.append(one)
    return numlist

#多参数预测模型(数集大)0 LinearRegression, 1 SGDRegressor,2 GradientBoostingRegressor,3 RandomForestRegressor,4 ExtraTreesRegressor
def auto_func_biglist(xlist,ylist,type=4):
    if type==0:
        model=linear_model.LinearRegression()
    elif type==1:
        model=linear_model.SGDRegressor()
    elif type==2:
        model=GradientBoostingRegressor()
    elif type==3:
        model=RandomForestRegressor()
    else:
        model=ExtraTreesRegressor()
    ylen=0
    try:
        model.fit(xlist, ylist)
        modeltype = 1
    except Exception:
        ylen=len(ylist[0])
        newxlist = []
        newylist = []
        for j,one in enumerate(xlist):
            for i in range(ylen):
                newxlist.append(one+[i])
                newylist.append(ylist[j][i])
        model.fit(newxlist,newylist)
        modeltype=2
        print("Exception in auto_func_biglist")

    return model,modeltype,ylen

#多参数模型预测数据(数集大)
def func_biglist_predict(model,xlist,inxlist=False):
    if(model[1]==1):
        numlist = []
        for i, one in enumerate(model[0].predict(xlist)):
            result = []
            try:
                for two in one:
                    if inxlist:
                        info = get_near(xlist[i], two)
                    else:
                        info = two
                    result.append(info)
                numlist.append(result)
            except:
                numlist.append(one)
        return numlist
    elif(model[1]==2):
        ylen=model[2]
        xnew=[]
        for one in xlist:
            for i in range(ylen):
                xnew.append(one+[i])
        temp_predict=model[0].predict(xnew)
        result = []
        subre = []
        for i,info in enumerate(temp_predict):

            if inxlist:
                tinfo = get_near(xlist[i], info)
            else:
                tinfo = info
            subre.append(tinfo)
            if((i+1)%ylen==0):
                result.append(subre)
                subre=[]

        return result

#保存模型
def save_func_list(model,path):
    joblib.dump(model, path)

#加载模型
def load_func_list(path):
    return joblib.load(path)


#数据关联性分析 获得相关系数和显著水平
def get_data_relation(x,y):
    return stats.pearsonr(x,y)

def combine(temp_list,rangelist):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    end_list = []
    for i in rangelist:
        temp_list2 = []
        for c in combinations(temp_list, i):
            temp_list2.append(c)
        end_list.extend(temp_list2)
    return end_list

#关联列分析[[]] []
def relation_result(xlist,y,rangelist):
    best_score,best_id=0,0
    lenlist=list(range(min([len(one) for one in xlist])))
    cb=combine(lenlist,rangelist)
    for i,cols in enumerate(cb):
        temp_x=[]
        for two in xlist:
            info = []
            for one in cols:
                info.append(two[one])
            temp_x.append(info)
        model=auto_func_list(temp_x,y)
        a,b=get_data_relation(model.predict(temp_x),y)
        score=a*(1-b)
        if(score>best_score):
            best_score=score
            best_id=i
    return cb[best_id],best_score

#获得n行数据
def get_relation_result(xlist,relation):
    xlist=np.array(xlist).T
    result=[]
    for one in relation[0]:
        result.append(xlist[one])
    return np.array(result).T

#得到模型r2评判分数
def get_model_r2score(info):
    return r2_score(info[-2],func_general(info,info[-1]))


# #规律寻找 (n^2+1)*2
# info=find_logical([490, 477, 467, 458, 450, 442, 433, 426, 419, 413, 411, 428, 445, 441, 434, 436, 446, 442, 427, 414, 402,
#      391, 381, 372, 366, 363, 363, 364, 366, 372, 382, 397, 414, 430, 444, 460, 481, 502, 522, 539, 551, 561,
#      567, 569, 568, 566, 570, 576, 578, 574, 565, 553, 541, 529, 519, 507, 496, 486, 494, 528, 551, 563, 576,
#      596, 612, 624, 631, 636, 639, 640, 640, 638, 635, 633, 630, 625, 620, 615, 609, 603, 597, 590, 584, 578,
#      571, 559, 541, 529, 524, 511, 486, 454, 422, 394, 372, 348, 340, 335, 334, 332, 332, 332, 332, 332, 333,
#      336, 339, 341, 344, 349, 355, 360, 366, 372, 383, 396, 408, 419, 432, 448, 463, 473, 482, 493, 511, 530,
#      551, 568, 580, 595, 597, 597, 595, 593, 598, 606, 619, 632, 642, 653, 659, 658, 653, 645, 640, 641, 643,
#      650, 656, 659, 659, 655, 649, 640, 632, 626, 621, 614, 603, 590, 575, 564, 550, 530, 519, 507, 495, 484,
#      472, 462, 452, 445, 437, 430, 423, 417, 423, 442, 445, 435, 423, 422, 431, 436, 428, 413, 401, 390, 381,
#      373, 367, 363, 364, 365, 367, 371, 378, 396, 411, 428])
# print("用此公式求得原值:",func_general(info,info[-1]),info[-3])
# print("用此公式计算的后3个值:",func_general(info,[info[-1][-1]+1,info[-1][-1]+2,info[-1][-1]+3]))
# show_func(info,1,len(info[-1])+1)
# print(get_func(info))
# print("r2分数：",get_model_r2score(info))


#  # 图像展示
# info = find_logical([8,18,32,48,65])
# print("用此公式求得原值:",func_general(info,[info[-1]]))
# print("预测下一个值：",func_general(info,[info[-1][-1]+1])) #补全None 和预测下一个值
# show_func(info, 1, 20)    #展示函数图像
# print(get_func(info)) #匹配曲线公式
# print(info)


# #曲线匹配
# x=[1,2,3,4,5]
# y=[0.11,0.21,0.33,0.44,0.54]
# info=auto_func(x,y)
# print("原值为:",info[-2])
# print("用此公式求得原值:",func_general(info,info[-1]),info[-3])
# # print("用此公式计算的后3个值:",func_general(info,[info[-1][-1]+1,info[-1][-1]+2,info[-1][-1]+3]))
# show_func(info,1,len(info[-1])+1)
# print(get_func(info))
# print("r2分数：",get_model_r2score(info))



# # 获得曲线表示函数(selectmode 设置特定函数)
# info = find_logical([1,2,3,7,46])
# print("下一个值",func_general(info,[info[-1][-1]+1]))
# print(get_func(info))


# #多参数预测
# model=auto_func_biglist([[94.76,313747,4234],[94.02,327676,5146],[105.54,405616,6039],[110.01,444361,6538],[106.00,366405,5463],[91.51,366874,6232],[89.02,321645,5773]]
#                         ,[[442.29],[505.31],[605.71],[658.90],[546.62],[594.04],[539.37]],
#                         4)
# print(np.round(func_biglist_predict(model,[[101.71,346026,5536],[112.81,391636,6639],[91.72,406190,6639]])))


# #趋势关联性和显著水平
# print(get_data_relation([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],[1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765]))


# #关联性分析
# result=relation_result([[3,2,1],[2,3,4],[1,3,2],[4,3,6],[9,5,6],[9,3,5]],
#                   np.array([[1,2,3],[2,3,4],[1,2,3],[3,4,6],[5,6,9],[3,5,9]]).T[0],range(1,4,1))
# x=get_relation_result([[3,2,1],[2,3,4],[1,3,2],[4,3,6],[9,5,6],[9,3,5]],result)
# print(result,x)




# #reshape resize
# info=auto_func_biglist(np.array([0,1.7,32,53.7,5.5,55.7,64.8]).reshape((-1,1))
#                         ,[[0,200],[50,201.7],[1159,247.2],[2310,316],[2429,324.7],[2444,325.8],[3131,384]],4)
# print(func_biglist_predict(info,[[0],[55.6]]))




# #数列求解
# from tools import myfunctool as func
#
# def get_func_value(n):
#     an=1
#     for i in range(n):
#         if(i==0):
#             continue
#         else:
#             an=2*i/an
#     return an
#
# #规律寻找
# rangev=7
# info=func.find_logical([get_func_value(i) for i in range(1,rangev+1)],selectmode=2)
# print([get_func_value(i) for i in range(1,rangev+1+3)])
# print("用此公式求得原值:",func.func_general(info,info[-1]),info[-3])
# print("用此公式计算的后3个值:",func.func_general(info,[info[-1][-1]+1,info[-1][-1]+2,info[-1][-1]+3]))
# func.show_func(info,1,rangev)
# print(func.get_func(info))



# info=find_logical([22,35,56,90,None,234],selectmode=2)
# print("求取缺失值:",func_general(info,[5]))
# print("拟合函数:",get_func(info))