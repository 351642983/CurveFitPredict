#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/7/22 22:14
#@Author: hdq
#@File  : funcmodel.py

import numpy as np
import scipy.stats as stats

#自动型最大幂次,减小计算量
maxn=20

#获得ndarray的从小到大的百分之多少的数
def get_npnum(x,inprecent):
    sortlist=np.sort(x)
    mnum=np.percentile(sortlist,inprecent)
    if len(np.where(x==mnum)[0])!=0:
        return mnum
    else:
        return sortlist[np.max(np.where(sortlist<=mnum))]

#自动多项式函数拟合(计算量相对较大) pointer:get_npnum
def auto_ax_bfit(x,y,inpercent=100):
    x=np.array(x)
    y=np.array(y)
    n=maxn if len(x)-1>maxn else len(x)-1
    alist=[]
    g_start=True
    suit=1
    minsuit=0
    import warnings
    warnings.filterwarnings("ignore")
    #仅仅适用递增数列
    #np.percentile(a,50) #中位数就是50%处的数字，也可以获得0%、100%处的数字
    pi=np.where(x==get_npnum(x,inpercent))[0][0]
    new_x = np.delete(x,pi)
    new_y = np.delete(y,pi)
    for i in range(1, n, 1):
        fit_coef, blist = ax_bfit(new_x, new_y, i)
        temp_minus = abs(func_general(fit_coef, [x[pi]])[0] - y[pi])
        if (g_start or temp_minus < minsuit):
            minsuit = temp_minus
            suit = i
            if (g_start):
                g_start = False
    warnings.filterwarnings("default")
    fit_coef, blist = ax_bfit(x, y, suit)
    a, b = stats.pearsonr(func_general(fit_coef, x), y)
    score = a * (1 - b)
    print("自适应最高幂次:",suit)
    minnum=1e-12
    for one in fit_coef:
        if(not g_start):
            if(abs(one)>minnum):
                g_start=True
                alist.append(one)
        else:
            alist.append(one)
    alist=[round(one,9) for one in list(alist)]
    return fit_coef,alist,score

#靠猜最高幂次多项式拟合，n是最高幂次，这个需要靠猜n的大小，n越大计算量也就越大，同时也越准确（泰勒展开公式，最高幂次越大越逼近准确函数），同时n需要小于len(x),当太大时直接取正整数型，想方便时也直接用自动型就好了
#但如果一个函数中包含幂函数，则需要将最高幂次调整至对应的范围使其符合幂函数的增长趋势才能让拟合函数在一定区间内符合函数增长，这点自动型做不到
def ax_bfit(x,y,n):
    x=np.array(x)
    y=np.array(y)
    fit_coef=np.polyfit(x,y,n)
    alist=[round(one,9) for one in list(np.polyfit(x,y,n))]
    return fit_coef,alist

#函数构建器
def func_general(fit_coef,predict_xlist):
    calculate_y = [np.round(one, 9).tolist() for one in list(np.polyval(fit_coef, predict_xlist))]
    return calculate_y



#数据关联性分析 获得相关系数和显著水平
def get_data_relation(x,y):
    return stats.pearsonr(x,y)



#数列规律寻找(数列中包含的None表示该位置没有数值标志，数列规律受数列顺序影响)
def auto_find_logical(y,inpercent=100):
    x=[]
    x_None=[]
    y_temp=[]
    for i,info in enumerate(y):
        if(info!=None):
            x.append(i+1)
            y_temp.append(info)
        else:
            x_None.append(i)
    suit=1
    minsuit=0
    g_start=True
    import warnings
    warnings.filterwarnings("ignore")
    pi = np.where(x == get_npnum(x, inpercent))[0][0]
    x_new=np.delete(x,pi)
    y_new=np.delete(y_temp,pi)
    for i in range(1,maxn,1):
        fit_coef, alist =ax_bfit(x_new, y_new, i)
        temp_minus=abs(func_general(fit_coef, [x[pi]])[0]-y_temp[pi])
        if(g_start or temp_minus<minsuit):
            minsuit=temp_minus
            suit=i
            if(g_start):
                g_start=False
        else:
            break
    warnings.filterwarnings("default")
    fit_coef, alist = ax_bfit(x, y_temp, suit)
    a, b = stats.pearsonr(func_general(fit_coef,x), y_temp)
    score = a * (1 - b)
    print("自适应最高幂次:",suit)
    for one in x_None:
        x.insert(one,None)
    return fit_coef,alist,x,score

#靠猜最高幂次型数列规律寻找(数列中包含的None表示该位置没有数值标志，数列规律受数列顺序影响)
def find_logical(y,n):
    x=[]
    x_None=[]
    y_temp=[]
    for i,info in enumerate(y):
        if(info!=None):
            x.append(i+1)
            y_temp.append(info)
        else:
            x_None.append(i)
    fit_coef,alist=ax_bfit(x,y_temp,n)
    for one in x_None:
        x.insert(one,None)
    return fit_coef,alist,x


import matplotlib.pyplot as plt
#展示函数图像
def show_func(fit_coef,x_min,x_max,exactvalue=201,info=None,show=True):
    np.set_printoptions(suppress=True)
    x = np.linspace(x_min, x_max, exactvalue)
    y = func_general(fit_coef, x)
    plt.plot(x, y, "r", lw=4, label="predict")
    if info:
        plt.plot([one for one in info[-1] if one!=None], [one for one in info[-2] if one!=None], "lightgreen", label="real", lw=2)
    plt.legend()
    if(show):
        plt.show()
    return plt

#------------函数匹配
# #数据列表
# x=[1,2,3,4,5]
# y=[4,21,64,145,276]
# #需要预测的x值
# calculate_x=[1,3,6]
# #期待值[4,64,469]
# # #拟合函数最高次幂(这部分可以靠猜，当为3时是准确值,计算量相对较小型)
# # n=3
# # fit_coef,alist=ax_bfit(x,y,n)
#
# #自动型，计算量相对较大
# fit_coef,alist,score=auto_ax_bfit(x,y)
#
# print("系数列表:",alist)
# print("calculate_x列表对应的预测值:",func_general(fit_coef,calculate_x))



# #数据列表
# x=[2,2.11,3.24,3,4,5,2.1,4.2,3.2,1.3,1.36,1.6,2.7,5.7,3.6]
# y=[np.exp(one) for one in x]
#
# #需要预测的x值
# calculate_x=[1]
# #期待值[2.7182818284590452353602874713526624977572]
# # #拟合函数最高次幂(这部分可以靠猜，当为3时是准确值,计算量相对较小型)
# # n=3
# # fit_coef,alist=ax_bfit(x,y,n)
#
# #自动型，计算量相对较大
# fit_coef,alist,score=auto_ax_bfit(x,y)
#
# print("系数列表:",alist)
# print("calculate_x列表对应的预测值:",func_general(fit_coef,calculate_x))



#------------数列规律寻找
# fit_coef,alist,x,score=auto_find_logical([1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765])
# print(func_general(fit_coef,[x[-1]+1,x[-1]+2]))
#
# fit_coef,alist,x,score=auto_find_logical([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99])
# print(func_general(fit_coef,[x[-1]+1]))


#------------相关系数分析
# x=[1,2,3,4,5]
# y=[4,21,64,145,276]
# print(get_data_relation(x,y))
