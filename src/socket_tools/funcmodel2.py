#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/7/24 17:19
#@Author: hdq
#@File  : funcmodel2.py

import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats

def func_exxaxx(x,a,b,c,d,e,f,g):
    x=np.array(x)
    y = a*np.exp(b*x)+c*x**d+e*x**(f*x)+g
    return y


def func_exxa(x,a,b,c,d,g):
    x=np.array(x)
    y = a*np.exp(b*x)+c*x**d+g
    return y


def func_exxx(x,a,b,e,f,g):
    x=np.array(x)
    y = a*np.exp(b*x)+e*x**(f*x)+g
    return y


def func_xaxx(x,c,d,e,f,g):
    x=np.array(x)
    y = c*x**d+e*x**(f*x)+g
    return y

def func_ex(x,a,b,g):
    x=np.array(x)
    y = a*np.exp(b*x)+g
    return y

def func_xa(x,c,d,g):
    x=np.array(x)
    y = c*x**d+g
    return y

def func_xx(x,e,f,g):
    x=np.array(x)
    y =e*x**(f*x)+g
    return y


import math

#简单周期函数
def func_sinx(x,a,b,c,d,g):
    x = np.array(x)
    y = a*np.sin(b*x*math.pi+c)+d*x+g
    return y

#高斯曲线拟合(正态函数)
def func_gauss(x, a, b, c, sigma):
    x = np.array(x)
    return a*np.exp(-(x-b)**2/(2*sigma**2)) + c

#傅里叶级数(周期函数拟合)
def func_fourier(x,*b):
    x=np.array(x)
    global t
    w = 2 * math.pi/(t*2)*b[-1]
    ret = 0
    a=b[:-1]
    for deg in range(0, int(len(a) / 2) + 1):
        ret += a[deg] * np.cos(deg * w * x) + a[len(a) - deg - 1] * np.sin(deg * w * x)
    return ret


allfunc=["func_exxaxx","func_exxa","func_exxx","func_xaxx","func_ex","func_xa","func_xx","func_sinx","func_gauss","func_fourier"]

def polyfit(x, y,limitfunc=lambda x:x**3):
    fit_coef_list, pcov_list,scorelist=[],[],[]
    maxnh=-1
    id=0
    suit=0
    suiti=0
    import warnings
    warnings.filterwarnings("ignore")
    for i,one in enumerate(allfunc):
        one=eval(one)
        try:
            fit_coef, pcov = curve_fit(one, x, y ,maxfev=15000)
            fit_coef_list.append(fit_coef)
            id += 1
            scorelist.append(0)
            pcov_list.append(pcov)
            result=one([x],*fit_coef)[0]
            finalone=one([len(x)+1],*fit_coef)[0]
            downs=0.5 if abs(finalone-result[-1])>limitfunc(max(result)-min(result)) else 0
            a,b=stats.pearsonr(result, y)
            score=a*(1-b)-downs
            scorelist[len(scorelist) - 1] = score
            # scorelist.append(score)

            if(score>maxnh):
                maxnh=score
                suit=len(fit_coef_list) -1
                suiti=i
        except Exception as e:
            print(e)
            pass
        finally:
            if one == func_fourier:
                try:
                    global t
                    ay=np.array(y)
                    t=abs(np.where(ay==max(y))[0][0]-np.where(ay==min(y))[0][0])
                    fit_coef, pcov = curve_fit(one, x, y,[1.0]*(80 if len(x)-2>80 else len(x)-2), maxfev=15000)
                    scorelist.append(0)
                    fit_coef_list.append(fit_coef)
                    id += 1
                    pcov_list.append(pcov)
                    result = one([x], *fit_coef)[0]
                    finalone = one([len(x) + 1], *fit_coef)[0]
                    downs = 0.5 if abs(finalone - result[-1]) > limitfunc(max(result) - min(result)) else 0
                    a, b = stats.pearsonr(result, y)
                    score = a * (1 - b) - downs
                    scorelist[len(scorelist) - 1] = score
                    # scorelist.append(score)

                    if (score > maxnh):
                        maxnh = score
                        suit = len(fit_coef_list) -1
                        suiti = i
                except Exception as w:
                    print(w)
                    pass
            pass
        # print(one, scorelist)
    warnings.filterwarnings("default")
    # print(allfunc[suiti])
    if(len(scorelist)==0):
        return 0,0,-1
    #参数
    return fit_coef_list[suit],allfunc[suiti],scorelist[suit]

def find_logical(y,limitfunc=lambda x:x**3):
    x_None=[]
    x=[]
    y_temp=[]
    for i,info in enumerate(y):
        if(info!=None):
            x.append(i+1)
            y_temp.append(info)
        else:
            x_None.append(i)
    fit_coef,func,score = polyfit(x, y_temp,limitfunc)
    for one in x_None:
        x.insert(one,None)
    return fit_coef,x,func,score

import matplotlib.pyplot as plt
#展示函数图像
def show_func(fit_coef,func,x_min,x_max,exactvalue=201,info=None,show=True):
    np.set_printoptions(suppress=True)
    x = np.linspace(x_min, x_max, exactvalue)
    y = eval(func)(x,*fit_coef)
    plt.plot(x, y, "r",label="predict", lw=2)
    if info:
        # print(info[-1],info[-2])
        plt.scatter([one for one in info[-1] if one!=None], [one for one in info[-2] if one!=None], c="blue",label="real", marker='o')
    plt.legend()
    if(show):
        plt.show()
    return plt

#数据关联性分析 获得相关系数和显著水平
def get_data_relation(x,y):
    return stats.pearsonr(x,y)

# #函数对应
# y=[4,21,64,145,276]
# x=list(range(1,len(y)+1,1))
# fit_coef,func,score = polyfit(x, y)
# print(func([6],*fit_coef))
#
#
#
# # 数列规律(找下一项是什么)
# fit_coef,x,func,score=find_logical([2 ,5 ,11 ,23 ,None,95],limitfunc=lambda x:x**3)
# print(func([5,len(x)+1],*fit_coef))
