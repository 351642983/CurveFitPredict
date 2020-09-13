# CurveFitPredict
对sklearn中的曲线拟合函数进行了简单整合，提供了方便曲线拟合的简单的统一接口

曲线函数拟合预测
====
主要功能
----
将多项式拟合和一些常见的函数进行了整合，提供了一个方便的统一接口

包含
----
* 1.拟合单变量曲线
* 2.预测曲线
* 3.预测曲线和真实数据对比图
* 4.预测曲线分数

拟合曲线包括
---
* 1.幂函数
* 2.周期函数
* 3.指数函数
* 4.上述函数的组合
* 5.多项式函数
* 6.符合上述函数的数列

其余
---
* 提供单变量拟合的曲线和具体的函数表达式和曲线图对比
* 根据skleran提供的函数拟合多变量函数

使用说明
====
1.导入文件
----
导入myfunctool
````python
from tools import myfunctool
````
2.数列拟合预测
----
下面是一个拟合斐波那契数列的例子,提供了斐波那契数列前19个值，其中第18个值是缺失的，下面是补全第18个值并预测20-22个值的斐波那契值
````python
#---------数列拟合并预测-------
print("---------数列拟合预测-------")
#下面是一个拟合斐波那契数列的例子
#斐波那契数列前19个数，并存在缺失值
x=[1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,None,6765]
#拟合斐波那契数列函数
info=myfunctool.find_logical(x,selectmode=2)#selectmode为1时表示使用多项式函数拟合，selectmode=2表示利用整合的函数拟合，selectmode=3表示自动判断二者拟合效果并给出拟合范围内最好的值，默认为3
print("用此公式求得原数列值:",myfunctool.func_general(info,list(range(1,len(info[-1])+1))),"  拟合得分:",info[-3])
#预测该斐波那契数列的之后的3个值 (真实值为:10946,17711,28657)
print("用此公式计算的后3个数列值:",myfunctool.func_general(info,[info[-1][-1]+1,info[-1][-1]+2,info[-1][-1]+3]))
#预测该斐波那契数列中的缺失值 (真实值为:4181)
print("用此公式预测缺失值:",myfunctool.func_general(info,[info[-1][-1]-1]))
print("预测函数公式为:",myfunctool.get_func(info))
# print("拟合的r2分数：",myfunctool.get_model_r2score(info))#r2暂时只能评判不存在缺失值的函数
#显示斐波那契数列拟合函数和真实值对比，展示预测函数x从1到比原数列预测多一个的值
myfunctool.show_func(info,1,len(info[-1])+1)
````
3.曲线拟合
----
下面是一个拟合任意曲线的例子，以y=4x^2+3x为例，给出当x=0,1,2,3,4时候以及对应的y值，根据给定的值拟合曲线
````python
#---------曲线拟合并预测-------
#随便想一个函数,例如:y=4x^2+3x,并给出任意对应的x和y的值
print("---------曲线拟合预测-------")
x=[0,1,2,3,4]
y=[0,7,22,45,76]
#曲线拟合
info=myfunctool.auto_func(x,y)
print("原值为:",info[-2])
print("用此公式求得原值:",myfunctool.func_general(info,info[-1]),"  拟合曲线函数得分：",info[-3])
print("x=5时，函数的预测值:",myfunctool.func_general(info,[5]))
print("预测拟合曲线函数为:",myfunctool.get_func(info))
print("r2分数：",myfunctool.get_model_r2score(info))
#显示x=0-6时的预测函数和真实值的对比图
myfunctool.show_func(info,0,6)
````
4.变量函数拟合
-----
同sklearn
