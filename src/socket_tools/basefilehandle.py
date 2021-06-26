#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/5/26 20:45
#@Author: hdq
#@File  : basefilehandle.py

# 添加内容到文件里面
import os
import shutil

# 复制文件
def copyfile(file,to):
    shutil.copyfile(file,to)

# 复制文件夹及其文件
def copy(postion,to):
    shutil.copy(postion, to)


# 创建文件夹
def createDir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        return True
    return False

#添加文件内容到文件前面
def addheadinfo(filepath,info):
    with open(filepath, "r+") as f:
        old = f.read()
        f.seek(0)
        f.write(info)
        f.write(old)

#替换文件列表
def replace_pointfile(tofilelist,replacelist):
    for i,one in enumerate(tofilelist):
        remove_file(one)
        copyfile(replacelist[i],one)

# 获得文件内容以行数呈递
def getfilelines(filepath, encoding="utf-8",replaceEnd=True):
    f = open(filepath, "r", encoding=encoding)
    str = f.readlines()
    f.close()
    if(replaceEnd):
        str=[one.replace("\n","") for one in str]
    return str


# 获得文件的所有内容，返回字符串
def getfileinfos(filepath, encoding="utf-8"):
    f = open(filepath, "r", encoding=encoding)
    str = f.read()
    f.close()
    return str

def appendfile(filepath, info, startchar="", encoding="utf-8"):
    with open(filepath, 'a', encoding=encoding) as file_obj:
        file_obj.write(startchar)
        file_obj.write(info)


# 输出文件内容
def outfile(filepath, info, encoding="utf-8"):
    with open(filepath, 'w', encoding=encoding) as file_obj:
        file_obj.write(info)


# 获得文件夹所有文件包含子目录文件
def get_all_files(dir):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_

#获得所有文件夹包括dir
def get_all_dirs(dir):
    dirs = []
    for dirpath, dirnames, filenames in os.walk(dir):
        dirs.append(dirpath)
    return dirs

#获得所有文件
def get_files_in_dir(dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for one in ([dirpath+"/"+one for one in filenames]):
            files.append(one)
    return files

#获得当前目录下的类型
def get_dir_infos(dir,type=2):
    result=[]
    for sub in os.walk(dir):
        for i in sub[type]:
            result.append(dir+"/"+i)
    return result

#删除文件夹以及下面所有文件
def remove_point_dirs(dir):
    shutil.rmtree(dir)

#删除文件夹目录文件
def remove_in_dirs(dir):
    filelists=get_dir_infos(dir)
    dirslist = get_dir_infos(dir,1)
    print(dirslist)
    for one in dirslist:
        remove_point_dirs(one)
    for one in filelists:
        remove_file(one)

#删除文件列表
def remove_files(lists):
    for one in lists:
        remove_files(one)

#删除单个文件
def remove_file(path):
    if(os.path.exists(path)):
        os.remove(path)
        return True
    return False

# 获得以endtype结尾的全部文件
def get_files_by_types(dir, endtype,type=0):
    if(type==1):
        return [one for one in get_all_files(dir) if one.endswith(endtype)]
    else:
        return [one for one in get_dir_infos(dir) if one.endswith(endtype)]

# 获得文件的字节
def get_file_size(path):
    return os.path.getsize(path)

# 获得路径完整文件名
def get_path_file_completebasename(path):
    return os.path.split(path)[1]


# 获得文件名中的路径
def get_path_file_subpath(path):
    return os.path.split(path)[0]


# 获得路径的文件名
def get_path_file_basename(path):
    return os.path.basename(path).replace(os.path.splitext(path)[1], "")


# 获得路径的后缀名
def get_path_file_append(path):
    return os.path.splitext(path)[1]

#获得文件的前一个目录名
def get_file_pre_dir(filename):
    return os.path.basename(get_path_file_subpath(filename))

#获得文件列表的文件名
def get_paths_file_basename(filelist):
    basenames=[]
    for name in filelist:
        basenames.append(get_path_file_basename(name))
    return basenames