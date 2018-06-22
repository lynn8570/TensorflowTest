# coding=utf-8
import sys
import IPython.display as display
import os
import imageio
# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display,Image


# 图片读取和显示示例
fn = os.listdir("notMNIST_small/A/")
for file in fn[:20]:
    path = 'notMNIST_small/A/' + file
    print("path:",path)
    # img = plt.imread(path)
    # plt.imshow(img)
    # plt.show()
    # img = Image.open(path)
    img = Image(filename=path)
    display(img) #没有输出图片    pip install jupyter 是什么  jupyter notebook notebook.ipynb 启动


