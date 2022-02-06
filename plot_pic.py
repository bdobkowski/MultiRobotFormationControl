#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 19:09:55 2022

@author: bdobkowski
"""
import numpy as np
from matplotlib import pyplot as plt

y = np.loadtxt('./x.txt')
temp = 10 - y
y = y + 2*temp
x = np.loadtxt('./y.txt')

plt.figure(1)
plt.scatter(x, y)
