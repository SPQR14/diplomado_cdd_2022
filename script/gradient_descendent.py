#!/home/sqpr14_/anaconda3/envs/dss python
# -*- coding: utf-8 -*-

"""gradient_descendent.py: Diplomado ed Ciencia de datos Módulo 2 - Pŕactica 3
An implementacion of the grandient descendent algorithm"""

__author__ = "Alberto Isaac Pico Lara"
__date__ = "friday 24/06/2022"

import numpy as np

# f_primitive = lambda x: x ** 2 - 2 * x - 3
# derivate_f = lambda x: 2 * x - 2


def f(x):
    return x ** 2 - 2 * x - 3


def f_prime(x):
    return 2 * x - 2


def gradient_descendent(f, gradient, x_0 = 0, learning_rate=0.01, n_iter=1000):
    x_i = x_0
    for i in range(n_iter):
        step = -learning_rate * gradient
        x_i += step
    return x_i
