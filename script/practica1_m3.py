#!/home/sqpr14_/anaconda3/envs/DSS2 python
# -*- coding: utf-8 -*-

"""practica1_m3.py: Diplomado ed Ciencia de datos Módulo 3 - Pŕactica 1"""

__author__ = "Alberto Isaac Pico Lara"
__date__ = "saturday 13/08/2022"

class Distancias:
    def __init__(self, id=''):
        """
        :type id: string
        """
        self.id = id
        pass

    def minkowski(self, i, j, q=2):
        """
        :param i:
        :param j:
        :param q:
        :return: Minkowski distance between a couple of lists
        """
        if len(i) > len(j):
            return("Error: list length must be equal.")
        elif isinstance(i[0], int) or isinstance(i[0], float):
            result = 0
            for x in range(0, len(i)):
                result += abs(i[x] - j[x]) ** q
            return result ** (1 / q)
        else:
            return("Error: list elements must be integer or float.")

    def weighted_minkowski(self, i, j, q=2, w=1):
        """
        :param i:
        :param j:
        :param q:
        :param w:
        :return: Weighted minkowski distance between a couple os lists
        """
        if len(i) > len(j):
            return("Error: list length must be equal.")
        elif isinstance(i[0], int) or isinstance(i[0], float):
            result = 0
            for x in range(0, len(i)):
                result += w[x] * (abs(i[x] - j[x]) ** q)
            return result ** (1/q)
        else:
            return("Error: list elements must be integer or float.")

    def euclidean(self, i, j):
        """
        :param i:
        :param j:
        :return: Euclidean distance between a couple of lists
        """
        if len(i) > len(j):
            return "Error: list length must be equal."
        elif isinstance(i[0], int) or isinstance(i[0], float):
            return (sum([abs(xi - xj) ** 2 for xi, xj in zip(i, j)])) ** (1/2)
        else:
            return "Error: list elements must be integer or float."


    def manhattan(self, i, j):
        """
        :param i:
        :param j:
        :return:  Manhattan distance between a couple of lists
        """
        if len(i) > len(j):
            return "Error: list length must be equal."
        elif isinstance(i[0], int) or isinstance(i[0], float):
            return sum([abs(xi - xj) for xi, xj in zip(i, j)])
        else:
            return "Error: list elements must be integer or float."

    def simetric_binary(self, i, j):
        if len(i) > len(j):
            return "Error: list length must be equal."
        if (i[0] == 0 or i[0] == 1) and (j[0] == 0 or j[0] == 1):
            q = 0
            r = 0
            s = 0
            t = 0
            for k in range(0, len(i)):
                if i[k] == 1:
                    if j[k] == 1:
                        q += 1
                    else:
                        r += 1
                elif i[k] == 0:
                    if j[k] == 1:
                        s += 1
                    else:
                        t += 1
            return (r + s) / (q + r + s + t)
        else:
            return "Elements in list are not binary"

    def non_simetric_binary(self, i, j):
        if len(i) > len(j):
            return "Error: list length must be equal."
        elif (i[0] == 0 or i[0] == 1) and (j[0] == 0 or j[0] == 1):
            q = 0
            r = 0
            s = 0
            for k in range(0, len(i)):
                if i[k] == 1:
                    if j[k] == 1:
                        q += 1
                    else:
                        r += 1
                elif i[k] == 0 and j[k] == 1:
                    s += 1
            return (r + s) / (q + r + s)
        else:
            return "Elements in list are not binary"

    def ordinal(self, i, j):
        if len(i) > len(j):
            return "Error"
        else:
            maximum_i = max(i)
            maximum_j = max(j)
            a = []
            b = []
            for k in range(0, len(i)):
                a.append((i[k] - 1) / maximum_i - 1)
                b.append((j[k] - 1) / maximum_j - 1)
            return self.euclidean(a, b)

    def nominal(self, i, j, w=None):
        if len(i) > len(j):
            return "error"
        elif w:
            p = sum(w)
            m = 0
            for x, y, z in zip(i, j, w):
                if x == y:
                    m += z
            return (p - m) / p
        else:
            m = 0
            p = len(i)
            for x, y in zip(i, j):
                if x == y:
                    m += 1
            return (p - m) / p
