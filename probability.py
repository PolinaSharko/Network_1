import numpy as np
from  numpy.core._exceptions import UFuncTypeError
import math
import random
import copy
import matplotlib.pyplot as plt
from itertools import *

def save(name='', fmt='png'):
    plt.savefig('{}.{}'.format(name, fmt))

def multipoly(n1, n2):

    n1 = int(n1, 2)
    s = int('0', 2)
    for c in n2[::-1]:
        if c == '1':
            s = s ^ n1
        n1 *= 2
    s = list(bin(s))
    s = s[2:]
    return s

def polydiv(dividend, divisor):
    out = list(dividend)
    normalizer = divisor[0]
    for i in range(len(dividend) - len(divisor) + 1):
        out[i] /= int(normalizer)
        coef = int(out[i])
        if coef != 0:
            for j in range(1, len(divisor)):
                out[i + j] += int(-divisor[j] * coef)
    for i in range(len(out)):
        out[i] = out[i] % 2
    separator = 1 - len(divisor)
    return out[:separator], out[separator:]


def sumpoly(n1, n2):
    tmp = []
    if len(n1) > len(n2):
        while len(n1) != len(n2):
            n2.insert(0, '0')
    if len(n2) > len(n1):
        while len(n1) != len(n2):
            n1.insert(0, '0')
    for i in range(len(n1)):
        tmp.append((int(n1[i])+int(n2[i])) % 2)
    return tmp


class Coder(object):

    def __init__(self, g, l, e):
        self.poly_g = list(str(g))
        self.len = l
        self.accuracy = e
        self.deg = len(self.poly_g)
        self.set = self.make_set()
        self.ci =[]

    def make_set(self): #Все возможные сообщения длины len
        set = []
        set1 = []
        for i in product('01', repeat=self.len):
            set.append(list(i))
        for i in set:
            set1.append(''.join(i))
        return set1

    def code_dict(self): #Множество кодовых слов
        xr = ''
        for i in range(self.deg):
            if i == 0:
                xr = xr + "1"
            else:
                xr = xr + '0'
        print(len(self.set))
        for j in self.set:
            mxr = multipoly(j, xr)
            if len(mxr) < self.len+self.deg-1:
                while len(mxr) != self.len+self.deg-1:
                    mxr.insert(0, '0')
            try:
                c = polydiv(np.array(mxr, dtype=np.int64), np.array(self.poly_g, dtype=np.int64))[1]
                #print(mxr)
                a = sumpoly(mxr, c)
                self.ci.append(a)
            except UFuncTypeError:
                pass

    def coder_worker(self):
        self.code_dict()
        pi = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        n = self.len+self.deg
        ai = []
        for i in self.ci:
            ai.append(i.count(1))
        ai1 = [0]*n
        for i in range(len(ai1)):
            ai1[i] = (ai.count(i))
        print(ai1)
        for i in range(1, len(ai1)):
            if ai1[i] != 0:
                d = i
                break

        pe1 = []

        for j in range(len(pi)):
            pe = 0
            for i in range(d, n):
                pe = pe + ai1[i]*pow(pi[j], i)*pow((1-pi[j]), (n - i - 1))
            pe1.append(pe)
        plt.grid()
        plot1 = plt.plot(pi, pe1, label = 'Точная оценка')
        print(pe1)

        pe_high1 = []
        for i in range(len(pi)):
            pe_high = 0
            for j in range(0,  d):
                pe_high = pe_high + (math.factorial(n)/(math.factorial(j)*math.factorial(n-j)))*pow(pi[i], j)*pow((1-pi[i]), (n-j))
            pe_high1.append(1-pe_high)
        plot2 = plt.plot(pi, pe_high1, '-.',  label = 'Верхняя граница')

        N = round(9/(4*self.accuracy*self.accuracy))
        for i in range(len(self.poly_g)):
            self.poly_g[i] = int(self.poly_g[i])
        pe_eps = []
        for i in range(len(pi)):
            count = 0
            err = 0

            while count < N:
                b = []
                e = []
                a = copy.deepcopy(self.ci[random.randint(0, len(self.ci) - 1)])
                for k in range(len(a)):
                    if random.uniform(0,1) < pi[i]:
                        e.append(1)
                    else:
                        e.append(0)
                    b.append((a[k] + e[k]) % 2)
                s1 = polydiv(b, self.poly_g)
                if sum(e) != 0 and sum(s1[1]) == 0:
                    err+=1
                count+=1
            pe_eps.append(err / N)

        print(pe_eps)
        plot3 = plt.plot(pi, pe_eps, '-.', label='Имитационное моделирование')
        plt.legend(loc=2)
        save('pe3')


Coder = Coder(10111, 6, 0.015)
Coder.coder_worker()
