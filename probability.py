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
        for i in product('01', repeat=self.len):
            set.append(list(i))
        return set

    def code_dict(self): #Множество кодовых слов
        for j in self.set:
            xr = ''
            for i in range(self.deg):
                if i == 0:
                    xr = xr + "1"
                else:
                    xr = xr + '0'
            mxr = multipoly(''.join(j), xr)
            try:
                c = polydiv(np.array(multipoly(''.join(j), xr), dtype=np.int64), np.array(self.poly_g, dtype=np.int64))[1]
                for i in range(len(c)):
                    c[i] = str(c[i] % 2)
                a = sumpoly(mxr, c)
                self.ci.append(a)
            except UFuncTypeError:
                pass

    def coder_worker(self):
        self.code_dict()
        pi = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65,0.75, 0.85, 0.95, 1]
        ai = []

        for i in self.ci:
            ai.append(i.count(1))
        ai1 = []
        for i in range(len(ai)):
            ai1.append(ai.count(i))
        if min(ai) == 0:
            ai.remove(min(ai))
        d = min(ai)
        pe1 = []
        n = self.len+self.deg

        for j in range(len(pi)):
            pe = 0
            for i in range(d, n):
                pe = pe + ai1[i]*pow(pi[j], i)*pow((1-pi[j]), (max(ai1)-i))
            pe1.append(pe)
        plot1 = plt.plot(pi, pe1)
        print(pe1)

        pe_high1 = []
        d = min(ai)
        for i in range(len(pi)):
            pe_high = 0
            for j in range(d):
                pe_high = pe_high + (math.factorial(max(ai))/(math.factorial(j)*math.factorial(max(ai)-j)))*pow(pi[i], j)*pow((1-pi[i]), (max(ai)-j))
            pe_high1.append(1-pe_high)
        plot2 = plt.plot(pi, pe_high1)
        print(pe_high1)
        #save('pe1')
        #plt.close()

        N = round(9/(4*self.accuracy*self.accuracy))
        for i in range(len(self.poly_g)):
            self.poly_g[i] = int(self.poly_g[i])
        pe_eps = []

        a = copy.deepcopy(self.ci[random.randint(0, len(self.ci) - 1)])

        for i in range(len(pi)):
            count = 0
            err = 0
            while count < N:
                b = []
                e = []
                for k in range(len(a)):
                    if pi[random.randint(0, len(pi) - 1)] <= pi[i]:
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
        plot3 = plt.plot(pi, pe_eps)
        save('pe1')


Coder = Coder(1011, 4, 0.015)
Coder.coder_worker()