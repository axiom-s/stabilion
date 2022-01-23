# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.fftpack import fft
import pandas as pd
import pylab as pl
from os import listdir
from os.path import isfile, join
import json

def average(lst): return sum(lst) / len(lst) 
def minmax(vec): return min([min(i) for i in vec]), max([max(i) for i in vec]) 
def normalize(raw,minn,maxx): return [(float(i)-minn)/(maxx-minn) for i in raw]

def plotSpectrum(y,Fs):
    'строим и рисуем  спектp'
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]

    'информативная часть спектра'
    obrez=700#3000 #-1
    frq=frq[10:obrez]
    Y=Y[10:obrez]
    #print len(frq)

    pl.plot(frq,abs(Y),'-r') # plotting the spectrum
    #pl.xlabel('Freq (Hz)')
    #pl.ylabel('|Y(freq)|')

def spectrum(y,Fs):
    'строим спектp'
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    'информативная часть спектра'
    obrez=300#3000 #-1
    frq=frq[0:obrez]
    Y=    Y[0:obrez] #print len(frq)
    return abs(Y) ### !!!!!!!!!!!!!!

def getfrom(i,db):
    'возвращает замеры и данные человека по номеру строки'
    fn='data_file/'+db['Fayl_dannych'][i][:-4]+'.csv'
    familia = db['Familiya'][i].decode('utf-8', 'ignore')
    Id= db['ID'][i]
    data = pd.read_csv(fn,sep='\t',decimal=",")
    V =  data.iloc[:, [2]]
    X =  data.iloc[:, [3]]
    Y =  data.iloc[:, [4]]
    X=X.values.tolist(); X=[i[0] for i in X]
    Y=Y.values.tolist(); Y=[i[0] for i in Y]
    V=V.values.tolist(); V=[i[0] for i in V]
    return X,Y,V,familia, Id


def read2json():
    length =6000# 6500# 7488
    def reader(fn):
        db=pd.read_csv("BD/"+fn)
        Xs = []; Ys = []; Vs=[]
        for i in range(1,len(db)):
            X,Y,V,familia, Id = getfrom(i,db); # print familia
            Xs.append(X[:length])
            Ys.append(Y[:length])
            Vs.append(V[:length])
        return Xs,Ys, Vs


    mypath='BD'; fns = [f for f in listdir(mypath) if isfile(join(mypath, f))]# print fns #fn = fns[11] #fn = "Александрова.csv" #fn = "Александров.csv" #fn = "Бабанов.csv" #fn = "Бибик.csv" #fn = "Болдинова.csv" #fn = "Гроховский.csv"

    dataX = []; dataY = []; dataV = []
    for i in range(1,len(fns)):
        fn=fns[i]
        Xs,Ys, Vs = reader(fn)
        if Xs!=[] or Ys!=[] or Vs!=[]:  
            print (fn, len(Xs))
            for L in Xs:
                if len(L)==length: #L=param[0] 
                    dataX.append(L)

            print (fn, len(Ys))
            for L in Ys:
                if len(L)==length: #L=param[0] 
                    dataY.append(L)

            print (fn, len(Vs))
            for L in Vs:
                if len(L)==length: #L=param[0] 
                    dataV.append(L) #print len(dataX), len(dataY), len(dataV)

    d = {}
    d['X'] = dataX
    d['Y'] = dataY
    d['V'] = dataV
    with open('data.json', 'w') as outfile: json.dump(d, outfile)
#data = dataX

