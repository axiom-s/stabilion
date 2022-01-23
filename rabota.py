# -*- coding: utf-8 -*-
from __future__ import division
from os import listdir
from os.path import isfile, join
from matplotlib import rc
#rc('font',**{'family':'serif'})
#rc('text', usetex=True)
#rc('text.latex',unicode=True)
#rc('text.latex',preamble='\usepackage[utf8]{inputenc}')
#rc('text.latex',preamble='\usepackage[russian]{babel}')
from functions import *

def splitter(L,n,m): return [L[i:i+n] for i in range(0, len(L), n-m)][:-2] #L = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] #print splitter(L,3,2) # n = 3  # group size # m = 2  # overlap size

def A3(troyka,m,t): 
    #траектория работы из трех точек
    T=troyka
    A=(m/2)*(t**2) * abs(  (T[2]-T[1])**2   -   (T[1]-T[0])**2   )
    #Ax=(m/2*t**2) * (|  (X[i]-X[i-1])**2   -   (X[i-1]-X[i-2])**2   |)
    return A

def rabota(X,Y,m,t):
    # траектория работы из всех точек по три
    troykaX = splitter(X,3,2)
    troykaY = splitter(Y,3,2)
    Ax = [A3(i,m,t) for i in troykaX]
    Ay = [A3(i,t,t) for i in troykaY]
    A=sum(Ax)+sum(Ay)
    return A,Ax,Ay

length =6000# 6500# 7488
def reader(fn):
    db=pd.read_csv(mypath+fn)
    Xs = []; Ys = []; Vs=[]
    for i in range(1,len(db)):
        X,Y,V,familia, Id = getfrom(i,db); # print familia
        Xs.append(X[:length])
        Ys.append(Y[:length])
        Vs.append(V[:length])
    return Xs,Ys, Vs

def phaze(x):
    'метод по Меклеру'
    x1=[];x2=[]
    for i in range(0, len(x), 2): x1.append(x[i])
    for i in range(1, len(x), 2): x2.append(x[i])
    return x1,x2

if __name__=="__main__":
    #кривая изменения кинетеческой энергии, интеграл этой кривой - это работа

    #mypath='BD/'
    #fns = [f for f in listdir(mypath) if isfile(join(mypath, f))]# print fns #fn = fns[11] #fn = "Александрова.csv" #fn = "Александров.csv" #fn = "Бабанов.csv" #fn = "Бибик.csv" #fn = "Болдинова.csv" #fn = "Гроховский.csv"
    #print fns[1]
    #Xs,Ys, Vs = reader(fns[1])

    mypath='BD/'
    Xs,Ys, Vs = reader('Сережина.csv')

    print len(Xs), '- всего измеренрий у этого человека, но взяли только одно'

    nomer_zamera = 1
    X,Y,V = Xs[nomer_zamera],Ys[nomer_zamera],Vs[nomer_zamera] # первая строка -  измерений много - (измерение пациента)

    m=sum(V)/len(V) #m-средняя масса по каналу V
    t=250*0.000001 # t*t=250*250*0.000001 остнется умножить на 2
    A,Ax,Ay = rabota(X,Y,m,t)

    print ('m,A=',m,A)#,Ax,Ay)
    
    #pl.plot(phaze(Ax)[0],phaze(Ax)[1]); pl.show()
    #pl.plot(Ax,Ay); pl.show()
    
    pl.plot(Ax);
    #pl.plot(X,Y);
    pl.tick_params(axis='both', which='major', labelsize=22)
    pl.show()

    #for i in range(len(Ax)): pl.plot(X[:-2][i], Ax[i], '-b'); pl.show()

    pl.plot(Y[:-2],Ay);
    pl.tick_params(axis='both', which='major', labelsize=22)
    pl.show()
    pl.plot(X[:-2],Ax); 
    pl.tick_params(axis='both', which='major', labelsize=22)
    pl.show()

    pl.show()
