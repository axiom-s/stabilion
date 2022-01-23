# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
import pylab as plt
import json
import math
from matplotlib import rc
#rc('font',**{'family':'serif'})
#rc('text', usetex=True)
#rc('text.latex',unicode=True)
#rc('text.latex',preamble='\usepackage[utf8]{inputenc}')
#rc('text.latex',preamble='\usepackage[russian]{babel}')
from functions import *
from rabota import *
import itertools
from mvpa2.suite import *




def preprocess_data(mypath):
    length =6000# 6500# 7488
    def getfrom(fn):
        fn=mypath+'/'+fn
        'возвращает замеры и данные человека по имени файла'
        data = pd.read_csv(fn,sep='\t',decimal=",")
        V =  data.iloc[:, [2]]
        X =  data.iloc[:, [3]]
        Y =  data.iloc[:, [4]]
        X=X.values.tolist(); X=[i[0] for i in X]
        Y=Y.values.tolist(); Y=[i[0] for i in Y]
        V=V.values.tolist(); V=[i[0] for i in V]
        return X,Y,V

    fns = [f for f in listdir(mypath) if isfile(join(mypath, f))]# print fns #fn = fns[11] #fn = "Александрова.csv" #fn = "Александров.csv" #fn = "Бабанов.csv" #fn = "Бибик.csv" #fn = "Болдинова.csv" #fn = "Гроховский.csv"
    '''
    Xs,Ys, Vs = getfrom(fns[1])
    print Xs
    '''
    d={};  d['X']=[]; d['Y']=[]; d['V']=[]
    for f in fns:
        X,Y,V = getfrom(f)
        d['X'].append(X)
        d['Y'].append(Y)
        d['V'].append(V)

    ###############
    dataX = d['X']
    dataY = d['Y']

    As=[]
    for i in range(len(dataX)):
        X=d['X'][i]
        Y=d['Y'][i]
        V=d['V'][i]
        m=sum(V)/len(V) #m-средняя масса по каналу V
        t=250*0.000001 # t*t=250*250*0.000001 остнется умножить на 2
        A,Ax,Ay = rabota(X,Y, m,t)
        A = round(A*100000,2) ####### домножили до вменяемости, перепроверить!
        As.append(A)
    #X,Y,V = Xs[1],Ys[1],Vs[1] 
    #m=sum(V)/len(V) #m-средняя масса по каналу V
    #t=250*0.000001 # t*t=250*250*0.000001 остнется умножить на 2
    #A,Ax,Ay = rabota(X,Y,m,t)

    def centers(data):
        sas=[]   # поиск среднего кажой линии
        Cdata=[] # центрированные данные
        spectrs = [] #  все спектры
        Fs=250#1024#    уточнить параметр для спектра
        for  line in data:
            sa = sum(line)/len(line) # среднее арифметическое
            signal = [i-sa for i in line]
            Cdata.append(signal)
            sas.append(sa)
            sp=spectrum(signal,Fs)
            spectrs.append(sp) #print len(sp)
        cdata = np.array(Cdata)
        spectrs = np.array(spectrs)
        return cdata,spectrs

    cdataX,spectrsX = centers(dataX)
    cdataY,spectrsY = centers(dataY)

    '''
    #кривые 2d
    pl.figure()
    for i in range(len(cdataX)): pl.plot(dataX[i],dataY[i],'b')
    #pl.show()
    '''

    return cdataX, spectrsX, cdataY, spectrsY, As


def clusterisation(data, x,y,sigma,learning_rate,epochs):
    #============SOM===============
    #def neuroclusterisation(x,y,cdata): #som = SimpleSOMMapper((x,y), 10, learning_rate=2.5) som = SimpleSOMMapper((x,y), 22, learning_rate=0.05) som.train(cdata) #pl.imshow(som.K, origin='lower') # для трехмерных mapped = som(cdata) #print mapped #pl.imshow(mapped, interpolation='nearest')# origin='lower') return mapped,som
    def neuroclusterisation2(x,y,cdata): #https://github.com/JustGlowing/minisom
        from minisom import MiniSom    
        SoM = MiniSom(x, y, len(cdata[0]), sigma=sigma, learning_rate=learning_rate)
        SoM.train(cdata, epochs)
        #mapped = som(cdata) #print mapped #pl.imshow(mapped, interpolation='nearest')# origin='lower')
        mapped=[]
        for i in cdata: mapped.append(SoM.winner(i))
        return mapped, SoM

    mapped,som = neuroclusterisation2(x,y,data)
    #print mapped

    Min=min([min(i) for i in data]); Max=max([max(i) for i in data]) #print Min, Max
    M = list(set([str(i) for i in mapped]))#print M
    lm=len(M)
    return lm, M, som
    
def conc(a,b):
    r=[]
    for i in range(len(a)): r.append(np.concatenate([a[i], b[i]]))
    return r

def statistica():
    stat={} # статистика по больным и здоровым
    for k in range(len(spectrs)): 
        for m in range(lm): #print  som.winner(spectrs[k]),'-',M[m]
            if str(som.winner([spectrs[k]])) == M[m]: stat[k] = eval(M[m]), inds[k]

    #print stat
    ill={}
    health={}
    all_klasterts=list(set([stat[i][0] for i in stat])) #print all_klasterts

    for klast in all_klasterts:
        ill[klast]  = 0  
        health[klast]= 0  

    for i in range(len(stat)):
       #if stat[i][1]=='ill':    pl.plot(stat[i][0], '+r', alpha=0.2, markersize=40)
       #if stat[i][1]=='health': pl.plot(stat[i][0], 'xb', alpha=0.2, markersize=40)
       diag = stat[i][1]
       klast= stat[i][0]
       if diag =='ill':    ill[klast]    +=1  
       if diag =='health': health[klast] +=1  #print 'ill',    ill #print 'health', health 

    stat={}
    for k in all_klasterts: 
        prob_pat=ill[k]*100/(ill[k]+health[k])
        prob_norm=health[k]*100/(ill[k]+health[k])

        prob=round(max(prob_pat,prob_norm),2)
        if prob_pat>prob_norm: diag='ill'
        if prob_pat<prob_norm: diag='health'

        if prob_pat==prob_norm: diag='none'

        #if diag=='none' or max(prob_pat, prob_norm)<65 or ill[k]+health[k]<7:
        #    pass
        #else:
        stat[k]={ 'ill': ill[k], 'health': health[k], 'all':ill[k]+health[k], 'diag': [diag, prob] }


        
        print (k, 'больных / здоровых: ', ill[k],health[k],) 
        print (' всего', ill[k]+health[k] )
        print (' вероятность патологии: ', prob_pat),
        print (' вероятность нормы: ',    prob_norm )
    
    return stat


def dov_interval(r,n): #n-всего,  p=r/n, r - оличество индивидуумов в выборке с интересующими нас характерными особенностями #http://statistica.ru/theory/doveritelnye-intervaly/ 
    p=r/n; e=math.sqrt(p*(1-p)/n) 
    return (round(p-(1.96*e),2), round(p+(1.96*e),2))

def visualisation (lm, M, som, As,    x, y, sigma, learning_rate, epochs):
    stat= statistica()
    print (stat)
    imgpath='img/'

    import shutil
    shutil.rmtree(imgpath) 
    os.mkdir(imgpath)
    #=======визуализация============

    '''спектры'''
    pl.figure('spectrs', figsize=(14, 9)) # рисуем спекты
    pl.subplots_adjust(left=0.1, bottom=0, right=0.9, top=0.9, wspace=0, hspace=0.7)
    #stat={} # статистика по больным и здоровым
    for k in range(len(spectrs)): 
        for m in range(lm): #print  som.winner(spectrs[k]),'-',M[m]
            #if str(som([spectrs[k]])[0]) == M[m]: #str(mapped[k]):    

            if str(som.winner([spectrs[k]])) == M[m]: #str(mapped[k]):    
                info=stat[eval(M[m])] #stat[k] = eval(M[m]), inds[k]
                pl.suptitle (str(len(spectrs)) + " vectors, SOM:  " +  str(x)+'x'+str(y)+ ' (s='+str(sigma)+ ' lr='+ str(learning_rate) +' e=' +str(epochs)+')' )
                ax = pl.subplot(lm+1, 1, m+1) #pl.title(M[m], fontsize=8) #pl.subplots_adjust(hspace = 0.1)
                ax.grid(b=True, which='major', color='b', linestyle='-')
                if inds[k]=='ill':    pl.plot(spectrs[k], 'red', alpha=0.3) 
                if inds[k]=='health': pl.plot(spectrs[k], 'blue' , alpha=0.3) #pl.plot(health_spectrsX[k-len(health_spectrsX)],health_spectrsY[k-len(health_spectrsX)], 'blue', alpha=0.1, linewidth=0.5)
                pl.legend(['ill/health='+str(info['ill'])+'/'+str(info['health'])+' ('+str(info['all']) +')' ], fontsize=10)  #pl.plot(ill_spectrsX[k],ill_spectrsY[k], 'red', alpha=0.1, linewidth=0.5)
                pl.xticks(fontsize=6); pl.yticks(fontsize=5) #ax.set_ylim([Min,Max]); #ax.set_xlim([0,lenSp])
                h=pl.ylabel(str(M[m])+': '+str(round(info['all']*100/len(spectrs), 2))+'\%', fontsize=8); h.set_rotation(0); ax.yaxis.labelpad = 30
                ax.xaxis.set_label_position('top')
                pl.xlabel(str(info['diag'][0])+' '+ str(info['diag'][1])+str( '%'), fontsize=10); ax.xaxis.labelpad=5
                #if k+1!=lm: pl.xticks([])
                #ax.yaxis.set_label_position("right"); #pl.subplots_adjust(left=0, bottom=0, right=100, top=None, wspace=0, hspace=0)
    pl.savefig(imgpath+'spectrs.png')
                

    '''исходные данные - кластеры кривых'''
    pl.figure( figsize=(9, 9)) # по спектрам соответсвующие кривые

    for k in range(len(cdata)):
        for m in range(lm): #print  som([cdata[k]])[0],'-',M[m]
            if str(som.winner([spectrs[k]])) == M[m]: #str(mapped[k]):    

                    info=stat[eval(M[m])]
                    if info['diag'][1]>60 and info['health']+info['ill']>7:
                        pl.figure(str(m))
                        #ax = pl.subplot(lm+1, 1, m+1) #pl.title(M[m], fontsize=8) #pl.subplots_adjust(hspace = 0.1)
                        #ax.grid(b=True, which='major', color='b', linestyle='-')
                        if inds[k]=='ill':    #pl.plot(cdata[k], 'red')
                            pl.subplot(121) #pl.title(M[m], fontsize=8) #pl.subplots_adjust(hspace = 0.1)
                            pl.plot(ill_cdataX[k],ill_cdataY[k], 'red', alpha=0.1, linewidth=0.5)
                            pl.legend([str(info['ill']) +' /'+str(info['all'])  ], fontsize=10)  #pl.plot(ill_spectrsX[k],ill_spectrsY[k], 'red', alpha=0.1, linewidth=0.5)
                        if inds[k]=='health': #pl.plot(cdata[k], 'gray')
                            pl.subplot(122) #pl.title(M[m], fontsize=8) #pl.subplots_adjust(hspace = 0.1)
                            pl.plot(health_cdataX[k-len(health_cdataX)],health_cdataY[k-len(health_cdataX)], 'blue', alpha=0.1, linewidth=0.5)
                            pl.legend([str(info['health'])+' /'+str(info['all']) ], fontsize=10)  #pl.plot(ill_spectrsX[k],ill_spectrsY[k], 'red', alpha=0.1, linewidth=0.5)
                        pl.xticks(fontsize=6); pl.yticks(fontsize=5) #ax.set_ylim([Min,Max]); #ax.set_xlim([0,lenSp])
                        #if k+1!=lm: pl.xticks([])
                        #ax.yaxis.set_label_position("right"); #pl.subplots_adjust(left=0, bottom=0, right=100, top=None, wspace=0, hspace=0)

                        t='Cluster '+str(M[m])+': ' + str(info['diag'][0]) +' ' + str(info['diag'][1]) +'\%; '
                        if info['diag'][0] == 'ill':    pl.suptitle(t+ ' 95\%CI='+str(dov_interval(info['ill'],   info['all'])))
                        if info['diag'][0] == 'health': pl.suptitle(t+ ' 95\%CI='+str(dov_interval(info['health'],info['all'])))
                        pl.savefig(imgpath+str(m)+'.png')

    
    '''отметим распознанные'''
    pl.figure( figsize=(9, 3)) # по спектрам соответсвующие кривые

    i=0
    for k in range(len(cdata)):
        for m in range(lm): #print  som([cdata[k]])[0],'-',M[m]
            if str(som.winner([spectrs[k]])) == M[m]: #str(mapped[k]):    
                    info=stat[eval(M[m])]
                    if info['diag'][1]>60 and info['health']+info['ill']>7:
                        if inds[k]=='ill':    pl.plot(k,1,'ro') 
                        if inds[k]=='health': pl.plot(k,1,'bo') 
                        i=i+1
                    else:
                        if inds[k]=='ill':    pl.plot(k,1,'rx') 
                        if inds[k]=='health': pl.plot(k,1,'bx') 

    pl.xticks(fontsize=6); pl.yticks(fontsize=1) 
    pl.xlim(0, k)
    pl.title(str(i)+'/'+str(k) +'('+str(round(i*100/k,2)) + '\%)') 
    pl.savefig(imgpath+'rasp.png')


    '''работа '''
    pl.figure('rabota', figsize=(14, 9))
    i=0
    Health=[]
    Ill=[]
    for k in range(len(cdata)):
        for m in range(lm): #print  som([cdata[k]])[0],'-',M[m]
            if str(som.winner([spectrs[k]])) == M[m]: #str(mapped[k]):    
                ax = pl.subplot(lm+1, 1, m+1) #pl.title(M[m], fontsize=8) #pl.subplots_adjust(hspace = 0.1)
                ax.grid(b=True, which='major', color='b', linestyle='-')
                print (As[k], M[m])
                if inds[k]=='health':
                    pl.plot(i,As[k],'bo', alpha=0.8)
                    Health.append(As[k])
                if inds[k]=='ill':    
                    pl.plot(i,As[k],'ro', alpha=0.8)
                    Ill.append(As[k])
                All=Ill+Health
                #print k, All,len(All)
                i+=1
                pl.xticks(fontsize=6); pl.yticks(fontsize=5) #ax.set_ylim([Min,Max]); #ax.set_xlim([0,lenSp])
                ax.yaxis.tick_left()
                pl.xlim(0, len(As))
                
                #h=pl.ylabel(str(M[m])+ ' SA='+str(round(sum(All)/(len(All)+0.00001),2)), fontsize=8); h.set_rotation(0); ax.yaxis.labelpad = 30 #if k+1!=lm: pl.xticks([])
                ax.yaxis.tick_right()
    pl.savefig(imgpath+'rabota.png')    #pl.show()

    '''
    #ax=pl.subplot(lm+1,1, lm+1)
    pl.figure()
    for i in data: 
        print len(i)
        pl.plot(i,'b')
    for i in som.K:
        for j in i:
            print len(j) # показывает центроиды спектров а не кривых
            pl.plot(j , 'r') 
    pl.ylim(Min, Max)
    '''

    #pl.subplot(lm+2,1,lm+2) #for sa in sas: pl.plot([sa] * length ,'b')
    #for line in data: pl.plot(line, 'b')


mypath_ill='Insult_train_data/ill_csv/'
ill_cdataX, ill_spectrsX, ill_cdataY, ill_spectrsY, ill_As = preprocess_data(mypath_ill)
mypath_health='Insult_train_data/health_csv/'
health_cdataX, health_spectrsX, health_cdataY, health_spectrsY, health_As = preprocess_data(mypath_health)

#print len(ill_spectrsX), len(ill_spectrsY), len (health_spectrsX), len(health_spectrsY)
ill_spectrs = conc(ill_spectrsX,ill_spectrsY); health_spectrs = conc(health_spectrsX, health_spectrsY)
spectrs = ill_spectrs + health_spectrs  #spectrs=[] #for i in range(len(ill_spectrsX)): spectrs.append(np.concatenate([ill_spectrsX[ilth],ill_spectrsY[i]])) #spectrs =  ill_spectrsX+ ill_spectrsY #+ [health_spectrsX   +health_spectrsY]]

inds = ['ill']*len(ill_spectrs) + ['health']*len(health_spectrs) #print inds

cdata   = conc(ill_cdataX,  ill_cdataY)  + conc(health_cdataX,   health_cdataY)   #cdata=[] #for i in range(len(ill_cdataX)): cdata.append(np.concatenate([ill_cdataX[i],ill_cdataY[i]])) #cdata   =  ill_cdataX   +ill_cdataY  #+ [health_cdataX     +health_cdataY]]

#cdata:
#1  ) illX_illY
#2  ) illX_illY
#...
#k-1) healthX_healthY
#k  ) healthX_healthY 

As      =  ill_As       +health_As 
#x=2; y=3; sigma=0.3; learning_rate=0.2; epochs=10000    ## Эталон
x=16; y=1; sigma=0.3; learning_rate=0.2; epochs=10000  ## Эталон
#x=5; y=1; sigma=0.1; learning_rate=0.1; epochs=10000
lm, M, som  = clusterisation(spectrs, x=x, y=y, sigma=sigma, learning_rate=learning_rate, epochs=epochs)
visualisation (lm, M, som, As, x, y, sigma, learning_rate, epochs)
pl.show()
