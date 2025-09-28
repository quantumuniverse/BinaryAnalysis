# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:44:02 2018

@author: Administrator
"""
import numpy as np
import userlib.util as tool
from joblib import Parallel, delayed


G = tool.get_G()
pi = np.pi
tablePath = './binary/dataTable/'
tableFn = 'theta(e,phase).npy'


def getPhasesByDeltaTs_T0(deltaTs, P, T0):
    ts = np.insert(np.cumsum(deltaTs),0,0.) + T0
    return (ts%P)/P


def getRandomDeltaTs_0_P(P, o):
    return np.round(np.random.uniform(0, P, o - 1), 3)  # day

def getRandomDeltaTs_1_P(P, o):
    return np.round(np.random.uniform(1, P, o - 1) ,3)# day

def getE(e, phase):
    E = 0.
    ranE = [0, 2*pi]
    while (True):
        E = sum(ranE)/2.
        r0 = ranE[0] - e*np.sin(ranE[0]) - (2*pi*phase)
        r1 = ranE[1] - e*np.sin(ranE[1]) - (2*pi*phase)
        r = E - e*np.sin(E) - (2*pi*phase)
        if (r0<r and r<0):
            ranE[0] = E
        if (0<r and r<r1):
            ranE[1] = E
        if (abs(r)<0.0001):
            break
    return E


def getTheta(e, phase):
    if (phase < 0):
        phase += 1
    if (e == 0.):
        E = 2*pi*phase
    else:
        E = getE(e, phase)
    factor = (np.cos(E)-e)/(1-e*np.cos(E))
    theta = np.arccos(factor)
    if (E >= pi):
        theta = - theta
    return theta
                   
         
def getRv(P,q,e,m1,i,omega,theta):
    part1 = 2.*pi*np.sin(i) / (P*np.sqrt(1-e**2))
    a = ((G*m1*(1+q)*P**2)/(4*pi**2))**(1./3.)
    a1 = a / (1.+1./q)
    part3 = np.cos(theta + omega)
    part4 = e*np.cos(omega)
    return part1 * a1 * (part3 + part4)
                   
         
def getRvMaxAndMin(P,q,e,m1,i,omega):
    part1 = 2.*pi*np.sin(i) / (P*np.sqrt(1-e**2))
    a = ((G*m1*(1+q)*P**2)/(4*pi**2))**(1./3.)
    a1 = a / (1.+1./q)
    part4 = e*np.cos(omega)
    return part1 * a1 * (part4+1), part1 * a1 * (part4-1)


def getK1K2(P,q,e,m1,i):
    part1 = 2.*pi*np.sin(i) / (P*np.sqrt(1-e**2))
    a = ((G*m1*(1+q)*P**2)/(4*pi**2))**(1./3.)
    a1 = a / (1.+1./q)
    a2 = a - a1
    return abs(part1 * a1), abs(part1 * a2)


def getRvsByPhases(P,q,e,m1,i,omega,phases):
    rvList = []
    part1 = 2.*pi*np.sin(i) / (P*np.sqrt(1-e**2))
    a = ( (G*m1*(1+q)*P**2)/(4*pi**2) )**(1./3.)
    a1 = a / (1.+1./q)
    part4 = e*np.cos(omega)
    factor1 = part1 * a1
    for ph in phases:
        theta = getTheta(e, ph)
        part3 = np.cos(theta + omega)
        rvList.append(factor1 * (part3 + part4))
    return np.array(rvList)


def getThetasByEandPhases_lookupTable(e,phases,dtbl_theta):
    index_e = round(e*10**4)
    if index_e == 10000:
        index_e = 9999
    index_ph = (phases*10**4).astype(int)
    return dtbl_theta[index_e][index_ph]


def getRvsByPhases_lookupTable(P, q, e, m1, i, omega, phases, dtbl_theta):
    index_e = round(e*10**4)
    if index_e == 10000:
        index_e = 9999
    index_ph = (phases*10**4).astype(int)
    thetas = dtbl_theta[index_e][index_ph]
    part1 = 2.*pi*np.sin(i) / (P*np.sqrt(1-e**2))
    a = (((G*m1*(1+q)*P**2))/(4*pi**2))**(1./3.)
    a1 = a / (1.+1./q)
    part3 = np.cos(thetas + omega)
    part4 = e*np.cos(omega)
    rvs = part1 * a1 * (part3 + part4)
    return rvs


def generateDataTable_theta():
    es = np.round(np.linspace(0,1,10**4, endpoint=0), 4)
    phases = np.round(np.linspace(0,1,10**4, endpoint=0), 4)
    Thetas = np.zeros([len(es),len(phases)])
    for e in es:
        print (e, end=', ', flush=True)
        for ph in phases:
            Thetas[round(e*10**4)][round(ph*10**4)] = getTheta(e, ph)
    tool.mkdir(tablePath)
    
    np.save(tablePath + tableFn,Thetas)
    print (tablePath + tableFn + ' is saved.')


def generateDataTable_theta_threading(index_e):
    print ("index_e =%.02f\n"%index_e)
    es = np.round(np.linspace(index_e, index_e + 0.01, 10**2, endpoint=0), 4)
    phases = np.round(np.linspace(0,1,10**4, endpoint=0), 4)
    Thetas = np.zeros([len(es),len(phases)])
    for i in range(len(es)):
        ind_e = round(es[i]*10**4)
        print (ind_e, end=', ', flush=True)
        for ph in phases:
            Thetas[i][round(ph*10**4)] = getTheta(es[i], ph)
    saveFn = str(index_e) + '_' + tableFn
    np.save(tablePath + saveFn,Thetas)
    print (tablePath + saveFn + ' is saved.')
    
def checkDataTable(dtbl_theta):
    print ('In checkDataTable.')
    for j in range(0,len(dtbl_theta)):
        row = dtbl_theta[j]
        print (j, sum(row==0.))
        for k in range(len(row)):
            if (row[k]==0):
                dtbl_theta[j,k] = getTheta(j/10**4, k/10**4)
        print (j, sum(row==0.))
        print ('- - - - - - - -')
#    fn = tableName + '_repaired'
#    np.save(tablePath + fn,dtbl_theta)
#    print (tablePath + fn + ' is saved.')
    

def loadDataTableTheta():
    fn = tablePath + tableFn
    if (not tool.isFileExisted(fn)):
        generateDataTable_theta()
    dtb = np.load(fn, allow_pickle=True)
    print ('-----------------------------------')
    print ('DataTable ' + tableFn + ' is loaded.')
    print ('-----------------------------------')
    return dtb


def constructDataTable():
    dtbl = np.array([])
    for index_e in range(0, 100, 1):
        temp = np.load(tablePath + str(round(index_e * 0.01, 2)) + '_' + tableFn, allow_pickle=True)
        if (dtbl.size == 0):
            dtbl = temp
        else:
            dtbl = np.concatenate((dtbl, temp), axis=0)
            
    print ('DataTable is constructed.')
    np.save(tablePath + tableFn, dtbl)
    print ('DataTable is saved.')
        

if (__name__ == '__main__'):
    deltaTs = getRandomDeltaTs_0_P(350, 2)#观测时间间隔序列
    print (deltaTs)
    phases = getPhasesByDeltaTs_T0(deltaTs, 350, 156)
    print (phases)