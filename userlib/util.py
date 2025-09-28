# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:05:34 2017

@author: Administrator
"""
import os
import sys
import os.path
import gc
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.dpi'] = 75
plt.rcParams['figure.figsize'] = (5, 3)



P_min, P_max = 1, 1400
q_min, q_max = 0.1, 1.0
e_min, e_max = 1e-5, 0.9
m1_min, m1_max = 1.0, 10.

def get_c():
    return 299792.458 # km/s

def get_G():
    return 1.327458213e11 #km^3 Msun^-1  s^-2

def get_Msun():
    return 1.989*10**30 #kg

def get_date_Ymd():
    return datetime.datetime.now().strftime('%Y-%m-%d')

def now():
    return datetime.datetime.now()

def argvs():
    return sys.argv

    
def collect():
    print (gc.collect())
    
def getRandomDeltaTs_1_P(P, o):
    return np.round(np.random.uniform(1, P, o-1) ,3)# day
# MNRAS Tian   & Salpeter (1955, APJ)
def get_m1(m1_min, m1_max):
    a = -2.35
    b = a + 1
    xi = (m1_max**b - m1_min**b) / b
    Fm = np.random.random()
    m1 = (b * xi * Fm + m1_min**b) ** (1 / b)  # Msun
    return m1

def secondsBetweenTwoDates(dateStr1, dateStr2):
    date1 = datetime.datetime.strptime(dateStr1, '%Y%m%d%H%M%S')
    date2 = datetime.datetime.strptime(dateStr2, '%Y%m%d%H%M%S')
    return ((date2-date1).total_seconds())


def loadDataTableTheta():
    fn = "./binary/dataTable/theta(e,phase).npy"
    if not isFileExisted(fn):
        fn = "D:/dataTable/theta(e,phase).npy"
    dtb = np.load(fn, allow_pickle=True)
    print("-----------------------------------")
    print("DataTable theta(e,phase) is loaded.")
    print("-----------------------------------")
    return dtb

def mkdir(path):
    if (os.path.isfile(path)):
        os.remove(path)
    if (not os.path.exists(path)):
        try:
            os.makedirs(path)
            print (path,' is created.')
        except Exception as e:
            print (e)

def clearDir(path):
    for f in os.listdir(path):
        os.remove(path+f)
    print (path, 'is cleared.')
        
def getPositions(Y):
    Y /= sum(Y)
    cum_Y = np.cumsum(Y)
    peak = np.argmax(Y)
    pos_P15 = np.argmin(np.abs(cum_Y-0.15))
    pos_P50 = np.argmin(np.abs(cum_Y-0.50))
    pos_P85 = np.argmin(np.abs(cum_Y-0.85))
    return [peak, pos_P15, pos_P50, pos_P85]
        
def saveResultParamPicData(paramName, list_prob, picDataPath):
    dtfn = f'{picDataPath}likelihood_{paramName}.npy'
    np.save(dtfn, list_prob)
    print(f'{dtfn} is saved.', flush=True)

    positions = getPositions(list_prob)
    posfn = f'positions_{paramName}.npy'
    np.save(picDataPath + posfn, positions)
    print(f'positions_{paramName}:', positions, flush=True)
    print(f'{posfn} is saved.\n', flush=True)
    

def saveParamGrids(picDataPath, grid_pi, grid_kappa, grid_eta, grid_fbin):
    np.save(picDataPath + 'grid_pi.npy', grid_pi)
    np.save(picDataPath + 'grid_kappa.npy', grid_kappa)
    np.save(picDataPath + 'grid_eta.npy', grid_eta)
    np.save(picDataPath + 'grid_fbin.npy', grid_fbin)
    print('Param grids are saved.\n', flush=True)
    

def savePics(basePath, truths):
    params_list = ['pi','fbin'] #'pi','kappa','eta','fbin'
    picDataPath = basePath + 'picData/'
    picPath = basePath + 'pics/'
    title = 'samples:%i, epochs:%i, $f_{bin}$:%.02f, epsilon:%.02f\n $\pi$:%.1f, $\kappa$:%.1f, $\eta$:%.1f' % \
    (truths['sample_size'], truths['epochs'], truths['fbin'], truths['epsilon'], truths['pi'], truths['kappa'], truths['eta'])
    for p in params_list:
        grid = np.load(picDataPath + f'grid_{p}.npy', allow_pickle=True)
        likelihood = np.load(picDataPath + f'likelihood_{p}.npy', allow_pickle=True)
        positions = np.load(picDataPath + f'positions_{p}.npy', allow_pickle=True)
        plt.figure()
        plt.title(title)
        plt.plot(grid, likelihood, 'k-', lw=1.5)
        plt.axvline(grid[positions[0]], ls='-', color='r', lw=1, label='Peak: %.2f' % grid[positions[0]])
        plt.axvline(grid[positions[1]], ls='--', color='gray', lw=1, label='P15: %.2f' % grid[positions[1]])
        plt.axvline(grid[positions[2]], ls='--', color='blue', lw=1, label='P50: %.2f' % grid[positions[2]])
        plt.axvline(grid[positions[3]], ls='--', color='gray', lw=1, label='P85: %.2f' % grid[positions[3]])
        plt.xlabel(f'{p}')
        plt.ylabel('Likelihood')
        # plt.title(f'Likelihood of {p}')
        plt.xticks()
        plt.yticks()
        plt.grid(alpha=0.3)
        plt.xlim(grid[0]-0.05, grid[-1]+0.05)
        plt.ylim(0, 1.1*likelihood.max())
        plt.legend(loc='best')
        plt.tight_layout()
        picFn = picPath + f'likelihood_{p}.png'
        plt.savefig(picFn)
        plt.close()
        print (f'{picFn} is saved.')


def getOrbitalParameters(pi, kappa, eta):
        P = np.random.uniform(P_min, P_max) # days
        if (pi == 0.):
            P = np.random.uniform(P_min, P_max)
        else:
            cdv_P = np.random.uniform(0,1)
            P = np.power(cdv_P*(P_max**(pi+1) - P_min**(pi+1))+ P_min**(pi+1), 1/(pi+1))

        if (kappa == 0.):
            q = np.random.uniform(q_min, q_max)
        else:
            cdv_q = np.random.uniform(0,1)
            q = np.power(cdv_q*(q_max**(kappa+1)-q_min**(kappa+1)) + q_min**(kappa+1), 1/(kappa+1))
        
        if (eta == 0.):
            e = np.random.uniform(e_min, e_max)
        else:
            cdv_e = np.random.uniform(0,1)
            e = np.power(cdv_e*(e_max**(eta+1)-e_min**(eta+1)) + e_min**(eta+1), 1/(eta+1))
        m1 = get_m1(m1_min, m1_max)
        
        i = np.random.uniform(-np.pi/2,np.pi/2)
        omega = np.random.uniform(0,2*np.pi)
        T0 = np.random.uniform(0,P) # days
        
        return np.round([P,q,e,m1,i,omega,T0],3)

def pause():
    sys.exit(0)

    
def isFileExisted(filename):
    if (os.path.exists(filename)):
        return True
    else:
        return False
    

# if (__name__ == '__main__'):
#     e_min, e_max = 1e-5, 0.9
#     eta = -1.8
#     cdv_e = np.random.uniform(0,1)
#     e = np.power(cdv_e*(e_max**(eta+1)-e_min**(eta+1)) + e_min**(eta+1), 1/(eta+1))
#     print (e)