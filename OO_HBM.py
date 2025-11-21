# -*- coding: utf-8 -*-
import numpy as np
import userlib.util as tool
from joblib import Parallel, delayed
import os
import userlib.RVgenerator as RVG

class HBM:
    def __init__(self, population, args_dict, dtbl_theta):
        self.binWidth = 0.1 # km/s
        
        # Initialize class attributes
        self.basePath = args_dict['results_path'] + 'HBM/' + args_dict['sub_path_str']
        
        self.dtbl_theta = dtbl_theta

        self.mc_times = args_dict['mc_times']
        self.n_jobs = args_dict['n_jobs']
        # Grid parameters
        self.grid_pi = args_dict['grid_pi']
        self.grid_kappa = args_dict['grid_kappa']
        self.grid_eta = args_dict['grid_eta']

        self.truths = args_dict['truths']
        self.epsilon = self.truths['epsilon']
        self.grid_fbin = args_dict['grid_fbin']
        self.fbin_intv = args_dict['fbin_intv']
        self.par_intv = args_dict['par_intv']
        self.population = population
        self.sampleSize = len(self.population)
        
        self.fragmentFileName = '(%s, %s, %s).npy'

        self.fragmentFilePath = self.basePath + 'fragments/'
        self.originalDataPath = self.basePath + f'originalData/'
        self.probDataPath = self.basePath + f'Prob/binWidth={self.binWidth}/'
        self.PBDataPath = self.probDataPath + 'PB/'
        self.PSDataPath = self.probDataPath + 'PS/'
        self.sigmaListPath = self.basePath + 'sigmaList/'
        
        self.picDataPath = self.basePath + 'picData/'
        self.picPath = self.basePath + 'pics/'
        
        # 生成参数列表
        self.params = []
        for a in self.grid_pi:
            for b in self.grid_kappa:
                for c in self.grid_eta:
                    self.params.append([a,b,c])
        
        # Create necessary directories
        for path in [self.fragmentFilePath, self.originalDataPath, self.PBDataPath,
                     self.PSDataPath, self.sigmaListPath, self.picDataPath, self.picPath]:
            tool.mkdir(path)
        
        tool.saveParamGrids(self.picDataPath, self.grid_pi, self.grid_kappa, self.grid_eta, self.grid_fbin)
        
        
    def getLogMax(self):
        max_log = float("-inf")
        for fn in os.listdir(self.originalDataPath):
            pbv_fbin = np.load(self.originalDataPath + fn, allow_pickle=True)
            mv = np.max(pbv_fbin)
            if (max_log < mv):
                max_log = mv
        return max_log
    
    def deductMaxLog(self):
        max_log = self.getLogMax()
        print(max_log)
        fragments = []
        for fn in os.listdir(self.originalDataPath):
            fragments.append(fn)
            log_PSigma = np.load(self.originalDataPath + fn, allow_pickle=True)
            np.save(self.fragmentFilePath + fn, log_PSigma - max_log)
        return fragments


    def generateMDFWithSingleParam(self, par_name, fragments):
        index = 0 if par_name == 'pi' else (1 if par_name == 'kappa' else 2)
        grid_par = np.load(self.picDataPath + f'grid_{par_name}.npy', allow_pickle=True)
        pbv = np.zeros(len(grid_par))
        for fn in fragments:
            li, ri = fn.find('(')+1, fn.find(')')
            p = fn[li:ri].split(',')[index]
            pbv_ind = np.where(grid_par == float(p))[0]
            log_pbv = np.load(self.fragmentFilePath + fn, allow_pickle=True)
            pbv[pbv_ind] += np.sum(np.exp(log_pbv))
        tool.saveResultParamPicData(par_name, pbv, self.picDataPath)


    def generateMDFfB(self, fragments):
        lklhd_fB = np.zeros(len(self.grid_fbin))
        for fn in fragments:
            log_PSigma = np.load(self.fragmentFilePath + fn, allow_pickle=True)
            lklhd_fB += np.exp(log_PSigma)
        tool.saveResultParamPicData('fbin', lklhd_fB, self.picDataPath)


    def marginalDistribution(self, fragments):
        if (len(fragments) == 0):
            fragments = os.listdir(self.fragmentFilePath)
        param_names = ['pi']#, 'kappa', 'eta']
        
        for p in param_names:
            self.generateMDFWithSingleParam(p, fragments)
        
        self.generateMDFfB(fragments)


    def deleteTempFragments(self):
        for fn in os.listdir(self.fragmentFilePath):
            os.remove(self.fragmentFilePath + fn)
        print('Fragments are removed.')
        for fn in os.listdir(self.PBDataPath):
            os.remove(self.PBDataPath + fn)
        print('Temporary PB data files are removed.')
        for fn in os.listdir(self.PSDataPath):
            os.remove(self.PSDataPath + fn)
        print('Temporary PS data files are removed.')


    def getPS_sigmaList(self, epsilon, epochs, mc_times):
        ''' data caching for single stars'''
        # fn = 'singleStar_ep=%0.2f_epochs=%s_mc_times=%s.npy'%(epsilon, epochs, int(mc_times))
        # if (tool.isFileExisted(self.sigmaListPath + fn)):
        #     return np.load(path + fn)

        PS_sigmaList = np.zeros(mc_times)
        for t in range(mc_times):
            rvs = np.random.normal(0, epsilon, epochs)
            PS_sigmaList[t] = np.std(rvs)
        # np.save(path + fn, PS_sigmaList)
        # print (path + fn + ' is saved.')
        return PS_sigmaList
    

    def paramNodeCalculation(self, param):
        s = tool.now()
        pi, kappa, eta = param[0], param[1], param[2]
        list_PB, list_PS = np.zeros(self.sampleSize), np.zeros(self.sampleSize)
        print ('sampleSize = %s, mc_times = %s, param =  %s'% (self.sampleSize, self.mc_times, param))
        for i in range(self.sampleSize):
            PB_sigmaList = np.zeros(self.mc_times)
            rv_std = np.std(self.population[i]['rvs'])
            for t in range(self.mc_times):
                p = tool.getOrbitalParameters(pi, kappa, eta)
                P, q, e, m1, inc, omega, T0 = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                phases = RVG.getPhasesByDeltaTs_T0(self.population[i]['deltaTs'], P, T0)
                phases = np.mod(phases,1)
                rvs = RVG.getRvsByPhases_lookupTable(P*86400., q, e, m1, inc, omega, phases,self.dtbl_theta)
                rvs += np.random.normal(0, self.epsilon, len(phases))#测量误差
                PB_sigmaList[t] = np.std(rvs)
            # kde_pb = gaussian_kde(PB_sigmaList)
            # pb = kde_pb(rv_std)
            pb = 1e-323
            if (rv_std<max(PB_sigmaList)):
                list_density, bins = np.histogram(PB_sigmaList, density=True, bins=np.arange(min(PB_sigmaList), max(PB_sigmaList)+self.binWidth, self.binWidth))
                pb_index = np.digitize(rv_std, bins)
                pb = list_density[pb_index-1]

            PS_sigmaList = self.getPS_sigmaList(self.epsilon, self.population[i]['epochs'], self.mc_times)
            # kde_ps = gaussian_kde(PS_sigmaList)
            # ps = kde_ps(rv_std)
            ps = 1e-323
            if (rv_std<max(PS_sigmaList)):
                list_density, bins = np.histogram(PS_sigmaList, density=True, bins=np.arange(min(PS_sigmaList), max(PS_sigmaList)+self.binWidth, self.binWidth))
                ps_index = np.digitize(rv_std, bins)
                ps = list_density[ps_index-1]
            
            list_PB[i], list_PS[i] = pb, ps
                
            print ('π:%.2f\tκ:%.2f\tη:%.2f\tgroup_id:%s\tpb:%0.6f\tps:%0.6f. Time consumed:%s'%\
                   (pi, kappa, eta, self.population[i]['id'], pb, ps, tool.now()-s), flush=True)
                
        return list_PB, list_PS


    def PBPSCalculation(self, param):
        pi, kappa, eta = param[0], param[1], param[2]
        filename = self.fragmentFileName%(pi, kappa, eta)
        PB, PS = self.paramNodeCalculation(param)
        np.save(self.PBDataPath + filename[:-4] + '_PB.npy', PB)
        np.save(self.PSDataPath + filename[:-4] + '_PS.npy', PS)
        print(self.probDataPath + filename[:-4] + ' PBPS is saved.')


    def multithreading(self):
        Parallel(n_jobs=self.n_jobs,verbose=10)(
            delayed(self.PBPSCalculation)(p) for p in self.params)


    def generateOriginalNodeData(self):
        for fn in os.listdir(self.PBDataPath):
            prefix = fn[:-7]
            PB = np.load(self.PBDataPath + fn, allow_pickle=True)
            PS = np.load(self.PSDataPath + prefix + '_PS.npy', allow_pickle=True)
            matrix_log_PSigma = []
            
            for fB in self.grid_fbin:
                PSigma = fB*PB + (1-fB)*PS
                log_PSigma = np.log(PSigma[PSigma>1e-323])
                sum_log_PSigma = np.sum(log_PSigma)
                matrix_log_PSigma.append(sum_log_PSigma)
            filename = self.originalDataPath + prefix + '.npy'
            np.save(filename, matrix_log_PSigma)
            print(filename + ' is saved. \n')


    def run(self):
        """Main processing method that orchestrates the entire workflow"""
        self.start = tool.now()
        self.deleteTempFragments()
        
        # Run main processing
        self.multithreading()
        self.generateOriginalNodeData()
        fragments = self.deductMaxLog()
        self.marginalDistribution(fragments)
        self.deleteTempFragments()
        
        end = tool.now()
        np.savetxt(f'{self.picDataPath}running_time.txt', [(end - self.start).total_seconds()], fmt='%.03f')
        tool.savePics(self.basePath, self.truths)
        tool.collect()
        print ("-------------- HBM is finished --------------", flush=1)
        print(end - self.start)
        print ("---------------------------------------------\n", flush=1)
        print(end)

