# -*- coding: utf-8 -*-
import numpy as np
import userlib.util as tool
from joblib import Parallel, delayed
import scipy.stats as st
import userlib.RVgenerator as RVG


class DVCD:
    def __init__(self, population, args_dict, dtbl_theta):
        self.template_epochs = 11
        # self.n_template_dtajrv = self.template_epochs - 1
        self.n_template_dtajrv = sum(range(1,self.template_epochs))
        self.dtbl_theta = dtbl_theta
        self.population = population
        self.n_orbits = 10**4
        self.truths = args_dict['truths']
        self.epsilon = self.truths['epsilon'] # km/s
        self.n_jobs = args_dict['n_jobs']
        # logProb linearProb
        self.basePath = args_dict['results_path'] + 'DVCD/noCaching/linearProb/everyPairs/' + args_dict['sub_path_str']

        ''' data caching '''
        # self.templateRootPath = self.basePath + 'dtajrvTemplate/n_orbits(%i)/epochs(%02i)/' % (self.n_orbits, self.template_epochs)
        # self.templateBinaryPath = self.templateRootPath + 'binary/pi(%.02f)/kappa(%.02f)/'
        # self.templateBinaryFn = 'eta(%.02f).npy'
        # self.templateSinglePath = self.templateRootPath + 'singleStar/'
        # self.templateSingleFn = 'n_orbits(%i)epochs(%02i)epsilon(%.03f).npy' % (self.n_orbits, self.template_epochs, self.epsilon)
        ''' data caching '''
        
        self.fragmentFileName = '(%.02f,%.02f,%.02f).npy'

        self.grid_pi = args_dict['grid_pi']
        self.grid_kappa = args_dict['grid_kappa']
        self.grid_eta = args_dict['grid_eta']
        self.grid_fbin = args_dict['grid_fbin']
        self.par_intv = args_dict['par_intv']
        self.fbin_intv = args_dict['fbin_intv']

        self.params = []
        for a in self.grid_pi:
            for b in self.grid_kappa:
                for c in self.grid_eta:
                    self.params.append([a, b, c])

        self.hypercube_prob_fn = 'hypercube_prob.npy'

        self.fragmentFilePath = self.basePath + 'fragments/'
        self.originalDataPath = self.basePath + 'originalData/'
        self.picDataPath = self.basePath + 'picData/'
        self.picPath = self.basePath + 'pics/'
        
        # Create necessary directories
        for path in [self.fragmentFilePath, # self.templateBinaryPath, self.templateSinglePath, 
                     self.originalDataPath, self.picDataPath, self.picPath]:
            tool.mkdir(path)
        
        tool.saveParamGrids(self.picDataPath, self.grid_pi, self.grid_kappa, self.grid_eta, self.grid_fbin)


    def getMDFEstimation(self):
        Z = np.load(self.originalDataPath + self.hypercube_prob_fn, allow_pickle=1)
        Z = np.reshape(Z, [len(self.grid_pi), len(self.grid_kappa), len(self.grid_eta), len(self.grid_fbin)])
        Z -= np.max(Z)
        linear_Z = np.exp(Z)
        linear_Z = Z/np.max(Z)

        lklhd_fbin = np.sum(linear_Z, axis=0)
        lklhd_fbin = np.sum(lklhd_fbin, axis=0)
        lklhd_fbin = np.sum(lklhd_fbin, axis=0)
        tool.saveResultParamPicData('fbin', lklhd_fbin, self.picDataPath)

        lklhd_pi = np.sum(linear_Z, axis=1)
        lklhd_pi = np.sum(lklhd_pi, axis=1)
        lklhd_pi = np.sum(lklhd_pi, axis=1)
        tool.saveResultParamPicData('pi', lklhd_pi, self.picDataPath)


    def fragmentsProcessing(self):
        if tool.isFileExisted(self.originalDataPath + self.hypercube_prob_fn):
            print(self.hypercube_prob_fn, 'was existed.')
            tool.clearDir(self.fragmentFilePath)
            return
        list_prob = []
        for p in self.params:
            list_lklhd_fbin = np.load(self.fragmentFilePath + self.fragmentFileName % (p[0], p[1], p[2]))
            list_prob.append(list_lklhd_fbin)
        saveFn = self.originalDataPath + self.hypercube_prob_fn
        np.save(saveFn, np.array(list_prob))
        print('hypercube_prob_fn was saved.')
        tool.clearDir(self.fragmentFilePath)

    def getBinaryDtajrvList(self, pi, kappa, eta):
        dtrvsq_list = []
        for _ in range(self.n_orbits):
            p = tool.getOrbitalParameters(pi, kappa, eta)
            P, q, e, m1, i, omega, T0 = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            phases = np.random.random(self.template_epochs) # This is a key difference to the S13 and HBM, which make the precalculated approach feasible.
            rvs = RVG.getRvsByPhases_lookupTable(P*86400., q, e, m1, i, omega, phases, self.dtbl_theta)
            rvs += np.random.normal(0, self.epsilon, self.template_epochs)
            dtrvsq_list.append(np.diff(rvs))
            
        return np.ndarray.flatten(np.array(dtrvsq_list))

    def getSingleStarDtajrvList(self):
        dtrvsq_list = []
        for _ in range(self.n_orbits):
            rvs = np.random.normal(0, self.epsilon, self.template_epochs)
            # dtrvsq_list.append(np.diff(rvs))
            
            for a in range(len(rvs)-1):
                for b in range(a+1, len(rvs)):
                    dtrvsq_list.append(rvs[b]-rvs[a])
        return np.ndarray.flatten(np.array(dtrvsq_list))


    def simulation(self, fbin, list_b, list_s):
        n_orbits_simul_b = round(self.n_orbits * fbin)
        n_orbits_simul_s = self.n_orbits - n_orbits_simul_b
        DtajrvList_tmpl = np.r_[
            list_b[:n_orbits_simul_b * self.n_template_dtajrv],
            list_s[:n_orbits_simul_s * self.n_template_dtajrv]
        ]
        KSpv = max(1e-323, st.ks_2samp(self.list_dtajrv_obs, DtajrvList_tmpl)[1])
        return np.log(KSpv)
    

    def KSpvCalculation(self, p):
        pi, kappa, eta = p[0], p[1], p[2]
        fragmentFn = self.fragmentFilePath + self.fragmentFileName % (pi, kappa, eta)
        print('DVCD: π=%.02f κ=%.02f η=%.02f started at:%s. n_orbits:%i, samples:%i, epochs:%i, ε:%.02f, fbin:%.02f' % (
            pi, kappa, eta, tool.now(), self.n_orbits, self.truths['sample_size'], self.truths['epochs'], self.epsilon, self.truths['fbin']), flush=1)
        
        s = tool.now()
        DtajrvList_binary = self.getBinaryDtajrvList(pi, kappa, eta)
        # DtajrvList_single = self.getSingleStarDtajrvList()
        lnKSpv = []
        for fbin in self.grid_fbin:
            lnKSpv.append(self.simulation(fbin, DtajrvList_binary, self.DtajrvList_single))

        print('DVCD: π=%.02f\tκ=%.02f\tη=%.02f\t was end.  n_orbits=%i, Time consumed:%s' % (
            pi, kappa, eta, self.n_orbits, tool.now() - s), flush=1)
        np.save(fragmentFn, lnKSpv)

    def multipleThreading(self):
        Parallel(n_jobs=self.n_jobs, verbose=10) \
            (delayed(self.KSpvCalculation)(p) for p in self.params)

    def run(self):
        self.start = tool.now()
        
        # 观测部分
        temp_list_dtajrv_obs = []
        for p in self.population: 
            # temp_list_dtajrv_obs.append(np.diff(p['rvs']))
            for a in range(len(p['rvs'])-1):
                for b in range(a+1, len(p['rvs'])):
                    temp_list_dtajrv_obs.append(p['rvs'][b]-p['rvs'][a])
        self.list_dtajrv_obs = np.ndarray.flatten(np.array(temp_list_dtajrv_obs))
        
        self.DtajrvList_single = self.getSingleStarDtajrvList()
        self.multipleThreading()
        self.fragmentsProcessing()
        self.getMDFEstimation()

        end = tool.now()
        np.savetxt(f'{self.picDataPath}running_time.txt', [(end - self.start).total_seconds()], fmt='%.03f')
        
        tool.savePics(self.basePath, self.truths)
        tool.collect()
        print ("-------------- DVCD is finished --------------", flush=1)
        print(end - self.start)
        print ("----------------------------------------------\n", flush=1)
        print(end)

