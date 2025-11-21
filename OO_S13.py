# -*- coding: utf-8 -*-
import numpy as np
import userlib.util as tool
import userlib.RVgenerator as RVG
import scipy.stats as st
from joblib import Parallel, delayed

class S13:
    def __init__(self, population_obs, args_dict, dtbl_theta):
        
        # 基本参数
        self.a_coef = 4.0
        self.C = 20.0
        self.population_obs = population_obs
        self.sampleSize = len(self.population_obs)

        # 文件名模板
        self.fragmentFileName = 'gmf(%s, %s, %s, %.03f).npy' # global merit function
        self.dicFileName = 'dic_logGMF.npy'
        
        self._setup_parameters(args_dict)
        self._setup_paths(args_dict)
        self.dtbl_theta = dtbl_theta
        
        # 保存参数范围
        np.save(f'{self.picDataPath}grid_pi.npy', self.grid_pi)
        np.save(f'{self.picDataPath}grid_kappa.npy', self.grid_pi)
        np.save(f'{self.picDataPath}grid_eta.npy', self.grid_pi)
        np.save(f'{self.picDataPath}grid_fbin.npy', self.grid_fbin)


    def getDistibution(self, population):
        """获取分布"""
        flag_binary = np.zeros(len(population))
        max_delta_rv_list, min_delta_hjd_list = [], []
        
        for p in range(len(population)):
            g = population[p]
            temp_deltaRvs, temp_deltaHJDs = [], []
            
            for i in range(len(g['rvs'])-1):
                for j in range(i+1, len(g['rvs'])):
                    abs_deltaRv = np.abs(g['rvs'][i]-g['rvs'][j])
                    error = np.sqrt(g['mmes'][i]**2 + g['mmes'][j]**2)
                    
                    if (abs_deltaRv > self.C and abs_deltaRv > self.a_coef*error):
                        temp_deltaRvs.append(abs_deltaRv)
                        temp_deltaHJDs.append(np.abs(g['obsTimes'][j] - g['obsTimes'][i])) # sec
                        
            if (len(temp_deltaRvs)>0):
                flag_binary[p] = 1
                max_delta_rv_list.append(np.max(temp_deltaRvs))  # km/s
                min_delta_hjd_list.append(np.min(temp_deltaHJDs)/86400.)  # day
        
        N_dtc = sum(flag_binary)
        return N_dtc, np.array(max_delta_rv_list), np.array(min_delta_hjd_list)


    def simulation(self, population_obs, par, fbin):
        """模拟计算"""
        pi, kappa, eta = par[0], par[1], par[2]
        Nbin = int(round(self.N * fbin))
        ln_gmf_list = []
        
        for t in range(self.mc_times):
            ind_binaries = np.sort(np.random.choice(self.N, Nbin, replace=False))
            binaries = population_obs[ind_binaries]
            
            for i in range(len(binaries)):
                b = binaries[i]
                p = tool.getOrbitalParameters(pi, kappa, eta)
                P, q, e, m1, inc, omega, T0 = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
                phases = RVG.getPhasesByDeltaTs_T0(b['deltaTs'], P, T0)
                phases = np.mod(phases,1)
                rvs = RVG.getRvsByPhases_lookupTable(P*86400., q, e, m1, inc, omega, phases, self.dtbl_theta)
                binaries[i]['rvs'] = rvs + np.random.normal(0, b['mmes'])
            
            singleStars = np.delete(population_obs, ind_binaries)
            for j in range(len(singleStars)):
                singleStars[j]['rvs'] = np.random.normal(0, singleStars[j]['mmes'])
            
            samples_simul = np.r_[binaries, singleStars]
            N_dtc_simul, list_dtrv_simul, list_dthjd_simul = self.getDistibution(samples_simul)
            
            fbin_dtc_simul = max(1e-323, N_dtc_simul/self.N)
            P_binomial = max(1e-323, st.binom.pmf(self.N_binaries_obs, self.N, fbin_dtc_simul))
            
            KSpV_dtrv = 1e-323
            if (len(list_dtrv_simul)>0):
                KSpV_dtrv = max(1e-323, st.ks_2samp(self.list_dtrv_obs, list_dtrv_simul)[1])
                
            KSpV_dthjd = 1e-323
            if(len(list_dthjd_simul)>0):
                KSpV_dthjd = max(1e-323, st.ks_2samp(self.list_dthjd_obs, list_dthjd_simul)[1])
            
            ln_gmf_list.append(np.log(KSpV_dtrv) + np.log(KSpV_dthjd) + np.log(P_binomial))
            
        return np.mean(ln_gmf_list)# np.sum(ln_gmf_list)

    def writeFragmentFiles(self, par):
        """写入片段文件"""
        for fbin in self.grid_fbin:
            s = tool.now()
            fn = self.fragmentFileName%(par[0], par[1], par[2], fbin)
            print(f"{fn} N = {self.N}, fbin_dtc_obs = {self.N_binaries_obs/self.N:.4f}, a = {self.a_coef:.1f} started. (Begin:{self.start})", flush=True)
            
            log_gmf = self.simulation(self.population_obs, par, fbin)
                
            np.save(self.fragmentFilePath + fn, log_gmf)
            print(f"{fn} is saved. log_gmf = {log_gmf:.4f}, time consumed:{tool.now()-s}", flush=True)

    def multithreading(self):
        """多线程处理"""
        Parallel(n_jobs=self.n_jobs,verbose=10)\
            ((delayed(self.writeFragmentFiles)(p)
            ) for p in self.params)

    def getLogMax(self):
        """获取最大对数值"""
        max_log_gmf = float("-inf")
        for p in self.params:
            for fbin in self.grid_fbin:
                fn = self.fragmentFileName%(p[0], p[1], p[2], fbin)
                d = np.load(self.fragmentFilePath + fn, allow_pickle=True)
                if (max_log_gmf < d):
                    max_log_gmf = d
        np.save(self.originalDataPath + 'max_log.npy', max_log_gmf)
        print('getLogMax is over.')

    def deductMaxLog(self):
        """扣除最大对数值"""
        dic_ln_GMF = {}
        max_log = np.load(self.originalDataPath + 'max_log.npy', allow_pickle=True)
        
        for p in self.params:
            for fbin in self.grid_fbin:
                fn = self.fragmentFileName%(p[0], p[1], p[2], fbin)
                d = np.load(self.fragmentFilePath + fn, allow_pickle=True)
                key = fn[5:-5]
                dic_ln_GMF[key] = d - max_log
                
        np.save(self.originalDataPath + self.dicFileName, dic_ln_GMF)
        print('deductMaxLog is over.')

    def generatePicData(self):
        """生成图片数据"""
        self._generatePicDataSingleParam(0)
        # self._generatePicDataSingleParam(1)
        # self._generatePicDataSingleParam(2)
        self._generatePicDataSingleParam(3)

    def _generatePicDataSingleParam(self, index):
        """生成单参数图片数据"""
        dic = np.load(self.originalDataPath + self.dicFileName, allow_pickle=1).item()
        par_name = self.param_name_list[index]
        par_vs = np.load(self.picDataPath + f'grid_{par_name}.npy', allow_pickle=True)
        list_prob = np.zeros(len(par_vs))
        
        for param in dic.keys():
            ind = np.where(par_vs == float(param.split(',')[index]))[0]
            list_prob[ind] += max(1e-323, np.exp(dic[param]))
            
        dtfn = f'{self.picDataPath}likelihood_{par_name}.npy'
        np.save(dtfn, list_prob)
        print(f'{dtfn} is saved.')

        positions = tool.getPositions(list_prob, self.par_intv)
        posfn = f'positions_{par_name}.npy'
        np.save(self.picDataPath + posfn, positions)
        print(f'positions_{par_name}:', positions)
        print(f'{posfn} is saved.\n')


    def _generatePicDatafB(self):
        """生成fB图片数据"""
        dic = np.load(self.originalDataPath + self.dicFileName, allow_pickle=True).item()
        list_prob = np.zeros(len(self.grid_fbin))
        
        for param in dic.keys():
            ind = np.where(self.grid_fbin == float(param[-5:]))[0]
            list_prob[ind] += max(1e-323, np.exp(dic[param]))
            
        dtfn = 'likelihood_fbin.npy'
        np.save(self.picDataPath + dtfn, list_prob)
        print(f'{dtfn} is saved.')

        positions = tool.getPositions(list_prob, self.fbin_intv)
        posfn = 'positions_fbin.npy'
        np.save(self.picDataPath + posfn, positions)
        print ('fB_positions:', positions)
        print(f'{posfn} is saved.')


    def _setup_parameters(self, args_dict):
        self.mc_times = args_dict['mc_times']
        self.n_jobs = args_dict['n_jobs']
        self.grid_pi = args_dict['grid_pi']
        self.grid_kappa = args_dict['grid_kappa']
        self.grid_eta = args_dict['grid_eta']
        self.grid_fbin = args_dict['grid_fbin']
        self.par_intv = args_dict['par_intv']
        self.fbin_intv = args_dict['fbin_intv']
        
        # 生成参数列表
        self.params = []
        for a in self.grid_pi:
            for b in self.grid_kappa:
                for c in self.grid_eta:
                    self.params.append([a,b,c])
        
        self.param_name_list = np.array(['pi', 'kappa', 'eta', 'fbin'])

    def _setup_paths(self, args_dict):
        """设置文件路径"""
        self.basePath = args_dict['results_path'] + 'S13/' + args_dict['sub_path_str']

        self.originalDataPath = self.basePath + 'originalData/'
        self.fragmentFilePath = self.basePath + 'fragments/'
        self.picDataPath = self.basePath + 'picData/'
        self.picPath = self.basePath + 'pics/'
        
        # 创建必要的目录
        for path in [self.originalDataPath, self.fragmentFilePath, 
                    self.picDataPath, self.picPath]:
            tool.mkdir(path)

    def run(self):
        """运行主程序"""
        self.start = tool.now()
        
        # 观测部分
        self.N_binaries_obs, self.list_dtrv_obs, self.list_dthjd_obs = self.getDistibution(self.population_obs)
        self.N = len(self.population_obs) # observed sample size
        print('Obs data has been processed.\n fB_obs = %.02f' % (self.N_binaries_obs/len(self.population_obs) ))

        # 运行主要流程
        self.multithreading()
        self.getLogMax()
        self.deductMaxLog()
        self.generatePicData()
        tool.clearDir(self.fragmentFilePath)
        
        end = tool.now()
        np.savetxt(f'{self.picDataPath}running_time.txt', [(end - self.start).total_seconds()], fmt='%.03f')
        tool.savePics(self.basePath)
        tool.collect()
        print ("-------------- S13 is finished --------------")
        print(end - self.start)
        print ("---------------------------------------------\n")
        print(end)