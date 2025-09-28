# -*- coding: utf-8 -*-
import numpy as np
import userlib.util as tool
from OO_S13 import S13 
from OO_HBM import HBM
from OO_DVCD import DVCD

sample_path = './binary/MockSamples/epsilon(%.02f)/'
binary_path = sample_path + 'binary/pi(%.02f)/kappa(%.02f)/eta(%.02f)/'
single_path = sample_path + 'singleStar/'

bfn = 'binary(2000)epochs(%02i)epsilon(%.02f)P(1,1400)q(0.1,1.0)e(0.0,0.9)m1(1,10)_(%.02f,%.02f,%.02f).npy'
sfn = 'singleStar(2000)epochs(%02i)epsilon(%.02f).npy'


# ================ Used for constructing the samples to be measured ================
ground_truth_samples = [200, 400, 800, 1600, 3200] #  sample size [200, 400, 800, 1600, 3200]
ground_truth_fbins = [0.1, 0.3, 0.5, 0.7, 0.9] # binary fraction [0.1, 0.3, 0.5, 0.7, 0.9]
ground_truth_epochs = [6] # number of observations [2, 3, 4, 5, 6, 8, 10, 20 ,30] 
ground_truth_epsilons = [1.0]#, 8, 16]  # RV measurement error (km/s) [0.1, 1.0, 2.0, 4.0, 8.0, 16.0]
ground_truth_pis = [-1.5, 0.0, 1.5] # parameter of the distribution of orbital period [-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
ground_truth_kappas = [0.0] # parameter of the distribution of mass ratio
ground_truth_etas = [0.0] # parameter of the distribution of eccentricity
#===================================================================================


# ======================== Used for parameter scanning ========================
par_intv = 0.1
grid_pi = np.round(np.arange(-3, 3.00 + par_intv, par_intv), 3)
grid_pi[ grid_pi == -1.] = -1.001
grid_kappa = np.round([0.0], 3)
grid_eta = np.round([0.0], 3)
fbin_intv = 0.01
grid_fbin = np.round(np.arange(0.01, 1.0 + fbin_intv, fbin_intv), 2)
#===============================================================================

mc_order = 4
mc_coefficent = 1.0
mc_times = round(mc_coefficent * 10**mc_order)
n_jobs = min(100, len(grid_pi)*len(grid_kappa)*len(grid_eta))

results_path = './results/n_jobs(%i)/mc_times(10^%ix%.01f)/'%(n_jobs, mc_order, mc_coefficent)
sub_path_str = 'epsilon(%.01f)/samples(%i)/fbin(%.01f)/epochs(%02i)/pi_kappa_eta(%.01f, %.01f, %.01f)/'

def constructPopulation(pi, kappa, eta, epsilon, fbin, sample_size, epochs):
    nb = round(sample_size * fbin)
    ns = sample_size - nb

    binarySampleFn = binary_path%(epsilon, pi, kappa, eta) + bfn%(epochs, epsilon, pi, kappa, eta)
    binary = np.load(binarySampleFn, allow_pickle=True)

    singleSampleFn = single_path%(epsilon) + sfn%(epochs, epsilon)
    single = np.load(singleSampleFn, allow_pickle=True)

    population = np.r_[binary[:nb], single[:ns]]
    return population


if __name__ == "__main__":

    # nohup python3.6 Comparison.py > S13mc10^4.log 2>&1 &
    # nohup python3.6 Comparison.py > HBMmc10^4.log 2>&1 &
    # nohup python3.6 Comparison.py > DVCDmc10^4.log 2>&1 &
    # tail S13mc10^4.log -f
    # ps aux | grep Comparison.py
    # pgrep -f Comparison.py 
    # ps aux | grep Comparison.py 
    # ps aux | grep nohup Comparison.py 
    # pgrep -f nohup
    
    dtbl_theta = tool.loadDataTableTheta()

    for epsilon in ground_truth_epsilons:
        for sample_size in ground_truth_samples:
            if (sample_size > 200):
                ground_truth_fbins = [0.5]
            for fbin in ground_truth_fbins:
                 for epochs in ground_truth_epochs:
                    for pi in ground_truth_pis:
                        for kappa in ground_truth_kappas:
                            for eta in ground_truth_etas:
                                args_dict = {
                                    'mc_times': mc_times,
                                    'n_jobs': n_jobs,
                                    'grid_pi': grid_pi,
                                    'grid_kappa': grid_kappa,
                                    'grid_eta': grid_eta,
                                    'grid_fbin': grid_fbin,
                                    'results_path': results_path,
                                    'truths': {
                                                'epsilon':epsilon,
                                                'sample_size':sample_size,
                                                'fbin':fbin,
                                                'epochs':epochs,
                                                'pi':pi,
                                                'kappa':kappa,
                                                'eta':eta
                                               },
                                    'sub_path_str':sub_path_str%(epsilon, sample_size, fbin, epochs, pi, kappa, eta),
                                    'par_intv':par_intv,
                                    'fbin_intv':fbin_intv,
                                }
                                population = constructPopulation(pi, kappa, eta, epsilon, fbin, sample_size, epochs)

                                # s13Processor = S13(population, args_dict, dtbl_theta)
                                # s13Processor.run()

                                hbmProcessor = HBM(population, args_dict, dtbl_theta)
                                hbmProcessor.run()
                                
                                # dvcdProcessor = DVCD(population, args_dict, dtbl_theta)
                                # dvcdProcessor.run()
