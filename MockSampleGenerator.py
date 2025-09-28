# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as ran
import userlib.RVgenerator as RVG
import userlib.util as tool
from joblib import Parallel, delayed



def getBinaryPopulation(N, epochs, pi, kappa, eta, epsilon):
    population = []
    for n in range(N):
        j = tool.getOrbitalParameters(pi, kappa, eta)
        P, q, e, m1, i, omega, T0 = j[0], j[1], j[2], j[3], j[4], j[5], j[6]
        deltaTs = RVG.getRandomDeltaTs_0_P(P_max, epochs)  # 观测时间间隔序列 days
        obsTimes = np.insert(np.cumsum(deltaTs), 0, 0.0) + T0
        phases = (obsTimes % P) / P
        phases = np.mod(phases, 1)
        rvs = RVG.getRvsByPhases_lookupTable(
            P*86400, q, e, m1, i, omega, phases, dtbl_theta
        )
        rvs += ran.normal(0, epsilon, epochs)
        dic = {
            "id": str(n) + "B",
            "P": P,  # d
            "q": q,
            "e": e,
            "m1": round(m1, 3),
            "i": i,
            "omega": omega,
            "T0": T0,
            "epochs": epochs,
            "obsTimes": obsTimes,
            "deltaTs": deltaTs,
            "rvs": rvs,
            "mmes": np.zeros(len(rvs)) + epsilon,
            "bmi": np.random.choice(epochs, 1)[0],
        }
        population.append(dic)
    return np.array(population)


def getSingleStarPopulation(N, epochs, epsilon):
    population = []
    for n in range(N):
        deltaTs = RVG.getRandomDeltaTs_0_P(P_max, epochs)  # 观测时间间隔序列
        obsTimes = np.insert(np.cumsum(deltaTs), 0, 0.0) + ran.uniform(0, P_max)
        rvs = ran.normal(0, epsilon, epochs)
        dic = {
            "id": str(n) + "S",
            "epochs": epochs,
            "obsTimes": obsTimes,
            "deltaTs": deltaTs,
            "rvs": rvs,
            "m1": round(tool.get_m1(), 3),
            "mmes": np.zeros(len(rvs)) + epsilon,
            "bmi": np.random.choice(epochs, 1)[0],
        }
        population.append(dic)
    return np.array(population)


def constructRandomPopulation(sampleSize, epochs, epsilon, pi, kappa, eta):
    filename = (
        "binary(%i)epochs(%02i)epsilon(%.02f)P(%i,%i)q(%.01f,%.01f)e(%.01f,%.01f)m1(%i,%i)_(%.02f,%.02f,%.02f)"
        % (
            sampleSize,
            epochs,
            epsilon,
            P_min,
            P_max,
            q_min,
            q_max,
            e_min,
            e_max,
            m1_min,
            m1_max,
            pi,
            kappa,
            eta,
        )
    )
    binary = getBinaryPopulation(sampleSize, epochs, pi, kappa, eta, epsilon)
    np.save(binaryPath + filename, binary)
    print(filename, "is saved.", flush=True)

    filename = "singleStar(%i)epochs(%02i)epsilon(%.02f)" % (
        sampleSize,
        epochs,
        epsilon,
    )
    singleStar = getSingleStarPopulation(sampleSize, epochs, epsilon)
    np.save(singleStarPath + filename, singleStar)
    print(filename, "is saved.", flush=True)


if __name__ == "__main__":
    rootPath = "./binary/MockSamples/"
    epochs = np.arange(2, 31, 1)
    epsilons = [1.]#[0.05, 0.1, 0.5, 1, 2, 4, 8, 16]

    P_min, P_max = 1, 1400
    q_min, q_max = 0.1, 1.0
    e_min, e_max = 0.0, 0.9
    m1_min, m1_max = 1.0, 10.

    params = [
        # [-2.5, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        # [-1.5, 0.0, 0.0],
        [-1.001, 0.0, 0.0],
        # [-0.5, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        # [1.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        # [2.5, 0.0, 0.0],
    ]
    dtbl_theta = tool.loadDataTableTheta()
    for x in params:
        pi_true, kappa_true, eta_true = x[0], x[1], x[2]
        for epsilon in epsilons:
            print ('---------------------- params = ',x , 'epsilon = ', epsilon, ' ----------------------\n')
            path = rootPath + "/epsilon(%.02f)/" % epsilon
            binaryPath = path + "binary/pi(%.02f)/kappa(%.02f)/eta(%.02f)/" % (
                pi_true,
                kappa_true,
                eta_true,
            )
            singleStarPath = path + "singleStar/"

            tool.mkdir(binaryPath)
            tool.mkdir(singleStarPath)
            Parallel(n_jobs=30, verbose=10)(
                delayed(constructRandomPopulation)(
                    2000, o, epsilon, pi_true, kappa_true, eta_true
                )
                for o in epochs
            )

    tool.collect()
