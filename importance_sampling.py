import numpy as np
import parameters
import scipy.stats as st
from numpy.random import normal
import matplotlib.pyplot as plt
from tqdm import tqdm



def euler_maruyama_IS(X0,dt,cx,cy):
    
    #### Variant of the homonymous standard_monte_carlo.py function ####
    #
    # INPUT:
    # X0 = starting point of the trajectory
    # dt = euler-maruyama discretization step
    # cx,cy = Importance Sampling parameters for the importance distribution shifting 
    #
    # OUTPUT:
    # entrance*w = Importance Sampling single sample
    
    sigma = parameters.sigma
    u = parameters.u
    T = parameters.T
    X = X0
    R = parameters.R
    entrance = False # boolean that states if the particle entered the well or not
    t = 0
    w = 1.0 # the correction term is a product of many terms, we then initialize it to 1

    while (t*dt<T and entrance == False): # the simulation is stopped only if it goes beyond the time horizon or it enters the well

        DW = np.array([sigma*np.sqrt(dt)*normal() + cx*dt, sigma*np.sqrt(dt)*normal() + cy*dt]) # 2D vecotr of the Brownian increments
        X = X + u(X)*dt + DW # main Euler-Maruyama step
        t = t + 1 
        
        w = w * np.exp( (pow(DW[0]-cx*dt,2) + pow(DW[1]-cy*dt,2) - pow(DW[0],2) - pow(DW[1],2))  /  (2*sigma**2*dt) ) # main step for the w calculation, see report for the explanation

        if (np.linalg.norm(X) < R): # i.e. if the particle is inside the well region
            entrance = True
    
    return entrance*w


def importance_sampling(n_iter,dt,cx,cy):
    
    #### Important sampling main function ####
    #
    # INPUT:
    # n_iter = cardinality of the Monte Carlo sample
    # dt = euler-maruyama discretization step
    # cx,cy = Importance Sampling parameters for the importance distribution shifting 
    #
    # OUTPUT:
    # results.mean() = Importance Sampling estimation 
    
    print ('\n ---- IMPORTANCE SAMPLING RESOLUTION ---- \n')
    
    X0 = parameters.X0_IS
    results = np.zeros(n_iter) # the vector with the probability samples
    for i in tqdm(range(n_iter), desc='X0=('+str(X0[0]) +','+ str(X0[1]) +')'):
        results[i] = euler_maruyama_IS(X0, dt, cx, cy)

    print ('X0=('+str(X0[1]) +','+ str(X0[1]) +'): ', results.mean() , '+-', parameters.Za*results.std()/np.sqrt(n_iter), end='\n')
        
    return results.mean()



def best_c(n_iter,dt,c_list,c_fixed,c_dir):
    
    #### To be used to compare the variances varying cx or cy parameter ####
    #
    # INPUT:
    # n_iter = cardinality of the Monte Carlo sample
    # dt = euler-maruyama discretization step
    # c_list = list of cx (or cy) parameters (Importance Sampling parameter for the importance distribution shifting)
    # c_fixed = the remaining parameter for the importance distribution shifting
    # c_dir = 'x' or 'y', the parameter that varies in the list (respectively, cx or cy)
    #
    # OUTPUT:
    # sigma_list = list of confidence interval semi-amplitude, each one corresponding to one cx value
    
    sigma_list = [] # list of confidence interval semi-amplitudes, each one corresponding to one cx value
    mean_list = [] # list of IS estimations, each one corresponding to one cx value
    X0 = parameters.X0_IS

    for c in tqdm(c_list):
        results = np.zeros(n_iter)
        for i in range(n_iter):
            if (c_dir == 'x'):
                results[i] = euler_maruyama_IS(X0, dt, c, c_fixed)
            elif (c_dir == 'y'):
                results[i] = euler_maruyama_IS(X0, dt, c_fixed, c)
                     
        mean_list.append(results.mean())
        sigma_list.append(parameters.Za*results.std()/np.sqrt(n_iter))
    
    plt.figure()
    ax = plt.axes()
    ax.set_title('Estimation comparison for Importance Sampling')
    plt.plot(c_list,mean_list)
    plt.xlabel('c_'+ c_dir)
    plt.ylabel('$p_{est}$')

    plt.savefig("./figures/IS_means")

    plt.figure()
    ax = plt.axes()
    ax.set_title('Sigma comparison for Importance Sampling')
    plt.plot(c_list,sigma_list)
    plt.xlabel('c_' + c_dir)
    plt.ylabel('C.I. semi-amplitude')
    plt.savefig("./figures/IS_sigmas")
    			        
    return sigma_list

        
        
        
if __name__ == '__main__':
    #importance_sampling(n_iter = 1000,cx=0,cy=0,dt=0.005) # standard monte carlo
    importance_sampling(n_iter = 50000,cx=-4,cy=-3,dt=0.0001) # IS with best parameters
    #best_c(n_iter=1000, dt=0.005, c_list=np.linspace(-6,0,50), c_fixed=0, c_dir='x') # to generate report plots

