import numpy as np
import parameters
from numpy.random import normal
from tqdm import tqdm
import matplotlib.pyplot as plt



def euler_maruyama_step(X,t,dt,W):
    
    #### Euler-Maruyama main step ####
    #
    # INPUT:
    # X = current point of the trajectory
    # t = current time of the state
    # dt = euler-maruyama discretization step
    # W = Brownian increments
    #
    # OUTPUT:
    # X = new point of the trajectory
    # t = new time of the state
    # entrance = boolean that states if the particle has entered the well or not
    
    X = X + parameters.u(X)*dt + parameters.sigma*np.sqrt(dt)*W # Euler-Maruyama increment formula
    t = t + 1
    entrance = False
        
    if (np.linalg.norm(X) < parameters.R): # i.e., if the particle has entered the well
        entrance = True
    
    return X,t,entrance


def euler_maruyama(X0,dt):
    
    #### Euler-Maruyama scheme ####
    #
    # INPUT:
    # X0 = initial point of the trajectory
    # dt = euler-maruyama discretization step
    #
    # OUTPUT:
    # entrance = boolean that states if the particle has entered the well or not
    
    T = parameters.T
    X = X0
    entrance = False
    t = 0
    
    while (t*dt<T and entrance == False):
        W = normal(size=2)
        X,t,entrance = euler_maruyama_step(X, t, dt, W)
    
    return entrance




def standard_monte_carlo(n_iter,dt):
    
    #### Standard Monte Carlo main function ####
    #
    # INPUT:
    # n_iter = cardinality of the Monte Carlo sample
    # dt = euler-maruyama discretization step
    #
    # OUTPUT:
    # result_list = list of MC estimations corresponding to each starting point
    
    print ('\n ---- MONTE CARLO METHOD ---- \n')
    
    result_list = []
    for i in range(parameters.X0_MC.shape[0]): # loop over all the starting points
        X0 = parameters.X0_MC[i,:]
        results = np.zeros(n_iter)
        for i in tqdm(range(n_iter), desc='X0=('+str(X0[0]) +','+ str(X0[1]) +')'):
            results[i] = euler_maruyama(X0,dt)
    
        print ('X0=('+str(X0[0]) +','+ str(X0[1]) +'): ', results.mean() , '+-', parameters.Za*results.std()/np.sqrt(n_iter), end='\n')
        result_list.append(results.mean())
    return result_list
        
        
        
def multiple_predictions(n_iter,dt,ind_list,ind_type):
    
    #### To be used to compare results varying n_iter or dt ####
    #
    # INPUT:
    # n_iter = cardinality of the Monte Carlo sample
    # dt = euler-maruyama discretization step
    # ind_list = list of n_iter or dt (w.r.t. the parameter the user wants to vary)
    # ind_type = 'iter' or 'time', the parameter the user wants to vary

    
    results = np.zeros([parameters.X0_MC.shape[0],len(ind_type)])
    for i,q in enumerate(ind_list):
        
        print ('\nSIMULATION ', i+1,'/',len(ind_list))
        if (ind_type == 'iter'):
            results[:,i] = standard_monte_carlo(q, dt)
        elif (ind_type == 'time'):
            results[:,i] = standard_monte_carlo(n_iter,q)
            
            
            
def convergence_study_dt(n_iter, dt_ref, n_halfs):
    
    #### To be used to study the convergence order of the Euler-Maruyama w.r.t dt ####
    #
    # INPUT:
    # n_iter = cardinality of the Monte Carlo sample
    # dt_ref = smallest dt to be adopted
    # n_halfs = number of times for which dt_ref is divided by 2
    
    
    halfs = np.array(range(n_halfs)) # a vector for the halvings
    results = np.zeros([n_halfs])
    X0 = parameters.X0_IS # for this study, we only use the starting point X0 = (2.5, 2.5)
    dt = dt_ref / pow(2,halfs) # vector with the discretization steps
    true_value = 0.0626 # almost exact value obtained from Finite Element method
    T = parameters.T
    
    for i in tqdm(range(n_iter)):
        X = np.tile(X0, (n_halfs,1))
        t = np.zeros([n_halfs])
        entrance = np.zeros([n_halfs])
        t_ref = 0 # the refence time for all the trajectories (it coincides with the time of the finest trajectory)
        counter = True
        while (counter == True): # one single Monte Carlo iteration is stopped when all the trajectories are stopped
            counter = False
            for j in range(n_halfs):
                W = normal(size=2) # important aspect: the Brownian increments are the same 
                if (t_ref % 2**halfs[n_halfs - j - 1] == 0 and t[j]*dt[j]<T and entrance[j] == False): # i.e. the standard conditions + the condition related to the number of halvings
                    X[j,:],t[j],entrance[j] = euler_maruyama_step(X[j,:],t[j],dt[j],W)
                    counter = True
            t_ref = t_ref + 1
        results = results + entrance # we sum the result of each MC iteration to compute later the mean
    results = results/n_iter # it is now the vector with the MC estimations for all the halvings
    plt.figure()
    plt.loglog(dt, abs(results-true_value), label='error')
    err0 = abs(results[0]-true_value)
    plt.loglog(dt,err0/pow(dt[0],0.5) * pow(dt,0.5),label='$\Delta t^{1/2}$')
    plt.loglog(dt,err0/dt[0]*dt,label='$\Delta t$')
    plt.loglog(dt,err0/pow(dt[0],2) * pow(dt, 2),label='$\Delta t^2$')
    plt.xlabel('$\Delta t$')
    plt.ylabel('$|p_{est}- p_{true}|$')
    plt.title('Convergence of Euler-Maruyama scheme')
    plt.legend()
    plt.savefig("./figures/euler-maruyama-convergence")
    
    

            
            
        
if __name__ == '__main__':
    standard_monte_carlo(n_iter = 5000,dt=0.0001)
    #multiple_predictions(n_iter=50000,dt=0.0005,ind_list=[0.01,0.001],ind_type='time')
    #convergence_study_dt(n_iter=500000, dt_ref=1, n_halfs=5)
    


