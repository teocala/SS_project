import numpy as np
import parameters
from numpy.random import normal
from tqdm import tqdm
import random
import standard_monte_carlo.euler_maruyama_step as euler_maruyama_step

def euler_maruyama(X0,dt,R): 
    
    #### Variant of the homonymous standard_monte_carlo.py function ####
    #
    # INPUT:
    # X0 = starting point of the trajectory
    # dt = euler-maruyama discretization step
    # R = radius that corresponds to the entrance region (here, for the splitting method, flexible)
    #
    # OUTPUT:
    # entrance = boolean that states if the particle has entered the well or not
    # np.append(X,t) = 3D vector that contains the location and time of the arrival state
    
    T = parameters.T
    X = X0[0:2]
    entrance = False
    t = X0[2]
    
    while (t*dt<T and entrance == False):
        
        W = normal(size=2)
        X,t,entrance = euler_maruyama_step(X, t, dt, W)
    
    return entrance, np.append(X,t)


def splitting_method(n_iter,dt,n_levels):
    
    #### Splitting method main function ####
    #
    # INPUT:
    # n_iter = cardinality of Monte Carlo samples
    # dt = euler-maruyama discretization step
    # n_levels = number of splitting method levels (optimal choice = -ln(true_value)/2 )
    #
    # OUTPUT:
    # p = estimated probability
    
    print ('\n ---- SPLITTING METHOD ---- \n')
    p = 1.0
    X0 = np.append(parameters.X0_SM, 0) # 3D vector: x,y,t
    starting_points = np.tile(X0, (n_iter,1))  # n_iter copies of X0
    dR = (np.linalg.norm(X0)-parameters.R)/(n_levels) # radius constant increment
    R = np.linalg.norm(X0) - dR # initial radius
    
    for l in tqdm(range(n_levels)): # loop over all levels
        p_l = np.zeros(n_iter) # initialization of the probabiliy estimations at level l
        starting_points_next = [] # the list that will be used in the next level as starting points
        for X_idx, X_l in enumerate(starting_points):
            entrance, X_final = euler_maruyama(X_l, dt, R)
            if (entrance==True):
                p_l[X_idx] = 1
                starting_points_next.append(X_final) # every state that reaches the next level is added to the starting point list
        p = p * p_l.mean() # main splitting method passage: product of conditional probabilities
        R = R - dR # update the radius for the new level
        print(" Level probability = ", p_l.mean())
        starting_points = random.choices(starting_points_next,k=n_iter) # to sample with repetitions
    
    print ('X0 = ('+str(X0[0]) +','+ str(X0[1]) +'): ', p)
        
    return p
        
        

            
        
if __name__ == '__main__':
    splitting_method(n_iter=10000,dt=0.0005,n_levels=8)
    
    


