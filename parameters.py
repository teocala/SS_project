import numpy as np
import scipy.stats as st

####### MAIN PARAMETERS FILE ########

X0_MC = np.array([[1.2, 1.1],[3.0,4.0],[2.5,2.5]])  # starting points to be used in standard Monte Carlo method
X0_FE = np.array([[1.2, 1.1],[3.0,4.0],[2.5,2.5],[7.0,7.0]]) # starting points to be used in Finite Element method
X0_IS = np.array([2.5,2.5]) # starting point to be used in Importance Sampling method
X0_SM = np.array([7.0,7.0]) # starting point to be used in Splitting Method
sigma = 2 # diffusion of the porous media
Q = 1 # extraction mass rate
R = 1 # well radius
T = 1 # time horizon
u = lambda X: np.array([1.,0.]) + (Q/(2*np.pi*(np.power(X[0],2) + np.power(X[1],2))))*X # water flow function
poly_order = 1 # by default, Finite Element method is used with first order polynomials
Za = st.norm.ppf(1 - 0.01 / 2) # multiplicative constant for confidence intervals