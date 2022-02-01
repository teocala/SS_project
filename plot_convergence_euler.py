# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 09:10:15 2022

@author: matte
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.1 / pow(2,np.array(range(6)))
results = np.array([[0.034578, 0.040672, 0.04581,  0.049374, 0.052158, 0.056676],
                    [0.034464, 0.040008, 0.045496, 0.049572, 0.053712, 0.05699 ],
                    [0.034522, 0.040258, 0.045508, 0.04944,  0.05355,  0.057152],
                    [0.034426, 0.040542, 0.045512, 0.04987,  0.052486, 0.05651 ],
                    [0.034522, 0.040684, 0.046106, 0.050118, 0.052388, 0.056962]])
plt.figure()
plt.loglog(dt, 1/abs(results[0,0] - 0.0626) * abs(results[0,:] - 0.0626), 'bx-', linewidth=0.5, label='errors')
plt.loglog(dt, 1/abs(results[1,0] - 0.0626) * abs(results[1,:] - 0.0626), 'bx-','tab:blue', linewidth=0.5)
plt.loglog(dt, 1/abs(results[2,0] - 0.0626) * abs(results[2,:] - 0.0626), 'bx-','tab:blue', linewidth=0.5)
plt.loglog(dt, 1/abs(results[3,0] - 0.0626) * abs(results[3,:] - 0.0626), 'bx-','tab:blue', linewidth=0.5)
plt.loglog(dt, 1/abs(results[4,0] - 0.0626) * abs(results[4,:] - 0.0626), 'bx-','tab:blue', linewidth=0.5)
plt.loglog(dt,1/pow(dt[0],0.5) * pow(dt,0.5), 'tab:red', label='$\Delta t^{1/2}$')
plt.loglog(dt,1/dt[0]*dt, 'tab:green', label='$\Delta t$')
plt.loglog(dt,1/pow(dt[0],2) * pow(dt, 2), 'tab:orange', label='$\Delta t^2$')
plt.xlabel('$\Delta t$')
plt.ylabel('$|p_{est}- p_{true}|$')
plt.title('Convergence of Euler-Maruyama scheme')
plt.legend()
plt.savefig("./euler-maruyama-convergence")