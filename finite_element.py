import fenics as f
import dolfin as d
import mshr as m
import numpy as np
import parameters
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D #even if the warning, keep it, it's needed for 3d plot
from matplotlib import cm
import matplotlib.pyplot as plt


def print_message(idx):
    
    #### A function for multiple printing messages ####
    
    if (idx==0):
        print ('\n ---- FINITE ELEMENT METHOD ---- \n')
    elif (idx==1):
        print("\r{0}".format("Mesh generation:                               started ----> "), end='')
    elif (idx==2):
        print("completed")
    elif (idx==3):
        print("\r{0}".format("Functions and BC definition:                   started ----> "), end='')
    elif (idx==4):
        print("\n\n\r{0}".format("Post-processing:                               started ----> "), end='')
        
        
def post_processing(u,mesh):
    
    #### Post-processing: 3D plot, 2D profile plot, mesh plot ####
    #
    # INPUT:
    # u = Finite Element solution
    # mesh = Finite Element mesh

    
    print_message(4)
    n_points = 200
    
    #------ 3D solution plot -------
    
    x = np.linspace(-5*parameters.R,5*parameters.R,n_points)
    y = np.linspace(-5*parameters.R,5*parameters.R,n_points)
    
    x,y = np.meshgrid(x,y) # creation of the 3D plot grid
    x = x.ravel()
    y = y.ravel()
    z = np.zeros(x.shape)
    inner_list = [] # the list of grid points inside the well region
    for i in range (len(x)):
        if (x[i]**2 + y[i]**2 >= parameters.R**2): # i.e. if the point is outside the well region
            P = f.Point(x[i],y[i]) 
            z[i] = u(P) # z is the evaluation of u(x,y)
        else:
            inner_list.append(i) 
    
    x = np.delete(x,inner_list) # we delete every coordinate corresponding to points inside the well
    y = np.delete(y,inner_list) 
    z = np.delete(z,inner_list)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_title('PDE solution at t=T')
    plt.xlim([-5*parameters.R,5*parameters.R])
    plt.ylim([-5*parameters.R,5*parameters.R])
    plt.savefig("./figures/u_3d")
    
    #--------------------------------
    
    #------ 2D solution plot --------
    
    xl = np.linspace(-5*parameters.R,-parameters.R,n_points) # i.e. x points at the left of the well
    xr = np.linspace(parameters.R,5*parameters.R,n_points) # i.e. x points at the right of the well
    zl = np.zeros(xl.shape)
    zr = np.zeros(xr.shape)
    for i in range (len(xl)):
        P = f.Point(xl[i],0) # every z is the evaluation of u(x,y=0), discerning left from right
        zl[i] = u(P)
        P = f.Point(xr[i],0)
        zr[i] = u(P)
    
    q = np.array([[-parameters.R, -parameters.R, parameters.R, parameters.R, -parameters.R], [0, 1, 1, 0, 0]]) # the array that will plot the square well
    plt.figure()
    ax = plt.axes()
    plt.plot(xl,zl, 'tab:blue')
    plt.plot(xr,zr, 'tab:blue')
    plt.plot(q[0,:],q[1,:], 'tab:red')
    plt.xlabel('x')
    plt.ylabel('u(x,0,T)')
    ax.set_title('PDE solution (2D profile) at t=T');
    plt.savefig("./figures/u_2d")
    #--------------------------------
    
    #--------- Mesh plot -----------
    plt.figure()
    d.plot(mesh,title="Finite Element mesh")
    plt.xlim([0,2*parameters.R])
    plt.ylim([0,2*parameters.R])
    plt.savefig("./figures/mesh")
    #-------------------------------
    print_message(2)
        
    
    

def probability_print (u):
    
    #### To be used to print on screen the probability estimations ####
    #
    # INPUT:
    # u = Finite Element solution
    
    for i in range(parameters.X0_FE.shape[0]):
        X0 = parameters.X0_FE[i,:]
        print ('X0=('+str(X0[0]) +','+ str(X0[1]) +'): ', u(X0))

    
    
    
        

def pde_resolution(mesh_size,num_steps,max_length):
    
    #### Finite Element main function ####
    #
    # INPUT:
    # mesh_size = number of mesh elements along one domain diagonal (it is not the total number of elements)
    # num_steps = number of time steps for the temporal discretization
    # max_length = radius of the domain, should be high enough to approximate the infinite domain
    
    print_message(0)
    
    
    #-- preliminaries --
    f.set_log_level(f.LogLevel.ERROR) # to print only error messages (set 13 to print full run information)
    T = parameters.T
    dt = T / num_steps # discretization time step length
    
    #-- mesh generation --
    print_message(1)
    D = m.Circle(f.Point(0,0),max_length) 
    B = m.Circle(f.Point(0,0),parameters.R)# i.e. the well
    domain = D  - B # the full domain is the big ball without the well
    mesh = m.generate_mesh(domain,mesh_size)
    print_message(2)
    
    
    print_message(3)
    #-- definition of the functional space --
    V = f.FunctionSpace(mesh, 'P', parameters.poly_order) # by default, first order polynomials
    
    #-- boundary conditions --
    infinite_distance = 'on_boundary && pow(x[0],2)+pow(x[1],2) > 2*pow(' + str(parameters.R) +',2)'
    cylinder = 'on_boundary && pow(x[0],2)+pow(x[1],2) < 2*pow(' + str(parameters.R) + ',2)'
    bc_infinite = f.DirichletBC(V,f.Constant(0),infinite_distance) # homogeneous dirichlet on the infinite distance
    bc_cylinder = f.DirichletBC(V,f.Constant(1),cylinder) # non-homogeneous dirichlet on the well border
    bc = [bc_infinite,bc_cylinder]
    
    #-- test functions --
    u_n = f.Expression('0',degree=1) # it is used to return the solution at the previous temporal step in the time discretization, initialized at 0 because of the initial conditions
    u_n = f.interpolate(u_n,V) 
    u = f.TrialFunction(V) # the FE solution
    v = f.TestFunction(V) # the FE test function
    
    #-- definition of expression needed in variational form --
    U = f.Expression(('1. + Q/(2*pi*(pow(x[0],2) + pow(x[1],2))) * x[0]', 'Q/(2*pi*(pow(x[0],2) + pow(x[1],2))) * x[1]'), degree=1, Q=parameters.Q, pi=np.pi) # the water flow function
    
    #-- definition of variational problem --
    F = u*v*f.dx - u_n*v*f.dx - dt*f.dot(U,f.grad(u))*v*f.dx + 0.5* parameters.sigma**2*dt*f.dot(f.grad(u),f.grad(v))*f.dx # weak formulation, see report for the derivation
    a, L = f.lhs(F), f.rhs(F)
    print_message(2)
    
    

    print ('\n\nSpace polynomial order: ', parameters.poly_order, '    Number of mesh nodes: ', mesh.num_vertices(),'\n\n')
    
    
    #-- main calculation steps --
    print ("FE system solving ("+ str(num_steps) +" temporal steps):")
    u = f.Function(V)
    t = 0
    for n in tqdm(range (num_steps)):
        
        t += dt # time increment
        f.solve(a==L,u,bc,solver_parameters={'linear_solver':'mumps'}) # F.E. system solving
        u_n.assign(u) # assign u to u_n for the next step
        
    print ('\n FINITE ELEMENT solving completed!\n')
    
    return u,mesh



def independence_study(mesh_size, num_steps, max_length, ind_list, ind_type):
    
    #### To be used to compare multiple solutions with different parameters (chosen from mesh_size, num_steps, max_length) ####
    #
    # INPUT:
    # mesh_size = number of mesh elements along one domain diagonal (it is not the total number of elements)
    # num_steps = number of time steps for the temporal discretization
    # max_length = radius of the domain, should be high enough to approximate the infinite domain
    # ind_list = list of parameter values
    # ind_type = 'grid' or 'time' or 'max-distance', the kind of parameter that corresponds to ind_list (i.e. the parameter the user wants to vary)
    
    # ------- 2D plot -------
    n_points = 200
    x =np.linspace(parameters.R,5*parameters.R,n_points) # the comparison plot is made on the 2D profile of the solution at the right of the well
    z = np.zeros(x.shape)
    plt.figure()
    ax = plt.axes()
    
    for i,q in enumerate(ind_list):
        
        print ('SIMULATION ', i+1,'/',len(ind_list))
        if (ind_type == 'grid'):
            u,mesh = pde_resolution(q, num_steps, max_length)
        elif (ind_type == 'time'):
            u,mesh = pde_resolution(mesh_size, q, max_length)
        elif (ind_type == 'max-distance'):
            u,mesh = pde_resolution(mesh_size, num_steps, q)
        
        probability_print(u)
            
        for j in range (len(x)):
            P = f.Point(x[j],0)
            z[j] = u(P)
            
        if (ind_type == 'grid'):
            plt.plot(x,z, label=str(mesh.num_vertices())+' nodes')
        elif (ind_type == 'time'):
            plt.plot(x,z, label=str(q)+' time steps')
        elif (ind_type == 'max-distance'):
            plt.plot(x,z, label=str(q/parameters.R)+' R')

        if (i==0):
            u_first = u
        elif(i==len(ind_list)-1):
            u_last = u

    plt.xlabel('x')
    plt.ylabel('u(x,0,T)')
    plt.legend()
    ax.set_title('FE '+ind_type+'-independence (t=T)');
    plt.savefig("./figures/u_"+ind_type+"_independence")
    
    
    #------- err_inf -------
    x = np.linspace(-5*parameters.R,5*parameters.R,n_points)
    y = np.linspace(-5*parameters.R,5*parameters.R,n_points)
    x,y = np.meshgrid(x,y)
    x = x.ravel()
    y = y.ravel()
    z_first = np.zeros(x.shape)
    z_last = np.zeros(x.shape)
    for i in range (len(x)):
        if (x[i]**2 + y[i]**2 >= parameters.R**2):
            P = f.Point(x[i],y[i])
            z_first[i] = u_first(P)
            z_last[i] = u_last(P)
    
    print ('\nErr_inf between first and last: ',np.linalg.norm(z_first-z_last, ord=np.inf),'\n')
    
    
    
def save_result (u,mesh):
    #### To save locally results from Finite Element Method ####
    #
    # INPUT:
    # u = Finite Element solution
    # mesh = Finite Element mesh

    mesh_file = d.File("./files/mesh.xml")
    mesh_file << mesh
    u_file = d.HDF5File(d.MPI.comm_world,"./files/f.h5","w")
    u_file.write(u,"/f")
    u_file.close()



def load_result():
    #### To load from local memory results there were saved with save_result.py ####
    #
    # OUTPUT:
    # u = Finite Element solution
    # mesh = Finite Element mesh
    
    mesh = d.Mesh('./files/mesh.xml')
    V = f.FunctionSpace(mesh, 'P', parameters.poly_order)
    u = f.Function(V)
    u_file = d.HDF5File(d.MPI.comm_world,"./files/f.h5","r")
    u_file.read(u,"/f")
    u_file.close()
    return u,mesh
    

if __name__ == '__main__':
    u,mesh = pde_resolution(mesh_size=600,num_steps=800,max_length=40)
    post_processing(u, mesh)
    probability_print(u)
    save_result(u,mesh)
    
    #independence_study(mesh_size=600, num_steps=800, max_length=40, ind_list=[300,424,520,600], ind_type='grid')
    #independence_study(mesh_size=600, num_steps=800, max_length=40, ind_list=[200,400,600,800], ind_type='time')
    #independence_study(mesh_size=600, num_steps=800, max_length=40, ind_list=[20,30,40], ind_type='max-distance')
    
    
    
    

