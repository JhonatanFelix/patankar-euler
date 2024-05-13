import numpy as np
import matplotlib.pyplot as plt


def explicit_euler(pos_funcs, neg_funcs, times, init_cond):
    """ 
    Explicit Euler method for differential equations with separate positive and negative terms.
    
    Parameters:
        pos_funcs (list): List of positive-valued functions. Each function must take two arguments: time t and state x.
        neg_funcs (list): List of negative-valued functions. Each function must take two arguments: time t and state x.
        times (list or array): Array of time points at which to solve the ODE.
        initial_condition (list): List of initial values for each state variable.
    
    Returns:
        numpy.ndarray: Array of solution values at each time in 'times'.
    """
    # Initialize the array to store state vectors over time
    x = np.zeros((len(init_cond), len(times)))
    x[:, 0] = init_cond  # Set the initial condition for all state variables at the first time point

    # Time step size
    h = times[1] - times[0]  
    
    # Euler method to update x for each time step
    for i in range(1, len(times)):
        t = times[i-1]
        x_current = x[:, i-1]
        positive_increment = sum(f(t, x_current) for f in pos_funcs)
        negative_increment = sum(f(t, x_current) for f in neg_funcs)
        
        x[:, i] = x_current + h * (positive_increment - negative_increment)
    
    return x



def euler_maruyama(pos_funcs, neg_funcs, diffusion_funcs ,times, init_cond):

    x = np.zeros((len(init_cond), len(times)))
    x[:,0] = init_cond

    h = times[1] - times[0]

    for i in range(1, len(times)):
        t = times[i-1]
        x_current = x[:, i-1]
        
        #positive_increment = np.zeros(i)
        #negative_increment = np.zeros(i)
        #diffusion_increment = np.zeros(i)

        positive_increment = sum(f(t, x_current) for f in pos_funcs)
        negative_increment = sum(d(t, x_current) for d in neg_funcs)
        diffusion_increment = sum(g(t,x_current) for g in diffusion_funcs)

        I = np.random.normal(0 , h, len(init_cond))

        x[:,i] = x_current + positive_increment - negative_increment + diffusion_increment*I
    
    return x


def deterministic_patankar_euler(pos_funcs, neg_funcs, diffusion_funcs ,times, init_cond):

    x = np.zeros((len(init_cond), len(times)))
    x[:,0] = init_cond

    h = times[1] - times[0]

    for i in range(1, len(times)):
        t = times[i-1]
        x_current = x[:, i-1]
        
        #positive_increment = np.zeros(i)
        #negative_increment = np.zeros(i)
        #diffusion_increment = np.zeros(i)

        positive_increment = sum(f(t, x_current) for f in pos_funcs)
        negative_increment = sum(d(t, x_current) for d in neg_funcs)
        diffusion_increment = sum(g(t,x_current) for g in diffusion_funcs)
        neg_over_x = sum(d(t,x_current)/x_current for d in diffusion_funcs) 

        I = np.random.normal(0 , h, len(init_cond))

        x[:,i] = (x_current + positive_increment*h + diffusion_increment*I)/(1+ neg_over_x) 
    
    return x


def stochastic_patankar_euler(pos_funcs, neg_funcs, diffusion_funcs ,times, init_cond):

    x = np.zeros((len(init_cond), len(times)))
    x[:,0] = init_cond

    h = times[1] - times[0]

    for i in range(1, len(times)):
        t = times[i-1]
        x_current = x[:, i-1]
        
        #positive_increment = np.zeros(i)
        #negative_increment = np.zeros(i)
        #diffusion_increment = np.zeros(i)

        positive_increment = sum(f(t, x_current) for f in pos_funcs)
        #negative_increment = sum(d(t, x_current) for d in neg_funcs)
        #diffusion_increment = sum(g(t,x_current) for g in diffusion_funcs)
        neg_over_x = sum(d(t,x_current)/x_current for d in neg_funcs) 
        diff_over_x = sum(g(t,x_current)/x_current for g in diffusion_funcs)


        I = np.random.normal(0 , h, len(init_cond))

        x[:,i] = (x_current + positive_increment*h )/(1+ neg_over_x + neg_over_x - diff_over_x
                                                    + (diff_over_x)**2 ) 
    
    return x





## I noticed that I'm doing the functions to calculate EVERY term of the methods, and not only the 
## next ones. I'll change it right now.


def euler_maruyama_term(pos_funcs, neg_funcs, diffusion_funcs ,time, h, x_current):
    '''
    time now is only a value
    h is the timestep
    '''
    t = time

    positive_increment = sum(f(t, x_current) for f in pos_funcs)
    negative_increment = sum(d(t, x_current) for d in neg_funcs)
    diffusion_increment = sum(g(t,x_current) for g in diffusion_funcs)

    I = np.random.normal(0 , h, len(x_current))

    x_current = x_current + positive_increment*h - negative_increment*h + diffusion_increment*I
    
    return x_current


def deterministic_patankar_euler_term(pos_funcs, neg_funcs, diffusion_funcs ,time, h,  x_current):

    t = time
        
    positive_increment = sum(f(t, x_current) for f in pos_funcs)
    negative_increment = sum(d(t, x_current) for d in neg_funcs)
    diffusion_increment = sum(g(t,x_current) for g in diffusion_funcs)
    neg_over_x = sum(d(t,x_current)/x_current for d in diffusion_funcs) 

    I = np.random.normal(0 , h, len(x_current))

    x_current= (x_current + positive_increment*h + diffusion_increment*I)/(1+ neg_over_x*h) 
    
    return x_current


def stochastic_patankar_euler_term(pos_funcs, neg_funcs, diffusion_funcs ,time, h, x_current):

    t = time

    positive_increment = sum(f(t, x_current) for f in pos_funcs)
    neg_over_x = sum(d(t,x_current)/x_current for d in neg_funcs) 
    diff_over_x = sum(g(t,x_current)/x_current for g in diffusion_funcs)


    I = np.random.normal(0 , h, len(x_current))

    x_current = (x_current + positive_increment*h )/(1+ neg_over_x *h - diff_over_x
                                                + (diff_over_x)**2 ) 
    
    return x_current



def composite_euler_method(pos_funcs, neg_funcs, diffusion_funcs, times, init_cond):    
    
    h = times[1] - times[0]
    x_current = init_cond
    x = np.zeros((len(times), len(init_cond)))


    for i in range(len(times)):
        time = times[i]

        dpet = deterministic_patankar_euler_term(pos_funcs, neg_funcs, diffusion_funcs ,time, h,  x_current)
        spet = stochastic_patankar_euler_term(pos_funcs, neg_funcs, diffusion_funcs ,time, h, x_current)

        if np.all(dpet > 0):
            x[i] = dpet
        else: 
            mask = dpet < 0
            dpet[mask] = spet[mask]
            x[i] =dpet

    return x
    

def fully_composite_euler_method(pos_funcs, neg_funcs, diffusion_funcs, times, init_cond):    
    
    h = times[1] - times[0]
    x = np.zeros((len(times), len(init_cond)))
    x_current = init_cond

    for i in range(len(times)):

        time = times[i]

        emt = euler_maruyama_term(pos_funcs, neg_funcs, diffusion_funcs, time, h, x_current)
        dpet = deterministic_patankar_euler_term(pos_funcs, neg_funcs, diffusion_funcs ,time, h,  x_current)
        spet = stochastic_patankar_euler_term(pos_funcs, neg_funcs, diffusion_funcs ,time, h, x_current)

        if np.all(emt >0):
            x[i] = emt
        elif np.all(dpet > 0):
            mask = emt < 0
            emt[mask] = dpet[mask]
            x[i] = emt
        else:
            mask1 = emt < 0
            emt[mask1] = dpet[mask1]
            mask2 = emt < 0
            emt[mask2] = spet[mask2]
            x[i] = emt      

    return x