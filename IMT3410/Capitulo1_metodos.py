from scipy.optimize import newton
import numpy as np

def Euler_explicito(odefun, t_span, y0, h=None, return_trajectory=False): 
    '''
    Metodo de Euler explicito
    Resuelve: dy/dy = odefun(t,y), y(t0) = y0
    Input  : odefun, t_span = (tiemp inicial, tiempo final)
             y0 condicion inicial, h paso en t, 
             return_trajectory returnar o no toda la trajectoria de la solucion
    Output : (t,y)
    '''
    t0, tf = t_span
    t = t0; y = y0
    if h is None: h = (tf-t0)/100
    yh = []; tn = [] 
    if return_trajectory: yh.append(y0); tn.append(0.0)
    
    while tf-t>1e-14:
        # Paso de Euler explicito
        y = y + h*odefun(t,y)
        t+=h
        if return_trajectory: yh.append(y); tn.append(t)
    if return_trajectory: 
        return (np.array(yh),tn) 
    else: 
        return y,t
    
def Euler_implicito(odefun, t_span, y0, h=None, return_trajectory=False, odefunprime=None ): 
    '''
    Metodo de Euler implicito
    Resuelve: dy/dy = odefun(t,y), y(t0) = y0
    Input  : odefun, t_span = (tiemp inicial, tiempo final)
             y0 condicion inicial, h paso en t, 
             return_trajectory returnar o no toda la trajectoria de la solucion
    Output : (t,y)
    '''
    t0, tf = t_span
    t = t0; y = y0
    
    if h is None: h = (tf-t0)/100
    if return_trajectory: yh = []; tn = []; yh.append(y0); tn.append(0.0)
    if np.isscalar(y0): I = 1.0 
    else: I = np.eye(y0.size)
    
    F = lambda z,t0,y0: z-h*odefun(t0+h,z) - y0
    if odefunprime is not None: J = lambda z,t0,y0: I - h*odefunprime(t0+h,z)
    else: J = None
        
    while tf-t>1e-14:
        # Paso de Euler implicito
        y = newton(F, y0, fprime=J, args=(t,y))
        t+=h
        if return_trajectory: yh.append(y); tn.append(t)
            
    if return_trajectory: return (np.array(yh),np.array(tn)) 
    else: return y,t
def trapezoidalrule(odefun, t_span, y0, h=None, return_trajectory=False, odefunprime=None ):
    '''
    Metodo de regla del trapecio
    Resuelve: dy/dy = odefun(t,y), y(t0) = y0
    Input  : odefun, t_span = (tiemp inicial, tiempo final)
             y0 condicion inicial, h paso en t, 
             return_trajectory returnar o no toda la trajectoria de la solucion
    Output : (t,y)
    '''
    t0, tf = t_span
    t = t0; y = y0
    if h is None: h = (tf-t0)/100
    if return_trajectory: yh = []; tn = []; yh.append(y0); tn.append(0.0)
    if np.isscalar(y0): I = 1.0 
    else: I = np.eye(y0.size)
    
    F = lambda z,t0,y0: z-0.5*h*(odefun(t0,y0)+odefun(t0+h,z)) - y0
    if odefunprime is not None: J = lambda z,t0,y0: I - 0.5*h*odefunprime(t0+h,z)*odefun(t0+h,z)
    else: J = None
    
    while tf-t>1e-14:
        # Paso de regla del trapecio
        y = newton(F, y0, fprime=J, args=(t,y))
        t+=h
        if return_trajectory: yh.append(y); tn.append(t)
            
    if return_trajectory: return (np.array(yh),np.array(tn)) 
    else: return y,t