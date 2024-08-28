from scipy.optimize import newton
from scipy.optimize import fsolve
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def plot_stability_region_multistep(a,b, xinter=[-3,0.5], yinter=[-3,3]):
    
    #a : coefficients
    nx, ny = (100, 200)
    x = np.linspace(xinter[0], xinter[1], nx)
    y = np.linspace(yinter[0], yinter[1], ny)
    xv, yv = np.meshgrid(x, y)
    A = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            z = x[i] +1j*y[j]
            ceval = [evalcoeff(z,a[k],b[k]) for k in range(len(a))]
            Z = np.roots(np.flip(np.asarray(ceval)))
            if np.any(np.abs(Z)>1):
                A[i,j] = -1 # no estable
            else:
                A[i,j] = 1
    plt.contourf(x, y, A.T, cmap='gnuplot')
    return A
def evalcoeff(z,ak,bk):
    return ak -z*bk

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
    
def Euler_implicito(odefun, t_span, y0, h, return_trajectory=False, odefunprime=None ): 
    
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
    
    if return_trajectory: yh = []; tn = []; yh.append(y0); tn.append(0.0)
    while tf-t>1e-14:
        # Paso de Euler implicito con iteracion de punto fijo
        y = paso_Euler_implicito(odefun, y, t, h)
        t+=h
        if return_trajectory: yh.append(y); tn.append(t)
            
    if return_trajectory: return (np.array(yh,dtype=object),np.array(tn,dtype=object)) 
    else: return y,t
from scipy.optimize import fsolve

def paso_Euler_implicito(odefun, y, t, h):
    F = lambda y_new: y_new - y - h*odefun(t+h, y_new)
    y_new = fsolve(F, y)

    return y_new
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
    if odefunprime is not None: J = lambda z,t0,y0: I - 0.5*h*odefunprime(t0+h,z)#*odefun(t0+h,z)
    else: J = None
    
    while tf-t>1e-14:
        # Paso de regla del trapecio
        y = newton(F, y0, fprime=J, args=(t,y))
        t+=h
        if return_trajectory: yh.append(y); tn.append(t)
            
    if return_trajectory: return (np.array(yh),np.array(tn)) 
    else: return y,t

def simpsonmult(odefun, t_span, y0, h=None, y1=None, return_trajectory=False, odefunprime=None):
    '''
    Metodo de dos pasos de regla de Simpson
    Resuelve: dy/dy = odefun(t,y), y(t0) = y0
    y_{n+1} = y_{n-1} + (h/3)(f_{n-1} + 4 f_{n} + f_{n+1})
    Input  : odefun, t_span = (tiemp0 inicial, tiempo final)
             y0 condicion inicial, h paso en t, 
             return_trajectory retornar o no toda la trajectoria de la solucion
    Output : (t,y)
    '''
    t0, tf = t_span
    t = t0; y = y0
    if h is None: h = (tf-t0)/100
    if return_trajectory: yh = []; th = []; yh.append(y0); yh.append(y1); th.append(t0); th.append(t0+h)
    if np.isscalar(y0): I = 1.0 
    else: I = np.eye(y0.size)
    
    F = lambda z,t0,y0,y1: z-(h/3.0)*(odefun(t0,y0)+4.0*odefun(t0+h,y1)+odefun(t0+2*h,z))-y0
    if odefunprime is not None: J = lambda z,t0,y0,y1: I-(h/3.0)*odefunprime(t0+2*h,z)#*odefun(t0+2*h,z)
    else: J = None
    
    ynm1=y0; yn=y1; t=t0
    while tf-t>1e-14:
        ynp1 = newton(F, x0=yn+h*odefun(t+h,yn), fprime=J, args=(t,ynm1,yn))
        t += h
        if return_trajectory: yh.append(ynp1); th.append(t+h)
        ynm1 = yn; yn = ynp1
    if return_trajectory: return np.asarray(yh), np.asarray(th)
    else: return yn, t+h

class ButcherTableau:
    def __init__(self, A=None, b=None, c=None):
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)
        
    @classmethod
    def implicitmidpoint(cls):
        A = [[0.5]]
        c = [0.5]
        b = [1.0]
        return cls(A, b, c)
    
    @classmethod
    def IRK2(cls):
        A = [[0.25, (3.0-2.0*sqrt(3.0))/12.0],[(3.0+2.0*sqrt(3.0))/12.0, 0.25]]
        b = [0.5, 0.5]
        c = [(3.0-sqrt(3.0))/6.0, (3.0+sqrt(3.0))/6.0]
        return cls(A, b, c)
    
    

def IRK(odefun, t_span, y0, h, return_trajectory, ButcherTab=None):
    
    # Butcher Tableau
    if ButcherTab is None: ButcherTab = ButcherTableau.implicitmidpoint()

    # time span
    t0, tfinal = t_span
    t_values = np.arange(t0, tfinal + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    # Set the initial condition
    y_values[0] = y0
    y = y0
    
    # Time stepping
    t = t0
    for i in range(1, len(t_values)):
        y = implicit_runge_kutta_step(odefun, y, t, h, ButcherTab)
        y_values[i] = y
    
    return t_values, y_values

def implicit_runge_kutta_step(f, y, t, h, ButcherTab):
    # parametros
    max_iter = 1000; tol = 1e-8
    
    # Butcher Tableau
    A = ButcherTab.A
    b = ButcherTab.b
    c = ButcherTab.c
    # Number of stages
    s = len(c)
    
    # Initialize k array to store the stages
    k = np.zeros((s, len(y)))
    k_new = np.zeros((s, len(y)))
    # Initial guesses for k (explicit Euler step)
    for j in range(s):
        k[j] = f(t + c[j]*h, y)
    
    for iteracion in range(max_iter):
        for j in range(s):
#             print(f(t + c[j]*h, y + h * (A[j,:].dot(k) ) ) )
            k_new[j] = f(t + c[j]*h, y + h * (A[j,:].dot(k)))
        if np.linalg.norm(k_new - k) < tol :
            break
    
        k= k_new
    print(iteracion)
    # Combine the stages to compute the next value of y
    y_new = y + h * b.dot(k)
    
    return y_new


def Euler_explicito_multivariate(f, y0, N, XM, x0=0):
    h = (XM- x0)/N
    yn = y0.copy()
    x = []
    y = []
    x.append(x0)
    y.append(y0)
    xn = x0

    for n in range(1,N+1):
        yn = yn + h * f(xn,yn)
        xn = x0+h*n
        x.append(xn)
        y.append(yn)
    return y, x
def Euler_implicito_multivariate(f, y0, N, XM, x0=0, dfdy=None):
    h = (XM- x0)/N
    yn = y0
    x = []
    y = []
    x.append(x0)
    y.append(y0)
    xn = x0
    
    if np.isscalar(y0):
        I = 1.0
    else:
        I = np.eye(y0.size)
    for n in range(1,N+1):
        F = lambda y: np.array([y[0] - yn[0] - h*f(xn+h,y)[0], y[1] - yn[1] - h*f(xn+h,y)[1]])
        if dfdy is None:
            Fprime = None
        else:
            Fprime = lambda y: I - h*dfdy(xn+h,y)
        yn,_ = newton_multivariate(F, X0=yn+h*f(xn,yn), Fprime=Fprime)
        xn = x0+h*n
        x.append(xn)
        y.append(yn)
    return y, x
def newton_multivariate(F, X0, Fprime, maxiter=50, tol=1e-8):
    i = 0
    error = 1.0
    X = X0
    
    while np.any(abs(error)>tol) and i <maxiter:
        funeval = F(X)
        Jac = Fprime(X)
        Xnew = X - np.linalg.solve(Jac, funeval)
        error = Xnew-X
        X = Xnew
        i += 1
    return X, F(X)

   