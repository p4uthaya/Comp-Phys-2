# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:58:20 2018

@author: Prabha
"""
# Assignment 5 
# ex. 5.10 
from gaussxw import gaussxw
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# potetital function
def V(x):
    v = x*x*x*x
    return v
# integrand function
def f(x,b):
    f = 1/np.sqrt(V(b)-V(x))
    return f 

# finding T  
Tlist = []


for i in np.arange(0.001, 2, 0.02):
    N = 20
    a = 0
    b = i
    m = 1
    
    x,w = gaussxw(N)
    xp = 0.5*(b-a)*x + 0.5*(b+a)
    wp = 0.5*(b-a)*w
    
    T = 0
    for k in range(N):
        T += wp[k]*f(xp[k],b)   
    Tlist.append(np.sqrt(8*m)*T)

x = np.arange(0.01,2,0.02)
print(Tlist[-1],x[-1])
plt.figure()
plt.plot(x,Tlist)
plt.xlabel('Amplitude (a)')
plt.ylabel('Period (T)')
plt.title('Period vs Amplitude')
    

# exercise 5.13
#Part a
def H(n,x):
    h = [1,2*x]
    for i in range (2,n+1):
        hn = 2*x*h[i-1]-2*(i-1)*h[i-2]
        h.append(hn)
    if n == 0 :
        return h[0]
    if n == 1:
        return h[1]
    else:
        return h[n]

print (H(3,1))

def Y(n,x):
    y = (1/(np.sqrt((2**n)*(factorial(n))*np.sqrt(np.pi))))*np.exp((-x*x)/2)*H(n,x)
    return y

x = np.arange(-4,4,0.01)

plt.figure()
for i in range(0,4):
    plt.plot(x,Y(i,x),label = "n = %d" %i)

plt.ylabel('$\Psi$(x)')
plt.xlabel('x')
plt.title('Harmonic Oscillator Wavefunctions')
plt.legend()

# part b
x2 = np.arange(-10,10,0.01)
plt.figure()
plt.plot(x2,Y(30,x2))
plt.ylabel('$\Psi$(x)')
plt.xlabel('x')
plt.title('Harmonic Oscillator Wavefunction for n = 30')

#part c
def f(n,x):
    f = x*x*Y(n,x)*Y(n,x)
    return f
def fz(n,z):
    fz = ((1+z*z)/((1-z*z)*(1-z*z)))*f(n,(z/(1-z*z)))
    return fz

N = 100
a = -1
b = 1

x,w = gaussxw(N)
xp = 0.5*(b-a)*x + 0.5*(b+a)
wp = 0.5*(b-a)*w

u = 0
for i in range(N):
    u += wp[i]*fz(5,xp[i])
    
print(np.sqrt(u))




        
        