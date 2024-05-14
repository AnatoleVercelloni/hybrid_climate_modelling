import numpy as np
from numpy import arcsin, arctan, tan, sqrt, cross
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import numba
jit=numba.jit(nopython=True, parallel=True)
from numba import njit

############### spherical primitives #################


@njit
def dotprod(a, b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
@njit
def norm2(a): return a[0]*a[0]+a[1]*a[1]+a[2]*a[2]
@njit
def norm(a): return np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
@njit
def unit_vector(a): return a/norm(a)
@njit
def arc(a, b): return 2.*np.arcsin(.5*norm(a-b))

@jit
def cross_jit(a,b, c):
    c[0]=a[1]*b[2]-b[1]*a[2]
    c[1]=a[2]*b[0]-b[2]*a[0]
    c[2]=a[0]*b[1]-b[0]*a[1]

def area(a,b,c):
# http://mathworld.wolfram.com/LHuiliersTheorem.html
    a,b,c = .25*arc(b,c), .25*arc(c,a), .25*arc(a,b)
    area = sqrt(tan(a+b+c)*tan(a+b-c)*tan(b+c-a)*tan(a+c-b))
#    print("area(a,b,c):",a,b,c, a+b-c, b+c-a, a+c-b, area)
    return 4.*arctan(area)

def areas(points, simplices):
    for simplex in simplices:
        a,b,c = points[simplex,:]
        yield area(a,b,c)
    
def circum(A,B,C):
# see Dubos et al., GMD 2015, appendix
# the computation is stable, and valid whether a,b,c is clockwise or not
    A,B,C=map(unit_vector, (A,B,C))
    b,c = B-A, C-A
    p1,p2 = cross(b,c), norm2(c)*b-norm2(b)*c
    p = A+.5*cross(p1,p2)/norm2(p1)
    p = p/norm(p)
    a,b,c=norm(p-A),norm(p-B),norm(p-C)
    if abs(a-b)>1e-15:
        print("Check circum", a-b, b-c, a-c)
        raise(ValueError)
    return p

@jit
def circum_fast(simp, points):
    """ simp : int[M,3] describing M simplices"""
    M=simp.shape[0]
    circ = np.zeros((M,3))
    p1, p2, p = np.zeros(3), np.zeros(3), np.zeros(3)
    for n in range(M):
        i,j,k = simp[n,:]
        A,B,C = points[i,:], points[j,:], points[k,:]
        b,c = B-A, C-A
        cross_jit(b,c, p1)
        p2 = norm2(c)*b-norm2(b)*c
        cross_jit(p1,p2, p)
        p = A+p*(.5/norm2(p1))
        circ[n,:]=p*(1./norm(p))
    return circ

def barycenters_fast(regions_size, vert, points, vertices, weight):
    # print("== 4 x number of edges =%d"%len(vert))
    N = points.shape[0]
    bary = np.zeros((N,3))
    i=0 # running index in vert1, vert2
    A=0.
    for n in range(N):
        point = points[n,:]
        b = 0.
        for j in range(regions_size[n]):
            v1 = vertices[vert[i],:]
            i = i+1
            v2 = vertices[vert[i],:]
            i = i+1
            dA=area(point, v1, v2)
            A = A+dA
            bb = unit_vector(point+v1+v2)
            b = b + dA*bb*weight(bb)
        bary[n,:] = unit_vector(b)
    print("== Total area = 4*pi+%f"%(A-4*np.pi)) 
    return bary
        
def barycenters(regions, points, edges, vertices):
    N = points.shape[0]
    bary = np.zeros((N,3))
    A=0.
    for n in range(N):
        point, region = points[n,:], regions[n] # generator and Voronoi region
        b = 0.
        for edge in region:
            (n1,p1), (n2,p2) = edges[edge]
            v1, v2 = vertices[n1,:], vertices[n2,:]
            dA=area(point, v1, v2)
            if N<10 :
                print("n, n1, n2", n,n1,n2)
                print("point,v1,v2,dA=",point,v1,v2,dA)
            A = A+dA
            bb = unit_vector(point+v1+v2)
            b = b + dA*bb
        bary[n,:] = unit_vector(b)
    if N<100: print("==== Total area = 4*pi+%f"%(A-4*np.pi))
    return bary

def plot_mesh(points,edges,orient, color):
    xy=points[:,0:2]
    segments=[ xy[[i, j],:] for i,j in edges if dotprod(orient,points[i,:])>0. ]
    ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    ax.add_collection(LineCollection(segments, color=color))

