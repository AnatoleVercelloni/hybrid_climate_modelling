from scvt.delaunay import delaunay
from netCDF4 import Dataset as NetCDFFile 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits import mplot3d
from netCDF4 import Dataset as NetCDFFile 
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math



def get_xyz(filename, Ai=False):
    #take a netcdf file filename and return the cartesian coordinates
    #from the lon lat coordiantes

    nc = NetCDFFile(filename)
    #print(Ai)

    if Ai:
        lat = nc.variables['lat_i'][:]
        lon = nc.variables['lon_i'][:]
    else: 
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]


    nlat = lat*np.pi/180
    nlon = lon*np.pi/180

    x = np.cos(nlat) * np.cos(nlon)
    y = np.cos(nlat) * np.sin(nlon)
    z = np.sin(nlat)

    points = np.array([x, y, z]).T

    return points






def centroid(D, points, v, simplicies):

    if v not in D: return []
    poly = list(D[v])
  
    v_ = points[v]
    BARY = []
    a = v
    ref_point = points[poly[0][1]]
    Ls = []
    tmp = []
    #print("POLY", poly)
    for (b, c) in poly: #compute the barycentre for each triangle 

        if a != v: print ("erreur, sorted method failed !")


        na = points[a]
        nb = points[b]
        nc = points[c]
           

        ac = na - nc
        bc = nb - nc
        cr = np.cross(ac,bc)


        bary = nc + np.cross((np.linalg.norm(ac)**2*bc - np.linalg.norm(bc)**2*ac), cr)/(2*np.linalg.norm(cr)**2) 

        bary = bary/np.linalg.norm(bary)

        tmp.append((b,c, bary))


        #print("bary", bary)
            
        if np.abs(bary[0]**2 + bary[1]**2 + bary[2]**2 - 1) > 0.001: print("bary not on the sphere !")
            
        v1 = bary - v_
        v2 = ref_point-v_
        
            
        cangle = math.acos((np.dot(v1,  v2)/np.linalg.norm(v1)*np.linalg.norm(v2)))
        sign = np.sign(np.dot(v_,np.cross(v1, v2)))
 
        if sign >= 0:
            angle = cangle
        else:
            angle = -cangle
           
        BARY.append([bary, angle])
            
        
    BARY.sort(key = lambda x: x[1])   #sort the barycenters accordind to the angle 
    i = 0
    n = len(Ls)
 
    return [BARY[i][0] for i in range(len(poly))]



def polygones (simplicies, D, points, nvertex, verbose=True):
    #construct the polygones associated to the triangulation "simplicies, points"
    POLY = []
    
    for v in range(points.shape[0]):  #find the neighborhood of each vertex v
        if v % 100 == 0 and verbose == True: print("step ",v)
        BARY = centroid(D, points, v, simplicies)  
        i = len(BARY) -1
        while len(BARY) < nvertex and len(BARY)>0:
            BARY.append(BARY[i])
        if len(BARY) != 0: POLY.append(BARY)
    
    return POLY, nvertex


def check_sphere(POLY):
    #check that all the verticies in POLY are on the sphere
    ok = 0
    for poly in POLY:
        for bary in poly:
            if np.abs(bary[0]**2 + bary[1]**2 + bary[2]**2 - 1) > 0.001: ok = 1
    if ok == 0: print("every barycenter is on the sphere")
    else : print("error, some barycenter is not on the sphere !!")
    return 



def mesh(filename, nvertex, verbose=True):
    #construct the set of polygones reprensenting the mesh
    points = get_xyz(filename)
    print("xyz coordinates from netcdf file ok")
    simplicies, D = delaunay(points)
    for k in D:
        D[k] = set(tuple(i) for i in D[k])
    #print("dimplicies = ", simplicies)
    print("=======================\n=============================")
    #print("D = ", D[27])
    #print("D = ", D[41])
    print("triangulation with delaunay ok")
    POLY, nvertex = polygones(simplicies, D, points, nvertex, verbose)
    #print(POLY[0])
    print("polygones ok")
    #check_sphere(POLY[0])
   
    
    return POLY, nvertex


def matplotlib_plot(POLY, C = ['b','g','r','c','m','y'], test = []):
    #a naive visualization implementation
    if test == []: test = POLY
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(-1.5,1.5))


    i = -1
    for poly in test:
            i = i+1
            (x, y, z) = np.array(poly).T

            verts = [list(zip(x,y,z))]
            ax.add_collection3d(Poly3DCollection(verts, color = C[i%len(C)]))
    plt.show()

def plot_triangulation(points, simplices):  
    # to vizualize the triangulation
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(-1.5,1.5))

    C = ['b'] + ['g'] + ['r'] + ['c'] + ['m'] + ['y']

    i = -1

    for tri in simplices[0]:
        #if 31 in tri or 27 in tri:
        if  True:
                #print("==============================================================\n===============================")
                (a_, b_, c_) = tri
                (a, b, c) = (list(points[a_]),list(points[b_]),list(points[c_]))
                #print("vertices: ", a_, b_, c_)
                (x, y, z) = np.array([a,b,c]).T
                i = i+1

                

                verts = [list(zip(x,y,z))]
                ax.add_collection3d(Poly3DCollection(verts, color = C[i%len(C)]))
    plt.show()



def format_(POLY):
    #take as input a list of x,y,z coordinate verticies and
    #return two list of lon (resp. lat) coordinates verticies for visu
    bounds_lat = []
    bounds_lon = []

    #print(POLY)

    for poly in POLY:
        lat_poly = []
        lon_poly = []
        for (x, y, z) in poly:
            lon = np.degrees(np.arctan2(y, x))
            p = np.sqrt(x**2 + y**2)
            lat = np.degrees(np.arctan2(z, p))
            #lat = np.degrees(lat)
            lon_poly.append(lon)
            lat_poly.append(lat)
            #if lat >=178 or lat<=-178: print(lon, lat)
            #if lon >=178 or lon<=-178: print(lon, lat)
        bounds_lon.append(lon_poly)
        bounds_lat.append(lat_poly)
 

    return bounds_lon, bounds_lat
    


