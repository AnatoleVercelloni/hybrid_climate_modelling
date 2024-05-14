import numpy as np
from scipy.spatial._qhull import _QhullUser
from scipy.spatial._qhull import _Qhull

from scvt.timer import Timer
from scvt.topology import simplex

########################################################

# custom Delaunay class, slightly more efficient

class Delaunay(_QhullUser):
    def __init__(self, points):
        qhull_options = b"Qbb Qc Qz Q12"
        # Run qhull
        qhull = _Qhull(b"d", points, qhull_options, required_options=b"Qt")
        _QhullUser.__init__(self, qhull)
    def _update(self, qhull):
        qhull.triangulate()
        self.paraboloid_scale, self.paraboloid_shift = \
                               qhull.get_paraboloid_shift_scale()
        junk = qhull.get_simplex_facet_array()
        self.simplices, self.neighbors, self.equations, self.coplanar, junk = junk
        _QhullUser._update(self, qhull)

########################################################        

def testn(z): return z>0
def tests(z): return z<=0


def north_south(points, verbose):
    with Timer('in stereographic projection', verbose):
        points_norm = np.linalg.norm(points, axis=1)
        
        points_norm = np.reshape(np.repeat(points_norm, 3), (points.shape[0], 3))


        normalized_points = points / points_norm

        for i in range(points.shape[0]):
            # TODO: it seems that points are modified by reference outside
            points[i, :] = normalized_points[i, :]

        zn = 1. + normalized_points[:, 2]
        zs = 1. - normalized_points[:, 2]

        mask_north = zn > 0.5
        mask_south = zs > 0.5

        north = normalized_points[mask_north, 0:2] / (zn[mask_north, None])
        south = normalized_points[mask_south, 0:2] / (zs[mask_south, None])
    
    with Timer('in planar Delaunay', verbose):
        trin, tris = Delaunay(north), Delaunay(south)
    return points[:, 2], (np.where(mask_north)[0], trin, testn), (np.where(mask_south)[0], tris, tests)
    #return points[:, 2], (np.where(mask_south)[0], tris, tests), (np.where(mask_south)[0], tris, tests)
    #return points[:, 2],  (np.where(mask_north)[0], trin, testn),  (np.where(mask_north)[0], trin, testn)


def enum_simplices(z, domains, D):
    # domains is a list of tuples (ind, tri, test)
    for ind, tri, test in domains:
        for i,j,k in tri.simplices:
            ii, jj, kk = ind[i], ind[j], ind[k] # global indices
            zi, zj, zk = z[ii], z[jj], z[kk]
            if test(zi) | test(zj) | test(zk): yield simplex(D, ii,jj,kk)

def delaunay(points, verbose=False):
    D = {}
    z, north, south = north_south(points, verbose)
    with Timer('enumerating simplices', verbose):
        simplices = list(set( enum_simplices(z, (north, south), D ) ))
    #print(D)
    return simplices, D
