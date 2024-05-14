cdef extern from "sphere_fast.h":
    cdef void fast_scale_vec(double, double[])
    cdef void circum_batch(double*, double*)
    cdef void circum_fast_(int, long*, double*, double*)
    cdef void barycenters_fast_(int, long*, long*, double*, double*, double*, double*, double*)

def scale_vec(double scale, double[::1] vec):
    fast_scale_vec(scale, &vec[0])

def circum_test(double[:,::1] pts, double[:,::1] circ):
    circum_batch(&pts[0,0], &circ[0,0])

def circum_fast(long[:,::1] ind, double[:,::1] pts, double[:,::1] circ):
    cdef int N = ind.shape[0] # number of circumcenters to compute
#    print('Computing %d circumcenters'%N)
    circum_fast_(N, &ind[0,0], &pts[0,0], &circ[0,0])

def barycenters_fast(long[::1] degree, long[::1] index, 
                     double[:,::1] point, double[::1] w_i,
		     double[:,::1] vertex, double[::1] w_v,
		     double[:,::1] bary):
    cdef int N=degree.shape[0] # number of voronoi cells

    cdef int sum=0
    for i in range(N):
      sum += degree[i]

#    print(degree[0:10])
#    print('Computing %d barycenters'%N)
#    print('degree.sum() = %d'%sum )
#    print('index.size = %d'%index.size )
#    print('point.shape = ', point.shape )
#    print('vertex.shape = ', vertex.shape )
#    print('bary.shape = ', bary.shape )
    
    return barycenters_fast_(N, &degree[0], &index[0], 
                             &point[0,0], &w_i[0],
			     &vertex[0,0], &w_v[0],
			     &bary[0,0])
