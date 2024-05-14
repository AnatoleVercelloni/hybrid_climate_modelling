#include <math.h>

enum { batchsize=8 };
static double Ax[batchsize], Ay[batchsize], Az[batchsize];
static double Bx[batchsize], By[batchsize], Bz[batchsize]; 
static double Cx[batchsize], Cy[batchsize], Cz[batchsize]; 
static double Dx[batchsize], Dy[batchsize], Dz[batchsize]; 

/*

def norm2(a): return a[0]*a[0]+a[1]*a[1]+a[2]*a[2]
def norm(a): return np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])

def cross_jit(a,b, c):
    c[0]=a[1]*b[2]-b[1]*a[2]
    c[1]=a[2]*b[0]-b[2]*a[0]
    c[2]=a[0]*b[1]-b[0]*a[1]

def circum(A,B,C):
# see Dubos et al., GMD 2015, appendix
# the computation is stable, and valid whether a,b,c is clockwise or not
    b,c = B-A, C-A
    p1,p2 = cross(b,c), norm2(c)*b-norm2(b)*c
    p = A+.5*cross(p1,p2)/norm2(p1)
    return p/norm(p)
*/

static inline void circum_main(int N)
{
  int i;
  // Any decent compiler should vectorize this
  for(i=0 ; i<N ; i++)
    {
      //    b,c = B-A, C-A
      double bx,by,bz,cx,cy,cz;
      bx=Bx[i]-Ax[i];
      by=By[i]-Ay[i];
      bz=Bz[i]-Az[i];
      cx=Cx[i]-Ax[i];
      cy=Cy[i]-Ay[i];
      cz=Cz[i]-Az[i];
      // p = cross(b,c)
      double px,py,pz;
      px = by*cz-bz*cy;
      py = bz*cx-bx*cz;
      pz = bx*cy-by*cx;
      // q = norm2(c)*b-norm2(b)*c
      double b2, c2, qx,qy,qz ;
      b2 = bx*bx+by*by+bz*bz;
      c2 = cx*cx+cy*cy+cz*cz;
      qx = c2*bx-b2*cx;
      qy = c2*by-b2*cy;
      qz = c2*bz-b2*cz;
      // r = A+.5*cross(p,q)/norm2(p)
      double p2,rx,ry,rz;
      p2 = .5/(px*px+py*py+pz*pz);
      rx = Ax[i]+p2*(py*qz-pz*qy);
      ry = Ay[i]+p2*(pz*qx-px*qz);
      rz = Az[i]+p2*(px*qy-py*qx);
      // circ[i]=r/norm(r)
      double r=1./sqrt(rx*rx+ry*ry+rz*rz);
      Dx[i] = r*rx;
      Dy[i] = r*ry;
      Dz[i] = r*rz;
    }
}

static inline void circum_copyin(int N, long *ind, double *points)
{
  int n;
  for(n=0 ; n<N ; n++)
    {
      int i; const double *ptr;
      i=*ind++; ptr=points+3*i;
      Ax[n]=*ptr++;
      Ay[n]=*ptr++;
      Az[n]=*ptr++;
      i=*ind++; ptr=points+3*i;
      Bx[n]=*ptr++;
      By[n]=*ptr++;
      Bz[n]=*ptr++;
      i=*ind++; ptr=points+3*i;
      Cx[n]=*ptr++;
      Cy[n]=*ptr++;
      Cz[n]=*ptr++;
    }
}

static inline void circum_copyout(int N, double *circ)
{
  int i;
  for(i=0 ; i<N ; i++)
    {
      *circ++ = Dx[i];
      *circ++ = Dy[i];
      *circ++ = Dz[i];
    }
}

static inline void circum_batch(double *points, double *circ)
{
  int i;
  for(i=0 ; i<batchsize ; i++)
    {
      Ax[i]=*points++;
      Ay[i]=*points++;
      Az[i]=*points++;
      Bx[i]=*points++;
      By[i]=*points++;
      Bz[i]=*points++;
      Cx[i]=*points++;
      Cy[i]=*points++;
      Cz[i]=*points++;
    }
  circum_main(batchsize);
  circum_copyout(batchsize, circ);
}

static inline void circum_fast_(int N, long *ind, double *points, double *circ)
{
  // work by batches of size batchsize
  while(N>batchsize)
    {
      circum_copyin(batchsize, ind, points);
      circum_main(batchsize);
      circum_copyout(batchsize, circ);
      ind  += 3*batchsize;
      circ += 3*batchsize;
      N-=batchsize;
    }
  // finish the job
  circum_copyin(N, ind, points);
  circum_main(N);
  circum_copyout(N, circ);
}

static inline void fast_scale_vec(double scale, double * vec)
{
  vec[0]*=scale;
  vec[1]*=scale;
  vec[2]*=scale;
}
 
/*

def arc(a,b): return 2.*np.arcsin(.5*norm(a-b))

def area(a,b,c):
# http://mathworld.wolfram.com/LHuiliersTheorem.html
    a,b,c = .25*arc(b,c), .25*arc(c,a), .25*arc(a,b)
    return 4.*arctan(sqrt(tan(a+b+c)*tan(a+b-c)*tan(b+c-a)*tan(a+c-b)))

def barycenters_fast(regions_size, vert, points, vertices, weight):
    N = points.shape[0]
    bary = np.zeros((N,3))
    i=0 # running index in vert1, vert2
    for n in range(N):
        point = points[n,:]
        b = 0.
        for j in range(regions_size[n]):
            v1 = vertices[vert[i],:]
            i = i+1
            v2 = vertices[vert[i],:]
            i = i+1
            dA=area(point, v1, v2)
            bb = unit_vector(point+v1+v2)
            b = b + dA*bb*weight(bb)
        bary[n,:] = unit_vector(b)
    return bary
        
*/

static inline double arc4(double dx, double dy, double dz) // 1/4 geodesic length, as used in L'Huillier's formula
{
  double dist;
  //  printf("dx dy dz = %f %f %f\n", dx,dy,dz);
  dist=dx*dx+dy*dy+dz*dz;
  //  printf("dist = %f\n", dist);
  dist=sqrt(dist);
  //  printf("dist = %f\n", dist);
  dist = .5*asin(.5*dist);
  //  printf("dist = %f\n", dist);
  return dist;
}


static inline double barycenters_fast_(int N, const long *degree, const long *ind, 
				       const double *point, const double *w_i,
				       const double *vertex, const double *w_v,
				       double *bary)
{
  int cell;
  double A=0; // total area
  // loop over voronoi cells
  for(cell=0; cell<N ; cell++)
    {
      double aw,ax,ay,az; // weight and xyz coordinates of generator of voronoi cell
      int deg, edge;
      double gx=0., gy=0., gz=0.; // barycenter of current voronoi cell
      aw=*w_i++;
      ax=*point++;
      ay=*point++;
      az=*point++;
      deg=*degree++;
      //      if(cell<10) printf("deg[%d]=%d\n", cell, deg);

      for(edge=0; edge<deg ; edge++) // loop over edges of voronoi cell
	{
	  int i,j; const double *ptr;
	  double bw,bx,by,bz, cw,cx,cy,cz;
	  i=*ind++; ptr=vertex+3*i; // first vertex
	  bw=w_v[i];
	  bx=*ptr++;
	  by=*ptr++;
	  bz=*ptr++;
	  j=*ind++; ptr=vertex+3*j; // second vertex
	  cw=w_v[j];
	  cx=*ptr++;
	  cy=*ptr++;
	  cz=*ptr++;
	  //	  printf("cell, vertex1, vertex2 : %d %d %d\n", cell, i, j);
	  //	  printf("cell    x y z : %f %f %f\n", ax,ay,az);
	  //	  printf("vertex1 x y z : %f %f %f\n", bx,by,bz);
	  //	  printf("vertex2 x y z : %f %f %f\n", cx,cy,cz);

	  double a,b,c,dA; // 1/4 geodesic distances, as needed in L'Huillier's formula
	  // http://mathworld.wolfram.com/LHuiliersTheorem.html
	  a=arc4(bx-cx,by-cy,bz-cz);
	  b=arc4(ax-cx,ay-cy,az-cz);
	  c=arc4(ax-bx,ay-by,az-bz);
	  dA=4.*atan(sqrt(tan(a+b+c)*tan(a+b-c)*tan(b+c-a)*tan(a+c-b)));
	  A+=dA;
	  gx+=dA*(aw*ax+bw*bx+cw*cx);
	  gy+=dA*(aw*ay+bw*by+cw*cy);
	  gz+=dA*(aw*az+bw*bz+cw*cz);
	  //	  printf("Triangle sides : %f %f %f\n", a,b,c);
	  //	  printf("Local area : dA=%f\n", dA);
	}
      double g=1./sqrt(gx*gx+gy*gy+gz*gz);
      *bary++ = gx*g;
      *bary++ = gy*g;
      *bary++ = gz*g;
    }
  // printf("Total area : A=%f\n", A);
  return A; // total area should be 4*pi
}
