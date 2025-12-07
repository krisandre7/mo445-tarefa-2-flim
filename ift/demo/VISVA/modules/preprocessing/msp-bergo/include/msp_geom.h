#ifndef msp_geom_h
#define msp_geom_h 1

#include <iostream>
#include <math.h>

using namespace std;

namespace MSP{

//! Point or Vector in 3D space.
//! Point/Vector coordinates are stored as 32-bit floating point.
class R3 {
 public:  
  float X,Y,Z;

  //! creates a point at the origin
  R3() { X=Y=Z=0.0; }
  //! Constructor, creates a point at the given coordinates
  R3(float x,float y,float z) { X=x; Y=y; Z=z; }
  //! Copy constructor
  R3(  R3 &p) { X=p.X; Y=p.Y; Z=p.Z; }
  //! Copy constructor
  R3(R3 *p) { X=p->X; Y=p->Y; Z=p->Z; }

  //! point assignment
  R3 & operator=(  R3 &b) { X=b.X; Y=b.Y; Z=b.Z; return(*this); }

  //! adds b to the point
  R3 & operator+=(  R3 &b) { X+=b.X; Y+=b.Y; Z+=b.Z; return(*this); }

  //! subtracts b from the point
  R3 & operator-=(  R3 &b) { X-=b.X; Y-=b.Y; Z-=b.Z; return(*this); }

  //! multiplies the point by a scalar
  R3 & operator*=(  float &b) { X*=b; Y*=b; Z*=b; return(*this); }

  //! divides the point by a scalar
  R3 & operator/=(  float &b) { X/=b; Y/=b; Z/=b; return(*this); }

  //! vector addition
  R3 & operator+(  R3 &b) {
    static R3 t; t.X = X+b.X; t.Y = Y+b.Y; t.Z = Z+b.Z; return(t); }

  //! vector subtraction
  R3 & operator-(  R3 &b) {
    static R3 t; t.X = X-b.X; t.Y = Y-b.Y; t.Z = Z-b.Z; return(t); }

  //! vector multiplication by a scalar
  R3 & operator*(  float &b) {
    static R3 t; t.X = X*b; t.Y = Y*b; t.Z = Z*b; return(t); }

  //! vector division by a scalar
  R3 & operator/(  float &b) {
    static R3 t; t.X = X/b; t.Y = Y/b; t.Z = Z/b; return(t); }
 
  //! vector inner product
  float inner(  R3 &b) { return(X*b.X+Y*b.Y+Z*b.Z); }

  //! Returns the vector cross product of this x b. Does not modify this point.
  R3 & cross(  R3 &b) {
    static R3 r;
    r.X = Y*b.Z - Z*b.Y;
    r.Y = Z*b.X - X*b.Z;
    r.Z = X*b.Y - Y*b.X;
    return r;
  }

  //! sets the coordinates of the point
  void set(float x, float y, float z) { X=x; Y=y; Z=z; }

  //! returns the length of the vector
  float length() { return(sqrt(X*X+Y*Y+Z*Z)); }

  //! normalizes this vector to unit-length
  void  normalize() { float l; l = length(); if (l!=0.0) (*this)/=l; }

  //! prints this vector to stdout
  void print()   { cout << "R3=(" << X << "," << Y << "," << Z << ")\n"; }
};

class Triangle {
 public:
  R3 A,B,C;

  Triangle() {
    A.set(0,0,0);
    B=A;
    C=A;
  }

  Triangle &operator=(  Triangle &b) {
    A = b.A;
    B = b.B;
    C = b.C;
    return(*this);
  }

  void set(  R3 &a,   R3 &b,   R3 &c) {
    A = a;
    B = b;
    C = c;
  }
  
};

//! 3x3 Linear Transformation.
//! 3x3 Linear Transformation, which does not allow translations.
class T3 {
 public:
  //! Constructor, creates a null transform
  T3() { zero(); }

  //! Copy constructor
  T3(  T3 &b) {
    int i,j; for(i=0;i<3;i++) for(j=0;j<3;j++) e[i][j] = b.e[i][j]; }

  //! Transform assignment
  T3 & operator=(  T3 &b) {
    int i,j; for(i=0;i<3;i++) for(j=0;j<3;j++) e[i][j] = b.e[i][j]; 
    return(*this); }

  //! Transform composition, multiplies this transform by b, and overwrites
  //! this transform. Returns a reference to this transform.
  T3 & operator*=(  T3 & b) { (*this) = (*this) * b; return(*this); }

  //! Multiplication by a scalar
  T3 & operator*(  T3 &b) {
    static T3 d;
    int i,j,k;
    d.zero();
    for(i=0;i<3;i++) for(j=0;j<3;j++) for(k=0;k<3;k++)
      d.e[i][j] += b.e[i][k] * this->e[k][j];
    return d;
  }
  
  //! Applies this transform to point a and returns the transformed point
  R3 & apply(  R3 & a) {
    static R3 b;
    b.X = a.X * e[0][0] + a.Y * e[0][1] + a.Z * e[0][2];
    b.Y = a.X * e[1][0] + a.Y * e[1][1] + a.Z * e[1][2];
    b.Z = a.X * e[2][0] + a.Y * e[2][1] + a.Z * e[2][2];
    return b;
  }

  //! Sets all coefficients to zero (null transform)
  void zero() {
    int i,j; for(i=0;i<3;i++) for(j=0;j<3;j++) e[i][j]=0.0; }

  //! Sets an identity transform
  void identity() { zero(); e[0][0] = e[1][1] = e[2][2] = 1.0; }

  //! Sets a rotation around X-axis transform, by angle degrees.
  //! Rotation center is the origin.
  void xrot(float angle) {
    identity();
    angle *= M_PI / 180.0;
    e[1][1] = cos(angle);  e[2][1] = sin(angle);
    e[1][2] = -sin(angle); e[2][2] = cos(angle);
  }

  //! Sets a rotation around Y-axis transform, by angle degrees.
  //! Rotation center is the origin.
  void yrot(float angle) {
    identity();
    angle *= M_PI / 180.0;
    e[0][0] = cos(angle);  e[2][0] = sin(angle);
    e[0][2] = -sin(angle); e[2][2] = cos(angle);
  }

  //! Sets a rotation around Z-axis transform, by angle degrees.
  //! Rotation center is the origin.
  void zrot(float angle) {
    identity();
    angle *= M_PI / 180.0;
    e[0][0] = cos(angle);  e[1][0] = sin(angle);
    e[0][1] = -sin(angle); e[1][1] = cos(angle);
  }

  //! Sets a scaling transform, by factor on all dimensions
  void scale(float factor) {
    zero(); e[0][0]=e[1][1]=e[2][2]=factor;
  }

  //! Sets a scaling transform, with different scaling factors
  //! for each dimension.
  void scale(float fx, float fy, float fz) {
    zero(); e[0][0]=fx; e[1][1]=fy; e[2][2]=fz;
  }

  //! Prints the transform matrix on stdout
  void print()   {
    int i,j;
    for(i=0;i<3;i++) {
      for(j=0;j<3;j++) {
	printf("%.2f ",e[j][i]);
      }
      printf("\n");
    }
    printf("\n");
  }

 private:
  float e[3][3];
};

//! 4x4 Linear Transformation.
//! 4x4 Linear Transformation.
class T4 {
 public:
  //! Constructor, creates a null transform
  T4() { zero(); }

  //! Copy constructor
  T4(  T4 &b) {
    int i,j; for(i=0;i<4;i++) for(j=0;j<4;j++) e[i][j] = b.e[i][j]; }

  //! Transform assignment
  T4 & operator=(  T4 &b) {
    int i,j; for(i=0;i<4;i++) for(j=0;j<4;j++) e[i][j] = b.e[i][j]; 
    return(*this); }

  //! Transform composition, multiplies this transform by b, overwrites this
  //! transform with the result, and returns a reference to this transform.
  T4 & operator*=(  T4 & b) { (*this) = (*this) * b; return(*this); }

  //! Transform multiplication. Multiplies this transform by b, and returns
  //! a reference to the result without modifying this transform
  T4 & operator*(  T4 &b) {
    static T4 d;
    int i,j,k;
    d.zero();
    for(i=0;i<4;i++) for(j=0;j<4;j++) for(k=0;k<4;k++)
      d.e[i][j] += b.e[i][k] * this->e[k][j];
    return d;
  }

  //! Applies this transform to point a and returns the transformed point
  R3 & apply(  R3 & a)   {
    static R3 b;
    float w;
    b.X = a.X * e[0][0] + a.Y * e[0][1] + a.Z * e[0][2] + e[0][3];
    b.Y = a.X * e[1][0] + a.Y * e[1][1] + a.Z * e[1][2] + e[1][3];
    b.Z = a.X * e[2][0] + a.Y * e[2][1] + a.Z * e[2][2] + e[2][3];
    w = e[3][0] + e[3][1] + e[3][2] + e[3][3];
    b /= w;
    return b;
  }

  //! Sets all coefficients to zero (null transform)
  void zero() {
    int i,j; for(i=0;i<4;i++) for(j=0;j<4;j++) e[i][j]=0.0; }

  //! Sets an identity transform
  void identity() { zero(); e[0][0] = e[1][1] = e[2][2] = e[3][3] = 1.0; }

  //! Sets a rotation around X-axis transform, by angle degrees.
  //! Rotation center is given by (cx,cy,cz)
  void xrot(float angle, float cx, float cy, float cz) {
    T4 a,b,c;
    a.translate(-cx,-cy,-cz);
    b.xrot(angle);
    c.translate(cx,cy,cz);
    a *= b;
    a *= c;
    (*this) = a;
  }

  //! Sets a rotation around Y-axis transform, by angle degrees.
  //! Rotation center is given by (cx,cy,cz)
  void yrot(float angle, float cx, float cy, float cz) {
    T4 a,b,c;
    a.translate(-cx,-cy,-cz);
    b.yrot(angle);
    c.translate(cx,cy,cz);
    a *= b;
    a *= c;
    (*this) = a;
  }

  //! Sets a rotation around Z-axis transform, by angle degrees.
  //! Rotation center is given by (cx,cy,cz)
  void zrot(float angle, float cx, float cy, float cz) {
    T4 a,b,c;
    a.translate(-cx,-cy,-cz);
    b.zrot(angle);
    c.translate(cx,cy,cz);
    a *= b;
    a *= c;
    (*this) = a;
  }

  //! Sets a rotation around X-axis transform, by angle degrees.
  //! Rotation center is the origin.
  void xrot(float angle) {
    identity();
    angle *= M_PI / 180.0;
    e[1][1] = cos(angle);  e[2][1] = sin(angle);
    e[1][2] = -sin(angle); e[2][2] = cos(angle);
  }

  //! Sets a rotation around Y-axis transform, by angle degrees.
  //! Rotation center is the origin.
  void yrot(float angle) {
    identity();
    angle *= M_PI / 180.0;
    e[0][0] = cos(angle);  e[2][0] = sin(angle);
    e[0][2] = -sin(angle); e[2][2] = cos(angle);
  }

  //! Sets a rotation around Z-axis transform, by angle degrees.
  //! Rotation center is the origin.
  void zrot(float angle) {
    identity();
    angle *= M_PI / 180.0;
    e[0][0] = cos(angle);  e[1][0] = sin(angle);
    e[0][1] = -sin(angle); e[1][1] = cos(angle);
  }
  
  //! Sets an arbitrary rotation around the given axis, by angle degrees
  //! Rotation center is the origin.
  //! Source: http://www.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
  void axisrot(float angle, R3 & axis) {

    if (axis.length() < 1e-7) {
      identity();
      return;
    }

    // {u,v,w} = axis.{X,Y,Z}
    double L,L2,ct,st,u,v,w,a=0.0,b=0.0,c=0.0;
    L2 = axis.inner(axis);
    L = sqrt(L2);
    ct = cos(angle * M_PI / 180.0);
    st = sin(angle * M_PI / 180.0);
    u = axis.X; v = axis.Y; w = axis.Z;

    identity();
    e[0][0] = ( u*u+(v*v+w*w)*ct ) / L2;
    e[0][1] = ( u*v*(1-ct)-w*L*st ) / L2;
    e[0][2] = ( u*w*(1-ct)+v*L*st ) / L2;
    e[0][3] = ( a*(v*v+w*w)-u*(b*v+c*w)+( u*(b*v+c*w)-a*(v*v+w*w) )*ct + (b*w-c*v)*L*st ) / L2;

    e[1][0] = ( u*v*(1-ct)+w*L*st ) / L2;
    e[1][1] = ( v*v+(u*u+w*w)*ct ) / L2;
    e[1][2] = ( v*w*(1-ct)-u*L*st ) / L2;
    e[1][3] = ( b*(u*u+w*w)-v*(a*u+c*w)+( v*(a*u+c*w)-b*(u*u+w*w) )*ct + (c*u-a*w)*L*st ) / L2;

    e[2][0] = ( u*w*(1-ct)-v*L*st ) / L2;
    e[2][1] = ( v*w*(1-ct)+u*L*st ) / L2;
    e[2][2] = ( w*w+(u*u+v*v)*ct ) / L2;
    e[2][3] = ( c*(u*u+v*v)-w*(a*u+b*v)+( w*(a*u+b*v)-c*(u*u+v*v) )*ct + (a*v-b*u)*L*st ) / L2;
  }

  //! Sets an arbitrary rotation around the given axis, by angle degrees
  //! Rotation center is (cx,cy,cz).
  //! Source: http://www.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
  void axisrot(float angle, R3 & axis, float cx, float cy, float cz) {

    if (axis.length() < 1e-7) {
      identity();
      return;
    }

    // {u,v,w} = axis.{X,Y,Z}
    double L,L2,ct,st,u,v,w,a=cx,b=cy,c=cz;
    L2 = axis.inner(axis);
    L = sqrt(L2);
    ct = cos(angle * M_PI / 180.0);
    st = sin(angle * M_PI / 180.0);
    u = axis.X; v = axis.Y; w = axis.Z;

    identity();
    e[0][0] = ( u*u+(v*v+w*w)*ct ) / L2;
    e[0][1] = ( u*v*(1-ct)-w*L*st ) / L2;
    e[0][2] = ( u*w*(1-ct)+v*L*st ) / L2;
    e[0][3] = ( a*(v*v+w*w)-u*(b*v+c*w)+( u*(b*v+c*w)-a*(v*v+w*w) )*ct + (b*w-c*v)*L*st ) / L2;

    e[1][0] = ( u*v*(1-ct)+w*L*st ) / L2;
    e[1][1] = ( v*v+(u*u+w*w)*ct ) / L2;
    e[1][2] = ( v*w*(1-ct)-u*L*st ) / L2;
    e[1][3] = ( b*(u*u+w*w)-v*(a*u+c*w)+( v*(a*u+c*w)-b*(u*u+w*w) )*ct + (c*u-a*w)*L*st ) / L2;

    e[2][0] = ( u*w*(1-ct)-v*L*st ) / L2;
    e[2][1] = ( v*w*(1-ct)+u*L*st ) / L2;
    e[2][2] = ( w*w+(u*u+v*v)*ct ) / L2;
    e[2][3] = ( c*(u*u+v*v)-w*(a*u+b*v)+( w*(a*u+b*v)-c*(u*u+v*v) )*ct + (a*v-b*u)*L*st ) / L2;
  }

  //! Sets a scaling transform, by factor on all dimensions
  void scale(float factor) {
    zero(); e[0][0]=e[1][1]=e[2][2]=factor;
  }

  //! Sets a scaling transform, by factors (fx,fy,fz)
  void scale(float fx, float fy, float fz) {
    zero(); e[0][0]=fx; e[1][1]=fy; e[2][2]=fz;
  }

  //! Sets a translation transform, by (dx,dy,dz)
  void translate(float dx, float dy, float dz) {
    identity(); e[0][3] = dx; e[1][3] = dy; e[2][3] = dz;
  }

  //! Computes the inverse transform
  void invert() { minv4(); }

  //! Prints the transform matrix on stdout
  void print()   {
    int i,j;
    for(i=0;i<4;i++) {
      for(j=0;j<4;j++) {
	printf("%.2f ",e[j][i]);
      }
      printf("\n");
    }
    printf("\n");
  }

  void printLine() {
    int i,j;
    printf("[ ");
    for(i=0;i<4;i++) {
      for(j=0;j<4;j++) {
	printf("%.3f ",e[j][i]);
      }
      if (i<3) printf(" | ");
    }
    printf(" ]\n");
  }

 private:
  float e[4][4];

  bool minv4() {
    int *sle;
    float *sq0, *a;
    int n = 4; // matrix size

    int lc,*le; float s,t,tq=0.,zr=1.e-15;
    float *pa,*pd,*ps,*p,*q,*q0;
    int i,j,k,m;
    
    sle = new int[n];
    sq0 = new float[n];
    a = new float[n*n];

    for(i=0;i<n*n;i++) a[i] = e[i%n][i/n];

    le = sle;
    q0 = sq0;
    
    for(j=0,pa=pd=a; j<n ;++j,++pa,pd+=n+1){
      if(j>0){
	for(i=0,q=q0,p=pa; i<n ;++i,p+=n) *q++ = *p;
	for(i=1; i<n ;++i){ lc=i<j?i:j;
	for(k=0,p=pa+i*n-j,q=q0,t=0.; k<lc ;++k) t+= *p++ * *q++;
	q0[i]-=t;
	}
	for(i=0,q=q0,p=pa; i<n ;++i,p+=n) *p= *q++;
      }
      
      s=fabs(*pd); lc=j;
      for(k=j+1,ps=pd; k<n ;++k){
	if((t=fabs(*(ps+=n)))>s){ s=t; lc=k;}
      }
      tq=tq>s?tq:s; if(s<zr*tq){ return false;}
      *le++ =lc;
      if(lc!=j){
	for(k=0,p=a+n*j,q=a+n*lc; k<n ;++k){
	  t= *p; *p++ = *q; *q++ =t;
	}
      }
      for(k=j+1,ps=pd,t=1./ *pd; k<n ;++k) *(ps+=n)*=t;
      *pd=t;
    }
    for(j=1,pd=ps=a; j<n ;++j){
      for(k=0,pd+=n+1,q= ++ps; k<j ;++k,q+=n) *q*= *pd;
    }
    for(j=1,pa=a; j<n ;++j){ ++pa;
    for(i=0,q=q0,p=pa; i<j ;++i,p+=n) *q++ = *p;
    for(k=0; k<j ;++k){ t=0.;
    for(i=k,p=pa+k*n+k-j,q=q0+k; i<j ;++i) t-= *p++ * *q++;
    q0[k]=t;
    }
    for(i=0,q=q0,p=pa; i<j ;++i,p+=n) *p= *q++;
    }
    for(j=n-2,pd=pa=a+n*n-1; j>=0 ;--j){ --pa; pd-=n+1;
    for(i=0,m=n-j-1,q=q0,p=pd+n; i<m ;++i,p+=n) *q++ = *p;
    for(k=n-1,ps=pa; k>j ;--k,ps-=n){ t= -(*ps);
    for(i=j+1,p=ps,q=q0; i<k ;++i) t-= *++p * *q++;
    q0[--m]=t;
    }
    for(i=0,m=n-j-1,q=q0,p=pd+n; i<m ;++i,p+=n) *p= *q++;
    }
    for(k=0,pa=a; k<n-1 ;++k,++pa){
      for(i=0,q=q0,p=pa; i<n ;++i,p+=n) *q++ = *p;
      for(j=0,ps=a; j<n ;++j,ps+=n){
	if(j>k){ t=0.; p=ps+j; i=j;}
	else{ t=q0[j]; p=ps+k+1; i=k+1;}
	for(; i<n ;) t+= *p++ *q0[i++];
	q0[j]=t;
      }
      for(i=0,q=q0,p=pa; i<n ;++i,p+=n) *p= *q++;
    }
    for(j=n-2,le--; j>=0 ;--j){
      for(k=0,p=a+j,q=a+ *(--le); k<n ;++k,p+=n,q+=n){
	t=*p; *p=*q; *q=t;
      }
    }
    delete sq0;
    delete sle;
    for(i=0;i<n*n;i++) e[i%n][i/n] = a[i];
    delete a;
    return true;
  } // minv4
};

class Plane {
 public:
  float z;
  T4    trans;

  Plane() {
    z = 0;
    trans.identity();
  }

  Plane &operator=(  Plane &b) {
    z     = b.z;
    trans = b.trans;
    return(*this);
  }

  void print()   {
    cout << "--- Plane Start ---\n";
    cout << "Z=" << z << endl;
    trans.print();
    cout << "--- Plane End ---\n";
  }

  // average displacement within a volume
  double distance(  Plane &b, int W, int H) {
    double zacc = 0.0;
    
    zacc += fabs(getZ(0,0)   - b.getZ(0,0));
    zacc += fabs(getZ(W/2,0) - b.getZ(W/2,0));
    zacc += fabs(getZ(W,0)   - b.getZ(W,0));

    zacc += fabs(getZ(0,H/2)   - b.getZ(0,H/2));
    zacc += fabs(getZ(W/2,H/2) - b.getZ(W/2,H/2));
    zacc += fabs(getZ(W,H/2)   - b.getZ(W,H/2));

    zacc += fabs(getZ(0,H)   - b.getZ(0,H));
    zacc += fabs(getZ(W/2,H) - b.getZ(W/2,H));
    zacc += fabs(getZ(W,H)   - b.getZ(W,H));

    return(zacc/9.0);
  }

  // get Z such that (x,y,z) is within the plane
  double getZ(int x, int y, int W=256, int H=256)   {
    double mind = W+H, bestz = 0.0, d;
    int i,j;
    R3 r;
    for(j=-H;j<2*H;j++)
      for(i=-W;i<2*W;i++) {
	r.set(i,j,z);
	r = trans.apply(r);
	d = dist2(r.X,r.Y,x,y);
	if (d < mind) {
	  mind = d;
	  bestz  = r.Z;
	}
      }
    return bestz;
  }

  // angle between two planes
  float angle(  Plane &b) {
    R3 v[6];
    T4 mix;
    float ang;

    v[0].set(0,0,0);
    v[1].set(0,0,1);

    v[2] = trans.apply(v[0]);
    v[3] = trans.apply(v[1]);
    v[3] -= v[2];

    v[4] = b.trans.apply(v[0]);
    v[5] = b.trans.apply(v[1]);
    v[5] -= v[4];

    v[3].normalize();
    v[5].normalize();

    ang = v[5].inner(v[3]);

    if (ang > 1.0) ang = 1.0;
    if (ang < -1.0) ang = -1.0;
    ang = 180.0 * acos(ang) / M_PI;

    return ang;
  }

 private:
  double dist2(double x1, double y1, double x2, double y2)   {
    return(sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)));
  }
};


} //end MSP namespace

#endif
