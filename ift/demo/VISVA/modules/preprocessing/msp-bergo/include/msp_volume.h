
#ifndef msp_volume_h
#define msp_volume_h 1

#include <iostream>
#include <fstream>
#include <queue>
#include "msp_location.h"
#include "msp_boundlocation.h"
#include "msp_volumedomain.h"
#include "msp_adjacency.h"
#include "msp_geom.h"
#include "msp_pq.h"
#include "msp_timestamp.h"

using namespace std;

namespace MSP{

typedef enum {
  SagittalOrientation = 0,
  CoronalOrientation  = 1,
  AxialOrientation    = 2,
  UnknownOrientation  = 99
} VolumeOrientation;

template <class T> class Volume : public VolumeDomain {

 public:
  float dx,dy,dz;

  typedef BoundLocation iterator;

  Volume() : VolumeDomain() {
    data = 0;
    dx = dy = dz = 1.0;
  }

  Volume(int w,int h,int d, 
	 float pdx=1.0, float pdy=1.0,float pdz=1.0) :  VolumeDomain(w,h,d) {
    int i;

    dx = pdx;
    dy = pdy;
    dz = pdz;
    if (N < 0) data=NULL; else data = new T[N];
    if (!data) return;
    for(i=0;i<N;i++) data[i] = 0;

    anyi.bind(*this);
    ibegin = anyi.begin(); ibegin.bind(*this);
    iend   = anyi.end();   iend.bind(*this);
  }

  Volume(const char *filename) : VolumeDomain() {
    data = 0;
    readSCN(filename);
  }
  
  // copy constructor
  Volume(Volume<T> *src) {
    W = src->W;
    H = src->H;
    D = src->D;
    N = W*H*D;
    dx = src->dx;
    dy = src->dy;
    dz = src->dz;
    resize(W,H,D);
    data = new T[N];
    if (data != NULL)
      for(int i=0;i<N;i++)
	data[i] = src->data[i];
  }


  virtual ~Volume() {
    if (data!=NULL)  delete data;
  }

  iterator & begin() { return(ibegin); }
  iterator & end()   { return(iend); }

  void resize(int w,int h,int d) {
    VolumeDomain::resize(w,h,d);
    anyi.bind(*this);
    ibegin = anyi.begin();
    iend   = anyi.end();
    ibegin.bind(*this);
    iend.bind(*this);
  }

  void clear() {
    if (data!=0) delete data;
    data = 0;
    resize(0,0,0);
  }

  void writeSCN(const char *filename) {
    ofstream f;
    int i;

    f.open(filename,ofstream::binary);
    if (!f.good()) return;
    
    f << "SCN\n" << W << " " << H << " " << D << "\n";
    f << dx << " " << dy << " " << dz << "\n";
    
    switch(sizeof(T)) {
    case 1: f << "8\n"; break;
    case 2: f << "16\n"; break;
    case 4: f << "32\n"; break;
    default: f.close(); return;
    }

    for(i=0;i<H*D;i++)
      f.write( (char *) (&data[i*W]), sizeof(T) * W);

    f.close();
  }

  void writeSCN(const char * filename, int bits, bool sign) {

    if (bits < 0) {
      int64_t a = (int64_t) minimum();
      int64_t b = (int64_t) maximum();
      if (a>=0 && b<=255) { 
	bits = 8; sign = false;
      } else if (a>=-128 && b<=127) {
	bits = 8; sign = true;
      } else if (a>=0 && b<=65535) {
	bits= 16; sign = false;
      } else if (a>=-32768 && b<=32767) {
	bits = 16; sign = true;
      } else {
	bits = 32;
	sign = (a < 0);
      }
    }

    if (bits!=8 && bits!=16 && bits!=32) {
      cerr << "** writeSCN: invalid bits value\n\n";
      return;
    }

    ofstream f;
    int i,j;

    f.open(filename,ofstream::binary);
    if (!f.good()) return;
    
    f << "SCN\n" << W << " " << H << " " << D << "\n";
    f << dx << " " << dy << " " << dz << "\n";
    f << bits << "\n";

    if (bits==8 && !sign) {      
      uint8_t *xd = new uint8_t[W];
      T x;

      for(i=0;i<H*D;i++) {
	for(j=0;j<W;j++) {
	  x = data[i*W + j];
	  xd[j] = (x < 0 ? 0 : (x > 255 ? 255 : (uint8_t) x ));
	}
	f.write( (char *) xd, 1 * W);
      }
      delete xd;
    } else if (bits==8 && sign) {
      int8_t *xd = new int8_t[W];
      T x;
      for(i=0;i<H*D;i++) {
	for(j=0;j<W;j++) {
	  x = data[i*W + j];
	  xd[j] = (x < -128 ? -128 : (x > 127 ? 127 : (int8_t) x ));
	}
	f.write( (char *) xd, 1 * W);
      }
      delete xd;
    } else if (bits==16 && !sign) {
      uint16_t *xd = new uint16_t[W];
      T x;

      for(i=0;i<H*D;i++) {
	for(j=0;j<W;j++) {
	  x = data[i*W + j];
	  xd[j] = (x < 0 ? 0 : (x > 65535 ? 65535 : (uint16_t) x ));
	}
	f.write( (char *) xd, 2 * W);
      }
      delete xd;
    } else if (bits==16 && sign) {
      int16_t *xd = new int16_t[W];
      T x;

      for(i=0;i<H*D;i++) {
	for(j=0;j<W;j++) {
	  x = data[i*W + j];
	  xd[j] = (x < -32768 ? -32768 : (x > 32767 ? 32767 : (int16_t) x ));
	}
	f.write( (char *) xd, 2 * W);
      }
      delete xd;
    } else if (bits==32 && !sign) {
      uint32_t *xd = new uint32_t[W];
      T x;

      for(i=0;i<H*D;i++) {
	for(j=0;j<W;j++) {
	  x = data[i*W + j];
	  xd[j] = (x < 0 ? 0 : (uint32_t) x );
	}
	f.write( (char *) xd, 4 * W);
      }
      delete xd;
    } else if (bits==32 && sign) {
      int32_t *xd = new int32_t[W];
      T x;

      for(i=0;i<H*D;i++) {
	for(j=0;j<W;j++) {
	  x = data[i*W + j];
	  xd[j] = (int32_t) x;
	}
	f.write( (char *) xd, 4 * W);
      }
      delete xd;
    }

    f.close();
  }


  void readSCN(const char *filename) {
    ifstream f;
    int i,w=0,h=0,d=0,c,state,bpp=0;
    float y;
    string x;

    clear();

    f.open(filename,ifstream::binary);
    if (!f.good()) return;

    if (f.get() != 'S') return;
    if (f.get() != 'C') return;
    if (f.get() != 'N') return;
    if (f.get() != '\n') return;

    x.clear();    
    for(state=0;state!=7;) {
      c = f.get();
      if (c==' ' || c=='\t' || c=='\n') {
	y = atof(x.c_str());
	x.clear();
	switch(state) {
	case 0: w = (int) y; break;
	case 1: h = (int) y; break;
	case 2: d = (int) y; break;
	case 3: dx = y; break;
	case 4: dy = y; break;
	case 5: dz = y; break;
	case 6: bpp = (int) y; break;
	}
	state++;
      } else {
	x += c;
      }
      if (state==7 && c!='\n') { do { c=f.get(); } while(c!='\n'); }
    }
    
    resize(w,h,d);
    data = new T[N];
    if (!data) return;

    switch(bpp) {
    case 8:
      uint8_t *td8;
      td8 = new uint8_t[N];
      if (!td8) return;
      for(i=0;i<H*D;i++) f.read( (char *) (&td8[i*W]), W);
      for(i=0;i<N;i++) data[i] = (T) (td8[i]);
      delete td8;
      break;
    case 16:
      uint16_t *td16;
      td16 = new uint16_t[N];
      if (!td16) return;
      for(i=0;i<H*D;i++) f.read( (char *) (&td16[i*W]), 2*W);
      for(i=0;i<N;i++) data[i] = (T) (td16[i]);
      delete td16;
      break;
    case 32:
      int32_t *td32;
      td32 = new int32_t[N];
      if (!td32) return;
      for(i=0;i<H*D;i++) f.read( (char *) (&td32[i*W]), 4*W);
      for(i=0;i<N;i++) data[i] = (T) (td32[i]);
      delete td32;
      break;
    }
    f.close();
  }

  bool ok() { return(data!=0); }

  T & voxel(int x,int y,int z) { return(data[address(x,y,z)]); }
  T & voxel(float x,float y,float z) { return(data[address((int)x,(int)y,(int)z)]); }
  T & voxel(int a) { return(data[a]); }
  T & voxel(Location & loc) { return(data[address(loc)]); }

  T maximum() {
    T x = data[0];
    for(int i=0;i<N;i++) if (data[i] > x) x = data[i];
    return x;
  }

  T minimum() {
    T x = data[0];
    for(int i=0;i<N;i++) if (data[i] < x) x = data[i];
    return x;
  }

  void fill(T val) { for(int i=0;i<N;i++) data[i] = val; }

  /* algorithms */

  int max(  int a,  int b) { return((a>b)?a:b); }

  void normalize(int maxval) {

    int i;
    T cmax, cmin, camp, v;

    cmax = maximum();
    cmin = minimum();
    camp = cmax-cmin;

    for(i=0;i<N;i++) {
      v = voxel(i);
      v = ((v-cmin)*maxval)/camp;
      voxel(i) = v;
    }

  }

  Volume<char> *binaryThreshold(T val, bool invert=false) {
    int i;
    Volume<char> *r;
    r = new Volume<char>(W,H,D,dx,dy,dz);
    r->fill(0);
    for(i=0;i<N;i++)
      if (voxel(i) >= val) r->voxel(i) = 1;
    if (invert)
      for(i=0;i<N;i++)
	r->voxel(i) = 1 - r->voxel(i);
    return r;
  }

  void incrementBorder(float radius=1.5) {
    SphericalAdjacency A(radius);
    iterator p;
    Location q;
    int i;

    for(p=begin();p!=end();p++) {
      if (voxel(p)!=0)
	for(i=0;i<A.size();i++) {
	  q = A.neighbor(p,i);
	  if (valid(q))
	    if (voxel(q)==0) {
	      voxel(p)++;
	      break;
	    }
	}
    }
  }

  void clamp(T minval, T maxval) {
    int i;
    for(i=0;i<N;i++)
      if (data[i] < minval) data[i] = minval;
      else if (data[i] > maxval) data[i] = maxval;
  }
  
  Volume<char> * treePruning(Volume<char> *seeds) {
    BasicPQ *Q;
    Volume<int> *cost, *pred, *dcount;
    Volume<char> *seg, *B;
    SphericalAdjacency A6(1.0);
    int i,j,k,maxval,inf,p,c1,c2,dmax,pmax=0,gmax;
    Location q,r;
    queue<int> fifo;

    /* compute IFT */
    cost = new Volume<int>(W,H,D);
    pred = new Volume<int>(W,H,D);

    pred->fill(-1);

    maxval = (int) maximum();
    inf = maxval + 1;

    Q = new BasicPQ(N,maxval+1);

    for(i=0;i<N;i++) {
      if (seeds->voxel(i) != 0) {
	cost->voxel(i) = 0;
	Q->insert(i,0);
      } else {
	cost->voxel(i) = inf;
      }
    }

    while(!Q->empty()) {
      p = Q->remove();
      q.set( xOf(p), yOf(p), zOf(p) );
      for(i=0;i<A6.size();i++) {
	r = A6.neighbor(q,i);
	if (valid(r)) {
	  c1 = cost->voxel(r);
	  c2 = max( cost->voxel(q), this->voxel(r) );	  
	  if (c2 < c1) {
	    cost->voxel(r) = c2;
	    pred->voxel(r) = p;
	    Q->insert(address(r), c2);
	  }
	}
      }
    }

    delete cost;


     /* compute descendant count (old method) */
    dcount = new Volume<int>(W,H,D);
    B = this->border();
    for(i=0;i<N;i++) {
      if (B->voxel(i) != 0) {
	j = i;
	while(pred->voxel(j) != -1) {
	  dcount->voxel(pred->voxel(j))++;
	  j = pred->voxel(j);
	}
      }
    }
    
    /* find leaking points */
    for(i=0;i<N;i++) {
      if (B->voxel(i) == 1) {
	j = i;
	dmax = 0;
	while(pred->voxel(j) != -1) {
	  if (dcount->voxel(j) > dmax) {
	    dmax = dcount->voxel(j);
	    pmax = j;
	  }
	  j = pred->voxel(j);
	}
	k = pmax;
	gmax = this->voxel(k);
	while(pred->voxel(k) != -1) {
	  if ( (dcount->voxel(k) == dmax) && (gmax < this->voxel(k)) ) {
	    pmax = k;
	    gmax = this->voxel(k);
	  }
	  k = pred->voxel(k);
	}

	/*
	if (B->voxel(pmax) != 2)
	  cout << "prune at " << pmax << endl;
	*/
	B->voxel(pmax) = 2; // set leaking point
      }
    }

    /* prune trees */
    seg = new Volume<char>(W,H,D);
    seg->dx = dx;
    seg->dy = dy;
    seg->dz = dz;

    for(i=0;i<N;i++) {
      if (pred->voxel(i) == -1) {
	seg->voxel(i) = 1;
	fifo.push(i);

	while(!fifo.empty()) {
	  j = fifo.front();
	  fifo.pop();
	  q.set( xOf(j), yOf(j), zOf(j) );
	  seg->voxel(j) = 1;

	  for(k=0;k<A6.size();k++) {
	    r = A6.neighbor(q,k);
	    if (valid(r))
	      if ( (pred->voxel(r) == j) && (B->voxel(r) != 2) )
		fifo.push(address(r));
	  }
	}
      }
    }

    delete pred;
    delete dcount;
    delete B;
    return seg;
  }

  Volume<char> * border() {
    Volume<char> *B;
    int i,j;
    B = new Volume<char>(W,H,D);
    for(i=0;i<W;i++)
      for(j=0;j<H;j++)
	B->voxel(i,j,0) = B->voxel(i,j,D-1) = 1;
    for(i=0;i<W;i++)
      for(j=0;j<D;j++)
	B->voxel(i,0,j) = B->voxel(i,H-1,j) = 1;
    for(i=0;i<H;i++)
      for(j=0;j<D;j++)
	B->voxel(0,i,j) = B->voxel(W-1,i,j) = 1;
    return B;
  }

  Volume<T> * erode(Adjacency &A) {
    int j;
    T val, upper, lower;
    Volume<T> * dest;
    iterator p;
    Location q;

    dest = new Volume<T>(W,H,D);
    upper = maximum();
    lower = minimum();

    for(p=begin();p!=end();p++) {
      val = upper;
      for(j=0;j<A.size();j++) {
	q = A.neighbor(p,j);
	if (valid(q)) if (voxel(q) < val) val = voxel(q);
	if (val == lower) break;
      }
      dest->voxel(p) = val;
    }
    dest->dx = dx;
    dest->dy = dy;
    dest->dz = dz;
    return dest;
  }

  Volume<T> * dilate(Adjacency &A) {
    int j;
    T val, upper, lower;
    Volume<T> * dest;
    iterator p;
    Location q;

    dest = new Volume<T>(W,H,D);
    upper = maximum();
    lower = minimum();

    for(p=begin();p!=end();p++) {
      val = lower;
      for(j=0;j<A.size();j++) {
	q = A.neighbor(p,j);
	if (valid(q)) if (voxel(q) > val) val = voxel(q);
	if (val == upper) break;
      }
      dest->voxel(p) = val;
    }
    dest->dx = dx;
    dest->dy = dy;
    dest->dz = dz;
    return dest;
  }

  Volume<T> * binaryErode(float radius) {
    Volume<int> *edt, *root;
    Volume<T> *dest;
    int rlimit;

    Location q,r,rr;
    iterator p;
    SphericalAdjacency A(1.5);
    BasicPQ *Q;
    int ap,ar,c1,c2,n,i,j;

    rlimit = 1 + (int) (radius * radius);

    edt = new Volume<int>(W,H,D);
    root = new Volume<int>(W,H,D);
    Q = new BasicPQ(N,W*W+H*H+D*D+1);

    for(p=begin();p!=end();p++) {
      ap = address(p);
      if (voxel(ap) == 0) {
	edt->voxel(ap) = 0;
	root->voxel(ap) = ap;
	Q->insert(ap, 0);
      } else {
	edt->voxel(ap) = rlimit;
      }
    }

    while(!Q->empty()) {
      n = Q->remove();
      q.set( xOf(n), yOf(n), zOf(n) );
      j = root->voxel(n);
      rr.set( xOf(j), yOf(j), zOf(j) );
      for(i=0;i<A.size();i++) {
        r = A.neighbor(q,i);
        if (valid(r)) {
          c2 = r.sqdist(rr);
          c1 = edt->voxel(r);
          if (c2 < c1) {
            ar = address(r);
            edt->voxel(ar) = c2;
            root->voxel(ar) = j;
            if (Q->colorOf(ar) == PriorityQueue::Gray)
              Q->update(ar,c1,c2);
            else
              Q->insert(ar,c2);
          }
        }
      }
    }
    delete Q;
    delete root;

    dest = new Volume<T>(W,H,D);
    dest->dx = dx;
    dest->dy = dy;
    dest->dz = dz;

    for(i=0;i<edt->N;i++)
      dest->voxel(i) = edt->voxel(i) < rlimit ? 0 : 1;
    delete edt;
    return dest;
  }

  Volume<T> * binaryDilate(float radius) {
    Volume<int> *edt, *root;
    Volume<T> *dest;
    int rlimit;

    Location q,r,rr;
    iterator p;
    SphericalAdjacency A(1.5);
    BasicPQ *Q;
    int ap,ar,c1,c2,n,i,j;

    rlimit = 1 + (int) (radius * radius);

    edt = new Volume<int>(W,H,D);
    root = new Volume<int>(W,H,D);
    Q = new BasicPQ(N,W*W+H*H+D*D+1);

    for(p=begin();p!=end();p++) {
      ap = address(p);
      if (voxel(ap) == 1) {
	edt->voxel(ap) = 0;
	root->voxel(ap) = ap;
	Q->insert(ap, 0);
      } else {
	edt->voxel(ap) = rlimit;
      }
    }

    while(!Q->empty()) {
      n = Q->remove();
      q.set( xOf(n), yOf(n), zOf(n) );
      j = root->voxel(n);
      rr.set( xOf(j), yOf(j), zOf(j) );
      for(i=0;i<A.size();i++) {
        r = A.neighbor(q,i);
        if (valid(r)) {
          c2 = r.sqdist(rr);
          c1 = edt->voxel(r);
          if (c2 < c1) {
            ar = address(r);
            edt->voxel(ar) = c2;
            root->voxel(ar) = j;
            if (Q->colorOf(ar) == PriorityQueue::Gray)
              Q->update(ar,c1,c2);
            else
              Q->insert(ar,c2);
          }
        }
      }
    }
    delete Q;
    delete root;

    dest = new Volume<T>(W,H,D);
    dest->dx = dx;
    dest->dy = dy;
    dest->dz = dz;

    for(i=0;i<edt->N;i++)
      dest->voxel(i) = edt->voxel(i) < rlimit ? 1 : 0;
    delete edt;
    return dest;
  }

  void binaryIntersection(Volume<T> *src) {
    unsigned int i,un;
    un = (unsigned int) N;
    for(i=0;i<un;i++)
      if (src->voxel(i) == 0)
	this->voxel(i) = 0;
  }

  Volume<T> * binaryClose(float radius) {
    Volume<T> *tb,*x,*y;
    int border,i,j,k;

    border = (int) ceil(radius);
    tb = new Volume<T>(W+2*border,H+2*border,D+2*border);
    for(k=0;k<D;k++)
      for(j=0;j<H;j++)
	for(i=0;i<W;i++)
	  tb->voxel(i+border,j+border,k+border) = voxel(i,j,k);

    x = tb->binaryDilate(radius);
    y = x->binaryErode(radius);
    delete tb;

    tb = new Volume<T>(W,H,D);
    tb->dx = dx;
    tb->dy = dy;
    tb->dz = dz;

    for(k=0;k<D;k++)
      for(j=0;j<H;j++)
	for(i=0;i<W;i++)
	  tb->voxel(i,j,k) = y->voxel(i+border,j+border,k+border);
    
    delete x;
    delete y;
    return tb;
  }

  Volume<T> * binaryOpen(float radius) {
    Volume<T> *tb,*x,*y;
    int border,i,j,k;

    border = (int) ceil(radius);
    tb = new Volume<T>(W+2*border,H+2*border,D+2*border);
    for(k=0;k<D;k++)
      for(j=0;j<H;j++)
	for(i=0;i<W;i++)
	  tb->voxel(i+border,j+border,k+border) = voxel(i,j,k);

    x = tb->binaryErode(radius);
    y = x->binaryDilate(radius);
    delete tb;

    tb = new Volume<T>(W,H,D);
    tb->dx = dx;
    tb->dy = dy;
    tb->dz = dz;

    for(k=0;k<D;k++)
      for(j=0;j<H;j++)
	for(i=0;i<W;i++)
	  tb->voxel(i,j,k) = y->voxel(i+border,j+border,k+border);
    
    delete x;
    delete y;
    return tb;
  }

  Volume<char> * brainMarkerComp() {
    int i,a=0,b,c=0,x;
    Volume<char> *dest,*ero;
    Volume<int> *tmp;
    SphericalAdjacency R5(5.0/dx,true), R1(1.0,false);

    b = computeOtsu();
    meansAboveBelowT(b,c,a);
    tmp = applySShape(a,b,c);
    dest = new Volume<char>(W,H,D);
    for(i=0;i<N;i++) {
      x = tmp->voxel(i);
      if (x >= b && x < 4096)
	dest->voxel(i) = 1;
    }
    delete tmp;

    ero = dest->erode(R5);
    delete dest;
    ero->selectLargestComp(R1);
    return ero;
  }

  void selectLargestComp(Adjacency &A) {
    Volume<int> * label;
    queue<int> Q;
    int i,j,k,L=1;
    Location p,q;
    int bestarea=0,bestcomp=0,area=0;

    label = new Volume<int>(W,H,D);
       
    for(i=0;i<N;i++) {
      if (this->voxel(i)==1 && label->voxel(i)==0) {
	area = 1;
	label->voxel(i) = L;
	Q.push(i);
	while(!Q.empty()) {
	  j = Q.front();
	  Q.pop();

	  p.set( xOf(j), yOf(j), zOf(j) );
	  for(k=0;k<A.size();k++) {
	    q = A.neighbor(p,k);
	    if (valid(q)) {
	      if (this->voxel(q)==1 && label->voxel(q)==0) {
		label->voxel(q) = label->voxel(p);
		++area;
		Q.push(address(q));
	      }
	    }
	  }
	}
      }
      if (area > bestarea) {
	bestarea = area;
	bestcomp = L;
      }
      L++;
    }

    for(i=0;i<N;i++)
      if (label->voxel(i) != bestcomp)
	this->voxel(i) = 0;

    delete label;
  }

  Volume<int> * labelBinComp(Adjacency &A) {
    Volume<int> * label;
    queue<int> Q;
    int i,j,k,L=1;
    Location p,q;

    label = new Volume<int>(W,H,D);
    
    for(i=0;i<N;i++) {
      if (voxel(i)==1 && label->voxel(i)==0) {
	label->voxel(i) = L;
	Q.push(i);
	while(!Q.empty()) {
	  j = Q.front();
	  Q.pop();

	  p.set( xOf(j), yOf(j), zOf(j) );
	  for(k=0;k<A.size();k++) {
	    q = A.neighbor(p,k);
	    if (valid(q)) {
	      if (voxel(q)==1 && label->voxel(q)==0) {
		label->voxel(q) = label->voxel(p);
		Q.push(address(q));
	      }
	    }
	  }
	}
      }
      L++;
    }
    return(label);
  }

  Volume<int> * otsuEnhance() {
    int a=0,b,c=0;
    b = computeOtsu();
    meansAboveBelowT(b,c,a);
    //    cout << "a=" << a << "  b=" << b << "  c=" << c << endl;
    return(applySShape(a,b,c));
  }

  int computeOtsu() {
    double *hist;
    int i,t,hn,imax,topt=0;
    double p1,p2,m1,m2,s1,s2,j,jmax=-1.0;

    imax = (int) maximum();
    hn = imax + 1;
    hist = new double[hn];
    for(i=0;i<hn;i++) hist[i] = 0;
    for(i=0;i<N;i++) ++hist[data[i]];
    for(i=0;i<hn;i++) hist[i] /= ((double)N);

    for(t=1;t<imax;t++) {
      for(i=0,p1=0.0;i<=t;i++) p1 += hist[i];
      p2 = 1.0 - p1;
      if ((p1>0.0)&&(p2>0.0)) {
	for(i=0,m1=0.0;i<=t;i++) m1 += hist[i] * i;
	m1 /= p1;
	for(i=t+1,m2=0.0;i<=imax;i++) m2 += hist[i] * i;
	m2 /= p2;
	for(i=0,s1=0.0;i<=t;i++) s1 += hist[i] * (i-m1) * (i-m1);
	s1 /= p1;
	for(i=t+1,s2=0.0;i<=imax;i++) s2 += hist[i] * (i-m2) * (i-m2);
	s2 /= p2;
	j = (p1*p2*(m1-m2)*(m1-m2))/(p1*s1+p2*s2);
      } else {
	j = 0.0;
      }
      if (j > jmax) { jmax = j; topt = t; }
    }
    delete hist;
    return topt;
  }

  int computeOtsu(Volume<char> *mask) {
    double *hist;
    int i,t,hn,imax,topt=0, RN;
    double p1,p2,m1,m2,s1,s2,j,jmax=-1.0;

    imax = (int) maximum();
    hn = imax + 1;
    hist = new double[hn];
    for(i=0;i<hn;i++) hist[i] = 0;
    RN = 0;
    for(i=0;i<N;i++) if (mask->voxel(i)!=0) { ++hist[data[i]]; ++RN; }
    for(i=0;i<hn;i++) hist[i] /= ((double)RN);

    for(t=1;t<imax;t++) {
      for(i=0,p1=0.0;i<=t;i++) p1 += hist[i];
      p2 = 1.0 - p1;
      if ((p1>0.0)&&(p2>0.0)) {
	for(i=0,m1=0.0;i<=t;i++) m1 += hist[i] * i;
	m1 /= p1;
	for(i=t+1,m2=0.0;i<=imax;i++) m2 += hist[i] * i;
	m2 /= p2;
	for(i=0,s1=0.0;i<=t;i++) s1 += hist[i] * (i-m1) * (i-m1);
	s1 /= p1;
	for(i=t+1,s2=0.0;i<=imax;i++) s2 += hist[i] * (i-m2) * (i-m2);
	s2 /= p2;
	j = (p1*p2*(m1-m2)*(m1-m2))/(p1*s1+p2*s2);
      } else {
	j = 0.0;
      }
      if (j > jmax) { jmax = j; topt = t; }
    }
    delete hist;
    return topt;
  }

  void meansAboveBelowT(int t, int &t1, int &t2) {
    int64_t m1=0, m2=0, nv1=0, nv2=0;
    int p,delta;
    for(p=0;p<N;p++) {
      if (data[p]<t) {
	m1 += data[p];
	++nv1;
      } else if (data[p]>t) {
	m2 += data[p];
	++nv2;
      }
    }
    delta = (int) ((m2/nv2 - m1/nv1)/2);
    t1 = t+delta;
    t2 = t-delta;
  }

  Volume<int> * applySShape(int a,int b,int c) {
    Volume<int> * dest;
    float x;
    int p;
    float weight,sqca;

    dest = new Volume<int>(W,H,D);
    dest->dx = dx;
    dest->dy = dy;
    dest->dz = dz;
    sqca = (float) ((c-a)*(c-a));

    for(p=0;p<N;p++) {
      x = voxel(p);
      if (x<=a) {
	weight = 0.0;
      } else {
	if (x>a && x<=b) {
	  weight = (2.0* (x-a) * (x-a)) / sqca;
	} else if (x>b && x<=c) {
	  weight = (1.0 - 2.0*(x-c)*(x-c)/sqca );
	} else {
	  weight = 1.0;
	}
      }
      dest->voxel(p) = (int) (weight * x);
    }
    return dest;
  }

  Volume<T> * featureGradient() {
    Volume<T> * dest;
    int i,ap;
    float *f,*mg,dist,gx,gy,gz,v,imax;
    SphericalAdjacency A6(1.0,false), A7(1.0,true);
    Location p, q;

    imax = (float) maximum();
    f = new float[N * 7];
    if (!f) return 0;
    mg = new float[6];

    dest = new Volume<T>(W,H,D);
    dest->dx = dx;
    dest->dy = dy;
    dest->dz = dz;
    
    for(p.Z=0;p.Z<D;p.Z++)
      for(p.Y=0;p.Y<H;p.Y++)
	for(p.X=0;p.X<W;p.X++) {
	  ap = address(p);
	  for(i=0;i<A7.size();i++) {
	    q = A7.neighbor(p,i);
	    if (valid(q)) {
	      f[(7*ap)+i] = voxel(q) / imax ;
	    } else
	      f[(7*ap)+i] = 0.0;
	  }
	}

    p.set(0,0,0);
    for(i=0;i<6;i++)
      mg[i] = sqrt(A6.neighbor(p,i).sqlen());

    for(p.Z=0;p.Z<D;p.Z++) {
      for(p.Y=0;p.Y<H;p.Y++) {
	for(p.X=0;p.X<W;p.X++) {
	  ap = address(p);
	  gx = gy = gz = 0.0;
	  for(i=0;i<A6.size();i++) {
	    q = A6.neighbor(p,i);
	    if (valid(q)) {
	      dist = featureDistance(&f[7*ap],&f[7*address(q)],7);
	      gx += (dist * A6[i].X) / mg[i];
	      gy += (dist * A6[i].Y) / mg[i];
	      gz += (dist * A6[i].Z) / mg[i];
	    }
	  }
	  v = 100000.0 * sqrt(gx*gx + gy*gy + gz*gz);
	  dest->voxel(ap) = (T) v;
	}
      }
    }

    delete f;
    delete mg;

    return dest;
  }

  float featureDistance(float *a,float *b,int n) {
      int i;
      float dist=0.0;
      for (i=0;i<n;i++)
	dist += (b[i]-a[i])/2.0;
      dist /= n;
      return(exp(-(dist-0.5)*(dist-0.5)/2.0));
  }

  void complement() {
    int i;
    T mval;

    mval = maximum();
    for(i=0;i<N;i++)
      voxel(i) = mval - voxel(i);
  }

  // input: 0=do not expand, 1=expand, 2=seed
  Volume<int> * edt() {
    Volume<int> *edt, *root;
    
    Location q,r,rr;
    iterator p;
    SphericalAdjacency A(1.5);
    BasicPQ *Q;
    int ap,ar,c1,c2,n,i,j;

    edt = new Volume<int>(W,H,D);
    root = new Volume<int>(W,H,D);
    Q = new BasicPQ(N,W*W+H*H+D*D+1);

    for(p=begin();p!=end();p++) {
      ap = address(p);
      switch(voxel(ap)) {
      case 0: edt->voxel(ap) = -1; break;
      case 1: edt->voxel(ap) = 2*(W*W+H*H+D*D); break;
      case 2: edt->voxel(ap) = 0; root->voxel(ap) = ap; Q->insert(ap,0); break;
      }
    }
 
    //cout << "initial state of 1,H-1,1 (this) = " << (int) (voxel(1,H-1,1)) << endl;
    //cout << "initial state of 1,H-1,1 (edt)  = " << edt->voxel(1,H-1,1) << endl;
   
    //cout << 2*(W*W+H*H+D*D);
    
    while(!Q->empty()) {
      n = Q->remove();
      q.set( xOf(n), yOf(n), zOf(n) );
      j = root->voxel(n);
      rr.set( xOf(j), yOf(j), zOf(j) );
      for(i=0;i<A.size();i++) {
	r = A.neighbor(q,i);
	if (valid(r)) {
	  c2 = r.sqdist(rr);
	  c1 = edt->voxel(r);
	  if (c2 < c1) {
	    ar = address(r);
	    edt->voxel(ar) = c2;
	    root->voxel(ar) = j;
	    if (Q->colorOf(ar) == PriorityQueue::Gray)
	      Q->update(ar,c1,c2);
	    else
	      Q->insert(ar,c2);
	  }
	}
      }
    }
    delete Q;
    delete root;

    //cout << "final state of 1,H-1,1 (edt) = " << edt->voxel(1,H-1,1) << endl;

    return edt;
  };

  Volume<T> * isometricInterpolation() {
    float md;

    if (dx==dy && dy==dz) {
      Volume<T> *v;
      int i;
      v = new Volume<T>(W,H,D);
      v->dx = dx;
      v->dy = dy;
      v->dz = dz;
      for(i=0;i<N;i++)
	v->voxel(i) = this->voxel(i);
      return v;
    }

    md = (dx < dy) ? dx : dy;
    md = (dz < md) ? dz : md;
    return(interpolate(md,md,md));
  }

  Volume<T> * interpolate(float ndx, float ndy, float ndz) {
     Volume<T> *dest, *tmp;
     int value;
     Location Q,P,R;
     float walked_dist,dist_PQ;

     if (ndx==dx && ndy==dy && ndz==dz) {
       return(new Volume<T>(this));
     }

     dest = this;

     if (ndx != dx) {
       tmp = new Volume<T>( (int)((float)(dest->W-1)*dest->dx/ndx)+1,
			    dest->H,
			    dest->D);
       for(Q.X=0; Q.X < tmp->W; Q.X++) {
	 for(Q.Z=0; Q.Z < tmp->D; Q.Z++)
	   for(Q.Y=0; Q.Y < tmp->H; Q.Y++) {
	     
	     walked_dist = (float)Q.X * ndx;
	     P.X = (int)(walked_dist/dx);
	     P.Y = Q.Y; P.Z = Q.Z;
	     R.X = P.X + 1;
	     if (R.X >= dest->W) R.X = dest->W - 1;
	     R.Y = P.Y; R.Z = P.Z;
	     dist_PQ =  walked_dist - (float)P.X * dx;
	     
	     value = (int)((( dx - dist_PQ)*
			    (float)(dest->voxel(P)) +
			    dist_PQ * (float) dest->voxel(R)) / dx);
	     tmp->voxel(Q) = (T) value;
	   }
       }
       dest = tmp;
     }

     if (ndy != dy) {
       tmp = new Volume<T>(dest->W,
			   (int)(((float)dest->H-1.0) * dy / ndy)+1,
			   dest->D);
       for(Q.Y=0; Q.Y < tmp->H; Q.Y++)
	 for(Q.Z=0; Q.Z < tmp->D; Q.Z++)
	   for(Q.X=0; Q.X < tmp->W; Q.X++) {
	     
	     walked_dist = (float)Q.Y * ndy;
	     
	     P.X = Q.X;
	     P.Y = (int)(walked_dist/dy);
	     P.Z = Q.Z;
	     R.X = P.X;
	     R.Y = P.Y + 1;
	     if (R.Y >= dest->H) R.Y = dest->H - 1;
	     R.Z = P.Z;
	     dist_PQ =  walked_dist - (float)P.Y * dy;
	     
	     value = (int)(((dy - dist_PQ)*
			    (float)(dest->voxel(P)) +
			    dist_PQ * (float)(dest->voxel(R))) / dy);
	     tmp->voxel(Q) = (T) value;
	   }
       if (dest != this)
	 delete dest;
       dest=tmp;
     }

     if (ndz != dz) {
       tmp = new Volume<T>(dest->W,
			   dest->H,
			   (int)(((float)dest->D-1.0) * dz / ndz)+1);
       for(Q.Z=0; Q.Z < tmp->D; Q.Z++)
	 for(Q.Y=0; Q.Y < tmp->H; Q.Y++)
	   for(Q.X=0; Q.X < tmp->W; Q.X++) {
	     
	     walked_dist = (float)Q.Z * ndz;
	     
	     P.X = Q.X; P.Y = Q.Y;
	     P.Z = (int)(walked_dist/dz);
	     
	     R.X = P.X; R.Y = P.Y;
	     R.Z = P.Z + 1;
	     if (R.Z >= dest->D) R.Z = dest->D - 1;	     
	     dist_PQ =  walked_dist - (float)P.Z * dz;
	     
	     value = (int)(((dz - dist_PQ)*
			    (float)(dest->voxel(P)) +
			    dist_PQ * (float)(dest->voxel(R))) / dz);
	     tmp->voxel(Q) = (T) value;
	   }
       if (dest != this)
	 delete dest;
       dest=tmp;
     }
     
     
     dest->dx=ndx;
     dest->dy=ndy;
     dest->dz=ndz;
     return(dest);
  }

  double KLMeasure(int z, double *href) {
    int i;
    double *h, kl=0.0;
    h  = histProb(z,4096);
    for (i=0;i<4096;i++)
      if (h[i]>1E-15 && href[i]>1E-15)
	kl += h[i]*log(h[i]/href[i]);
    delete h;
    return kl;
  }

  double KLMeasure(R3 &p1, R3 &p2, R3 &p3, double *href) {
    int i;
    double *h, kl=0.0;
    h  = histProb(p1,p2,p3,4096);
    for (i=0;i<4096;i++)
      if (h[i]>1E-15 && href[i]>1E-15)
	kl += h[i]*log(h[i]/href[i]);
    delete h;
    return kl;
  }


  double *histProb(int z, int bins) {
    double *h;
    int i, j, k, n=0;
    h = new double[bins];
    for(i=0;i<bins;i++)
      h[i] = 0.0;
    for(j=0;j<H;j++)
      for(i=0;i<W;i++) {
	k = voxel(i,j,z);
	if (k >= bins) k = bins - 1;
	h[k]++;
	n++;
      }
    for(i=0;i<bins;i++)
      h[i] /= n;
    return h;	
  }

  double *histProb(R3 &p1, R3 &p2, R3 &p3, int bins) {
    double *h;
    int i, j, k, z, n=0;
    h = new double[bins];
    for(i=0;i<bins;i++)
      h[i] = 0.0;
    for(j=0;j<H;j++)
      for(i=0;i<W;i++) {
	z = (int) planeZ(i,j,p1,p2,p3);
	if (valid(i,j,z)) {
	  k = voxel(i,j,z);
	  if (k >= bins) k = bins - 1;
	  h[k]++;
	  n++;
	}
      }
    for(i=0;i<bins;i++)
      h[i] /= n;
    return h;	
  }

  int volkauReferencePlane() {
    return(D/2 - (int)(20.0/dz));
  }

  double * volkauReferenceHist() {
    return(histProb(volkauReferencePlane(),4096));
  }

  double planeZ(double x, double y, R3 &p1, R3 &p2, R3 &p3) {
    double res,a,b,c,d;
    a = p1.Y*(p2.Z - p3.Z) + p2.Y*(p3.Z - p1.Z) + p3.Y*(p1.Z - p2.Z);
    b = p1.Z*(p2.X - p3.X) + p2.Z*(p3.X - p1.X) + p3.Z*(p1.X - p2.X);
    c = p1.X*(p2.Y - p3.Y) + p2.X*(p3.Y - p1.Y) + p3.X*(p1.Y - p2.Y);
    d = -(p1.X*(p2.Y*p3.Z - p3.Y*p2.Z) + p2.X*(p3.Y*p1.Z - p1.Y*p3.Z) + p3.X*(p1.Y*p2.Z - p2.Y*p1.Z));
    res = - (d + b*y + a*x) / c;
    return res;
  }

  int volkauFirst() {
    double kl,maxkl=0.0;
    double *href;
    int i, zref, zi, zf, delta, zmax=0;

    delta = (int) (20.0 / dz);
    zref = (D / 2) - delta;
    href = histProb(zref, 4096);
    zi = zref + 1;
    zf = zref + 2 * delta;
    for(i=zi;i<=zf;i++) {
      kl = KLMeasure(i,href);
      if (kl > maxkl) {
	maxkl = kl;
	zmax = i;
      }
    }
    delete href;

    cout << "stage 1: delta=" << delta << ", zref=" << zref << ", zmax=" << zmax << ", kl=" << maxkl << endl;

    return zmax;
  }

  Triangle & volkauSecond(int z1) {
    static Triangle r;
    R3 p1,p2,p3;
    int i,j,k,delta,mi=0,mj=0,mk=0;
    double kl,klmax=0.0;
    double *href;

    delta = (int) (5.0 / dz);

    p1.set(0,0,0);
    p2.set(W,0,0);
    p3.set(0,H,0);

    href = volkauReferenceHist();

    for(i=-4;i<=4;i++)
      for(j=-4;j<=4;j++)
	for(k=-4;k<=4;k++) {
	  p1.Z = z1 + i * delta;
	  p2.Z = z1 + j * delta;
	  p3.Z = z1 + k * delta;
	  kl = KLMeasure(p1,p2,p3,href);
	  if (kl>klmax) {
	    klmax = kl;
	    mi=i; mj=j; mk=k;
	  }
	}
    p1.Z = z1 + mi*delta;
    p2.Z = z1 + mj*delta;
    p3.Z = z1 + mk*delta;

    delete href;

    r.set(p1,p2,p3);
    return r;
  }

  Triangle & volkauThird(Triangle & t2, int maxdelta=3) {
    static Triangle r;
    R3 p1,p2,p3;
    int i,j,k,delta,mi=0,mj=0,mk=0;
    double kl,klmax=0.0;
    double *href;

    p1.set(0,0,0);
    p2.set(W,0,0);
    p3.set(0,H,0);

    href = volkauReferenceHist();

    for(delta=maxdelta;delta>=1;delta--)
      for(i=-2;i<=2;i++)
	for(j=-2;j<=2;j++)
	  for(k=-2;k<=2;k++) {
	    p1.Z = t2.A.Z + i * delta;
	    p2.Z = t2.B.Z + j * delta;
	    p3.Z = t2.C.Z + k * delta;
	    kl = KLMeasure(p1,p2,p3,href);
	    if (kl>klmax) {
	      klmax = kl;
	      mi=i; mj=j; mk=k;
	    }
	  }
    p1.Z = t2.A.Z + mi*delta;
    p2.Z = t2.B.Z + mj*delta;
    p3.Z = t2.C.Z + mk*delta;

    delete href;

    r.set(p1,p2,p3);
    return r;
  }

  // find darker XY slice in this volume, constrained by
  // the given mask. Ignore slices with less than minArea
  // voxels within the mask
  int darkerSlice(Volume<char> *mask, int minArea) {
    int i,j,k,n,eW,eH,eD;
    float acc;
    float minval;
    int minz;
    
    minval = (float) maximum();
    minz = 0;
    
    eW = W < mask->W ? W : mask->W;
    eH = H < mask->H ? H : mask->H;
    eD = D < mask->D ? D : mask->D;

    for(k=0;k<eD;k++) {
      acc = 0.0;
      n = 0;

      for(j=0;j<eH;j++)
	for(i=0;i<eW;i++) {
	  if (mask->voxel(i,j,k) != 0) {
	    acc += (float) voxel(i,j,k);
	    n++;
	  }
	}
      
      if (n < minArea) continue;
      acc /= (float) n;
      if (acc < minval) {
	minval = acc;
	minz = k;
      }
    }
    return minz;
  }

  // darkness score for finding the plane between hemispheres
  float planeScore(int z, Volume<char> *mask, T4 &X) {
    int i,j,n=0;
    R3 r;
    float acc=0.0;

    for(j=-H;j<2*H;j++)
      for(i=-W;i<2*W;i++) {
	  r.set(i,j,z);
	  r = X.apply(r);
	  if (valid(r.X,r.Y,r.Z) && mask->valid(r.X,r.Y,r.Z))
	    if (mask->voxel(r.X,r.Y,r.Z) != 0) {
	      acc += voxel(r.X,r.Y,r.Z);
	      n++;
	    }
      }

    if (n!=0) acc /= n;
    return acc;
  }

  // edt-based score for finding the plane between hemispheres
  float planeScore2(int z, Volume<char> *mask, Volume<float> *csfedt, T4 &X, int minArea=0) {
    int i,j,n=0;
    R3 r;
    float acc=0.0, acc2=0.0;

    for(j=-H;j<2*H;j++)
      for(i=-W;i<2*W;i++) {
	  r.set(i,j,z);
	  r = X.apply(r);
	  if (valid(r.X,r.Y,r.Z) && mask->valid(r.X,r.Y,r.Z) && csfedt->valid(r.X,r.Y,r.Z))
	    if (mask->voxel(r.X,r.Y,r.Z) != 0) {
	      acc  += voxel(r.X,r.Y,r.Z);
	      acc2 += csfedt->voxel(r.X,r.Y,r.Z);
	      n++;
	    }
      }

    if (n<minArea) return(1.0E15);

    if (n!=0) { acc /= n; acc2 /= n; }
    
    return(acc*acc2);
  }

  int bestSlice2(Volume<char> *mask, Volume<float> *csfedt, int minArea) {
    int k, minz;
    float acc, minval;
    T4 I;
    
    minval = 1.0E15;
    minz = 0;
    I.identity();

    for(k=0;k<D;k++) {
      acc = planeScore2(k,mask,csfedt,I,minArea);

      //cout << "k=" << k << " acc=" << acc << endl;

      if (acc < minval) {
	minval = acc;
	minz = k;
      }
    }
    return minz;
  }

  void planeFit2(int z, Volume<char> *mask, Volume<float> *csfedt, T4 &X,int &niter,const char *outfile=NULL,bool verbose=false) {
    T4 lib[42], at, bestat;
    float myscore, c, best, fscore;
    float cx,cy,cz;
    int i,j,whobest;

    cx = W/2.0;
    cy = H/2.0;
    cz = (float) z;

    const char *desc[42] = {
      "TX/-1","TX/+1",
      "TY/-1","TY/+1",
      "TZ/-1","TZ/+1",
      "RX/-1","RX/+1",
      "RY/-1","RY/+1",
      "RZ/-1","RZ/+1",
      "TX/-5","TX/+5",
      "TY/-5","TY/+5",
      "TZ/-5","TZ/+5",
      "RX/-5","RX/+5",
      "RY/-5","RY/+5",
      "RZ/-5","RZ/+5",
      "RX/-10","RX/+10",
      "RY/-10","RY/+10",
      "RZ/-10","RZ/+10",
      "TX/-10","TX/+10",
      "TY/-10","TY/+10",
      "TZ/-10","TZ/+10",
      "RX/-0.5","RX/+0.5",
      "RY/-0.5","RY/+0.5",
      "RZ/-0.5","RZ/+0.5",
    };

    lib[0].translate(-1.0, 0.0, 0.0);
    lib[1].translate( 1.0, 0.0, 0.0);
    lib[2].translate( 0.0,-1.0, 0.0);
    lib[3].translate( 0.0, 1.0, 0.0);
    lib[4].translate( 0.0, 0.0,-1.0);
    lib[5].translate( 0.0, 0.0, 1.0);

    lib[6].xrot(1.0, cx,cy,cz);
    lib[7].xrot(-1.0, cx,cy,cz);
    lib[8].yrot(1.0, cx,cy,cz);
    lib[9].yrot(-1.0, cx,cy,cz);
    lib[10].zrot(1.0, cx,cy,cz);
    lib[11].zrot(-1.0, cx,cy,cz);

    lib[12].translate(-5.0, 0.0, 0.0);
    lib[13].translate( 5.0, 0.0, 0.0);
    lib[14].translate( 0.0,-5.0, 0.0);
    lib[15].translate( 0.0, 5.0, 0.0);
    lib[16].translate( 0.0, 0.0,-5.0);
    lib[17].translate( 0.0, 0.0, 5.0);

    lib[18].xrot(5.0, cx,cy,cz);
    lib[19].xrot(-5.0, cx,cy,cz);
    lib[20].yrot(5.0, cx,cy,cz);
    lib[21].yrot(-5.0, cx,cy,cz);
    lib[22].zrot(5.0, cx,cy,cz);
    lib[23].zrot(-5.0, cx,cy,cz);

    lib[24].xrot(10.0, cx,cy,cz);
    lib[25].xrot(-10.0, cx,cy,cz);
    lib[26].yrot(10.0, cx,cy,cz);
    lib[27].yrot(-10.0, cx,cy,cz);
    lib[28].zrot(10.0, cx,cy,cz);
    lib[29].zrot(-10.0, cx,cy,cz);

    lib[30].translate(-10.0, 0.0, 0.0);
    lib[31].translate( 10.0, 0.0, 0.0);
    lib[32].translate( 0.0,-10.0, 0.0);
    lib[33].translate( 0.0, 10.0, 0.0);
    lib[34].translate( 0.0, 0.0,-10.0);
    lib[35].translate( 0.0, 0.0, 10.0);

    lib[36].xrot(0.5, cx,cy,cz);
    lib[37].xrot(-0.5, cx,cy,cz);
    lib[38].yrot(0.5, cx,cy,cz);
    lib[39].yrot(-0.5, cx,cy,cz);
    lib[40].zrot(0.5, cx,cy,cz);
    lib[41].zrot(-0.5, cx,cy,cz);

    fscore = myscore = planeScore2(z,mask,csfedt,X);

    j = 0;
    do {
      whobest = -1;
      best = myscore;
      bestat = X;
    
      for(i=0;i<42;i++) {
	at = X * lib[i];
	c = planeScore2(z,mask,csfedt,at);
	if (c < best && fabsf(c-best)>1.0E-4f) {
	  best = c;
	  whobest = i;
	  bestat = at;
	}
      }

      if (whobest >= 0) {
	X = bestat;
	j++;
	if (verbose) {
	  cout << "planeFit2 iteration " << j << ", " << desc[whobest] << ", score " << myscore << " => " << best << endl;
	}

	myscore = best;

      }

    } while(whobest >= 0);

    niter = j;
    if (verbose)
      cout << ">> " << j << " alignment iterations, optimization: " << fscore << " => " << best << endl; 

    if (outfile!=NULL)
      planeWrite(z,X,outfile);
  }

  float stdev(vector<float> &v) {
    unsigned int n,i;
    float m=0.0f,s=0.0f;
    n = v.size();
    for(i=0;i<n;i++)
      m += v[i];
    m /= n;
    for(i=0;i<n;i++)
      s += (v[i] - m)*(v[i] - m);
    return(sqrt(s/n));
  }

  Location &planeCenter(int z,T4 &X) {
    static Location P(0,0,0);
    int i,j;
    R3 r;
    int mv=0;

    for(j=0;j<H;j++)
      for(i=0;i<W;i++) {
	r.set(i,j,z);
	r = X.apply(r);
	if (valid(r.X,r.Y,r.Z))
	  if (voxel(r.X,r.Y,r.Z) > mv) {
	    mv = voxel(r.X,r.Y,r.Z);
	    P.set( (int) (r.X), (int) (r.Y), (int) (r.Z) );
	  }
      }

    return(P);
  }

  void planeFit(int z, Volume<char> *mask, T4 &X,int &niter,const char *outfile=NULL,bool verbose=false) {
    T4 lib[42], at, bestat;
    float myscore, c, best, fscore;
    float cx,cy,cz;
    int i,j,whobest;

    cx = W/2.0;
    cy = H/2.0;
    cz = (float) z;

    const char *desc[42] = {
      "TX/-1","TX/+1",
      "TY/-1","TY/+1",
      "TZ/-1","TZ/+1",
      "RX/-1","RX/+1",
      "RY/-1","RY/+1",
      "RZ/-1","RZ/+1",
      "TX/-5","TX/+5",
      "TY/-5","TY/+5",
      "TZ/-5","TZ/+5",
      "RX/-5","RX/+5",
      "RY/-5","RY/+5",
      "RZ/-5","RZ/+5",
      "RX/-10","RX/+10",
      "RY/-10","RY/+10",
      "RZ/-10","RZ/+10",
      "TX/-10","TX/+10",
      "TY/-10","TY/+10",
      "TZ/-10","TZ/+10",
      "RX/-0.5","RX/+0.5",
      "RY/-0.5","RY/+0.5",
      "RZ/-0.5","RZ/+0.5",
    };

    lib[0].translate(-1.0, 0.0, 0.0);
    lib[1].translate( 1.0, 0.0, 0.0);
    lib[2].translate( 0.0,-1.0, 0.0);
    lib[3].translate( 0.0, 1.0, 0.0);
    lib[4].translate( 0.0, 0.0,-1.0);
    lib[5].translate( 0.0, 0.0, 1.0);

    lib[6].xrot(1.0, cx,cy,cz);
    lib[7].xrot(-1.0, cx,cy,cz);
    lib[8].yrot(1.0, cx,cy,cz);
    lib[9].yrot(-1.0, cx,cy,cz);
    lib[10].zrot(1.0, cx,cy,cz);
    lib[11].zrot(-1.0, cx,cy,cz);

    lib[12].translate(-5.0, 0.0, 0.0);
    lib[13].translate( 5.0, 0.0, 0.0);
    lib[14].translate( 0.0,-5.0, 0.0);
    lib[15].translate( 0.0, 5.0, 0.0);
    lib[16].translate( 0.0, 0.0,-5.0);
    lib[17].translate( 0.0, 0.0, 5.0);

    lib[18].xrot(5.0, cx,cy,cz);
    lib[19].xrot(-5.0, cx,cy,cz);
    lib[20].yrot(5.0, cx,cy,cz);
    lib[21].yrot(-5.0, cx,cy,cz);
    lib[22].zrot(5.0, cx,cy,cz);
    lib[23].zrot(-5.0, cx,cy,cz);

    lib[24].xrot(10.0, cx,cy,cz);
    lib[25].xrot(-10.0, cx,cy,cz);
    lib[26].yrot(10.0, cx,cy,cz);
    lib[27].yrot(-10.0, cx,cy,cz);
    lib[28].zrot(10.0, cx,cy,cz);
    lib[29].zrot(-10.0, cx,cy,cz);

    lib[30].translate(-10.0, 0.0, 0.0);
    lib[31].translate( 10.0, 0.0, 0.0);
    lib[32].translate( 0.0,-10.0, 0.0);
    lib[33].translate( 0.0, 10.0, 0.0);
    lib[34].translate( 0.0, 0.0,-10.0);
    lib[35].translate( 0.0, 0.0, 10.0);

    lib[36].xrot(0.5, cx,cy,cz);
    lib[37].xrot(-0.5, cx,cy,cz);
    lib[38].yrot(0.5, cx,cy,cz);
    lib[39].yrot(-0.5, cx,cy,cz);
    lib[40].zrot(0.5, cx,cy,cz);
    lib[41].zrot(-0.5, cx,cy,cz);

    fscore = myscore = planeScore(z,mask,X);

    j = 0;
    do {
      whobest = -1;
      best = myscore;
      bestat = X;
    
      for(i=0;i<42;i++) {
	at = X * lib[i];
	c = planeScore(z,mask,at);
	if (c < best && fabsf(c-best)>1.0E-4f) {
	  best = c;
	  whobest = i;
	  bestat = at;
	}
      }

      if (whobest >= 0) {
	X = bestat;
	j++;
	if (verbose) {
	  cout << "planeFit iteration " << j << ", " << desc[whobest] << ", score " << myscore << " => " << best << endl;
	}

	myscore = best;

      }

    } while(whobest >= 0);

    niter = j;
    if (verbose)
      cout << ">> " << j << " alignment iterations, optimization: " << fscore << " => " << best << endl; 

    if (outfile!=NULL)
      planeWrite(z,X,outfile);
  }

  void drawPlane(Plane &p, T base, T inc) {
    int i,j;
    R3 r;
    for(j=-H;j<2*H;j++)
      for(i=-W;i<2*W;i++) {
	r.set(i,j,p.z);
	r = p.trans.apply(r);
	if (valid(r.X,r.Y,r.Z))
	  if (voxel(r.X,r.Y,r.Z) > base)
	    voxel(r.X,r.Y,r.Z) += inc;
	  else
	    voxel(r.X,r.Y,r.Z) = base;
      }    
  }

  void planeWrite(int z,T4 &X,const char *outfile) {
    
    int i,j;
    R3 r;
    Volume<short int> *tmp;

    tmp = new Volume<short int>(W,H,D,dx,dy,dz);
    for(i=0;i<N;i++)
      tmp->voxel(i) = (short int) (this->voxel(i));

    for(j=-H;j<2*H;j++)
      for(i=-W;i<2*W;i++) {
	r.set(i,j,z);
	r = X.apply(r);
	if (valid(r.X,r.Y,r.Z))
	  tmp->voxel(r.X,r.Y,r.Z) = 2000;
      }

    tmp->writeSCN(outfile);
    delete tmp;
  }

  void transform(T4 &X) {
    transformNN(X);
  }

  Volume<T> *paddedTransformLI(T4 &X) {
    R3 corner[8],a;
    Volume<T> *tmp;
    int x[2],y[2],z[2];
    int nw,nh,nd,hz;
    int i,j,k,ix,iy,iz,jx,jy,jz;
    float ws,vs,w;
    float sq3;
    T4 A,IX;

    corner[0].set(0,0,0);
    corner[1].set(W-1,0,0);
    corner[2].set(0,H-1,0);
    corner[3].set(0,0,D-1);
    corner[4].set(W-1,H-1,0);
    corner[5].set(W-1,0,D-1);
    corner[6].set(0,H-1,D-1);
    corner[7].set(W-1,H-1,D-1);

    for(i=0;i<8;i++)
      corner[i] = X.apply(corner[i]);

    x[0] = x[1] = (int) (corner[0].X);
    y[0] = y[1] = (int) (corner[0].Y);
    z[0] = z[1] = (int) (corner[0].Z);    

    for(i=0;i<8;i++) {
      if (corner[i].X < x[0]) x[0] = (int) (corner[i].X);
      if (corner[i].X > x[1]) x[1] = (int) (corner[i].X);
      if (corner[i].Y < y[0]) y[0] = (int) (corner[i].Y);
      if (corner[i].Y > y[1]) y[1] = (int) (corner[i].Y);
      if (corner[i].Z < z[0]) z[0] = (int) (corner[i].Z);
      if (corner[i].Z > z[1]) z[1] = (int) (corner[i].Z);
    }

    hz = D/2;
    if (abs(z[1] - hz) > abs(hz - z[0])) {
      z[0] = hz - abs(z[1] - hz);
    } else {
      z[1] = hz + abs(hz - z[0]);
    }

    nw = x[1] - x[0] + 1;
    nh = y[1] - y[0] + 1;
    nd = z[1] - z[0] + 1;

    IX = X;
    A.translate(-x[0],-y[0],-z[0]);
    IX *= A;    
    IX.invert();

    tmp = new Volume<T>(nw,nh,nd,dx,dy,dz);

    sq3 = sqrt(3.0);

    for(k=0;k<nd;k++)
      for(j=0;j<nh;j++)
	for(i=0;i<nw;i++) {
	  a.set(i,j,k);
	  a = IX.apply(a);

	  ws = 0.0f;
	  vs = 0.0f;

	  for(iz=0;iz<2;iz++)
	    for(iy=0;iy<2;iy++)
	      for(ix=0;ix<2;ix++) {
		jx = ((int)floor(a.X)) + ix;
		jy = ((int)floor(a.Y)) + iy;
		jz = ((int)floor(a.Z)) + iz;
		if (valid(jx,jy,jz)) {
		  w = sq3 - sqrt( (jx-a.X)*(jx-a.X) +
				  (jy-a.Y)*(jy-a.Y) +
				  (jz-a.Z)*(jz-a.Z) );
		  vs += w * (this->voxel(jx,jy,jz)) ;
		  ws += w;
		}
	      }
	  
	  if (ws != 0.0f) {
	    vs /= ws;
	    tmp->voxel(i,j,k) = (T) vs;
	  }
	}
    return tmp;
  }

  void transformLI(T4 &X) {
    R3 a;
    T4 IX;
    int i,j,k,ix,iy,iz,jx,jy,jz;
    float ws,vs,w;
    Volume<T> *tmp;
    float sq3;

    tmp = new Volume<T>(W,H,D,dx,dy,dz);

    IX = X;
    IX.invert();
    
    sq3 = sqrt(3.0);

    for(k=0;k<D;k++)
      for(j=0;j<H;j++)
	for(i=0;i<W;i++) {
	  a.set(i,j,k);
	  a = IX.apply(a);

	  ws = 0.0f;
	  vs = 0.0f;

	  for(iz=0;iz<2;iz++)
	    for(iy=0;iy<2;iy++)
	      for(ix=0;ix<2;ix++) {
		jx = ((int)floor(a.X)) + ix;
		jy = ((int)floor(a.Y)) + iy;
		jz = ((int)floor(a.Z)) + iz;
		if (valid(jx,jy,jz)) {
		  w = sq3 - sqrt( (jx-a.X)*(jx-a.X) +
				  (jy-a.Y)*(jy-a.Y) +
				  (jz-a.Z)*(jz-a.Z) );
		  vs += w * (this->voxel(jx,jy,jz)) ;
		  ws += w;
		}
	      }
	  
	  if (ws != 0.0f) {
	    vs /= ws;
	    tmp->voxel(i,j,k) = (T) vs;
	  }
	}

    for(i=0;i<N;i++)
      this->voxel(i) = tmp->voxel(i);
    delete tmp;   
  }

  void transformNN(T4 &X) {
    R3 a;
    T4 IX;
    int i,j,k;
    Volume<T> *tmp;

    tmp = new Volume<T>(W,H,D,dx,dy,dz);

    IX = X;
    IX.invert();
    
    for(k=0;k<D;k++)
      for(j=0;j<H;j++)
	for(i=0;i<W;i++) {
	  a.set(i,j,k);
	  a = IX.apply(a);
	  if (valid(a.X,a.Y,a.Z))
	    tmp->voxel(i,j,k) = this->voxel(a.X,a.Y,a.Z);
	}

    for(i=0;i<N;i++)
      this->voxel(i) = tmp->voxel(i);
    delete tmp;
  }

 private:
  T *data;
  iterator anyi,ibegin,iend;


 public:

  // area: mm^2
  VolumeOrientation guessOrientation(Volume<char> *mask, float area) {
    int i;
    float min[3],a;
    int pmin[3] = {-1,-1,-1};

    min[0] = min[1] = min[2] = (float) maximum();
    for(i=0;i<W;i++) {
      a = yzScore(i,mask,area);
      if (a < min[0]) { min[0] = a; pmin[0] = i; }
    }
    for(i=0;i<H;i++) {
      a = xzScore(i,mask,area);
      if (a < min[1]) { min[1] = a; pmin[1] = i; }
    }
    for(i=0;i<D;i++) {
      a = xyScore(i,mask,area);
      if (a < min[2]) { min[2] = a; pmin[2] = i; }
    }

    cout << min[0] << "," << min[1] << "," << min[2] << endl;
    cout << pmin[0] << "," << pmin[1] << "," << pmin[2] << endl;

    if (min[0] < min[1] && min[0] < min[2]) return AxialOrientation;
    if (min[1] < min[2] && min[1] < min[0]) return CoronalOrientation;
    return SagittalOrientation;
  }

 private:

  float xyScore(int z, Volume<char> *mask, float area) {
    float sum=0.0, va;
    int i,j,n=0;

    for(j=0;j<H;j++)
      for(i=0;i<W;i++)
	if (mask->voxel(i,j,z)!=0) {
	  sum += this->voxel(i,j,z);
	  n++;
	}

    va = dx*dy;
    if ( (n/va) < area) return 10000.0;

    sum /= n;
    return sum;
  }

  float xzScore(int y, Volume<char> *mask, float area) {
    float sum=0.0, va;
    int i,j,n=0;

    for(j=0;j<D;j++)
      for(i=0;i<W;i++)
	if (mask->voxel(i,y,j)!=0) {
	  sum += this->voxel(i,y,j);
	  n++;
	}

    va = dx*dz;
    if ( (n/va) < area) return 10000.0;

    sum /= n;
    return sum;
  }

  float yzScore(int x, Volume<char> *mask, float area) {
    float sum=0.0, va;
    int i,j,n=0;

    for(j=0;j<D;j++)
      for(i=0;i<H;i++)
	if (mask->voxel(x,i,j)!=0) {
	  sum += this->voxel(x,i,j);
	  n++;
	}

    va = dy*dz;
    if ( (n/va) < area) return 10000.0;

    sum /= n;
    return sum;
  }



};


} //end MSP namespace

#endif
