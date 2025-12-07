#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "color.h"

int triplet(int a,int b,int c) {
  return(((a&0xff)<<16)|((b&0xff)<<8)|(c&0xff));
}

int RGB2YCbCr(int v) {
  float y,cb,cr;

  y=0.299*(float)t0(v)+
    0.587*(float)t1(v)+
    0.114*(float)t2(v);

  cb=-0.169*(float)t0(v)-
    0.331*(float)t1(v)+
    0.500*(float)t2(v)+128;

  cr=0.500*(float)t0(v)-
     0.419*(float)t1(v)-
     0.081*(float)t2(v)+128;


  if (y<0.0) y=0.0;
  if (cb<0.0) cb=0.0;
  if (cr<0.0) cr=0.0;
  if (y>255.0) y=255.0;
  if (cb>255.0) cb=255.0;
  if (cr>255.0) cr=255.0;

  return(triplet((int)y,(int)cb,(int)cr));
}

int YCbCr2RGB(int v) {
  float r,g,b;

  r=1.000*(float)t0(v)-
    0.001*((float)t1(v)-128)+
    1.402*((float)t2(v)-128);

  g=1.000*(float)t0(v)-
    0.344*((float)t1(v)-128)-
    0.714*((float)t2(v)-128);

  b=1.000*(float)t0(v)+
    1.772*((float)t1(v)-128)+
    0.001*((float)t2(v)-128);


  if (r<0.0) r=0.0;
  if (g<0.0) g=0.0;
  if (b<0.0) b=0.0;
  if (r>255.0) r=255.0;
  if (g>255.0) g=255.0;
  if (b>255.0) b=255.0;

  return(triplet((int)r,(int)g,(int)b));
}

#define GMAX(a,b,c) (((a>=b)&&(a>=c))?a:((b>c)?b:c))
#define GMIN(a,b,c) (((a<=b)&&(a<=c))?a:((b<c)?b:c))

int RGB2HSV(int vi) {
  float r = (float) t0(vi),
        g = (float) t1(vi),
        b = (float) t2(vi), v, x, f;
  float a[3];

  int i;

  r/=255.0;
  g/=255.0;
  b/=255.0;

  // RGB are each on [0, 1]. S and V are returned on [0, 1] and H is
  // returned on [0, 6].

  x = GMIN(r, g, b);
  v = GMAX(r, g, b);
  if (v == x) {
    a[0]=0.0;
    a[1]=0.0;
    a[2]=v;
  } else {
    f = (r == x) ? g - b : ((g == x) ? b - r : r - g);
    i = (r == x) ? 3 : ((g == x) ? 5 : 1);
    a[0]=((float)i)-f/(v-x);
    a[1]=(v-x)/v;
    //a[2]=v;
    a[2]=0.299*r+0.587*g+0.114*b;
  }

  // (un)normalize

  a[0]*=60.0;
  a[1]*=255.0;
  a[2]*=255.0;

  return(triplet(ROUND(a[0]),ROUND(a[1]),ROUND(a[2])));
}

int HSV2RGB(int vi) {
  // H is given on [0, 6]. S and V are given on [0, 1].
  // RGB are each returned on [0, 1].
  float h = (float)t0(vi),
        s = (float)t1(vi),
        v = (float)t2(vi), m, n, f;
  float a[3]={0,0,0};
  int i;

  h/=60.0;
  s/=255.0;
  v/=255.0;

  if (s==0.0) {
    a[0]=a[1]=a[2]=v;
  } else {
    i = (int) floor(h);
    f = h - (float)i;
    if(!(i & 1)) f = 1 - f; // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);
    switch (i) {
    case 6:
    case 0: a[0]=v; a[1]=n; a[2]=m; break;
    case 1: a[0]=n; a[1]=v; a[2]=m; break;
    case 2: a[0]=m; a[1]=v; a[2]=n; break;
    case 3: a[0]=m; a[1]=n; a[2]=v; break;
    case 4: a[0]=n; a[1]=m; a[2]=v; break;
    case 5: a[0]=v; a[1]=m; a[2]=n; break;
    }
  }

  // (un)normalize
  for(i=0;i<3;i++)
    a[i]*=255;

  return(triplet(ROUND(a[0]),ROUND(a[1]),ROUND(a[2])));
}

int lighter(int c, int n) {
  int r,g,b,r0,g0,b0,i;
  r=c>>16;
  g=(c>>8)&0xff;
  b=c&0xff;

  for(i=0;i<n;i++) {
    r0=r+r/10; if (r0==r) r0=r+1;
    g0=g+g/10; if (g0==g) g0=g+1;
    b0=b+b/10; if (b0==b) b0=b+1;

    if (r0>255) r0=255;
    if (b0>255) b0=255;
    if (g0>255) g0=255;

    r=r0; g=g0; b=b0;
  }

  return( (r<<16)|(g<<8)|b );
}

int darker(int c, int n) {
  int r,g,b,r0,g0,b0,i;
  r=c>>16;
  g=(c>>8)&0xff;
  b=c&0xff;

  for(i=0;i<n;i++) {
    r0=r-r/10; if (r0==r) r0=r-1;
    g0=g-g/10; if (g0==g) g0=g-1;
    b0=b-b/10; if (b0==b) b0=b-1;

    if (r0<0) r0=0;
    if (b0<0) b0=0;
    if (g0<0) g0=0;

    r=r0; g=g0; b=b0;
  }

  return( (r<<16)|(g<<8)|b );
}

/*
   inverseColor
   Returns the inverse in RGB-space.

   input:  RGB triplet
   output: RGB triplet
   author: bergo
*/
int          inverseColor(int c) {
  int r,g,b;
  r=c>>16;
  g=(c>>8)&0xff;
  b=c&0xff;
  r=255-r; g=255-g; b=255-b;
  return( (r<<16)|(g<<8)|b );
}

/*
  mergeColorsRGB
  Merges 2 colors, (1-ratio) of a with (ratio) of b.
  Works in RGB space.

  input:  2 RGB triplets, merge ratio
  output: RGB triplet
  author: bergo
*/
int mergeColorsRGB(int a,int b,float ratio) {
  float c[6];
  int r[3];

  c[0] = (float) (a >> 16);
  c[1] = (float) (0xff & (a >> 8));
  c[2] = (float) (0xff & a);

  c[3] = (float) (b >> 16);
  c[4] = (float) (0xff & (b >> 8));
  c[5] = (float) (0xff & b);

  c[0] = (1.0-ratio) * c[0] + ratio * c[3];
  c[1] = (1.0-ratio) * c[1] + ratio * c[4];
  c[2] = (1.0-ratio) * c[2] + ratio * c[5];

  r[0] = (int) c[0];
  r[1] = (int) c[1];
  r[2] = (int) c[2];

  return( (r[0]<<16) | (r[1]<<8) | r[2] );
}

/*
  mergeColorsYCbCr
  Merges 2 colors, (1-ratio) of a with (ratio) of b.
  Works in YCbCr space.

  input:  2 RGB triplets, merge ratio
  output: RGB triplet
  author: bergo
*/
int          mergeColorsYCbCr(int a,int b,float ratio) {
  float c[6];
  int ya,yb;
  int r[3];

  ya=RGB2YCbCr(a);
  yb=RGB2YCbCr(b);

  c[0] = (float) (ya >> 16);
  c[1] = (float) (0xff & (ya >> 8));
  c[2] = (float) (0xff & ya);

  c[3] = (float) (yb >> 16);
  c[4] = (float) (0xff & (yb >> 8));
  c[5] = (float) (0xff & yb);

  c[0] = (1.0-ratio) * c[0] + ratio * c[3];
  c[1] = (1.0-ratio) * c[1] + ratio * c[4];
  c[2] = (1.0-ratio) * c[2] + ratio * c[5];

  r[0] = (int) c[0];
  r[1] = (int) c[1];
  r[2] = (int) c[2];

  ya = (r[0]<<16) | (r[1]<<8) | r[2];
  return(YCbCr2RGB(ya));
}

/*
  gray
  builds the gray triplet with R=c, G=c, B=c

  input:  gray level (0-255)
  output: RGB triplet
  author: bergo
*/
int gray(int c) {
  return( (c<<16) | (c<<8) | c );
}

int randomColor() {
  static int seeded = 0;
  int a,b,c;

  if (!seeded) {
    srand(time(0));
    seeded = 1;
  }

  a = 1+(int) (255.0*rand()/(RAND_MAX+1.0));
  b = 1+(int) (255.0*rand()/(RAND_MAX+1.0));
  c = 1+(int) (255.0*rand()/(RAND_MAX+1.0));
  return(triplet(a,b,c));
}


double ColorDistance(int color1, int color2){
  float teta, dx, dy;
  float fh1,fs1,fv1;
  float fh2,fs2,fv2;
  float dv,D;
  double distance;
  int   color;

  color = RGB2HSV(color1);
  fh1 = t0(color)/255.0;
  fs1 = t1(color)/255.0;
  fv1 = t2(color)/255.0;

  color = RGB2HSV(color2);
  fh2 = t0(color)/255.0;
  fs2 = t1(color)/255.0;
  fv2 = t2(color)/255.0;

  teta = (fh2-fh1)*2.0*PI;
  if(teta<0.0) teta = -teta;
  if(teta>PI) teta = 2*PI-teta;

  teta *= 2.0;
  if(teta>PI) teta = PI;

  fs1 *= fv1;
  fs2 *= fv2;

  dx = fs1 - cos(teta)*fs2;
  dy = 0.0 - sin(teta)*fs2;

  D = dx*dx+dy*dy;
  dv = fv2-fv1;

  distance = sqrt(D + dv*dv)/2.0;

  return distance;
}


double ColorDistance2(int color1, int color2){
  float fr1,fg1,fb1;
  float fr2,fg2,fb2;
  float dr,dg,db;
  double distance;

  fr1 = t0(color1)/255.0;
  fg1 = t1(color1)/255.0;
  fb1 = t2(color1)/255.0;

  fr2 = t0(color2)/255.0;
  fg2 = t1(color2)/255.0;
  fb2 = t2(color2)/255.0;

  dr = fr1-fr2;
  dg = fg1-fg2;
  db = fb1-fb2;
  dr *= dr;
  dg *= dg;
  db *= db;

  distance = sqrt((dr + dg + db)/3.0);

  return distance;
}


void RGB2Lab(float* r, float* g, float * b)
{

    float tresh = 0.008856;

    float Xn  = 0.950456;
    float Yn  = 1.0;
    float Zn  = 1.088854;

    float r13    = 1.0/3.0;
    float r16116 = 16.0/116.0;
    float cnst   = 7.787;

    float R0 = (*r)/255.0;
    float G0 = (*g)/255.0;
    float B0 = (*b)/255.0;

    // sRGB (d65) Matrix Cxr
    float X =  0.412453 * R0 + 0.357580 * G0 + 0.180423 * B0;
    float Y =  0.212671 * R0 + 0.715160 * G0 + 0.072169 * B0;
    float Z =  0.019334 * R0 + 0.119193 * G0 + 0.950227 * B0;

    float xr = X/Xn;
    float yr = Y/Yn;
    float zr = Z/Zn;

    float L, A, B;
    float xr0, yr0, zr0;

//    R0 = powf(R0,2.2); /// gamma
//    G0 = powf(G0,2.2); /// gamma
//    B0 = powf(B0,2.2); /// gamma


    if (xr > tresh)
    {
        xr0 = pow(xr, r13);
    }
    else
    {
        xr0 = cnst * xr + r16116;
    }

    if (yr > tresh)
    {
        yr0 = pow(yr, r13);
    }
    else
    {
        yr0 = cnst * yr + r16116;
    }

    if (zr > tresh)
    {
        zr0 = pow(zr, r13);
    }
    else
    {
        zr0 = cnst * zr + r16116;
    }

    if (yr > tresh)
    {
        L = 116.0 * pow(yr, r13) - 16.0;
    }
    else
    {
        L = 903.3 * yr;
    }

    A = 500.0 * (xr0 - yr0);
    B = 200.0 * (yr0 - zr0);

    *r = L;
    *g = A;
    *b = B;
}

void Lab2RGB(float* l, float* a, float* b)
{

    float thresh1 = 0.008856;
    float thresh2 = 0.206893;

    float Xn = 0.950456;
    float Yn = 1.000;
    float Zn = 1.088854;

    float r13    = 1.0/3.0;
    float r16116 = 16.0/116.0;
    float cnst   = 7.787;

    float X, Y, Z;
    float L = *l;
    float A = *a;
    float B = *b;

    float P = (L+16.0)/116.0;
    float yr,fy;

    float fx;
    float fz;
    float R1,G1,B1;

    if (L > 7.9996)
    {
        Y = Yn * P * P * P;
    }
    else
    {
        Y = L / 903.3;
    }

    yr = Y/Yn;
    if ( yr > thresh1 )
    {
        fy = pow(yr, r13);
    }
    else
    {
        fy = cnst * yr + r16116;
    }

    fx = A / 500.0 + fy;
    fz = fy - B / 200.0;

    if (fx > thresh2 )
    {
        X = Xn * fx * fx * fx;
    }
    else
    {
        X = Xn/cnst * ( fx - r16116 );
    }

    if (fz > thresh2 )
    {
        Z = Zn * fz * fz * fz;
    }
    else
    {
        Z = Zn/cnst * ( fz - r16116 );
    }

    // sRGB (d65) Matrix Crx
    R1 =   3.240479 * X - 1.537150 * Y - 0.498535 * Z;
    G1 = - 0.969256 * X + 1.875992 * Y + 0.041556 * Z;
    B1 =   0.055648 * X - 0.204043 * Y + 1.057311 * Z;

//    R1 = powf(R1,1/2.2); /// gamma
//    G1 = powf(G1,1/2.2); /// gamma
//    B1 = powf(B1,1/2.2); /// gamma

    *l = MAX(0.0, MIN(255.0, 255.0*R1));
    *a = MAX(0.0, MIN(255.0, 255.0*G1));
    *b = MAX(0.0, MIN(255.0, 255.0*B1));
}


