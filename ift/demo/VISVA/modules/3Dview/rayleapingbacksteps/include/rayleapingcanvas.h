
#ifndef _RAYLEAPINGCANVAS_H_
#define _RAYLEAPINGCANVAS_H_

#include "startnewmodule.h"
#include "rendercanvas.h"


namespace Rayleaping{

  typedef struct _Face {
    Point V;  //vertex
    Vector N; //normal
    int fixedindex; //x=0,y=1,z=2
    int fixedval;
  } Face;

  class RayleapingCanvas : public RenderCanvas {
  public:
    RayleapingCanvas(wxWindow *parent);

    //Draw phong shading.
    CImage *Render2CImage(int skip);
    void drawRender(int skip);
    void SetInteractionHandler(InteractionHandler *handler);
    int  GetSurfaceVoxel(int p);
    Vector Vpj,Vpi;

  protected:
    InteractionHandler *defaulthandler;
    Vector light;
    float ka;
    float kd;
    float ks;
    int   sre;
    float zgamma;
    float zstretch;
    float *zbuf;
    int   *vxbuf;
    float *normals;
    
    float   phong_specular(float angcos, int n);
    void    render_calc_normal(int x,int y,int rw,
			       int *vxbuf, float *out);
    CImage *Anaglyph3D(CImage *cimg_l, CImage *cimg_r);
  };
} //end Rayleaping namespace

#include "rayleapingview.h"
#include "rayleapinghandler.h"

#endif

