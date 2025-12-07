
#ifndef _CURVILINEARCANVAS_H_
#define _CURVILINEARCANVAS_H_

#include "startnewmodule.h"
#include "rendercanvas.h"


namespace Curvilinear{

  void MyDistTrans3(Scene *bin, int *dmap, int *BinCount);


  class CurvilinearCanvas : public RenderCanvas {
  public:
    CurvilinearCanvas(wxWindow *parent);

    CImage *Render2CImage(bool dataChanged, int skip);
    void drawRender(bool dataChanged,
		    int skip);
    void SetInteractionHandler(InteractionHandler *handler);
    int  GetSurfaceVoxel(int p);

  protected:
    InteractionHandler *defaulthandler;
    Vector light;
    float ka;
    float kd;
    float ks;
    int   sre;
    float zgamma;
    float zstretch;
    int   *vxbuf;

    int     xgray(int value);
    float   phong_specular(float angcos, int n);
    CImage *render_map(int *bmap, int maplen,
		       Scene *vol, 
		       bool dataChanged,
		       int skip);
    void    render_calc_normal(int x,int y,int rw,
			       float *xb,float *yb,
			       float *zb,float *out);
    CImage *Anaglyph3D(CImage *cimg_l, CImage *cimg_r);
  };

} //end Curvilinear namespace

#include "curvilinearview.h"
#include "curvilinearhandler.h"

#endif

