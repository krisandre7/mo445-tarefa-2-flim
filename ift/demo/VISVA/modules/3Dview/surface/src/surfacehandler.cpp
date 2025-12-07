
#include "surfacehandler.h"

namespace Surface{

  SurfaceHandler :: SurfaceHandler(SurfaceCanvas *canvas){
    this->canvas = canvas;
  }

  SurfaceHandler :: ~SurfaceHandler(){}

  void SurfaceHandler :: OnLeftClick(int p){
    Voxel Cut;
    int q;

    q = canvas->GetSurfaceVoxel(p);
    if(q!=NIL){
      Cut.x = VoxelX(APP->Data.orig, q);
      Cut.y = VoxelY(APP->Data.orig, q);
      Cut.z = VoxelZ(APP->Data.orig, q);
      APP->Set2DViewSlice(Cut);
      APP->Refresh2DCanvas();
    }
  }

  void SurfaceHandler :: OnRightClick(int p){}

  void SurfaceHandler :: OnMiddleClick(int p){}

  void SurfaceHandler :: OnLeftDrag(int p, int q){}

  void SurfaceHandler :: OnRightDrag(int p, int q){
    int x1,x2,y1,y2,w,h;
    float ax,ay;

    canvas->GetImageSize(&w, &h);
    x1 = p%w;
    y1 = p/w;
    x2 = q%w;
    y2 = q/w;
    ax = (float)(x2 - x1)*(PI / 180.0);
    ay = (float)(y2 - y1)*(PI / 180.0);

    canvas->RotateY(ax);
    canvas->RotateX(ay);
    canvas->drawRender(false, 12);
  }


  void SurfaceHandler :: OnLeftRelease(int p){
  }

  void SurfaceHandler :: OnRightRelease(int p){
    canvas->drawRender(false, 1);
  }

  void SurfaceHandler :: OnMiddleRelease(int p){
  }


  void SurfaceHandler :: OnMouseMotion(int p){
    APP->Window->SetStatusText(_T("Mouse Left: Navigate, Mouse Right: Rotate"));
  }

} //end Surface namespace

