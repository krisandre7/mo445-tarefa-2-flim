
#include "rayleapinghandler.h"

namespace Rayleaping{

  RayleapingHandler :: RayleapingHandler(RayleapingCanvas *canvas){
    this->canvas = canvas;
  }

  RayleapingHandler :: ~RayleapingHandler(){}

  void RayleapingHandler :: OnLeftClick(int p){
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

  void RayleapingHandler :: OnRightClick(int p){}

  void RayleapingHandler :: OnMiddleClick(int p){}

  void RayleapingHandler :: OnLeftDrag(int p, int q){}

  void RayleapingHandler :: OnRightDrag(int p, int q){
    int x1,x2,y1,y2,w,h;
    float ax,ay;

    canvas->GetImageSize(&w, &h);
    x1 = p%w;
    y1 = p/w;
    x2 = q%w;
    y2 = q/w;
    ax = (float)(x2 - x1)*(PI / 180.0);
    ay = (float)(y2 - y1)*(PI / 180.0);

    canvas->Rotate( ax, canvas->Vpi);
    canvas->Rotate(-ay, canvas->Vpj);
    canvas->drawRender(12);
  }


  void RayleapingHandler :: OnLeftRelease(int p){
  }

  void RayleapingHandler :: OnRightRelease(int p){
    canvas->drawRender(1);
  }

  void RayleapingHandler :: OnMiddleRelease(int p){
  }


  void RayleapingHandler :: OnMouseMotion(int p){
    APP->Window->SetStatusText(_T("Mouse Left: Navigate, Mouse Right: Rotate"));
  }

} //end Rayleaping namespace

