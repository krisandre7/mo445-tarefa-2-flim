
#include "navbasichandler.h"

namespace Basic{

  NavigatorHandler :: NavigatorHandler(){}

  NavigatorHandler :: ~NavigatorHandler(){}

  void NavigatorHandler :: OnEnterWindow(){
    wxCursor *cross=NULL;
    float zoom = APP->GetZoomLevel();
    cross = CrossCursor(ROUND(zoom));
    //wxCROSS_CURSOR
    APP->Set2DViewCursor(cross, 'x');
    APP->Set2DViewCursor(cross, 'y');
    APP->Set2DViewCursor(cross, 'z');
  }


  void NavigatorHandler :: OnLeaveWindow(){
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'x');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'y');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'z');
  }

  void NavigatorHandler :: OnLeftClick(int p){
    Voxel v;
    
    v.x = VoxelX(APP->Data.orig, p);
    v.y = VoxelY(APP->Data.orig, p);
    v.z = VoxelZ(APP->Data.orig, p);
    APP->Set2DViewSlice(v);
    APP->Refresh2DCanvas();
  }
  
  void NavigatorHandler :: OnRightClick(int p){}
  
  void NavigatorHandler :: OnMiddleClick(int p){
    printf("Navigator middle\n");

    CImage *cimg;

    APP->Refresh2DCanvas();

    cimg = APP->Copy2DCanvasAsCImage('x');
    WriteCImage(cimg, (char *)"slice_x.ppm");
    DestroyCImage(&cimg);

    cimg = APP->Copy2DCanvasAsCImage('y');
    WriteCImage(cimg, (char *)"slice_y.ppm");
    DestroyCImage(&cimg);

    cimg = APP->Copy2DCanvasAsCImage('z');
    WriteCImage(cimg, (char *)"slice_z.ppm");
    DestroyCImage(&cimg);
  }
  
  void NavigatorHandler :: OnLeftDrag(int p, int q) {
    Voxel v;
    v.x = VoxelX(APP->Data.orig, q);
    v.y = VoxelY(APP->Data.orig, q);
    v.z = VoxelZ(APP->Data.orig, q);
    APP->Set2DViewSlice(v);
    APP->Refresh2DCanvas();
  }
  
  void NavigatorHandler :: OnRightDrag(int p, int q){}

  void NavigatorHandler :: OnMouseMotion(int p){
    APP->Window->SetStatusText(_T("Mouse Left: Navigate"));
  }

} //end Basic namespace
