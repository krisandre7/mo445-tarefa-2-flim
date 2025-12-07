
#include "curvilinearrender.h"

namespace Curvilinear{

  CurvilinearRender :: CurvilinearRender(wxWindow *parent) 
    : SurfaceRender(parent){
  }


  void CurvilinearRender :: draw_isosurfaces(int skip, bool labelChanged){
    Scene *bin,*bin2;
    Scene *closed = NULL;
    static int *dmap=NULL;
    static int *BinCount=NULL;
    int p,n,D,mapi,Dmax;
    int *bmap=NULL;
    int maplen=0;
    int distance, sel_shading; 
    timer tic,toc;
    float totaltime,fdist;

    Window->Get3DViewOptions(&sel_shading, &distance);

    D = APP->Data.w*APP->Data.w + APP->Data.h*APP->Data.h + APP->Data.nframes*APP->Data.nframes;
    n = APP->Data.w*APP->Data.h*APP->Data.nframes;
    if( labelChanged ){
      wxWindowDisabler disableAll;
      wxBusyInfo busy(_T("Please wait, working..."));
      wxBusyCursor wait;
      Window->SetStatusText(_T("Please wait - Computation in progress..."));
      wxTheApp->Yield();

      gettimeofday(&tic,NULL);

      if(dmap!=NULL) free(dmap);
      if(BinCount!=NULL) free(BinCount);

      dmap = (int *)malloc(sizeof(int)*n);
      BinCount = (int *)malloc(sizeof(int)*D);
      for(p=0;p<D;p++)
	BinCount[p] = 0;

      bin = CopyScene(APP->Data.mark);
      //Get label from mark
      for(p=0; p<n; p++)
	bin->data[p] = GetMarkerLabel(bin->data[p]);
      bin2 = AddFrame3(bin, 20, 0);
      DestroyScene(&bin);
      closed = CloseBin3(bin2, 20.0);
      DestroyScene(&bin2);
      bin2 = RemFrame3(closed, 20);
      DestroyScene(&closed);
      closed = bin2;

      MyDistTrans3(closed, dmap, BinCount);
      DestroyScene(&closed);

      Dmax = 0;
      for(p=0;p<D;p++)
	if(BinCount[p]>0)
	  Dmax = p;
      Window->sDist->SetRange(0, ROUND(sqrt((float)Dmax)));
      Window->sDist->SetValue(0);

      APP->Data.RenderInvalid = true;

      gettimeofday(&toc,NULL);
      totaltime = (toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001;
      printf("\nCloseBin3 Time: %f ms\n",totaltime);
    }

    mapi=p=0;
    fdist = (float)distance;  
    while(p<D){
      if(sqrt((float)p)>=fdist-0.5)
	break;
      else
	mapi += BinCount[p];
      p++;
    }
    maplen=0;
    while(p<D){
      if(sqrt((float)p)>=fdist+0.5)
	break;
      else
	maplen += BinCount[p];
      p++;
    }
    bmap = (int *)malloc(sizeof(int)*maplen);
    memcpy(bmap, dmap+mapi, sizeof(int)*maplen);

    render_map(bmap, maplen, NIL, APP->Data.orig, skip);    
    free(bmap);
  }


  void CurvilinearRender :: OnMouseEvent(wxMouseEvent& event) {
    Matrix *C,*Rx,*Ry;
    float dx,dy;
    int x,y,ix,iy,p,q;
    int scx,scy,xUnit,yUnit;
    float fx,fy;

    if(!APP->Data.loaded) return;

    event.GetPosition(&x, &y);

    switch (event.GetButton()) {
    case wxMOUSE_BTN_RIGHT : 
      if(event.GetEventType() == wxEVT_RIGHT_DOWN){
	drag_x = x;
	drag_y = y;
      }
      else if (event.GetEventType() == wxEVT_RIGHT_UP){
	APP->Data.RenderInvalid = true;
	View3D->draw_render(1);
      }
      break;
    case wxMOUSE_BTN_MIDDLE :
      if (event.GetEventType() == wxEVT_MIDDLE_DOWN){}
      else if (event.GetEventType() == wxEVT_MIDDLE_UP){}
      break;
    case wxMOUSE_BTN_LEFT : 
      if (event.GetEventType() == wxEVT_LEFT_DOWN){
	GetViewStart(&scx, &scy);
	GetScrollPixelsPerUnit(&xUnit, &yUnit);
	scx *= xUnit;
	scy *= yUnit;
	ix = x + scx;
	iy = y + scy;
      
	fx = (float)ix / zoom;
	fy = (float)iy / zoom;
	ix = ROUND(fx);
	iy = ROUND(fy);

	// Só processa o evento caso o mouse esteja dentro da imagem.
	if(ix>=imageWidth || iy>=imageHeight) return;

	p = ix + iy*imageWidth;
	q = vxbuf[p];
	if(q!=NIL){
	  APP->Data.cut_x = VoxelX(APP->Data.orig, q);
	  APP->Data.cut_y = VoxelY(APP->Data.orig, q);
	  APP->Data.cut_z = VoxelZ(APP->Data.orig, q);
	  app_draw_frames();
	}
      }
      else if (event.GetEventType() == wxEVT_LEFT_UP){}
      break;
    case wxMOUSE_BTN_NONE :
      if (event.GetEventType() == wxEVT_ENTER_WINDOW) {}
      else if (event.GetEventType() == wxEVT_LEAVE_WINDOW){}
      else if (event.GetEventType() == wxEVT_MOTION) {
	if (event.Dragging()) {
	  if(event.RightIsDown()) {
	    dx = (float)(drag_x - x)/ zoom;
	    dy = (float)(drag_y - y)/ zoom;
	    Rx = RotateY(-dx);
	    Ry = RotateX(-dy);
	    C  = MultMatrix(Rx, rot);
	    DestroyMatrix(&rot);
	    rot = MultMatrix(Ry, C);
	    DestroyMatrix(&C);
	    DestroyMatrix(&Rx);
	    DestroyMatrix(&Ry);
	    drag_x = x;
	    drag_y = y;
	    APP->Data.RenderInvalid = true;
	    View3D->draw_render(12);
	  }
	  else if(event.LeftIsDown()) {}
	  else if(event.MiddleIsDown()) {}
	}
	Window->SetStatusText(_T("Mouse Left: Navigate, Mouse Right: Rotate"));
      }
      else if (event.GetEventType() == wxEVT_MOUSEWHEEL) {}
      break;
    }
  }

} //end Curvilinear namespace


