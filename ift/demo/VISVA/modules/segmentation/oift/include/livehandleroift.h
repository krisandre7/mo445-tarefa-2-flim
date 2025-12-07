
#ifndef _LIVEHANDLEROIFT_H_
#define _LIVEHANDLEROIFT_H_

#include "startnewmodule.h"
#include "moduleoift.h"
#include "livewireoift.h"

namespace OIFT{

  class LiveHandlerOIFT : public InteractionHandler {
  public:
    LiveHandlerOIFT(char axis,
		ModuleOIFT *mod);
    ~LiveHandlerOIFT();
  
  protected:
    char axis;
    ModuleOIFT *mod;
    PriorityQueue *frontier;
    Image *arcw;
    Image *pred;
    Image *cost;
    Image *label;
    AdjRel *A;
    int  axis_val;
    int  Wmax;
    bool closed;
    int *path_voxels;
    int *path_prevlb;
    //AnchorPoints:
    int anchorpoints[1024];
    int npoints;

    void OnLeftClick(int p);
    void OnRightClick(int p);
    void OnMiddleClick(int p);
    void OnMouseWheel(int p, int rotation, int delta);    
    void OnLeftDrag(int p, int q);
    void OnRightDrag(int p, int q);

    void OnMouseMotion(int p);
    void OnEnterWindow();
    void OnLeaveWindow();

    int *LiveWire(int px, 
		  bool add_click, 
		  bool close_click, 
		  bool del_click);

    Pixel GetCorrespondingPixel(int p_scn);
    Voxel GetCorrespondingVoxel(int p_img);

    Image *GetXSlice(Scene *scn, int x);
    Image *GetYSlice(Scene *scn, int y);
    Image *GetZSlice(Scene *scn, int z);

    void RefreshPath(int *boundary);

    void InsertPathSeeds(int *cpath);

    bool IsAnchorPoint(int px);
    int  GetLastAnchorPoint();
    int  GetFirstAnchorPoint();
    void AppendAnchorPoint(int px);
    void DelLastAnchorPoint();
    void ClearAnchorPoints();
  };
} //end OIFT namespace

#endif

