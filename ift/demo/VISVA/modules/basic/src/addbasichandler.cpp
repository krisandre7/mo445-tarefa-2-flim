
#include "addbasichandler.h"

namespace Basic{

  AddHandler :: AddHandler(char axis,
			   BrushPicker *brush){
    this->axis = axis;
    this->markerID = 1;
    this->brush = brush;
  }

  AddHandler :: ~AddHandler(){}

  void AddHandler :: OnLeftClick(int p){
    AdjRel *A;

    markerID++;
    A = brush->GetBrush();
    APP->AddSeedsInBrush(p, 1, unique(markerID),
			 A, axis);
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void AddHandler :: OnRightClick(int p){
    AdjRel *A;

    markerID++;
    A = brush->GetBrush();
    APP->AddSeedsInBrush(p, 0, unique(markerID),
			 A, axis);
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void AddHandler :: OnMiddleClick(int p){
    //Reset:
    /*
      APP->ResetData();
      this->markerID = 1;

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
    */
  }

  void AddHandler :: OnLeftDrag(int p, int q){
    AdjRel *A;

    A = brush->GetBrush();
    APP->AddSeedsInBrushTrace(p,q,1,unique(markerID),
			      A,axis);
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void AddHandler :: OnRightDrag(int p, int q){
    AdjRel *A;

    A = brush->GetBrush();
    APP->AddSeedsInBrushTrace(p,q,0,unique(markerID),
			      A,axis);
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void AddHandler :: OnMouseMotion(int p){
    APP->Window->SetStatusText(_T("Mouse Left: Add Object Marker, Mouse Center: Reset Segmentation, Mouse Right: Add Background Marker"));
  }


  int AddHandler :: unique(int id){
    id *= 4;
    if(axis=='x')      id += 1;
    else if(axis=='y') id += 2;
    else               id += 3;
    return id;
  }

} //end Basic namespace
