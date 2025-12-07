
#include "poshandler.h"

namespace IntensityProfile{

  PosHandler :: PosHandler(ProfileDialog *dialog){
    this->dialog = dialog;
  }

  PosHandler :: ~PosHandler(){}

  void PosHandler :: OnEnterWindow(){
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'x');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'y');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'z');
  }


  void PosHandler :: OnLeaveWindow(){}

  void PosHandler :: OnLeftClick(int p){}
  
  void PosHandler :: OnRightClick(int p){}
  
  void PosHandler :: OnMiddleClick(int p){}
  
  void PosHandler :: OnLeftDrag(int p, int q) {}
  
  void PosHandler :: OnRightDrag(int p, int q){}

  void PosHandler :: OnMouseMotion(int p){
    int margin_left,margin_right;
    int length,w,h,x,sizex,pos;

    length = dialog->GetProfileLength();
    dialog->GetPlotSize(&w, &h);

    margin_left   = 70;
    margin_right  = 40;
    sizex = w-(margin_left+margin_right);

    x = p%w - margin_left;
    if(x>=sizex || x<0) return;

    pos = ROUND(length*((float)x/(float)sizex));
    if(pos>=length) pos = length-1;
    dialog->SetPosition(pos);
  }

} //end IntensityProfile namespace


