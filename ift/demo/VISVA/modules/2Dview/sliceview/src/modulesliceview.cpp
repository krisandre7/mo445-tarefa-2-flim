
#include "modulesliceview.h"

namespace SliceView{

  void freewxPanel(void **mem){
    if(*mem!=NULL){
      //Module *mod = (Module *)*mem;
      //delete mod;
      wxPanel*pan = (wxPanel*)*mem;
      delete pan;
      *mem=NULL;
    }
  }


  ModuleSliceView :: ModuleSliceView()
    : View2DModule(){
    SetName((char *)"Slice view canvas");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);

    Panels = CreateArrayList(4);
    SetArrayListCleanFunc(Panels, freewxPanel);

    SliceViewOpt *sliceopt = new SliceViewOpt(APP->Window, this);
    optPanel = (wxPanel *)sliceopt;
  }


  ModuleSliceView :: ~ModuleSliceView(){
    DestroyArrayList(&Panels);
    delete optPanel;
  }


  void ModuleSliceView :: Start(){
    APP->PrependOptPanel(optPanel, this->GetType());
  }

  bool ModuleSliceView :: Stop(){
    APP->DetachOptPanel(optPanel);
    return true;
  }

  wxPanel* ModuleSliceView :: GetViewPanel(wxWindow *parent, 
					   char axis){
    SliceView *view = new SliceView(parent, axis);
    AddArrayListElement(Panels, (void *)view);
    return (wxPanel*)view;
  }

  wxPanel* ModuleSliceView :: GetCustomViewPanel(wxWindow *parent, 
						 Scene *scn,
						 char axis){
    SliceView *view = new SliceView(parent, axis);
    view->canvas->SetCustomData(scn);
    AddArrayListElement(Panels, (void *)view);
    return (wxPanel*)view;
  }


  void ModuleSliceView :: DelViewPanel(wxPanel **view){
    DelArrayListElement_2(Panels, (void **)view);
  }


  void ModuleSliceView :: SetInteractionHandler(InteractionHandler *handler,
						char axis){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      if(view->canvas->GetSliceAxis()==axis)
	view->canvas->SetInteractionHandler(handler);
    }
  }

  void ModuleSliceView :: SetCursor(  wxCursor *cursor,
				    char axis){
    SliceView *view;
    int i;
    
    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      if(view->canvas->GetSliceAxis()==axis)
	view->canvas->SetCursor(*cursor);
    }
  }

  void ModuleSliceView :: Refresh(){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->SliceRefresh();
    }
  }

  void ModuleSliceView :: Refresh(char axis){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      if(view->canvas->GetSliceAxis()==axis)
	view->SliceRefresh();
    }
  }

  void ModuleSliceView :: SetRefreshHandler(RefreshHandler *handler){
    SliceView *view;
    int i;
    
    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->canvas->SetRefreshHandler(handler);
    }
  }

  void ModuleSliceView :: Clear(){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->canvas->ClearFigure();
      view->canvas->Refresh();
      //CImage *tmp = CreateCImage(10,10);
      //view->canvas->DrawCImage(tmp);
      //DestroyCImage(&tmp);
    }
  }

  void ModuleSliceView :: Zoomin(){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->canvas->Zoomin();
    }
  }

  void ModuleSliceView :: Zoomout(){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->canvas->Zoomout();
    }
  }

  void ModuleSliceView :: ChangeDrawMarker(){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->canvas->ChangeDrawMarker();
    }
  }

  void ModuleSliceView :: SetZoomLevel(float zoom){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->canvas->SetZoomLevel(zoom);
    }
  }

  float ModuleSliceView :: GetZoomLevel(){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      return view->canvas->GetZoomLevel();
    }
    return 1.0;
  }

  void ModuleSliceView :: Set2DViewOptions(HighlightType highlight, 
					   DataType data, 
					   MarkerType marker){
    SliceView *view;
    int i;
  
    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->canvas->Set2DViewOptions(highlight, 
				     data, marker);
    }
  }


  void ModuleSliceView :: SetSliceVoxel(Voxel Cut){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      view->SetCutVoxel(Cut);
    }
  }


  Voxel ModuleSliceView :: GetSliceVoxel(){
    SliceView *view;

    view = (SliceView *)GetArrayListElement(Panels, 0);
    return view->GetCutVoxel();
  }


  CImage * ModuleSliceView :: CopyAsCImage(char axis){
    SliceView *view;
    int i;

    for(i=0; i<Panels->n; i++){
      view = (SliceView *)GetArrayListElement(Panels, i);
      if(view->canvas->GetSliceAxis()==axis)
	return view->canvas->CopyAsCImage();
    }
    return NULL;
  }

} //end SliceView namespace

