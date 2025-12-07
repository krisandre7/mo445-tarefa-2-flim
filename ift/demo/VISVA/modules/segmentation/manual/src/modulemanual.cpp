
#include "modulemanual.h"
//#include "manualopt.h"
#include "manualdialog.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Manual{

  ModuleManual :: ModuleManual()
    : SegmentationModule(){
    SetName((char *)"Manual");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(2,0,0);
    SetVersion(ver);

    optDialog = NULL;

    obj_visibility[0] = true;
    obj_color[0] = 0xffff00;
    obj_name[0][0] = '\0';
    obj_sel = 0;
    nobjs = 0;
  }


  ModuleManual :: ~ModuleManual(){
  }

  void ModuleManual :: Start(){
    SegmObject *obj=NULL;
    int r;

    APP->EnableObjWindow(false);
    APP->ResetData();

    obj_visibility[0] = true;
    r = APP->ShowNewObjDialog(&this->obj_color[0],
			      this->obj_name[0]);
    if (r!=0) return;

    this->active = true;
    this->nobjs = 1;
    this->obj_sel = 0;
    APP->SetLabelColour(0, NIL);
    APP->SetLabelColour(1, this->obj_color[0]);

    obj = APP->SearchObjByName(this->obj_name[0]);
    if(obj!=NULL)
      CopyBMap2SceneMask(APP->Data.label,obj->mask);

    //APP->AppendOptPanel(optPanel, this->GetType());
    this->AllocData();

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);

  }


  void ModuleManual :: AllocData(){
    int x,y,w,h;

    if(optDialog!=NULL) delete optDialog;
    ManualDialog *dialog = new ManualDialog(APP->Window, this);
    optDialog = (BaseDialog*)dialog;
    optDialog->Show(true);
    APP->Window->GetPosition(&x, &y);
    APP->Window->GetSize(&w, &h);
    optDialog->Move(MAX(x-20,0), h/2); //wxDefaultCoord);
  }



  bool ModuleManual :: Stop(){
    static const char *title = {"Keep segmentation?"};
    static const char *msg = {
      "You are about to leave the manual module.\nSave changes?"};

    if(!this->active) return true;

    wxString wxtitle(title, wxConvUTF8);
    wxString wxmsg(msg, wxConvUTF8);

    wxMessageDialog dialog(APP->Window,
			   wxmsg, wxtitle,
			   wxYES_NO | wxICON_QUESTION, 
			   wxDefaultPosition);

    if(dialog.ShowModal() == wxID_YES)
      this->Finish();
    else
      this->FreeData();

    return true;
  }


  void ModuleManual :: FreeData(){
    //APP->DetachOptPanel(optPanel);
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    this->active = false;

    optDialog->Show(false);
    if(optDialog!=NULL)
      optDialog->Destroy(); //delete optDialog;
    optDialog = NULL;
  }


  void ModuleManual :: Finish(){
    int p,n,i;
    Scene *label = APP->Data.label;
    SegmObject *obj;

    n = APP->Data.w*APP->Data.h*APP->Data.nframes;

    // set visibility
    for(i=0; i<APP->GetNumberOfObjs(); i++){
      SegmObject *so;
      so = APP->GetObjByIndex(i);
      so->visibility=false;
    }

    for(i=0; i<nobjs; i++){

      obj = CreateSegmObject(this->obj_name[i],
			     this->obj_color[i]);
      obj->mask = BMapNew(n);
      BMapFill(obj->mask, 0);
      for(p=0; p<n; p++){
	if(label->data[p]==i+1)
	  _fast_BMapSet1(obj->mask, p);
      }
      obj->seed = APP->CopyMarkerList();
      APP->AddCustomObj(obj);
      APP->SetObjVisibility(obj->name, true);
    }
    this->FreeData();
  }



  void ModuleManual :: Reset(){
    wxMessageDialog dialog(APP->Window, 
			   _T("Current segmentation will be lost.\nAre you sure you want to reset?"), 
			   _T("Reset segmentation?"), 
			   wxYES_NO | wxICON_QUESTION, 
			   wxDefaultPosition);

    if(dialog.ShowModal() == wxID_YES){
      APP->ResetData();
      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
    }
  }


  AdjRel *ModuleManual :: GetBrush(){
    ManualDialog *opt = (ManualDialog *)optDialog;
    return opt->GetBrush();
  }

  wxCursor *ModuleManual :: GetBrushCursor(int zoom){
    ManualDialog *opt = (ManualDialog *)optDialog;
    return opt->GetBrushCursor(zoom);
  }

  void   ModuleManual :: NextBrush(){
    ManualDialog *opt = (ManualDialog *)optDialog;
    opt->NextBrush();
  }

  void   ModuleManual :: PrevBrush(){
    ManualDialog *opt = (ManualDialog *)optDialog;
    opt->PrevBrush();
  }

  void ModuleManual :: DeleteObj(int obj){
    int o,p,lb;

    for(o=obj ; o<this->nobjs-1; o++){
      APP->SetLabelColour(o+1, this->obj_color[o+1]);
      obj_color[o] = obj_color[o+1];
      obj_visibility[o] = obj_visibility[o+1];
      strcpy(obj_name[o], obj_name[o+1]);
    }
    this->nobjs--;
    if(obj_sel>nobjs-1) this->obj_sel--;

    for(p=0; p<(APP->Data.label)->n; p++){
      lb = (APP->Data.label)->data[p];
      if(lb==obj+1){
	(APP->Data.label)->data[p] = 0;
      }
      else if(lb>obj+1)
	(APP->Data.label)->data[p] = lb-1;	
    }
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);
  }

} //end Manual namespace




