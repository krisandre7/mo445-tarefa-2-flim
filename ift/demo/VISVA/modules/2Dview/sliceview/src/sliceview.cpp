
#include "sliceview.h"

#include "../xpm/rotate.xpm"
#include "../xpm/save_canvas.xpm"

namespace SliceView{

    Voxel SliceView::Cut = {0,0,0};


    SliceView :: SliceView(wxWindow *parent, char axis)
            : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize,wxSIMPLE_BORDER){
        int id;

        owner = parent;
        //SetBackgroundColour(*wxBLUE);
        ntimes = 0;
        if (axis=='x') ntimes=3;
        if (axis=='y') ntimes=2;
        //Cut.x = Cut.y = Cut.z = 0;
        wxBoxSizer *vsizer = new wxBoxSizer(wxVERTICAL);
        canvas  = new SliceCanvas(this);
        canvas->SetSliceAxis(axis);
        vsizer->Add(canvas, 1, wxEXPAND);

        balloon = new wxFrame(canvas, -1, _T(""), wxDefaultPosition, wxDefaultSize, wxFRAME_NO_TASKBAR | wxNO_BORDER | wxFRAME_FLOAT_ON_PARENT | wxCLIP_CHILDREN, _T("Balloon help"));
        wxColour Colour(255, 255, 220);
        balloon->SetBackgroundColour(Colour);
        balloon->SetForegroundColour(*wxBLACK);
        wxBoxSizer *bsizer = new wxBoxSizer(wxHORIZONTAL);
        balloontext = new wxStaticText(balloon, -1, _T("Balloon"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("balloontext"));
        bsizer->Add(balloontext, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
        balloon->SetSizer(bsizer, true);
        bsizer->SetSizeHints(balloon);
        MoveBalloon(0, 0);

        wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);

        id = APP->idManager->AllocID();
        spinSlice = new wxSpinCtrl(this, id, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS|wxSP_VERTICAL, 0, 0, 0, _T("ChangeSlices"));
        hsizer->Add(spinSlice, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
        hsizer->AddSpacer(10);

        wxBitmap *bmRot = new wxBitmap(rotate_xpm);
        butRotate = new wxBitmapButton(this, id, *bmRot, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("butRotate"));
        hsizer->Add(butRotate, 0, wxALIGN_LEFT);
        hsizer->AddSpacer(10);

        int save_id = APP->idManager->AllocID();
        wxBitmap *bmSave = new wxBitmap(save_canvas_xpm);
        butSave = new wxBitmapButton(this, save_id, *bmSave, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("butSaveCanvas"));
        hsizer->Add(butSave, 0, wxALIGN_LEFT);
        hsizer->AddSpacer(20);


        coordinate = new wxStaticText(this, -1, _T(""), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("coordinate"));
        hsizer->Add(coordinate, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);

        vsizer->Add(hsizer, 0, wxALIGN_LEFT|wxEXPAND);

        SetSizer(vsizer, true);
        vsizer->SetSizeHints(this);

        Connect( id, wxEVT_COMMAND_SPINCTRL_UPDATED,
                 wxSpinEventHandler(SliceView::OnSliceSpin),
                 NULL, NULL );
        Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(SliceView::OnSliceRotate),
                 NULL, NULL );

        Connect( save_id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(SliceView::OnSaveCanvasView),
                 NULL, NULL );

    }


    SliceView::~SliceView(){
        int id = spinSlice->GetId();

        Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(SliceView::OnSliceRotate),
                    NULL, NULL );
        Disconnect( id, wxEVT_COMMAND_SPINCTRL_UPDATED,
                    wxSpinEventHandler(SliceView::OnSliceSpin),
                    NULL, NULL );
        APP->idManager->FreeID(id);
    }

    void SliceView :: SliceRefresh(){
        char axis = this->canvas->GetSliceAxis();
        switch(axis){
            case 'z':
                spinSlice->SetRange(0, APP->Data.nframes-1);
                spinSlice->SetValue(Cut.z);
                break;
            case 'x':
                spinSlice->SetRange(0, APP->Data.w-1);
                spinSlice->SetValue(Cut.x);
                break;
            case 'y':
                spinSlice->SetRange(0, APP->Data.h-1);
                spinSlice->SetValue(Cut.y);
                break;
        }
        this->canvas->Draw();
    }


    int SliceView :: GetSliceNumber() {
        return spinSlice->GetValue();
    }

    void SliceView :: SetCutVoxel(Voxel Cut){
        this->Cut = Cut;
    }

    Voxel SliceView :: GetCutVoxel(){
        return this->Cut;
    }

    void SliceView :: ShowCoordinate(Voxel *v){
        char voxel[100];
        int p,val,lb;
        int gx,gy,gz;

        if(!APP->Data.loaded) return;

        if(v!=NULL){
            p = VoxelAddress(APP->Data.orig,v->x,v->y,v->z);

            if(this->canvas->data == DATA_ORIG){
                val = GetVoxel(APP->Data.orig, p);
                sprintf(voxel,"voxel (%d,%d,%d) = %d",v->x,v->y,v->z,val);
            }
            else if(this->canvas->data == DATA_ARCW){
                val = GetVoxel(APP->Data.arcw, p);
                sprintf(voxel,"voxel (%d,%d,%d) = %d",v->x,v->y,v->z,val);
            }
            else if(this->canvas->data == DATA_GRAD){
                if(APP->Data.grad!=NULL){
                    gx = GetVoxel((APP->Data.grad)->Gx, p);
                    gy = GetVoxel((APP->Data.grad)->Gy, p);
                    gz = GetVoxel((APP->Data.grad)->Gz, p);
                }
                else{
                    gx = 0;
                    gy = 0;
                    gz = 0;
                }
                sprintf(voxel,"(%d,%d,%d) = (%d,%d,%d)",v->x,v->y,v->z,gx,gy,gz);
            }
            else if(this->canvas->data == DATA_CUSTOM){
                val = GetVoxel(this->canvas->custom_scn, p);
                sprintf(voxel,"voxel (%d,%d,%d) = %d",v->x,v->y,v->z,val);
            }
            else{
                val = GetVoxel(APP->Data.orig,  p);
                lb  = GetVoxel(APP->Data.label, p);
                if(APP->GetLabelColour(lb)==NIL) val = 0;
                sprintf(voxel,"voxel (%d,%d,%d) = %d",v->x,v->y,v->z,val);
            }
            wxString wxvoxel(voxel, wxConvUTF8);
            coordinate->SetLabel(wxvoxel);
        }
        else
            coordinate->SetLabel(_T(""));
    }

    void SliceView :: OnSliceRotate(wxCommandEvent & event){
        if(!APP->Data.loaded) return;

        ntimes++;
        ntimes %= 4;
        this->canvas->Draw();
    }

    void SliceView :: OnSliceSpin(wxSpinEvent& event) {
        char axis = this->canvas->GetSliceAxis();
        int value = this->spinSlice->GetValue();
        switch(axis){
            case 'z': Cut.z = value; break;
            case 'x': Cut.x = value; break;
            case 'y': Cut.y = value; break;
        }
        //this->canvas->Draw();
        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(false, 1.0);
    }

    void SliceView :: OnSaveCanvasView(wxCommandEvent & event) {
        wxString path;

        path = wxSaveFileSelector(_("Save canvas view"),_(""), _(""), APP->Window);

        if(path != _("")) {
            wxImage *img = canvas->GetImage();

            if(!img->SaveFile(path))
                wxMessageBox(_("Invalid file extension"), _("Error saving image"));
        }
    }

    void SliceView :: EnableBalloon(){
        /*
          if(this==Views[0]){
          Views[1]->DisableBalloon();
          Views[2]->DisableBalloon();
          }
          else if(this==Views[1]){
          Views[0]->DisableBalloon();
          Views[2]->DisableBalloon();
          }
          else if(this==Views[2]){
          Views[0]->DisableBalloon();
          Views[1]->DisableBalloon();
          }
        */
        if(!balloon->IsShown()){
            balloon->Show(true);
            //this->SetFocusIgnoringChildren();
            APP->Window->SetFocus();
        }
    }

    void SliceView :: DisableBalloon(){
        balloon->Show(false);
    }

    void SliceView :: SetBalloonText(char *text){
        wxSizer *sizer;
        //balloon->SetTip(text);
        wxString wxtext(text, wxConvUTF8);
        balloontext->SetLabel(wxtext);
        sizer = balloon->GetSizer();
        sizer->SetSizeHints(balloon);
        sizer->Layout();
    }

    void SliceView :: MoveBalloon(int x, int y){
        int sx,sy;

        sx = x; sy = y;
        canvas->ClientToScreen(&sx, &sy);
        balloon->Move(sx+20, sy+20);
    }

} //end SliceView namespace
