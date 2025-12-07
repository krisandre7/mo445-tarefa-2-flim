
#include "surfaceview.h"

#include "../xpm/glasses.xpm"
#include "../xpm/planes.xpm"
#include "../xpm/save_canvas.xpm"

namespace Surface{

    SurfaceView :: SurfaceView(wxWindow *parent)
            : BasePanel(parent){
        int id1,id2;

        canvas  = new SurfaceCanvas(this);

        id1 = APP->idManager->AllocID();
        id2 = APP->idManager->AllocID();

        wxBitmap *bm = new wxBitmap(glasses_xpm);
        but3D = new BitmapToggleButton(this, id1, bm, bm);
        but3D->SetBkgColor(true, 0x808080);
        but3D->SetValue(false);

        wxBitmap *bm2 = new wxBitmap(planes_xpm);
        butPlanes = new BitmapToggleButton(this, id2, bm2, bm2);
        butPlanes->SetBkgColor(true, 0x808080);
        butPlanes->SetValue(true);

        wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
        hsizer->AddSpacer(5);
        hsizer->Add(but3D,     0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
        hsizer->Add(butPlanes, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);

        int save_id = APP->idManager->AllocID();
        wxBitmap *bmSave = new wxBitmap(save_canvas_xpm);
        butSave = new wxBitmapButton(this, save_id, *bmSave, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("butSaveCanvas"));
        hsizer->Add(butSave, 0, wxALIGN_LEFT);
        hsizer->AddSpacer(20);

        sizer->Add(canvas, 1, wxEXPAND);
        sizer->Add(hsizer, 0, wxALIGN_LEFT|wxEXPAND);

        SetSizer(sizer, true);
        sizer->SetSizeHints(this);
        sizer->Layout();

        Connect( id1, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(SurfaceView::OnChange3D),
                 NULL, NULL );
        Connect( id2, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(SurfaceView::OnChangeViewPlanes),
                 NULL, NULL );

        Connect( save_id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(SurfaceView::OnSaveCanvasView),
                 NULL, NULL );

    }


    SurfaceView :: ~SurfaceView(){
        int id1 = but3D->GetId();
        int id2 = butPlanes->GetId();
        Disconnect( id1, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(SurfaceView::OnChange3D),
                    NULL, NULL );
        Disconnect( id2, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(SurfaceView::OnChangeViewPlanes),
                    NULL, NULL );
        APP->idManager->FreeID(id1);
        APP->idManager->FreeID(id2);
    }


    void SurfaceView :: OnChange3D(wxCommandEvent& event){
        APP->Refresh3DCanvas(false, 1.0);
    }

    void SurfaceView :: OnChangeViewPlanes(wxCommandEvent& event){
        APP->Refresh3DCanvas(true, 1.0);
    }

    void SurfaceView :: OnSaveCanvasView(wxCommandEvent & event) {
        wxString path;

        path = wxSaveFileSelector(_("Save canvas view"),_(""), _(""), APP->Window);

        if(path != _("")) {
            wxImage *img = canvas->GetImage();

            if(!img->SaveFile(path))
                wxMessageBox(_("Invalid file extension"), _("Error saving image"));
        }
    }

} //end Surface namespace




