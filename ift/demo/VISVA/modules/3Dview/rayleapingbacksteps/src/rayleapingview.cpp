
#include "rayleapingview.h"

#include "../xpm/glasses.xpm"
#include "../xpm/save_canvas.xpm"

namespace Rayleaping{

    RayleapingView :: RayleapingView(wxWindow *parent)
            : BasePanel(parent){
        int id1;

        canvas  = new RayleapingCanvas(this);

        id1 = APP->idManager->AllocID();

        wxBitmap *bm = new wxBitmap(glasses_xpm);
        but3D = new BitmapToggleButton(this, id1, bm, bm);
        but3D->SetBkgColor(true, 0x808080);
        but3D->SetValue(false);

        wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
        hsizer->AddSpacer(5);
        hsizer->Add(but3D, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);

        sizer->Add(canvas, 1, wxEXPAND);
        sizer->Add(hsizer, 0, wxALIGN_LEFT|wxEXPAND);
        hsizer->AddSpacer(10);

        int save_id = APP->idManager->AllocID();
        wxBitmap *bmSave = new wxBitmap(save_canvas_xpm);
        butSave = new wxBitmapButton(this, save_id, *bmSave, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("butSaveCanvas"));
        hsizer->Add(butSave, 0, wxALIGN_LEFT);
        hsizer->AddSpacer(20);


        SetSizer(sizer, true);
        sizer->SetSizeHints(this);
        sizer->Layout();

        Connect( id1, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(RayleapingView::OnChange3D),
                 NULL, NULL );

        Connect( save_id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(RayleapingView::OnSaveCanvasView),
                 NULL, NULL );

    }


    RayleapingView :: ~RayleapingView(){
        int id1 = but3D->GetId();
        Disconnect( id1, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(RayleapingView::OnChange3D),
                    NULL, NULL );
        APP->idManager->FreeID(id1);
    }

    void RayleapingView :: OnChange3D(wxCommandEvent& event){
        APP->Refresh3DCanvas(false, 1.0);
    }

    void RayleapingView :: OnSaveCanvasView(wxCommandEvent & event) {
        wxString path;

        path = wxSaveFileSelector(_("Save canvas view"),_(""), _(""), APP->Window);

        if(path != _("")) {
            wxImage *img = canvas->GetImage();

            if(!img->SaveFile(path))
                wxMessageBox(_("Invalid file extension"), _("Error saving image"));
        }
    }

} //end Rayleaping namespace




