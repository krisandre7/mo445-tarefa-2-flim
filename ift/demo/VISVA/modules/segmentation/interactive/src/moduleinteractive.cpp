
#include "moduleinteractive.h"
//#include "interactiveopt.h"
#include "interactivedialog.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Interactive
{

ModuleInteractive :: ModuleInteractive()
    : SegmentationModule()
{
    SetName((char *)"Semi-automatic");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(2, 0, 0);
    SetVersion(ver);

    //DIFT data:
    cost = NULL;
    pred = NULL;

    optDialog = NULL;

    obj_visibility[0] = true;
    obj_color[0] = 0xffff00;
    obj_name[0][0] = '\0';
    obj_sel = 0;
    nobjs = 0;
}


ModuleInteractive :: ~ModuleInteractive()
{
    //DIFT data:
    if (cost != NULL) bia::Scene16::Destroy(&cost);
    if (pred != NULL) DestroyScene(&pred);
}


void ModuleInteractive :: Start()
{
    static const char *title = "Resume segmentation?";
    static const char *msg = "The selected object already exists.\nDo you want to resume the previous segmentation?\n ('Yes' to resume, 'No' to start from scratch).";
    SegmObject *obj = NULL;
    Scene *bin = NULL;
    Set *S = NULL, *aux;
    int r, p, id, cod;
    timer tic;

    this->markerID = 4;
    APP->EnableObjWindow(false);
    APP->ResetData();

    obj_visibility[0] = true;
    r = APP->ShowNewObjDialog(&this->obj_color[0],
                              this->obj_name[0]);
    if (r != 0) return;

    this->active = true;
    this->nobjs = 1;
    this->obj_sel = 0;
    APP->SetLabelColour(0, NIL);
    APP->SetLabelColour(1, this->obj_color[0]);
    //APP->AppendOptPanel(optPanel, this->GetType());

    this->ComputeArcWeight();
    this->AllocData();

    obj = APP->SearchObjByName(this->obj_name[0]);
    if (obj == NULL) return;

    //----- The object already exists: ------------
    wxString wxtitle(title, wxConvUTF8);
    wxString wxmsg(msg, wxConvUTF8);

    wxMessageDialog dialog(APP->Window,
                           wxmsg, wxtitle,
                           wxYES_NO | wxICON_QUESTION,
                           wxDefaultPosition);

    if (dialog.ShowModal() != wxID_YES) return;

    //----- resuming previous segmentation: ------------
    APP->Busy((char *)"Please wait, resuming previous segmentation...");
    APP->StatusMessage((char *)"Please wait - Resuming Segmentation...");
    StartTimer(&tic);

    bin = CreateScene(APP->Data.w,
                      APP->Data.h,
                      APP->Data.nframes);
    CopySegmObjectMask2Scene(obj, bin);

    this->SetWf(APP->Data.arcw, bin);

    //ResumeFromScratchDIFT_Prototype(this->Wf, bin,
    //                                cost, pred,
    //                    APP->Data.label, &S);



    ResumeFromScratchDIFT_MinSeeds(APP->Data.orig,
                                   this->Wf, bin,
                                   cost, pred,
                                   APP->Data.label, &S);
    printf("After remove from scratch\n");


    for (p = 0; p < bin->n; p++)
    {
        id = (APP->Data.label)->data[p] + 4;
        cod = this->GetCodeValue(id, bin->data[p]);
        (APP->Data.label)->data[p] = cod;
    }
    this->markerID = 0;
    aux = S;
    while (aux != NULL)
    {
        p  = aux->elem;
        cod = (APP->Data.label)->data[p];
        id = this->GetCodeID(cod);

        if (bin->data[p] > 0) APP->SetLabelColour(cod, this->obj_color[0]);
        else               APP->SetLabelColour(cod, NIL);

        APP->AddSeed(p, bin->data[p], id);
        this->markerID = MAX(this->markerID, id);
        aux = aux->next;
    }
    this->markerID++;
    DestroyScene(&bin);

    printf("\nResuming previous segmentation - Time: ");
    StopTimer(&tic);

    //this->PrintSeedReport();

    APP->StatusMessage((char *)"Done");
    APP->Unbusy();

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
}

void ModuleInteractive :: SetWf(Scene *Wl, Scene *bin)
{
    int Imax = MaximumValue3(Wl);
    int p, q, i;
    bia::Voxel u, v;
    bia::AdjRel3::AdjRel3 *A = bia::AdjRel3::Spheric(1.0);
    float a, w_c, w_l;

    if (bin == NULL) a = 0.0;
    else          a = 0.4;

    for (u.c.z = 0; u.c.z < Wl->zsize; u.c.z++)
    {
        for (u.c.y = 0; u.c.y < Wl->ysize; u.c.y++)
        {
            for (u.c.x = 0; u.c.x < Wl->xsize; u.c.x++)
            {
                p = VoxelAddress(Wl, u.c.x, u.c.y, u.c.z);
                w_c = 0.0;
                if (bin != NULL)
                {
                    for (i = 1; i < A->n; i++)
                    {
                        v.v = u.v + A->d[i].v;
                        if (!ValidVoxel(Wl, v.c.x, v.c.y, v.c.z)) continue;
                        q = VoxelAddress(Wl, v.c.x, v.c.y, v.c.z);
                        if (bin->data[p] != bin->data[q])
                        {
                            w_c = 1.0;
                            break;
                        }
                    }
                }
                if (Imax == 0) w_l = 0.0;
                else        w_l = ((float)Wl->data[p]) / ((float)Imax);

                (this->Wf)->data[p] = ROUND(65000 * ((1.0 - a) * w_l + a * w_c));
            }
        }
    }
    bia::Scene16::GetMaximumValue(Wf);
    bia::AdjRel3::Destroy(&A);
}



void ModuleInteractive :: ComputeArcWeight()
{
    timer tic;

    if (MaximumValue3(APP->Data.arcw) == 0)
        DestroyScene(&(APP->Data.arcw));

    if (APP->Data.arcw == NULL)
    {
        APP->Busy((char *)"Please wait, computing gradient...");
        APP->StatusMessage((char *)"Please wait - Computing Gradient...");

        StartTimer(&tic);
        APP->Data.arcw = BrainGrad3(APP->Data.orig);

        printf("\nGradient Time: ");
        StopTimer(&tic);

        APP->Unbusy();
        APP->StatusMessage((char *)"Done");
    }
    else
        printf("Gradient already loaded...\n");

    MaximumValue3(APP->Data.arcw);

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
}


void ModuleInteractive :: AllocData()
{
    int x, y, w, h;
    //DIFT data:
    if (pred != NULL) DestroyScene(&pred);
    if (cost != NULL) bia::Scene16::Destroy(&cost);
    if (Wf  != NULL) bia::Scene16::Destroy(&Wf);
    pred = CreateScene(APP->Data.w,
                       APP->Data.h,
                       APP->Data.nframes);
    cost = bia::Scene16::Create(APP->Data.w,
                                APP->Data.h,
                                APP->Data.nframes);
    Wf   = bia::Scene16::Create(APP->Data.w,
                                APP->Data.h,
                                APP->Data.nframes);
    SetWf(APP->Data.arcw, NULL);

    bia::Scene16::Fill(cost, USHRT_MAX);
    SetScene(pred,     NIL);
    APP->ResetData();
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);

    if (optDialog != NULL) delete optDialog;
    InteractiveDialog *dialog = new InteractiveDialog(APP->Window, this);
    optDialog = (BaseDialog *)dialog;
    optDialog->Show(true);
    APP->Window->GetPosition(&x, &y);
    APP->Window->GetSize(&w, &h);
    optDialog->Move(MAX(x - 20, 0), h / 2); //wxDefaultCoord);
}


bool ModuleInteractive :: Stop()
{
    static const char *title = {"Keep segmentation?"};
    static const char *msg =
    {
        "You are about to leave the semi-automatic module.\nSave changes?"
    };

    if (!this->active) return true;

    wxString wxtitle(title, wxConvUTF8);
    wxString wxmsg(msg, wxConvUTF8);

    wxMessageDialog dialog(APP->Window,
                           wxmsg, wxtitle,
                           wxYES_NO | wxICON_QUESTION,
                           wxDefaultPosition);

    if (dialog.ShowModal() == wxID_YES)
        this->Finish();
    else
        this->FreeData();

    return true;
}


void ModuleInteractive :: FreeData()
{
    //DIFT data:
    if (pred != NULL) DestroyScene(&pred);
    if (cost != NULL) bia::Scene16::Destroy(&cost);
    if (Wf  != NULL) bia::Scene16::Destroy(&Wf);

    //APP->DetachOptPanel(optPanel);
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    this->active = false;

    optDialog->Show(false);
    if (optDialog != NULL)
        optDialog->Destroy(); //delete optDialog;
    optDialog = NULL;
}


void ModuleInteractive :: PrintSeedReport()
{
    int p, l, Lmax = 0;
    int *hist = NULL;
    Set *S, *tmp;

    S = APP->CopySeeds();

    tmp = S;
    while (tmp != NULL)
    {
        p = tmp->elem;
        l = APP->GetSeedLabel(p);
        if (l > Lmax) Lmax = l;
        tmp = tmp->next;
    }

    hist = AllocIntArray(Lmax + 1);

    tmp = S;
    while (tmp != NULL)
    {
        p = tmp->elem;
        l = APP->GetSeedLabel(p);
        if (l >= 0 && l <= Lmax)
            hist[l]++;
        tmp = tmp->next;
    }
    printf("SeedReport:\n");
    for (l = 0; l <= Lmax; l++)
    {
        printf("\tLabel%02d: %d\n", l, hist[l]);
    }
    free(hist);
    DestroySet(&S);
}


void ModuleInteractive :: Finish()
{
    int p, n, i;
    Scene *label = APP->Data.label;
    SegmObject *obj, *tmp;

    n = APP->Data.w * APP->Data.h * APP->Data.nframes;

    // set visibility
    for (i = 0; i < APP->GetNumberOfObjs(); i++)
    {
        SegmObject *so;
        so = APP->GetObjByIndex(i);
        so->visibility = false;
    }

    for (i = 0; i < nobjs; i++)
    {

        obj = CreateSegmObject(this->obj_name[i],
                               this->obj_color[i]);
        obj->mask = BMapNew(n);
        BMapFill(obj->mask, 0);
        for (p = 0; p < n; p++)
        {
            if (this->GetCodeLabel(label->data[p]) == i + 1)
                _fast_BMapSet1(obj->mask, p);
        }
        obj->seed = APP->CopyMarkerList();
        APP->AddCustomObj(obj);
        APP->SetObjVisibility(obj->name, true);
    }

    // Fix brain:
    bool isnew = false, fix = false;
    for (i = 0; i < nobjs; i++)
    {
        if (strcasecmp(this->obj_name[i],
                       (char *)"Left Hemisphere") == 0) fix = true;
        if (strcasecmp(this->obj_name[i],
                       (char *)"Right Hemisphere") == 0) fix = true;
        if (strcasecmp(this->obj_name[i],
                       (char *)"Cerebellum") == 0) fix = true;
    }
    if (fix &&
            APP->SearchObjByName((char *)"Left Hemisphere") != NULL &&
            APP->SearchObjByName((char *)"Right Hemisphere") != NULL &&
            APP->SearchObjByName((char *)"Cerebellum") != NULL )
    {

        obj = APP->SearchObjByName((char *)"Brain");
        if (obj == NULL)
        {
            isnew = true;
            obj = CreateSegmObject((char *)"Brain",
                                   0xffff00);
            obj->seed = NULL;
            obj->mask = BMapNew(n);
        }
        BMapFill(obj->mask, 0);

        tmp = APP->SearchObjByName((char *)"Left Hemisphere");
        for (p = 0; p < n; p++)
        {
            if (_fast_BMapGet(tmp->mask, p) > 0)
                _fast_BMapSet1(obj->mask, p);
        }

        tmp = APP->SearchObjByName((char *)"Right Hemisphere");
        for (p = 0; p < n; p++)
        {
            if (_fast_BMapGet(tmp->mask, p) > 0)
                _fast_BMapSet1(obj->mask, p);
        }

        tmp = APP->SearchObjByName((char *)"Cerebellum");
        for (p = 0; p < n; p++)
        {
            if (_fast_BMapGet(tmp->mask, p) > 0)
                _fast_BMapSet1(obj->mask, p);
        }

        if (isnew)
            APP->AddCustomObj(obj);
    }

    this->FreeData();
}


void ModuleInteractive :: Reset()
{
    static const char *title = "Reset segmentation?";
    static const char *msg = "Current segmentation will be lost.\nAre you sure you want to reset?";

    wxString wxtitle(title, wxConvUTF8);
    wxString wxmsg(msg, wxConvUTF8);

    wxMessageDialog dialog(APP->Window,
                           wxmsg, wxtitle,
                           wxYES_NO | wxICON_QUESTION,
                           wxDefaultPosition);

    if (dialog.ShowModal() == wxID_YES)
    {
        bia::Scene16::Fill(this->cost, USHRT_MAX);
        SetScene(this->pred,     NIL);
        APP->ResetData();
        this->markerID = 4;

        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);
    }
}


void ModuleInteractive :: Run()
{
    timer tic;

    APP->Busy((char *)"Please wait, working...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    StartTimer(&tic);

    this->Run_DIFT();

    printf("\nDIFT Time: ");
    StopTimer(&tic);

    this->MarkForRemovalIsolatedSeeds();

    this->Run_DIFT();

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);

    APP->StatusMessage((char *)"Done");
    APP->Unbusy();
}


void ModuleInteractive :: Run_DIFT()
{
    Set *S = NULL;
    Set *delSet = NULL, *aux;
    int id, cod;

    S = APP->CopySeeds();
    delSet = APP->CopyIdsMarkedForRemoval();
    aux = delSet;
    while (aux != NULL)
    {
        id = aux->elem;
        cod = GetCodeValue(id, APP->GetSeedLabelById(id));
        aux->elem = cod;
        aux = aux->next;
    }

    RunDIFT(this->Wf, cost,
            pred, APP->Data.label,
            &S, &delSet);

    APP->DelMarkedForRemoval();
    DestroySet(&S);
    DestroySet(&delSet);
}



void ModuleInteractive :: DeleteObj(int obj)
{
    int o, p, n, cod, id, newcod, newlb;

    n = (APP->Data.label)->n;
    for (o = obj ; o < this->nobjs - 1; o++)
    {
        obj_color[o] = obj_color[o + 1];
        obj_visibility[o] = obj_visibility[o + 1];
        strcpy(obj_name[o], obj_name[o + 1]);
    }
    this->nobjs--;
    if (obj_sel > nobjs - 1) this->obj_sel--;

    for (p = 0; p < n; p++)
    {
        cod = (APP->Data.label)->data[p];
        id = GetCodeID(cod);
        if (GetCodeLabel(cod) == obj + 1)
        {
            newcod = GetCodeValue(id, 0);
            (APP->Data.label)->data[p] = newcod;
            if (APP->IsSeed(p))
                APP->AddSeed(p, 0, id);
            APP->SetLabelColour(newcod, NIL);
        }
        else if (GetCodeLabel(cod) > obj + 1)
        {
            newlb  = GetCodeLabel(cod) - 1;
            newcod = GetCodeValue(id, newlb);
            (APP->Data.label)->data[p] = newcod;
            if (APP->IsSeed(p))
                APP->AddSeed(p, newlb, id);

            if (obj_visibility[newlb - 1])
                APP->SetLabelColour(newcod, obj_color[newlb - 1]);
            else
                APP->SetLabelColour(newcod, NIL);
        }
    }
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
}


void ModuleInteractive :: MarkForRemovalIsolatedSeeds()
{
    Scene *scn = APP->Data.orig;
    int i, p, q, l;
    Voxel u, v;
    AdjRel3 *A = Spheric(1.0);
    Set *S = APP->CopySeeds();
    Set *tmp = NULL;
    bool alldiff;

    tmp = S;
    while (tmp != NULL)
    {
        p = tmp->elem;
        l = APP->GetSeedLabel(p);

        u.x = VoxelX(scn, p);
        u.y = VoxelY(scn, p);
        u.z = VoxelZ(scn, p);
        alldiff = true;
        for (i = 1; i < A->n; i++)
        {
            v.x = u.x + A->dx[i];
            v.y = u.y + A->dy[i];
            v.z = u.z + A->dz[i];
            if (ValidVoxel(scn, v.x, v.y, v.z))
            {
                q = v.x + scn->tby[v.y] + scn->tbz[v.z];
                if (l == this->GetCodeLabel(APP->Data.label->data[q]))
                    alldiff = false;
            }
        }
        if (alldiff)
        {
            APP->MarkForRemoval(APP->GetSeedId(p));
        }
        tmp = tmp->next;
    }
    DestroySet(&S);
    DestroyAdjRel3(&A);
}


AdjRel *ModuleInteractive :: GetBrush()
{
    //InteractiveOpt *iopt = (InteractiveOpt *)optPanel;
    InteractiveDialog *iopt = (InteractiveDialog *)optDialog;
    return iopt->GetBrush();
}

wxCursor *ModuleInteractive :: GetBrushCursor(int zoom)
{
    InteractiveDialog *iopt = (InteractiveDialog *)optDialog;
    return iopt->GetBrushCursor(zoom);
}

void ModuleInteractive :: NextBrush()
{
    InteractiveDialog *iopt = (InteractiveDialog *)optDialog;
    iopt->NextBrush();
}

void ModuleInteractive :: PrevBrush()
{
    InteractiveDialog *iopt = (InteractiveDialog *)optDialog;
    iopt->PrevBrush();
}

int ModuleInteractive :: GetCodeID(int cod)
{
    return (cod >> 8); //(int)label/256;
}

int ModuleInteractive :: GetCodeLabel(int cod)
{
    return (cod & 0xff);  //(label % 256);
}

int ModuleInteractive :: GetCodeValue(int id, int lb)
{
    return ((id << 8) | lb); //(id*256 + (int)lb);
}

} //end Interactive namespace




