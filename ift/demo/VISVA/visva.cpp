
//#define BIADEBUG  1

#include "visva.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <wx/wx.h>

//Modules
#include "register_modules.h"

#include "xpm/segmentation.xpm"
#include "xpm/visualization.xpm"
#include "xpm/reg.xpm"
#include "xpm/curve.xpm"
#include "xpm/preproc.xpm"
#include "xpm/eye.xpm"
#include "xpm/noteye.xpm"
#include "xpm/trash.xpm"
#include "xpm/zoomin.xpm"
#include "xpm/zoomout.xpm"
#include "xpm/brightcontr.xpm"
#include "xpm/brain.xpm"
#include "xpm/new.xpm"
#include "xpm/open.xpm"
#include "xpm/save.xpm"
#include "xpm/measure.xpm"
#include "xpm/group.xpm"
#include "xpm/tools.xpm"

#include <string>
#include <iostream>
#include <vector>
using namespace std;



void app_selective_refresh(Voxel v);

#define NPREOBJ 7
static const char *PredefinedObjs[] = {"Left Lung",
                                       "Right Lung",
                                       "Heart",
                                       "Knee",
                                       "Cerebellum",
                                       "Left Hemisphere",
                                       "Right Hemisphere",
};
//"Lateral Ventricles"};
// "Cortical Dysplasias",
// "Corpus Callosum",
// "Cortex",
// "Caudate Nuclei",
// "Hipocampi",
// "Talamus",

static const char *ModalityNames[] = {"T1-weighted",
                                      "T2-weighted",
                                      "PD (proton density)",
                                      "BIN (binary mask)",
                                      "MLM (multi-label mask)",
                                      "Unknown"
};


ModuleRegistration *modManager;
BIAClass *APP;

//---------------------------------------------------------

enum
{
    ID_Quit = wxID_HIGHEST + 1,
    ID_About,
    ID_VolInfo,
#ifdef BIADEBUG
    ID_ParamEstimation,
    ID_LoadGrad,
    ID_LoadGradTool,
#endif
    ID_LoadArcWeight,
    ID_Preproc,
    ID_Segmentation,
    ID_Analysis,
    ID_Measurements,
    ID_GroupStudy,
    ID_3DVisu,
    ID_Registration,
    ID_NewProj,
    ID_OpenProj,
    ID_SaveProj,
    ID_SaveProjAs,
    ID_LoadVol,
    ID_LoadLabel,
    ID_SaveLabel,
    ID_SaveVolume,
    ID_SaveObject,
    ID_SaveObjectMask,
    ID_SaveObjectMap,
    ID_LoadMark,
    ID_SaveMark,
    ID_LoadVolTool,
    ID_ObjEye,
    ID_Zoomin,
    ID_Zoomout,
    ID_ChangeDrawMarker,
    ID_BriContr,
    ID_BriContrDi,
    ID_View0,
    ID_View1,
    ID_View2,
    ID_View3,
    ID_View4,
    ID_Last
};

//----------------------------------------------------------

// int ModalityToInt(ModalityType m) {
//   switch( m ) {
//   case T1_WEIGHTED:
//     return 0;
//   case T2_WEIGHTED:
//     return 1;
//   case PD_PROTONDENSITY:
//     return 2;
//   default:
//     return UNKNOWN_PROTOCOL;
//   }
// }
//----------------------------------------------------------

void CheckFileExtension(char *path, int filter)
{
    int s = strlen(path);
    char *out_path = NULL;
    if ((s < 4 || !iftEndsWith(path,".scn")) && filter == 1 )
        out_path = iftConcatStrings(2,path,".scn");
    else if ((s < 4 || !iftEndsWith(path,".hdr")) && filter == 2 )
        out_path = iftConcatStrings(2,path,".hdr");
    else if ((s < 8 || !iftEndsWith(path,".scn.bz2")) && filter == 3 )
        out_path = iftConcatStrings(2,path,".scn.bz2");
    else if ((s < 4 || !iftEndsWith(path,".nii")) && filter == 4 )
        out_path = iftConcatStrings(2,path,".nii");
    else if ((s < 7 || !iftEndsWith(path,".nii.gz")) && filter == 5 )
        out_path = iftConcatStrings(2,path,".nii.gz");

    if (out_path != NULL) {
        strcpy(path, out_path);
        iftFree(out_path);
    }
}

void CopyVoxelSizeFromScene(Scene *dst, Scene *src)
{
    dst->dx = src->dx;
    dst->dy = src->dy;
    dst->dz = src->dz;
}

Scene *ImageToScene(  iftImage *img) {
    Scene *scn = CreateScene(img->xsize, img->ysize, img->zsize);
    scn->dx = img->dx;
    scn->dy = img->dy;
    scn->dz = img->dz;
    scn->maxval = iftMaximumValue(img);
    scn->nii_hdr = NULL;

    #pragma omp parallel for
    for (int p = 0; p < img->n; p++)
        scn->data[p] = img->val[p];


    return scn;
}


iftImage *SceneToImage(  Scene *scn) {
    iftImage *img = iftCreateImage(scn->xsize, scn->ysize, scn->zsize);
    img->dx = scn->dx;
    img->dy = scn->dy;
    img->dz = scn->dz;

    #pragma omp parallel for
    for (int p = 0; p < scn->n; p++)
        img->val[p] = scn->data[p];

    return img;
}

BIAClass :: BIAClass()
{
    View2DModule *view2D;
    View3DModule *view3D;
    int n;

    wxInitAllImageHandlers();

    APP = this;
    this->disableAll = NULL;
    this->busycursor = NULL;
    this->busyinfo   = NULL;
    this->LoadPreferences();

    Window = new MyFrame(_T("VISVA - Volumetric Image Segmentation for Visualization and Analysis"),
                         wxPoint(50, 50), wxSize(800, 600));

    APP->measures = CreateMeasures();
    this->InitData();
    idManager  = new DynamicID(ID_Last);
    modManager = new ModuleRegistration();
    app_register_modules(modManager);

    n = modManager->GetNumberModules(Module::VIEW2D);
    if (n == 0) Error((char *)"BIAClass", (char *)"View2DModule not loaded");
    view2D = (View2DModule *)modManager->GetModule(Module::VIEW2D, 0);

    n = modManager->GetNumberModules(Module::VIEW3D);
    if (n == 0) Error((char *)"BIAClass", (char *)"View3DModule not loaded");
    view3D = (View3DModule *)modManager->GetModule(Module::VIEW3D, 0);

    this->Set2DViewModule(view2D);
    this->Set3DViewModule(view3D);

    Window->Show(TRUE);
    if ( (this->Preferences).IsMaximized )
    {
        Window->Maximize(true);
        Window->UniformView();
    }
}


BIAClass :: ~BIAClass()
{
    if (labelcolor != NULL) free(labelcolor);
    if (labelalpha != NULL) free(labelalpha);
    this->SavePreferences();
}





//***** Old IFT to New IFT functions: ****
Scene *BIAClass :: iftImageToScene(iftImage *img)
{
    int p = 0;
    Scene *scn = NULL;
    scn = CreateScene(img->xsize, img->ysize, img->zsize);

    scn->dx = img->dx;
    scn->dy = img->dy;
    scn->dz = img->dz;
    scn->n = img->n;

    for (p = 0; p < img->n; p++)
        scn->data[p] = img->val[p];

    for (p = 0; p < img->ysize; p++)
        scn->tby[p] = img->tby[p];

    for (p = 0; p < img->zsize; p++)
        scn->tbz[p] = img->tbz[p];

    scn->nii_hdr = NULL;

    return scn;
}
Scene *BIAClass :: LabelImageToScene(iftImageForest *fst)
{
    int p = 0;
    Scene *scn = NULL;
    int root, label, id, cod;
    scn = CreateScene(fst->label->xsize, fst->label->ysize, fst->label->zsize);

    scn->dx = fst->label->dx;
    scn->dy = fst->label->dy;
    scn->dz = fst->label->dz;
    scn->n = fst->label->n;

    APP->DelAllSeeds();
    for (p = 0; p < fst->label->n; p++)
    {
        // root = fst->root->val[p];
        // label = this->GetSeedLabel(root); /* Root is the seed that conquered p*/
        // id = this->GetSeedId(root); /* Marker id at the root */
        // cod = ((id << 8) | label); /* 'cod' is a code that contains the Marker ID (24 bits) + Label (8 bits) */
        // scn->data[p] = cod; /* This is required for the object color render be precise */
        // if (cod > max)
        //     max = cod;

        label = fst->label->val[p];
        id = fst->marker->val[p];

        if (fst->pred->val[p] == NIL)
            APP->AddSeed(p, label, id);

        cod = ((id << 8) | label);
        scn->data[p] = cod;
    }

    for (p = 0; p < fst->label->ysize; p++)
        scn->tby[p] = fst->label->tby[p];

    for (p = 0; p < fst->label->zsize; p++)
        scn->tbz[p] = fst->label->tbz[p];

    scn->nii_hdr = NULL;

    return scn;
}

Scene *BIAClass :: IGraphLabelImageToScene(iftIGraph *igraph)
{
    int p = 0;
    Scene *scn = NULL;
    int root, label, id, cod;
    scn = CreateScene(igraph->index->xsize, igraph->index->ysize, igraph->index->zsize);

    scn->dx = igraph->index->dx;
    scn->dy = igraph->index->dy;
    scn->dz = igraph->index->dz;
    scn->n = igraph->index->n;

    for (p = 0; p < igraph->index->n; p++) {
        if(igraph->index->val[p] != IFT_NIL) {
            root = igraph->root[p];
            label = igraph->label[p];
            id = this->GetSeedId(root); /* Marker id at the root */

            cod = ((id << 8) | label); /* 'cod' is a code that contains the Marker ID (24 bits) + Label (8 bits) */
            scn->data[p] = cod; /* This is required for the object color render be precise */
        }
    }

    for (p = 0; p < igraph->index->ysize; p++)
        scn->tby[p] = igraph->index->tby[p];

    for (p = 0; p < igraph->index->zsize; p++)
        scn->tbz[p] = igraph->index->tbz[p];

    scn->nii_hdr = NULL;

    return scn;
}

void BIAClass :: SetLabeledSetAsSeeds(iftLabeledSet *seeds)
{
    int p = 0;
    iftLabeledSet *aux = NULL;

    APP->DelAllSeeds();
    for (aux = seeds; aux != NULL; aux = aux->next)
    {
        // root = fst->root->val[p];
        // label = this->GetSeedLabel(root); /* Root is the seed that conquered p*/
        // id = this->GetSeedId(root); /* Marker id at the root */
        // cod = ((id << 8) | label); /* 'cod' is a code that contains the Marker ID (24 bits) + Label (8 bits) */
        // scn->data[p] = cod; /* This is required for the object color render be precise */
        // if (cod > max)
        //     max = cod;

        p = aux->elem;

        APP->AddSeed(p, aux->label, aux->marker);

        int cod = ((aux->marker << 8) | aux->label);
        APP->Data.label->data[p] = cod;
    }
}

iftImage *BIAClass :: SceneToiftImage(Scene *scn)
{
    iftImage *img = iftCreateImage(scn->xsize, scn->ysize, scn->zsize);
    int p = 0;

    if (!img)
        Error((char *)"BIAClass", (char *)"iftImage not created");

    img->n = scn->n;
    img->dx = scn->dx;
    img->dy = scn->dy;
    img->dz = scn->dz;

    for (p = 0; p < scn->n; p++)
        img->val[p] = scn->data[p];

    for (p = 0; p < img->ysize; p++)
        img->tby[p] = scn->tby[p];

    for (p = 0; p < img->zsize; p++)
        img->tbz[p] = scn->tbz[p];

    return img;
}


void BIAClass :: Busy(char *msg)
{
    wxString wxmsg(msg, wxConvUTF8);

    if (disableAll == NULL)
        disableAll = new wxWindowDisabler();
    if (busycursor == NULL)
        busycursor = new wxBusyCursor(wxHOURGLASS_CURSOR);
    if (busyinfo != NULL)
        delete busyinfo;
    if (msg[0] != '\0')
        busyinfo   = new wxBusyInfo(wxmsg, this->Window);
    wxTheApp->Yield();
}

void BIAClass :: Unbusy()
{
    if (disableAll != NULL)
        delete disableAll;
    if (busyinfo != NULL)
        delete busyinfo;
    if (busycursor != NULL)
        delete busycursor;
    disableAll = NULL;
    busyinfo   = NULL;
    busycursor = NULL;
    //wxTheApp->Yield();
}

void BIAClass :: StatusMessage(char *msg)
{
    wxString wxmsg(msg, wxConvUTF8);
    this->Window->SetStatusText(wxmsg);
}


int BIAClass :: GetColorDialog()
{
    wxColourDialog *colourDialog;
    wxColourData cdata;
    wxColour c;
    int r, g, b;

    colourDialog = new wxColourDialog(this->Window, NULL);
    if (colourDialog->ShowModal() == wxID_OK)
    {
        cdata = colourDialog->GetColourData();
        c = cdata.GetColour();
        r = c.Red();
        g = c.Green();
        b = c.Blue();
        return triplet(r, g, b);
    }
    return NIL;
}




void BIAClass :: ResetData()
{
    if (APP->Data.loaded)
    {
        SetScene(APP->Data.label, 0);
        this->DelAllSeeds();
        this->UnmarkAllForRemoval();
    }
}

void BIAClass :: DestroyData()
{
    modManager->ClearDependencies();
    if (APP->Data.loaded)
    {
        DestroyScene(&(APP->Data.orig));
        DestroyScene(&(APP->Data.label));
        DestroyScene(&(APP->Data.arcw));
        if (APP->Data.grad != NULL)
            DestroyScnGradient(&(APP->Data.grad));
        if (APP->Data.objmap != NULL)
            DestroyScene(&(APP->Data.objmap));
        if (APP->Data.marker != NULL)
            DestroyScene(&(APP->Data.marker));
        if ( APP->Data.msp != NULL )
            DestroyPlane(&(APP->Data.msp));
        this->DelAllObjs();
        this->DelAllSeeds();
        this->UnmarkAllForRemoval();
        APP->Data.loaded = 0;
        APP->Data.corrected = 0;
        APP->Data.oriented = 0;
        APP->Data.aligned = 0;
    }
}

void BIAClass :: SetDataVolume(Scene *scn)
{
    Voxel Cut;
    this->DestroyData();

    APP->Data.w = scn->xsize;
    APP->Data.h = scn->ysize;
    APP->Data.nframes = scn->zsize;

    if (APP->Data.orig != NULL) DestroyScene(&(APP->Data.orig));
    if (APP->Data.arcw != NULL) DestroyScene(&(APP->Data.arcw));
    if (APP->Data.label != NULL) DestroyScene(&(APP->Data.label));
    if (APP->Data.grad != NULL) DestroyScnGradient(&(APP->Data.grad));
    if (APP->Data.objmap != NULL) DestroyScene(&(APP->Data.objmap));
    APP->Data.orig  = scn;
    APP->Data.arcw  = CreateScene(APP->Data.w, APP->Data.h, APP->Data.nframes);
    APP->Data.label = CreateScene(APP->Data.w, APP->Data.h, APP->Data.nframes);
    APP->Data.grad   = NULL;
    APP->Data.objmap = NULL;
    SetVoxelSize(APP->Data.arcw,  scn->dx, scn->dy, scn->dz);
    SetVoxelSize(APP->Data.label, scn->dx, scn->dy, scn->dz);
    APP->ReallocSeedStructures();

    Cut.x = APP->Data.w / 2;
    Cut.y = APP->Data.h / 2;
    Cut.z = APP->Data.nframes / 2;
    APP->Set2DViewSlice(Cut);

    APP->Data.loaded = 1;

}


void BIAClass :: SetDataVolume_NoDestroy(Scene *scn)
{
    Voxel Cut;
    APP->Data.w = scn->xsize;
    APP->Data.h = scn->ysize;
    APP->Data.nframes = scn->zsize;

    if (APP->Data.orig != NULL) DestroyScene(&(APP->Data.orig));
    if (APP->Data.arcw != NULL) DestroyScene(&(APP->Data.arcw));
    if (APP->Data.label != NULL) DestroyScene(&(APP->Data.label));
    if (APP->Data.grad != NULL) DestroyScnGradient(&(APP->Data.grad));
    if (APP->Data.objmap != NULL) DestroyScene(&(APP->Data.objmap));
    APP->Data.grad = NULL;
    APP->Data.objmap = NULL;
    APP->Data.orig = scn;
    APP->Data.arcw = CreateScene(APP->Data.w, APP->Data.h, APP->Data.nframes);
    APP->Data.label = CreateScene(APP->Data.w, APP->Data.h, APP->Data.nframes);
    SetVoxelSize(APP->Data.arcw,  scn->dx, scn->dy, scn->dz);
    SetVoxelSize(APP->Data.label, scn->dx, scn->dy, scn->dz);
    APP->ReallocSeedStructures();

    Cut.x = APP->Data.w / 2;
    Cut.y = APP->Data.h / 2;
    Cut.z = APP->Data.nframes / 2;
    APP->Set2DViewSlice(Cut);

    APP->Data.loaded = 1;
}


int BIAClass :: ShowImageModalityDialog()
{
    int ret, id;
    char title[512];

    strcpy(title, "Select Modality");
    id = this->idManager->AllocID();
    ModalityPickerDialog dialog(this->Window,
                                id, title);

    ret = dialog.ShowModal();
    this->idManager->FreeID(id);
    if (ret == wxID_OK)
    {
        APP->Data.modality = dialog.GetModality();
        return 0;
    }
    else
    {
        APP->Data.modality = UNKNOWN_PROTOCOL;
        return 1;
    }
}



void BIAClass :: Set2DViewModule(View2DModule *mod)
{
    wxPanel *panel;

    this->mod2D = mod;

    panel = mod->GetViewPanel(Window, 'z');
    Window->SetViewPanel(panel, 0);

    panel = mod->GetViewPanel(Window, 'y');
    Window->SetViewPanel(panel, 1);

    panel = mod->GetViewPanel(Window, 'x');
    Window->SetViewPanel(panel, 2);

    mod->Start();
}


void BIAClass :: Set3DViewModule(View3DModule *mod)
{
    wxPanel *panel;

    this->mod3D = mod;
    panel = mod->GetViewPanel(Window);
    Window->SetViewPanel(panel, 3);
    mod->Start();
}


void BIAClass :: MakePalette()
{
    labelcolor[0] = NIL;
    labelcolor[1] = 0xffff00; //0xccbb99;
    labelcolor[2] = 0x00ff00;
    labelcolor[3] = 0x00ffff;
    labelcolor[4] = 0xff80ff;
    labelcolor[5] = 0xff8000;
    labelcolor[6] = 0x008000;
    labelcolor[7] = 0xee00ff;
    labelcolor[8] = 0xff0000;
    labelcolor[9] = 0x800000;
}


void BIAClass :: InitData()
{
    this->seedSet = NULL;
    this->seedIdLabel = NULL;
    this->removalIdSet = NULL;
    this->Data.msp = NULL;
    this->Data.loaded = 0;
    this->Data.oriented = 0;
    this->Data.aligned = 0;
    this->Data.corrected = 0;
    this->Data.modality = UNKNOWN_PROTOCOL;
    this->Lmax = 1024;
    this->labelcolor = AllocIntArray(this->Lmax);
    this->labelalpha = AllocIntArray(this->Lmax);
    this->segmobjs = CreateArrayList(32);
    SetArrayListCleanFunc(this->segmobjs, freeSegmObject);
    this->MakePalette();
    this->Data.labelname[ 0 ] = 0;
    this->Data.projectfilename[ 0 ] = 0;

    this->Data.orig = NULL;
    this->Data.arcw = NULL;
    this->Data.label = NULL;
    this->Data.grad = NULL;
    this->Data.objmap = NULL;
    this->Data.marker = NULL;
}


void BIAClass :: InitPreferences()
{
    (this->Preferences).IsMaximized        = true; //false;
    (this->Preferences).color_preproc      = NIL; //0xeeeeff;
    (this->Preferences).color_segmentation = NIL; //0xeeffee;
    (this->Preferences).color_view         = NIL; //0xffffdd;
    (this->Preferences).color_analysis     = NIL; //0xffeeee;
    (this->Preferences).dir_LoadVolume[0]  = '.';
    (this->Preferences).dir_LoadLabel[0]   = '.';
    (this->Preferences).dir_SaveVolume[0]  = '.';
    (this->Preferences).dir_SaveLabel[0]   = '.';
    (this->Preferences).dir_SaveObject[0]  = '.';
    (this->Preferences).dir_LoadMarkers[0] = '.';
    (this->Preferences).dir_SaveMarkers[0] = '.';
    (this->Preferences).ext_LoadVolume     = 0;
    (this->Preferences).ext_LoadLabel      = 0;
    (this->Preferences).ext_SaveVolume     = 0;
    (this->Preferences).ext_SaveLabel      = 0;
    (this->Preferences).ext_SaveObject     = 0;
    (this->Preferences).ext_SaveObjectMap  = 0;
    (this->Preferences).ext_LoadMarkers    = 0;
    (this->Preferences).ext_SaveMarkers    = 0;
}


void BIAClass :: SavePreferences()
{
    FILE *fp = NULL;
    char *value = NULL;
    char file[512];

    value = getenv("HOME");
    if (value == NULL)
    {
        Warning((char *)"Unable to save preferences", (char *)"SavePreferences");
        return;
    }
    sprintf(file, "%s/.bia_conf", value);
    fp = fopen(file, "w");
    if (fp == NULL)
    {
        Warning((char *)"Unable to save preferences", (char *)"SavePreferences");
        return;
    }
    fprintf(fp, "IsMaximized         %d\n", (this->Preferences).IsMaximized);
    fprintf(fp, "color_preproc       %d\n", (this->Preferences).color_preproc);
    fprintf(fp, "color_segmentation  %d\n", (this->Preferences).color_segmentation);
    fprintf(fp, "color_view          %d\n", (this->Preferences).color_view);
    fprintf(fp, "color_analysis      %d\n", (this->Preferences).color_analysis);
    fprintf(fp, "dir_LoadVolume      %s\n", (this->Preferences).dir_LoadVolume);
    fprintf(fp, "dir_LoadLabel       %s\n", (this->Preferences).dir_LoadLabel);
    fprintf(fp, "dir_SaveVolume      %s\n", (this->Preferences).dir_SaveVolume);
    fprintf(fp, "dir_SaveLabel       %s\n", (this->Preferences).dir_SaveLabel);
    fprintf(fp, "dir_SaveObject      %s\n", (this->Preferences).dir_SaveObject);
    fprintf(fp, "dir_LoadMarkers     %s\n", (this->Preferences).dir_LoadMarkers);
    fprintf(fp, "dir_SaveMarkers     %s\n", (this->Preferences).dir_SaveMarkers);
    fprintf(fp, "ext_LoadVolume      %d\n", (this->Preferences).ext_LoadVolume);
    fprintf(fp, "ext_LoadLabel       %d\n", (this->Preferences).ext_LoadLabel);
    fprintf(fp, "ext_SaveVolume      %d\n", (this->Preferences).ext_SaveVolume);
    fprintf(fp, "ext_SaveLabel       %d\n", (this->Preferences).ext_SaveLabel);
    fprintf(fp, "ext_SaveObject      %d\n", (this->Preferences).ext_SaveObject);
    fprintf(fp, "ext_SaveObjectMap   %d\n", (this->Preferences).ext_SaveObjectMap);
    fprintf(fp, "ext_LoadMarkers     %d\n", (this->Preferences).ext_LoadMarkers);
    fprintf(fp, "ext_SaveMarkers     %d\n", (this->Preferences).ext_SaveMarkers);

    fclose(fp);
}


void BIAClass :: LoadPreferences()
{
    FILE *fp = NULL;
    char option[512], value[512];
    char *envvalue = NULL;
    char file[512];

    this->InitPreferences();

    envvalue = getenv("HOME");
    if (envvalue == NULL)
        return;

    sprintf(file, "%s/.bia_conf", envvalue);
    fp = fopen(file, "r");
    if (fp == NULL)
        return;

    while (fscanf(fp, "%s  %s\n", option, value) != EOF)
    {
        if (strcmp(option, "IsMaximized") == 0)
            (this->Preferences).IsMaximized = atoi(value);

        else if (strcmp(option, "color_preproc") == 0)
            (this->Preferences).color_preproc = atoi(value);
        else if (strcmp(option, "color_segmentation") == 0)
            (this->Preferences).color_segmentation = atoi(value);
        else if (strcmp(option, "color_view") == 0)
            (this->Preferences).color_view = atoi(value);
        else if (strcmp(option, "color_analysis") == 0)
            (this->Preferences).color_analysis = atoi(value);
        else if (strcmp(option, "dir_LoadVolume") == 0)
            strcpy((this->Preferences).dir_LoadVolume, value);
        else if (strcmp(option, "dir_LoadLabel") == 0)
            strcpy((this->Preferences).dir_LoadLabel, value);
        else if (strcmp(option, "dir_SaveVolume") == 0)
            strcpy((this->Preferences).dir_SaveVolume, value);
        else if (strcmp(option, "dir_SaveLabel") == 0)
            strcpy((this->Preferences).dir_SaveLabel, value);
        else if (strcmp(option, "dir_SaveObject") == 0)
            strcpy((this->Preferences).dir_SaveObject, value);
        else if (strcmp(option, "dir_LoadMarkers") == 0)
            strcpy((this->Preferences).dir_LoadMarkers, value);
        else if (strcmp(option, "dir_SaveMarkers") == 0)
            strcpy((this->Preferences).dir_SaveMarkers, value);

        else if (strcmp(option, "ext_LoadVolume") == 0)
            (this->Preferences).ext_LoadVolume = atoi(value);
        else if (strcmp(option, "ext_LoadLabel") == 0)
            (this->Preferences).ext_LoadLabel = atoi(value);
        else if (strcmp(option, "ext_SaveVolume") == 0)
            (this->Preferences).ext_SaveVolume = atoi(value);
        else if (strcmp(option, "ext_SaveLabel") == 0)
            (this->Preferences).ext_SaveLabel = atoi(value);
        else if (strcmp(option, "ext_SaveObject") == 0)
            (this->Preferences).ext_SaveObject = atoi(value);
        else if (strcmp(option, "ext_SaveObjectMap") == 0)
            (this->Preferences).ext_SaveObjectMap = atoi(value);
        else if (strcmp(option, "ext_LoadMarkers") == 0)
            (this->Preferences).ext_LoadMarkers = atoi(value);
        else if (strcmp(option, "ext_SaveMarkers") == 0)
            (this->Preferences).ext_SaveMarkers = atoi(value);
    }
    fclose(fp);
}


void BIAClass :: AppendOptPanel(wxPanel *panel,
                                Module::ModuleType type)
{
    Window->AppendOptPanel(panel, type);
}

void BIAClass :: PrependOptPanel(wxPanel *panel,
                                 Module::ModuleType type)
{
    Window->PrependOptPanel(panel, type);
}

bool BIAClass :: DetachOptPanel(wxPanel *panel)
{
    return Window->DetachOptPanel(panel);
}


void BIAClass :: EnableObjWindow(bool enable)
{
    Window->EnableObjWindow(enable);
}


void BIAClass :: Set2DViewCursor(  wxCursor *cursor, char axis)
{
    View2DModule *view2D;
    int n;

    n = modManager->GetNumberModules(Module::VIEW2D);
    if (n == 0) Error((char *)"BIAClass", (char *)"View2DModule not loaded");
    view2D = (View2DModule *)modManager->GetModule(Module::VIEW2D, 0);
    view2D->SetCursor(cursor, axis);
}


void BIAClass :: Set2DViewInteractionHandler(InteractionHandler *handler,
                                             char axis)
{
    View2DModule *view2D;
    int n;

    n = modManager->GetNumberModules(Module::VIEW2D);
    if (n == 0) Error((char *)"BIAClass", (char *)"View2DModule not loaded");
    view2D = (View2DModule *)modManager->GetModule(Module::VIEW2D, 0);
    view2D->SetInteractionHandler(handler, axis);
}


void BIAClass :: SetDefaultInteractionHandler()
{
    APP->Set2DViewInteractionHandler(NULL, 'x');
    APP->Set2DViewInteractionHandler(NULL, 'y');
    APP->Set2DViewInteractionHandler(NULL, 'z');
}


void BIAClass :: Set2DViewSlice(Voxel Cut)
{
    View2DModule *view2D;
    int n;

    n = modManager->GetNumberModules(Module::VIEW2D);
    if (n == 0) Error((char *)"BIAClass", (char *)"View2DModule not loaded");
    view2D = (View2DModule *)modManager->GetModule(Module::VIEW2D, 0);
    view2D->SetSliceVoxel(Cut);
    APP->Refresh3DCanvas(false, 1.0);
}


Voxel BIAClass :: Get2DViewSlice()
{
    View2DModule *view2D;
    int n;

    n = modManager->GetNumberModules(Module::VIEW2D);
    if (n == 0) Error((char *)"BIAClass", (char *)"View2DModule not loaded");
    view2D = (View2DModule *)modManager->GetModule(Module::VIEW2D, 0);
    return view2D->GetSliceVoxel();
}


void BIAClass :: Refresh2DCanvas()
{
    this->mod2D->Refresh();
}


void BIAClass :: Refresh2DCanvas(char axis)
{
    this->mod2D->Refresh(axis);
}


void BIAClass :: Refresh3DCanvas(bool dataChanged,
                                 float quality)
{
    if (this->mod3D != NULL)
        this->mod3D->Refresh(dataChanged, quality);
}


void BIAClass :: SetRefresh2DHandler(RefreshHandler *handler)
{
    this->mod2D->SetRefreshHandler(handler);
}


void BIAClass :: Clear2DCanvas()
{
    this->mod2D->Clear();
}

void BIAClass :: ChangeDrawMarker()
{
    this->mod2D->ChangeDrawMarker();
    this->mod2D->Refresh();
}


CImage *BIAClass :: Copy3DCanvasAsCImage()
{
    return this->mod3D->CopyAsCImage();
}


CImage *BIAClass :: Copy2DCanvasAsCImage(char axis)
{
    return this->mod2D->CopyAsCImage(axis);
}


void BIAClass :: DrawSegmObjects()
{
    SegmObject *segm;
    Scene *label = APP->Data.label;
    int i, no, n, p;

    n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    SetScene(label, 0);
    APP->SetLabelColour(0, NIL);
    no = this->segmobjs->n;
    for (i = 0; i < no; i++)
    {
        segm = (SegmObject *)GetArrayListElement(this->segmobjs, i);
        if (segm->visibility)
        {
            APP->SetLabelColour(i + 1, segm->color);
            for (p = 0; p < n; p++)
            {
                if (_fast_BMapGet(segm->mask, p))
                    label->data[p] = i + 1;
            }
        }
    }
    this->Refresh2DCanvas();
    this->Refresh3DCanvas(true, 1.0);
}

void BIAClass :: SetBriContr(int B, int C)
{
    APP->Window->SetBriContr(B, C);
}

void BIAClass :: GetBriContr(int *B, int *C)
{
    APP->Window->GetBriContr(B, C);
}

void BIAClass :: SetZoomLevel(float zoom)
{
    this->mod2D->SetZoomLevel(zoom);
    this->mod3D->SetZoomLevel(zoom);
}

float BIAClass :: GetZoomLevel()
{
    return this->mod2D->GetZoomLevel();
}


void BIAClass :: SetObjVisibility(char *obj_name, bool enable)
{
    SegmObject *obj = NULL;
    obj = SearchObjByName(obj_name);
    if (obj != NULL)
        obj->visibility = enable;
}


SegmObject *BIAClass :: SearchObjByName(char *obj_name)
{
    int i, n = this->segmobjs->n;
    SegmObject *obj = NULL;

    for (i = 0; i < n; i++)
    {
        obj = (SegmObject *)GetArrayListElement(this->segmobjs, i);
        if (strcmp(obj->name, obj_name) == 0)
            return (obj);
    }
    return (NULL);
}

SegmObject *BIAClass :: GetObjByIndex(int index)
{
    SegmObject *obj = NULL;
    obj = (SegmObject *)GetArrayListElement(this->segmobjs, index);
    return obj;
}

SegmObject *BIAClass :: GetObjByDialog()
{
    SegmObject *obj = NULL;
    int ret, id, i, n = this->segmobjs->n;
    char **choices;

    choices = (char **)malloc(n * sizeof(char *));
    for (i = 0; i < n; i++)
    {
        obj = this->GetObjByIndex(i);
        choices[i] = obj->name;
    }
    id = this->idManager->AllocID();
    ObjectPickerDialog dialog(this->Window, id,
                              (char *)"Select Object",
                              (const char **)choices, n,
                              true, false, true);
    ret = dialog.ShowModal();

    if (ret == wxID_OK)
    {
        i = (dialog.texName)->GetCurrentSelection();

        if (i == wxNOT_FOUND) obj = NULL;
        else               obj = GetObjByIndex(i);
    }
    else
        obj = NULL;
    this->idManager->FreeID(id);
    free(choices);

    return obj;
}

ArrayList *BIAClass::GetSegmObjsArrayList()
{
    return this->segmobjs;
}


int BIAClass :: AddObj(char *name,
                       int color)
{
    SegmObject *obj = NULL;
    Scene *label = APP->Data.label;
    int p, n;

    n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    obj = CreateSegmObject(name, color);
    obj->visibility = true;
    obj->mask = BMapNew(n);
    BMapFill(obj->mask, 0);
    for (p = 0; p < n; p++)
    {
        if (label->data[p] > 0)
            _fast_BMapSet1(obj->mask, p);
    }
    obj->seed = APP->CopyMarkerList();

    return AddCustomObj(obj);
}


int BIAClass :: AddCustomObj(SegmObject *obj)
{
    SegmObject *aux = NULL;
    int i;
    for (i = 0; i < APP->GetNumberOfObjs(); i++)
    {
        aux = APP->GetObjByIndex(i);
        if (strcmp(aux->name, obj->name) == 0)
            break;
    }
    if (i < APP->GetNumberOfObjs())
        DelObjByIndex(i);

    obj->visibility = true;
    AddArrayListElement(this->segmobjs,
                        (void *)obj);
    return (this->segmobjs->n - 1);
}

void BIAClass :: DelObjByIndex(int index)
{
    DelArrayListElement(this->segmobjs, index);
}

void BIAClass :: DelAllObjs()
{
    DestroyArrayList(&segmobjs);
    this->segmobjs = CreateArrayList(32);
    SetArrayListCleanFunc(this->segmobjs,
                          freeSegmObject);
}

int  BIAClass :: GetNumberOfObjs()
{
    return (this->segmobjs->n);
}

int BIAClass :: ShowNewObjDialog(int *color,
                                 char *name)
{
    int ret, id, i;
    SegmObject *obj = NULL;
    char **choices;
    int nobjs = APP->GetNumberOfObjs();

    id = this->idManager->AllocID();
    if (nobjs == 0)
    {
        choices = (char **)malloc(NPREOBJ * sizeof(char *));
        for (i = 0; i < NPREOBJ; i++)
            choices[i] = const_cast<char *>(PredefinedObjs[i]);
        nobjs = NPREOBJ;
    }
    else
    {
        choices = (char **)malloc((nobjs + 1) * sizeof(char *));
        for (i = 0; i < nobjs; i++)
        {
            obj = this->GetObjByIndex(i);
            choices[i] = obj->name;
        }
        choices[i] = "All";
        nobjs++;
    }
    ObjectPickerDialog dialog(this->Window, id, (char *)"Select Name&Color", (const char **) choices, nobjs, false, true, true);
    free(choices);
    ret = dialog.ShowModal();
    this->idManager->FreeID(id);
    if (ret == wxID_OK)
    {
        dialog.GetName(name);
        *color = dialog.GetColor();
        return 0;
    }
    else
    {
        name[0] = '\0';
        *color = NIL;
        return 1;
    }

}

int BIAClass :: ShowObjColorDialog(int *color,
                                   char *name)
{
    int ret, id;
    char title[512];

    sprintf(title, "Choose %s color:", name);
    id = this->idManager->AllocID();
    ObjectPickerDialog dialog(this->Window,
                              id, title,
                              PredefinedObjs, NPREOBJ,
                              false, true, false);
    ret = dialog.ShowModal();
    this->idManager->FreeID(id);
    if (ret == wxID_OK)
    {
        *color = dialog.GetColor();
        return 0;
    }
    else
    {
        *color = NIL;
        return 1;
    }
}

bool BIAClass::CheckDependency(char *name,
                               Module::ModuleType type)
{

    if (type == Module::SEGMENTATION)
    {
        if (this->SearchObjByName(name) != NULL)
            return true;
        else
            return false;
    }
    else
        return modManager->CheckDependency(name, type);
}


bool BIAClass::SolveDependency(char *name,
                               Module::ModuleType type)
{

    if ( this->CheckDependency(name, type) == true )
        return true;

    return modManager->SolveDependency(name, type);
}


void BIAClass::SetDependencyStatus(char *name,
                                   Module::ModuleType type,
                                   bool solved)
{
    modManager->SetDependencyStatus(name, type, solved);
}


bool  BIAClass::IsSeed(int p)
{
    return (seedIdLabel->data[p] != 0);
}

int   BIAClass::GetSeedLabel(int p)
{
    int mk;
    mk = seedIdLabel->data[p];
    return _GetMarkerLabel(mk);
}

int   BIAClass::GetSeedLabelById(int id)
{
    Set *tmp = NULL;
    int q, mk;

    tmp = seedSet;
    while (tmp != NULL)
    {
        q = tmp->elem;
        mk = seedIdLabel->data[q];

        if (_GetMarkerID(mk) == id)
        {
            return _GetMarkerLabel(mk);
        }
        tmp = tmp->next;
    }
    return 0;
}

int   BIAClass::GetSeedId(int p)
{
    int mk;
    mk = seedIdLabel->data[p];
    return _GetMarkerID(mk);
}

void  BIAClass::AddSeed(int p, int label, int id)
{
    int mk;
    if (id == 0) return;
    if (seedIdLabel->data[p] == 0)
        InsertSet(&seedSet, p);

    mk = _GetMarkerValue(id, label);

    seedIdLabel->data[p] = mk;
}

void  BIAClass::AddSeedsInSet(Set *S, int label, int id)
{
    Set *tmp = NULL;
    int p;

    tmp = S;
    while (tmp != NULL)
    {
        p = tmp->elem;
        this->AddSeed(p, label, id);
        tmp = tmp->next;
    }
}

void BIAClass::AddSeedsInBrush(int p,
                               int label, int id,
                               AdjRel *A, char axis)
{
    Voxel u, v;
    int q, i;

    v.x = VoxelX(APP->Data.orig, p);
    v.y = VoxelY(APP->Data.orig, p);
    v.z = VoxelZ(APP->Data.orig, p);
    for (i = 0; i < A->n; i++)
    {
        if (axis == 'z')
        {
            u.x = v.x + A->dx[i];
            u.y = v.y + A->dy[i];
            u.z = v.z;
        }
        else if (axis == 'x')
        {
            u.x = v.x;
            u.y = v.y + A->dy[i];
            u.z = v.z + A->dx[i];
        }
        else if (axis == 'y')
        {
            u.x = v.x + A->dx[i];
            u.y = v.y;
            u.z = v.z + A->dy[i];
        }

        if (ValidVoxel(APP->Data.orig, u.x, u.y, u.z))
        {
            q = VoxelAddress(APP->Data.orig, u.x, u.y, u.z);
            if (APP->Data.marker == NULL)
            {
                APP->Data.marker = CreateScene(APP->Data.orig->xsize, APP->Data.orig->ysize, APP->Data.orig->zsize);
                for (int p = 0 ; p < APP->Data.marker->n; p++)
                    APP->Data.marker->data[p] = -1;
            }
            APP->Data.marker->data[q] = id;
            APP->AddSeed(q, label, id);
        }
    }
}

void BIAClass::AddSeedsInBrushTrace(int p, int q, int label, int id, AdjRel *A, char axis)
{
    Scene *scn = APP->Data.orig;
    Voxel v0, v1, v2;
    int t;

    v0.x = VoxelX(scn, p);
    v0.y = VoxelY(scn, p);
    v0.z = VoxelZ(scn, p);
    v1.x = VoxelX(scn, q);
    v1.y = VoxelY(scn, q);
    v1.z = VoxelZ(scn, q);
    v2.x = (v0.x + v1.x) / 2;
    v2.y = (v0.y + v1.y) / 2;
    v2.z = (v0.z + v1.z) / 2;
    t = VoxelAddress(scn, v2.x, v2.y, v2.z);

    if ((v2.x == v1.x && v2.y == v1.y && v2.z == v1.z) ||
        (v2.x == v0.x && v2.y == v0.y && v2.z == v0.z))
    {
        AddSeedsInBrush(p, label, id, A, axis);
        AddSeedsInBrush(q, label, id, A, axis);
        return;
    }
    AddSeedsInBrushTrace(p, t, label, id, A, axis);
    AddSeedsInBrushTrace(t, q, label, id, A, axis);
}


void BIAClass::AddSeedsInMarkerList(MarkerList *ML)
{
    int i, n, p;

    n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    for (i = 0; i < ML->n; i++)
    {
        p = ML->data[i].p;
        if (p < 0 || p >= n) continue;

        this->AddSeed(ML->data[i].p,
                      ML->data[i].label,
                      ML->data[i].id);
    }
}


void  BIAClass::DelSeed(int p)
{
    if (seedIdLabel->data[p] > 0)
    {
        seedIdLabel->data[p] = 0;
        RemoveSetElem(&seedSet, p);
    }
}

void  BIAClass::DelSeedsById(int id)
{
    Set **S = NULL;
    Set *tmp = NULL;
    int q, mk;

    S = &(seedSet);
    while (*S != NULL)
    {
        q = (*S)->elem;
        mk = seedIdLabel->data[q];

        if (_GetMarkerID(mk) == id)
        {
            seedIdLabel->data[q] = 0;
            tmp = *S;
            *S = (*S)->next;
            free(tmp);
        }
        else
            S = &((*S)->next);
    }
}

void  BIAClass::DelSeedsInSet(Set *S)
{
    Set *tmp = NULL;
    int p;

    tmp = S;
    while (tmp != NULL)
    {
        p = tmp->elem;
        this->DelSeed(p);
        tmp = tmp->next;
    }
}

void  BIAClass::DelSeedsNotInSet(Set *S)
{
    BMap *bmap = NULL;
    Set **SP = NULL;
    Set *tmp = NULL;
    int max, min, q, n;
    bool del;

    if (S != NULL)
    {
        min = MinimumSetValue(S);
        max = MaximumSetValue(S);
        n = max - min + 1;
        bmap = BMapNew(n);
        tmp = S;
        while (tmp != NULL)
        {
            q = tmp->elem;
            _fast_BMapSet1(bmap, (q - min));
            tmp = tmp->next;
        }
    }
    else
    {
        min = INT_MAX;
        max = INT_MIN;
    }

    SP = &(seedSet);
    while (*SP != NULL)
    {
        q = (*SP)->elem;

        del = false;
        if (q > max || q < min)
            del = true;
        else if ( !_fast_BMapGet(bmap, (q - min)) )
            del = true;

        if (del)
        {
            seedIdLabel->data[q] = 0;
            tmp = *SP;
            *SP = (*SP)->next;
            free(tmp);
        }
        else
            SP = &((*SP)->next);
    }
    if (bmap != NULL)
        BMapDestroy(bmap);
}


void  BIAClass::DelAllSeeds()
{
    if (APP->Data.loaded)
    {
        SetScene(this->seedIdLabel, 0);
        DestroySet(&(this->seedSet));
        this->seedSet = NULL;
    }
}


Set  *BIAClass::CopyInternalSeeds()
{
    Set *tmp = NULL;
    Set *S = NULL;
    int p;

    tmp = seedSet;
    while (tmp != NULL)
    {
        p = tmp->elem;
        if (GetSeedLabel(p) > 0)
            InsertSet(&S, p);
        tmp = tmp->next;
    }
    return S;
}

Set  *BIAClass::CopyExternalSeeds()
{
    Set *tmp = NULL;
    Set *S = NULL;
    int p;

    tmp = seedSet;
    while (tmp != NULL)
    {
        p = tmp->elem;
        if (GetSeedLabel(p) == 0)
            InsertSet(&S, p);
        tmp = tmp->next;
    }
    return S;
}


Set  *BIAClass::CopySeeds()
{
    return CloneSet(seedSet);
}


MarkerList *BIAClass::CopyMarkerList()
{
    MarkerList *ML;
    int p, n, nseeds, i;

    nseeds = 0;
    n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    for (p = 0; p < n; p++)
    {
        if (APP->IsSeed(p))
            nseeds++;
    }

    ML = CreateMarkerList(nseeds);

    i = 0;
    for (p = 0; p < n; p++)
    {
        if (APP->IsSeed(p))
        {
            ML->data[i].p = p;
            ML->data[i].id = APP->GetSeedId(p);
            ML->data[i].label = APP->GetSeedLabel(p);
            i++;
        }
    }

    return ML;
}


void BIAClass :: ReallocSeedStructures()
{
    DestroySet(&seedSet);
    seedSet = NULL;

    if (seedIdLabel != NULL)
        DestroyScene(&seedIdLabel);

    seedIdLabel = CreateScene(APP->Data.w,
                              APP->Data.h,
                              APP->Data.nframes);
}


void BIAClass :: MarkForRemoval(int id)
{
    bool flag;
    flag = IsInSet(removalIdSet, id);
    if (flag)
        return;
    else
        InsertSet(&removalIdSet, id);
}

void BIAClass :: UnmarkForRemoval(int id)
{
    bool flag;
    flag = IsInSet(removalIdSet, id);
    if (!flag)
        return;
    else
        RemoveSetElem(&removalIdSet, id);
}

void BIAClass :: UnmarkAllForRemoval()
{
    if (removalIdSet != NULL)
        DestroySet(&removalIdSet);
}

bool BIAClass :: IsMarkedForRemoval(int p)
{
    int id = APP->GetSeedId(p);
    return IsInSet(removalIdSet, id);
}


bool BIAClass :: IsIdMarkedForRemoval(int id)
{
    return IsInSet(removalIdSet, id);
}


void BIAClass :: DelMarkedForRemoval()
{
    int id;

    while (removalIdSet != NULL)
    {
        id = RemoveSet(&removalIdSet);
        APP->DelSeedsById(id);
    }
}

Set  *BIAClass :: CopyIdsMarkedForRemoval()
{
    return CloneSet(removalIdSet);
}


int BIAClass :: GetLabelColour(int label)
{
    if (label >= Lmax)
        return NIL;
    return labelcolor[label];
}

int BIAClass :: GetLabelAlpha(int label)
{
    if (label >= Lmax)
        return NIL;
    return labelalpha[label];
}


void BIAClass :: SetLabelColour(int label, int color)
{
    if (label >= Lmax)
    {
        Lmax = MAX(MAX(2 * Lmax, label + 1), 1024);
        labelcolor = (int *)realloc(labelcolor, sizeof(int) * Lmax);
        labelalpha = (int *)realloc(labelalpha, sizeof(int) * Lmax);
    }
    labelcolor[label] = color;
    labelalpha[label] = 76; //30%
}


void BIAClass :: SetLabelColour(int label,
                                int color,
                                int alpha)
{
    if (label >= Lmax)
    {
        Lmax = MAX(MAX(2 * Lmax, label + 1), 1024);
        labelcolor = (int *)realloc(labelcolor,
                                    sizeof(int) * Lmax);
        labelalpha = (int *)realloc(labelalpha,
                                    sizeof(int) * Lmax);
    }
    labelcolor[label] = color;
    labelalpha[label] = alpha;
}


void BIAClass::DrawBrush(int p, int label,
                         AdjRel *A, char axis)
{
    DrawBrushCustom(p, label, A, axis,
                    APP->Data.label);
}


void BIAClass::DrawBrushTrace(int p, int q,
                              int label,
                              AdjRel *A, char axis)
{
    DrawBrushTraceCustom(p, q, label, A, axis,
                         APP->Data.label);
}


void BIAClass::DrawBrushCustom(int p, int label,
                               AdjRel *A, char axis,
                               Scene *scn)
{
    Voxel u, v;
    int q, i;

    if (scn->n != (APP->Data.orig)->n) return;
    v.x = VoxelX(scn, p);
    v.y = VoxelY(scn, p);
    v.z = VoxelZ(scn, p);
    for (i = 0; i < A->n; i++)
    {
        if (axis == 'z')
        {
            u.x = v.x + A->dx[i];
            u.y = v.y + A->dy[i];
            u.z = v.z;
        }
        else if (axis == 'x')
        {
            u.x = v.x;
            u.y = v.y + A->dy[i];
            u.z = v.z + A->dx[i];
        }
        else if (axis == 'y')
        {
            u.x = v.x + A->dx[i];
            u.y = v.y;
            u.z = v.z + A->dy[i];
        }

        if (ValidVoxel(scn, u.x, u.y, u.z))
        {
            q = VoxelAddress(scn, u.x, u.y, u.z);
            scn->data[q] = label;
        }
    }
}

void BIAClass::DrawBrushCustom(int p, int label,
                               AdjRel *A, char axis,
                               bia::Scene16::Scene16 *scn)
{
    bia::Voxel u, v;
    int q, i;

    if (scn->n != (APP->Data.orig)->n) return;
    v.c.x = bia::Scene16::GetAddressX(scn, p);
    v.c.y = bia::Scene16::GetAddressY(scn, p);
    v.c.z = bia::Scene16::GetAddressZ(scn, p);
    for (i = 0; i < A->n; i++)
    {
        if (axis == 'z')
        {
            u.c.x = v.c.x + A->dx[i];
            u.c.y = v.c.y + A->dy[i];
            u.c.z = v.c.z;
        }
        else if (axis == 'x')
        {
            u.c.x = v.c.x;
            u.c.y = v.c.y + A->dy[i];
            u.c.z = v.c.z + A->dx[i];
        }
        else if (axis == 'y')
        {
            u.c.x = v.c.x + A->dx[i];
            u.c.y = v.c.y;
            u.c.z = v.c.z + A->dy[i];
        }

        if (bia::Scene16::IsValidVoxel(scn, u))
        {
            q = bia::Scene16::GetVoxelAddress(scn, u);
            scn->data[q] = label;
        }
    }
}

void BIAClass::DrawBrushTraceCustom(int p, int q,
                                    int label,
                                    AdjRel *A, char axis,
                                    Scene *scn)
{
    Voxel v0, v1, v2;
    int t;

    if (scn->n != (APP->Data.orig)->n) return;
    v0.x = VoxelX(scn, p);
    v0.y = VoxelY(scn, p);
    v0.z = VoxelZ(scn, p);
    v1.x = VoxelX(scn, q);
    v1.y = VoxelY(scn, q);
    v1.z = VoxelZ(scn, q);
    v2.x = (v0.x + v1.x) / 2;
    v2.y = (v0.y + v1.y) / 2;
    v2.z = (v0.z + v1.z) / 2;
    t = VoxelAddress(scn, v2.x, v2.y, v2.z);

    if ((v2.x == v1.x && v2.y == v1.y && v2.z == v1.z) ||
        (v2.x == v0.x && v2.y == v0.y && v2.z == v0.z))
    {
        DrawBrushCustom(p, label, A, axis, scn);
        DrawBrushCustom(q, label, A, axis, scn);
        return;
    }
    DrawBrushTraceCustom(p, t, label, A, axis, scn);
    DrawBrushTraceCustom(t, q, label, A, axis, scn);
}

void BIAClass::DrawBrushTraceCustom(int p, int q,
                                    int label,
                                    AdjRel *A, char axis,
                                    bia::Scene16::Scene16 *scn)
{
    bia::Voxel v0, v1, v2;
    int t;

    if (scn->n != (APP->Data.orig)->n) return;
    v0.c.x = bia::Scene16::GetAddressX(scn, p);
    v0.c.y = bia::Scene16::GetAddressY(scn, p);
    v0.c.z = bia::Scene16::GetAddressZ(scn, p);
    v1.c.x = bia::Scene16::GetAddressX(scn, q);
    v1.c.y = bia::Scene16::GetAddressY(scn, q);
    v1.c.z = bia::Scene16::GetAddressZ(scn, q);
    v2.c.x = (v0.c.x + v1.c.x) / 2;
    v2.c.y = (v0.c.y + v1.c.y) / 2;
    v2.c.z = (v0.c.z + v1.c.z) / 2;
    t = bia::Scene16::GetVoxelAddress(scn, v2);

    if ((v2.c.x == v1.c.x && v2.c.y == v1.c.y && v2.c.z == v1.c.z) ||
        (v2.c.x == v0.c.x && v2.c.y == v0.c.y && v2.c.z == v0.c.z))
    {
        DrawBrushCustom(p, label, A, axis, scn);
        DrawBrushCustom(q, label, A, axis, scn);
        return;
    }
    DrawBrushTraceCustom(p, t, label, A, axis, scn);
    DrawBrushTraceCustom(t, q, label, A, axis, scn);
}



Scene *BIAClass :: CopyVisibleLabelUnion()
{
    int p, n;
    Scene *bin, *label = APP->Data.label;

    if (!APP->Data.loaded) return NULL;

    bin = CreateScene(APP->Data.w,
                      APP->Data.h,
                      APP->Data.nframes);
    n = APP->Data.w * APP->Data.h * APP->Data.nframes;

    for (p = 0; p < n; p++)
    {
        if (APP->GetLabelColour(label->data[p]) != NIL)
            bin->data[p] = 1;
        else
            bin->data[p] = 0;
    }
    return bin;
}


Scene *BIAClass :: CopyDistinctVisibleLabels()
{
    int p, q, n, color, lb, id = 1;
    Scene *label = CopyScene(APP->Data.label);
    Scene *flabel;

    if (!APP->Data.loaded) return NULL;

    flabel = CreateScene(APP->Data.w,
                         APP->Data.h,
                         APP->Data.nframes);
    n = APP->Data.w * APP->Data.h * APP->Data.nframes;

    for (p = 0; p < n; p++)
    {
        lb = label->data[p];
        if (lb == NIL) continue;

        color = APP->GetLabelColour(lb);
        if (color != NIL)
        {

            for (q = 0; q < n; q++)
            {
                if (label->data[q] == NIL) continue;
                if (APP->GetLabelColour(label->data[q]) == color)
                {
                    flabel->data[q] = id;
                    label->data[q] = NIL;
                }
            }
            id++;
        }
        else
            flabel->data[p] = 0;
    }
    DestroyScene(&label);

    flabel->dx = APP->Data.orig->dx;
    flabel->dy = APP->Data.orig->dy;
    flabel->dz = APP->Data.orig->dz;

    return flabel;
}


int BIAClass :: _GetMarkerID(int mk)
{
    return (mk >> 8); //(int)label/256;
}

int BIAClass :: _GetMarkerLabel(int mk)
{
    return (mk & 0xff);  //(label % 256);
}

int BIAClass :: _GetMarkerValue(int id, int lb)
{
    return ((id << 8) | lb); //(id*256 + (int)lb);
}

int BIAClass :: GetMarkerValue(int id, int lb)
{
    return _GetMarkerValue(id, lb);
}

//----------------------------------------------------------

class MyApp: public wxApp
{
    virtual int  OnExit();
    virtual bool OnInit();
};


BEGIN_EVENT_TABLE(MyFrame, wxFrame)
EVT_MENU(ID_Quit, MyFrame::OnQuit)
EVT_MENU(ID_About, MyFrame::OnAbout)
EVT_MENU(ID_VolInfo, MyFrame::OnVolumeInformation)
#ifdef BIADEBUG
EVT_MENU(ID_ParamEstimation, MyFrame::OnParamEstimation)
#endif
EVT_MENU(ID_NewProj, MyFrame::OnNewProj)
EVT_MENU(ID_OpenProj, MyFrame::OnOpenProj)
EVT_MENU(ID_SaveProj, MyFrame::OnSaveProj)
EVT_MENU(ID_SaveProjAs, MyFrame::OnSaveProjAs)
EVT_MENU(ID_LoadVol, MyFrame::OnLoadVolume)
EVT_MENU(ID_LoadLabel, MyFrame::OnLoadLabel)
#ifdef BIADEBUG
EVT_MENU(ID_LoadGrad, MyFrame::OnLoadScnGradient)
#endif
EVT_MENU(ID_LoadArcWeight, MyFrame::OnLoadArcWeight)
EVT_MENU(ID_SaveVolume, MyFrame::OnSaveVolume)
EVT_MENU(ID_SaveLabel, MyFrame::OnSaveLabel)
EVT_MENU(ID_SaveObject, MyFrame::OnSaveObject)
EVT_MENU(ID_SaveObjectMask, MyFrame::OnSaveObjectMask)
EVT_MENU(ID_SaveObjectMap, MyFrame::OnSaveObjectMap)
EVT_MENU(ID_LoadMark, MyFrame::OnLoadMarkers)
EVT_MENU(ID_SaveMark, MyFrame::OnSaveMarkers)
EVT_MENU(ID_View0, MyFrame::OnSelectView)
EVT_MENU(ID_View1, MyFrame::OnSelectView)
EVT_MENU(ID_View2, MyFrame::OnSelectView)
EVT_MENU(ID_View3, MyFrame::OnSelectView)
EVT_MENU(ID_View4, MyFrame::OnSelectView)
EVT_MENU(ID_Zoomin, MyFrame::OnZoomin)
EVT_MENU(ID_Zoomout, MyFrame::OnZoomout)
EVT_MENU(ID_ChangeDrawMarker, MyFrame::OnChangeDrawMarker)

EVT_SIZE(           MyFrame::OnSize)

EVT_TOOL(ID_Preproc, MyFrame::OnPreprocessing)
EVT_TOOL(ID_Segmentation, MyFrame::OnSegmentation)
EVT_TOOL(ID_Analysis, MyFrame::OnAnalysis)
EVT_TOOL(ID_3DVisu, MyFrame::On3DVisualization)
EVT_TOOL(ID_Registration, MyFrame::OnRegistration)
EVT_TOOL(ID_LoadVolTool, MyFrame::OnLoadVolume)
#ifdef BIADEBUG
EVT_TOOL(ID_LoadGradTool, MyFrame::OnLoadScnGradient)
#endif
EVT_TOOL(ID_LoadArcWeight, MyFrame::OnLoadArcWeight)
EVT_TOOL(ID_Measurements, MyFrame::OnMeasurements)
EVT_TOOL(ID_GroupStudy, MyFrame::OnGroupStudy)

EVT_BUTTON(ID_ObjEye, MyFrame::OnShowObjWindow)
EVT_BUTTON(ID_Zoomin, MyFrame::OnZoomin)
EVT_BUTTON(ID_Zoomout, MyFrame::OnZoomout)
EVT_BUTTON(ID_BriContr, MyFrame::OnShowBriContr)
EVT_COMMAND_SCROLL(ID_BriContrDi, MyFrame::OnChangeBriContr)
//EVT_MOUSEWHEEL(MyFrame::OnChangeBriContr)
END_EVENT_TABLE()

//DECLARE_APP(MyApp)
IMPLEMENT_APP(MyApp)


int  MyApp::OnExit()
{
    delete APP;
    return 0;
}


bool MyApp::OnInit()
{
    char path[1024];

    vector<char *> colors;
    vector<int> color_values;

    colors.push_back("red");
    color_values.push_back(10824234);

    colors.push_back("blue");
    color_values.push_back(2003199);

    colors.push_back("green");
    color_values.push_back(9498256);

    colors.push_back("yellow");
    color_values.push_back(16776960);

    colors.push_back("gray");
    color_values.push_back(8355711);

    colors.push_back("orange");
    color_values.push_back(16753920);

    colors.push_back("cyan");
    color_values.push_back(11393254);


#ifdef _WIN32
    LPWSTR *szArglist;
  szArglist = CommandLineToArgvW(GetCommandLineW(), &argc);
  int __i;
  if ( NULL == szArglist )
  {
    wprintf(L"CommandLineToArgvW failed\n");
    return 0;
  }
  else
  {
    argv = (wxChar **) malloc(sizeof(char *) * argc);
    for ( __i = 0; __i < argc; __i++)
    {
      int tmp = wcslen(szArglist[__i]);
      char *c = (char *) malloc((sizeof(char) * tmp) + 2);
      wcstombs(c, szArglist[__i], tmp);
      argv[__i] = (wxChar *) c;
      argv[__i][tmp] = '\0';
    }
  }
  // Free memory allocated for CommandLineToArgvW arguments.
  LocalFree(szArglist);
#endif


    APP = new BIAClass();
    SetTopWindow(APP->Window);

    if (argc == 2)
    {
        wxString str(argv[1]); //wxSTRING_MAXLEN);
// #ifdef _WIN32
//     wxString str(argv[1], strlen((char *)argv[1])); //wxSTRING_MAXLEN);
// #else
//     wxString str(argv[1]);
// #endif
        if (strcmp(str.mb_str(), "-h") == 0 || strcmp(str.mb_str(), "help") == 0 || strcmp(str.mb_str(), "-help") == 0)
        {
            printf("\tvisva\n");
            printf("\tvisva -h\n");
            printf("\tvisva scene.scn\n");
            printf("\tvisva scene.scn -label label.scn\n");
            printf("\tvisva scene.scn -marker marker.txt\n");
            printf("\tvisva scene.scn -binary label1.scn o1 c1 label2.scn o2 c2 [...]\n");

            printf("\nColor parameter should be one of:\n");
            for (int i = 0; i < colors.size(); i++)
                cout << "\t" << colors[i] << endl;
            exit(0);
        }
        strcpy(path, str.mb_str());
        int s = strlen(path);
        if (s > 3 && strcasecmp(path + s - 4, ".bia") == 0)
            APP->Window->OpenProj(path);
        else
            APP->Window->LoadVolume(path);
    }
    else if (argc == 3)
    {
            printf("\tvisva\n");
            printf("\tvisva -h\n");
            printf("\tvisva scene.scn\n");
            printf("\tvisva scene.scn -label label.scn\n");
            printf("\tvisva scene.scn -marker marker.txt\n");
            printf("\tvisva scene.scn -binary label1.scn o1 c1 label2.scn o2 c2 [...]\n");

            printf("\nColor parameter should be one of:\n");
            for (int i = 0; i < colors.size(); i++)
                cout << "\t" << colors[i] << endl;
            exit(0);

    }
    else if (argc == 4)
    {
        wxString str2(argv[2]);
        wxString aux(_T("-label"));
        wxString aux2(_T("-marker"));
        if (str2.Cmp(aux) == 0)
        {
            wxString str(argv[1]);
            strcpy(path, str.mb_str());
            APP->Window->LoadVolume(path);
            wxString str3(argv[3]);
            strcpy(path, str3.mb_str());
            APP->Window->LoadLabel(path, NULL, 0);
        }
        else if (str2.Cmp(aux2) == 0)
        {
        	wxString str(argv[1]);
            strcpy(path, str.mb_str());
            APP->Window->LoadVolume(path);
            wxString str3(argv[3]);
            strcpy(path, str3.mb_str());
            APP->Window->LoadMarkers(path);
        } 
        else
        {
            printf("\tvisva\n");
            printf("\tvisva -h\n");
            printf("\tvisva scene.scn\n");
            printf("\tvisva scene.scn -label label.scn\n");
            printf("\tvisva scene.scn -marker marker.txt\n");
            printf("\tvisva scene.scn -binary label1.scn o1 c1 label2.scn o2 c2 [...]\n");

            printf("\nColor parameter should be one of:\n");
            for (int i = 0; i < colors.size(); i++)
                cout << "\t" << colors[i] << endl;
            exit(0);
        }
    }
    else if (argc >= 6 && argc % 3 == 0)
    {
        wxString str2(argv[2]);
        wxString aux(_T("-binary"));
        if (str2.Cmp(aux) == 0)
        {
            wxString str(argv[1]);
            strcpy(path, str.mb_str());
            APP->Window->LoadVolume(path);

            char label_name[1024];
            char color_str[1024];
            int label_color = 0;
            int next_object = 0;
            SegmObject *aux = NULL;
            for (int i = 3 ; i < argc; i += 3)
            {
                wxString str3(argv[i]);
                strcpy(path, str3.mb_str());
                wxString obj(argv[i + 1]);
                wxString color(argv[i + 2]);
                strcpy(label_name, obj.mb_str());
                strcpy(color_str, color.mb_str());
                label_color = -1;
                for (int j = 0; j < colors.size(); j++)
                {
                    if (strcmp(colors[j], color_str) == 0)
                        label_color = color_values[j];

                }
                if (label_color < 0)
                {
                    printf("Wrong color, please use visva help for the complete list\n");
                    exit(0);
                }
                APP->Window->LoadLabel(path, label_name, label_color);
                if (next_object < APP->GetNumberOfObjs())
                {
                    aux = APP->GetObjByIndex(next_object++);
                    strcpy(aux->name, label_name);
                    aux->color = label_color;
                }
            }
            APP->DrawSegmObjects();
        }
        else
        {
            printf("\tvisva\n");
            printf("\tvisva -h\n");
            printf("\tvisva scene.scn\n");
            printf("\tvisva scene.scn -label label.scn\n");
            printf("\tvisva scene.scn -marker marker.txt\n");
            printf("\tvisva scene.scn -binary label1.scn o1 c1 label2.scn o2 c2 [...]\n");

            printf("\nColor parameter should be one of:\n");
            for (int i = 0; i < colors.size(); i++)
                cout << "\t" << colors[i] << endl;
            exit(0);
        }
    }
    else if ((argc >= 6) && ((argc % 2) == 0))
    {
        wxString str2(argv[2]);
        wxString aux(_T("-label"));
        if (str2.Cmp(aux) == 0)
        {

            wxString str(argv[1]);
            strcpy(path, str.mb_str());
            APP->Window->LoadVolume(path);
            wxString str3(argv[3]);
            strcpy(path, str3.mb_str());
            APP->Window->LoadLabel(path, NULL, 0);

            char label_name[1024];
            char color_str[1024];
            int label_color = 0;
            int next_object = 0;
            SegmObject *aux = NULL;
            for (int i = 4 ; i < argc; i += 2)
            {
                wxString obj(argv[i]);
                wxString color(argv[i+1]);
                strcpy(label_name, obj.mb_str());
                strcpy(color_str, color.mb_str());
                label_color = -1;
                for(int j=0; j < colors.size(); j++)
                {
                    if(strcmp(colors[j], color_str) == 0)
                        label_color = color_values[j];

                }
                if (label_color < 0)
                {
                    printf("Wrong color, please use visva help for the complete list\n");
                    exit(0);
                }
                if (next_object < APP->GetNumberOfObjs())
                {
                    aux = APP->GetObjByIndex(next_object++);
                    strcpy(aux->name, label_name);
                    aux->color = label_color;
                }
            }
        }
        APP->DrawSegmObjects();
    }
    else if (argc > 1)
    {
            printf("\tvisva\n");
            printf("\tvisva -h\n");
            printf("\tvisva scene.scn\n");
            printf("\tvisva scene.scn -label label.scn\n");
            printf("\tvisva scene.scn -marker marker.txt\n");
            printf("\tvisva scene.scn -binary label1.scn o1 c1 label2.scn o2 c2 [...]\n");

            printf("\nColor parameter should be one of:\n");
            for (int i = 0; i < colors.size(); i++)
                cout << "\t" << colors[i] << endl;
            exit(0);
    }

    return TRUE;
}


//---------------------------------------------------

MyFrame::MyFrame(  wxString &title,   wxPoint &pos,   wxSize &size)
        : wxFrame((wxFrame *)NULL, -1, title, pos, size)
{

    SetIcon(brain_xpm);

    //Menubar
    wxMenu *menuFile = new wxMenu;
    menuFile->Append(ID_NewProj, _T("&New Project...\tCTRL+n"));
    menuFile->Append(ID_OpenProj, _T("&Open Project...\tCTRL+o"));
    menuFile->Append(ID_SaveProj, _T("&Save Project\tCTRL+s"));
    menuFile->Append(ID_SaveProjAs, _T("&Save Project As...\tCTRL+a"));
    menuFile->AppendSeparator();

#if wxCHECK_VERSION(2,8,10)
    wxMenu *menuImport = new wxMenu;
    menuImport->Append(ID_LoadVol, _T("Import Volume...\tCTRL+i"));
#ifdef BIADEBUG
    menuImport->Append(ID_LoadGrad, _T("Import Gradient...\tCTRL+g"));
#endif
    menuImport->Append(ID_LoadArcWeight, _T("Import ArcWeight...\tCTRL+g"));
    menuImport->Append(ID_LoadLabel, _T("Import Label...\tCTRL+b"));
    menuImport->Append(ID_LoadMark,  _T("Import Markers...\tCTRL+m"));
    wxMenu *menuExport = new wxMenu;
    menuExport->Append(ID_SaveVolume, _T("Export Volume...\tCTRL+e"));
    menuExport->Append(ID_SaveLabel, _T("Export Label...\tALT+b"));
    menuExport->Append(ID_SaveObject, _T("Export Object...\tALT+o"));
    menuExport->Append(ID_SaveObjectMask, _T("Export Object Mask...\tALT+o"));
    menuExport->Append(ID_SaveObjectMap, _T("Export Object Map..."));
    menuExport->Append(ID_SaveMark,  _T("Export Markers...\tALT+m"));
    menuFile->AppendSubMenu(menuImport, _T("Import..."), _T("Import data"));
    menuFile->AppendSubMenu(menuExport, _T("Export..."), _T("Export data"));
#else
    menuFile->Append(ID_LoadVol, _T("Import Volume...\tCTRL+i"));
#ifdef BIADEBUG
    menuFile->Append(ID_LoadGrad, _T("Import Gradient...\tCTRL+g"));
#endif
    menuImport->Append(ID_LoadArcWeight, _T("Import ArcWeight...\tCTRL+g"));
    menuFile->Append(ID_LoadLabel, _T("Import Label...\tCTRL+b"));
    menuFile->Append(ID_LoadMark,  _T("Import Markers...\tCTRL+m"));
    menuFile->Append(ID_SaveVolume, _T("Export Volume...\tCTRL+e"));
    menuFile->Append(ID_SaveLabel, _T("Export Label...\tALT+b"));
    menuFile->Append(ID_SaveObjectMask, _T("Export Object Mask...\tALT+o"));
    menuFile->Append(ID_SaveObjectMap, _T("Export Object Map..."));
    menuFile->Append(ID_SaveMark,  _T("Export Markers...\tALT+m"));
#endif

    menuFile->AppendSeparator();
    menuFile->Append(ID_VolInfo,  _T("Volume &Information\tF1"));
#ifdef BIADEBUG
    menuFile->AppendSeparator();
    menuFile->Append(ID_ParamEstimation, _T("Parameters Estimation\tF2"));
#endif
    menuFile->AppendSeparator();
    menuFile->Append(ID_Quit, _T("E&xit"));


    wxMenu *menuView = new wxMenu;
    menuView->Append(ID_View0, _T("&All views...\tCTRL+0"));
    menuView->Append(ID_View1, _T("View&1...\tCTRL+1"));
    menuView->Append(ID_View2, _T("View&2...\tCTRL+2"));
    menuView->Append(ID_View3, _T("View&3...\tCTRL+3"));
    menuView->Append(ID_View4, _T("View&4...\tCTRL+4"));
    menuView->AppendSeparator();
    menuView->Append(ID_Zoomin, _T("Zoom &In\tCTRL+="));
    menuView->Append(ID_Zoomout, _T("Zoom &Out\tCTRL+-"));
    menuView->AppendSeparator();
    menuView->Append(ID_ChangeDrawMarker, _T("Show/Hide Cursor\tspace"));

    wxMenu *menuHelp = new wxMenu;
    menuHelp->Append(ID_About,  _T("&About...\tF12"));

    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append( menuFile,   _T("&File") );
    menuBar->Append( menuView,   _T("&View") );
    menuBar->Append( menuHelp,   _T("&Help") );

    SetMenuBar( menuBar );

    //Toolbar
    wxBitmap *bmNew    = new wxBitmap(new_xpm);
    wxBitmap *bmOpen  = new wxBitmap(open_xpm);
    wxBitmap *bmSave  = new wxBitmap(save_xpm);

    wxBitmap *bmVisual    = new wxBitmap(visualization_xpm);
    wxBitmap *bmAnalysis  = new wxBitmap(curve_xpm);
    wxBitmap *bmRegistration  = new wxBitmap(reg_xpm);
    wxBitmap *bmPreproc   = new wxBitmap(preproc_xpm);
    wxBitmap *bmSegment   = new wxBitmap(segmentation_xpm);

    wxBitmap *bmMeasurements  = new wxBitmap(measure_xpm);
    wxBitmap *bmGroup  = new wxBitmap(group_xpm);

    toolBar = CreateToolBar(wxTB_HORIZONTAL | wxTB_TEXT, -1, _T("toolBar"));

    //********************************
    //wxBitmapButton *butPreproc = new wxBitmapButton(toolBar, ID_Preproc, *bmPreproc, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("Preprocessing"));

    /*
    LabeledBitmapButton *butPreproc;
    butPreproc = new LabeledBitmapButton(toolBar, ID_Preproc,
                     *bmPreproc, _T("Preproc"));
    //butPreproc->Update();

    wxColour wxcolor;
    SetColor(&wxcolor, 0xffffdd);
    butPreproc->SetBackgroundColour(wxcolor);

    //butPreproc->SetBitmapDisabled(*bmAnalysis);
    //butPreproc->Disable();

    toolBar->AddControl((wxControl*)butPreproc);

    //wxBitmap bmdis = butPreproc->GetBitmapDisabled();
    //bmdis.SaveFile(_T("disa.bmp"), wxBITMAP_TYPE_BMP, NULL);
    */

    //********************************

    toolBar->AddSeparator();

    toolBar->AddTool(ID_NewProj, _T("New Project"), *bmNew,
                     _T("New Project"), wxITEM_NORMAL);
    toolBar->AddTool(ID_OpenProj, _T("Open Project"), *bmOpen,
                     _T("Open Project"), wxITEM_NORMAL);
    toolBar->AddTool(ID_SaveProj, _T("Save Project"), *bmSave,
                     _T("Save Project"), wxITEM_NORMAL);

    toolBar->AddSeparator();

    toolBar->AddTool(ID_Preproc, _T("Transformations"), *bmPreproc,
                     _T("Transformations"), wxITEM_NORMAL);
    toolBar->AddTool(ID_Segmentation, _T("Segmentation"), *bmSegment,
                     _T("Segmentation"), wxITEM_NORMAL);
    toolBar->AddTool(ID_3DVisu, _T("Visualization"), *bmVisual,
                     _T("Visualization"), wxITEM_NORMAL);
    toolBar->AddTool(ID_Registration, _T("Registration"), *bmRegistration,
                     _T("Registration"), wxITEM_NORMAL);
    toolBar->AddTool(ID_Analysis, _T("Analysis"), *bmAnalysis,
                     _T("Analysis"), wxITEM_NORMAL);
    toolBar->AddSeparator();

    // toolBar->AddTool(ID_Measurements, _T("Measurements"), *bmMeasurements,_T("Measurements"), wxITEM_NORMAL);
    // toolBar->AddTool(ID_GroupStudy, _T("Group Study"), *bmGroup, _T("Group Study"), wxITEM_NORMAL);

    toolBar->Realize();


    //Display
    wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);

    panel = new wxPanel(this, wxID_ANY, wxDefaultPosition,
                        wxDefaultSize, wxBORDER_NONE);//wxBORDER_SIMPLE);
    //panel->SetBackgroundColour(*wxBLUE);
    splitter = new wxSplitterWindow(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D, _T("hsplitter"));

    topWindow = new wxSplitterWindow(splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D, _T("top"));
    topWindow->Show(true);

    bottomWindow = new wxSplitterWindow(splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D, _T("bottom"));
    bottomWindow->Show(true);

    splitter->SetMinimumPaneSize(20);  // Set this to prevent unsplitting
    splitter->SplitHorizontally( topWindow, bottomWindow );
    splitter->SetSashGravity(0.5);

    Views[0] = NULL;
    Views[1] = NULL;
    Views[2] = NULL;
    Views[3] = NULL;

    vsizer = new wxBoxSizer(wxVERTICAL);

    wxStaticBoxSizer *dissizer = new wxStaticBoxSizer(wxVERTICAL, panel, _T("Display"));
    wxStaticBoxSizer *objsizer = new wxStaticBoxSizer(wxVERTICAL, panel, _T("Objects"));
    wxBitmap *bmeye      = new wxBitmap(tools_xpm); //eye_xpm);
    wxBitmap *bmZoomin   = new wxBitmap(zoomin_xpm);
    wxBitmap *bmZoomout  = new wxBitmap(zoomout_xpm);
    wxBitmap *bmBriContr = new wxBitmap(brightcontr_xpm);

    buteye = new wxBitmapButton(panel, ID_ObjEye, *bmeye, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("button_eye"));
    objsizer->Add(buteye, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);

    BCDialog = new BriContrDialog(this, ID_BriContrDi);
    buZoomin = new wxBitmapButton(panel, ID_Zoomin, *bmZoomin, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("butZoomin"));
    buZoomout = new wxBitmapButton(panel, ID_Zoomout, *bmZoomout, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("butZoomout"));
    buBriContr = new wxBitmapButton(panel, ID_BriContr, *bmBriContr, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("butBriContr"));
    wxBoxSizer *hbsButton    = new wxBoxSizer(wxHORIZONTAL);
    hbsButton->Add(buZoomin,   1, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
    hbsButton->Add(buZoomout,  1, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
    hbsButton->Add(buBriContr, 1, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
    dissizer->Add(hbsButton, 0, wxEXPAND);

    vsizer->AddSpacer(15);
    vsizer->Add(dissizer, 0, wxEXPAND);
    vsizer->AddSpacer(5);
    vsizer->Add(objsizer, 0, wxEXPAND);
    panel->SetSizer(vsizer, true);
    vsizer->SetSizeHints(panel);

    hsizer->Add(panel,    0, wxALIGN_LEFT | wxEXPAND);
    hsizer->Add(splitter, 1, wxEXPAND);
    SetSizer(hsizer, true);
    hsizer->SetSizeHints(this);

    wxSystemScreenType sst = wxSystemSettings::GetScreenType();
    if (sst == wxSYS_SCREEN_TINY || sst == wxSYS_SCREEN_PDA)
    {
        SetSize(320, 240);
    }
    else if (sst == wxSYS_SCREEN_SMALL)
    {
        SetSize(640, 480);
    }
    else if (sst == wxSYS_SCREEN_DESKTOP || sst == wxSYS_SCREEN_NONE)
    {
        SetSize(800, 600);
    }
    else
    {
        SetSize(800, 600);
    }

    this->B = 50;
    this->C = 0;
    this->zoom = 1.0;
    vsizer->Layout();

    //Statusbar
    CreateStatusBar();
    SetStatusText( _T("Welcome to VISVA - Volumetric Image Segmentation for Visualization and Analysis!") );
}


MyFrame::~MyFrame()
{
    (APP->Preferences).IsMaximized = this->IsMaximized();
}


void MyFrame::AppendOptPanel(wxPanel *panel,
                             Module::ModuleType type)
{
    //wxWindow *parent = vsizer->GetContainingWindow();
    wxWindow *parent = (wxWindow *)this->panel;
    if (parent == NULL || panel == NULL) return;

    panel->Reparent(parent);
    panel->Show(true);

    if (vsizer->GetItem(panel, false) != NULL)
        vsizer->Detach(panel);

    this->SetOptPanelBkgColor(panel, type);
    vsizer->Add(panel, 0, wxEXPAND);
    vsizer->SetSizeHints(parent);
    vsizer->Layout();
}


void MyFrame::PrependOptPanel(wxPanel *panel,
                              Module::ModuleType type)
{
    //wxWindow *parent = vsizer->GetContainingWindow();
    wxWindow *parent = (wxWindow *)this->panel;
    if (parent == NULL || panel == NULL) return;

    panel->Reparent(parent);
    panel->Show(true);

    if (vsizer->GetItem(panel, false) != NULL)
        vsizer->Detach(panel);

    this->SetOptPanelBkgColor(panel, type);
    vsizer->Prepend(panel, 0, wxEXPAND);
    vsizer->SetSizeHints(parent);
    vsizer->Layout();
}


bool MyFrame::DetachOptPanel(wxPanel *panel)
{
    //wxWindow *parent = vsizer->GetContainingWindow();
    wxWindow *parent = (wxWindow *)this->panel;
    bool ret;

    if (parent == NULL || panel == NULL) return false;

    panel->Hide();
    ret = vsizer->Detach(panel);
    vsizer->SetSizeHints(parent);
    vsizer->Layout();
    return ret;
}


void MyFrame::EnableObjWindow(bool enable)
{
    this->buteye->Enable(enable);
}

void MyFrame::SetBriContr(int B, int C)
{
    BCDialog->SetBCLevel(B, C);
}

void MyFrame::GetBriContr(int *B, int *C)
{
    BCDialog->GetBCLevel(B, C);
}

void MyFrame::SetOptPanelBkgColor(wxPanel *panel,
                                  Module::ModuleType type)
{
    wxColour wxcolor;
    int color;

    switch (type)
    {
        case Module::PREPROC:
            color = (APP->Preferences).color_preproc;
            break;
        case Module::SEGMENTATION:
            color = (APP->Preferences).color_segmentation;
            break;
        case Module::ANALYSIS:
            color = (APP->Preferences).color_analysis;
            break;
        case Module::REGISTRATION:
            color = (APP->Preferences).color_analysis;
            break;
        case Module::VIEW2D:
            color = (APP->Preferences).color_view;
            break;
        case Module::VIEW3D:
            color = (APP->Preferences).color_view;
            break;
        default:
            return;
    }
    if (color == NIL) return;
    SetColor(&wxcolor, color);
    panel->SetBackgroundColour(wxcolor);
}


void MyFrame::SetViewPanel(wxPanel *view, int pos)
{
    wxSplitterWindow *splitter;

    if (pos == 0 || pos == 1)
        splitter = topWindow;
    else if (pos == 2 || pos == 3)
        splitter = bottomWindow;
    else return;

    // Set this to prevent unsplitting
    splitter->SetMinimumPaneSize(20);

    view->Reparent(splitter);
    if (Views[pos] != NULL)
    {
        splitter->ReplaceWindow(Views[pos], view);
        delete Views[pos];
        Views[pos] = view;
        Views[pos]->Show(true);
    }
    else
    {
        Views[pos] = view;
        Views[pos]->Show(true);
        if (splitter->IsSplit())
            splitter->Unsplit(NULL);

        if (pos % 2 == 0)
        {
            if (Views[pos + 1] != NULL)
                splitter->SplitVertically(view, Views[pos + 1]);
            else
                splitter->Initialize(view);
        }
        else
        {
            if (Views[pos - 1] != NULL)
                splitter->SplitVertically(Views[pos - 1], view);
            else
                splitter->Initialize(view);
        }
    }
    splitter->SetSashGravity(0.5);
}


void MyFrame::UniformView()
{
    int w, h;
    /*
    this->splitter->Layout();
    this->topWindow->Layout();
    this->bottomWindow->Layout();
    this->splitter->Update();
    this->topWindow->Update();
    this->bottomWindow->Update();
    */
    splitter->GetSize(&w, &h);
    splitter->SetSashPosition(h / 2, true);

    topWindow->GetSize(&w, &h);
    topWindow->SetSashPosition(w / 2, true);

    bottomWindow->GetSize(&w, &h);
    bottomWindow->SetSashPosition(w / 2, true);
}

/*
void MyFrame :: OnShowObjWindow(wxCommandEvent & event){
  static ObjectDialog *dialog = NULL;

  if(!APP->Data.loaded) return;

  if(APP->GetNumberOfObjs()==0){
    wxMessageBox(_T("No objects available."), _T("Warning"),
         wxOK | wxICON_EXCLAMATION, this);
    return;
  }
  else{
    //if(dialog!=NULL)
    //  delete dialog;
    dialog = new ObjectDialog(this);
    dialog->ShowModal();
    wxTheApp->Yield();
  }
}
*/

void MyFrame :: OnShowObjWindow(wxCommandEvent &event)
{
    static ObjectDialog *dialog = NULL;

    if (!APP->Data.loaded) return;

    if (APP->GetNumberOfObjs() == 0)
    {
        wxMessageBox(_T("No objects available."), _T("Warning"),
                     wxOK | wxICON_EXCLAMATION, this);
        return;
    }
    else
    {
        //if(dialog!=NULL)
        //  delete dialog;
        dialog = new ObjectDialog(this);
        wxSize size(iftMax(dialog->GetMinSize().GetWidth(), 500), iftMax(dialog->GetMinSize().GetHeight(), 400));
        dialog->SetSize(wxDefaultSize.GetWidth(), size.GetWidth());
        dialog->SetMaxSize(size);
        dialog->ShowModal();
        wxTheApp->Yield();
    }
}


void MyFrame::OnZoomin(wxCommandEvent &WXUNUSED(event))
{
if (!APP->Data.loaded) return;

if (this->zoom >= 15.9999) return;
this->zoom *= 1.25;
APP->SetZoomLevel(this->zoom);
}


void MyFrame::OnZoomout(wxCommandEvent &WXUNUSED(event))
{
if (!APP->Data.loaded) return;

if (this->zoom <= 0.5001) return;
this->zoom *= 0.8;
APP->SetZoomLevel(this->zoom);
}

void MyFrame::OnChangeDrawMarker(wxCommandEvent &WXUNUSED(event))
{
if (!APP->Data.loaded) return;

APP->ChangeDrawMarker();
}


void MyFrame::OnChangeBriContr(wxScrollEvent &event)
{
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(false, 1.0);
}


void MyFrame::OnShowBriContr(wxCommandEvent &WXUNUSED(event))
{
BCDialog->ShowWindow();
}


void MyFrame::OnSaveVolume(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_SaveVolume;
    int filter;

    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.scn;*.hdr;*.scn.bz2;*.nii;*.nii.gz)|*.scn;*.hdr;*.scn.bz2;*.nii;*.nii.gz|"
                                     "Scene files (*.scn)|*.scn|"
                                     "Analyze files (*.hdr)|*.hdr|"
                                     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
                                     "Nifti-1 files (*.nii)|*.nii|"
                                     "Compressed Nifti-1 files (*.nii.gz)|*.nii.gz|"
                                     "All files|*.*"
    );

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    wxFileDialog *saveFileDialog =
            new wxFileDialog ( this,
                               _T("Save Volume"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_SAVE | wxFD_OVERWRITE_PROMPT,
                               wxDefaultPosition);

    saveFileDialog->SetFilterIndex((APP->Preferences).ext_SaveVolume);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        saveFileDialog->SetDirectory(wxdir);
    }
    if (saveFileDialog->ShowModal() == wxID_OK)
    {
        wxBusyCursor wait;
        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename());

        (APP->Preferences).ext_SaveVolume = saveFileDialog->GetFilterIndex();

        char path[512];
        strcpy( dir,  saveFileDialog->GetDirectory().ToAscii() );
        strcpy( path, saveFileDialog->GetPath().ToAscii() );

        // Set the extension if missing
        filter = saveFileDialog->GetFilterIndex();
        CheckFileExtension(path,filter);

        // Save
        iftImage *img = SceneToImage(APP->Data.orig);
        iftWriteImageByExt(img,path);
        iftDestroyImage(&img);

        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename() + _T("... Done"));
    }
}

void MyFrame::OnSaveLabel(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_SaveLabel;
    Scene *flabel;
    int filter;

    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.scn;*.hdr;*.scn.bz2;*.nii;*.nii.gz)|*.scn;*.hdr;*.scn.bz2;*.nii;*.nii.gz|"
                                     "Scene files (*.scn)|*.scn|"
                                     "Analyze files (*.hdr)|*.hdr|"
                                     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
                                     "Nifti-1 files (*.nii)|*.nii|"
                                     "Compressed Nifti-1 files (*.nii.gz)|*.nii.gz|"
                                     "All files|*.*"
    );

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    wxString wxlabelname((APP->Data).labelname, wxConvUTF8);

    wxFileDialog *saveFileDialog =
            new wxFileDialog ( this,
                               _T("Save Label"),
                               _T(""),
                               wxlabelname,
                               FILETYPES,
                               wxFD_SAVE | wxFD_OVERWRITE_PROMPT,
                               wxDefaultPosition);

    saveFileDialog->SetFilterIndex((APP->Preferences).ext_SaveLabel);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        saveFileDialog->SetDirectory(wxdir);
    }
    if (saveFileDialog->ShowModal() == wxID_OK)
    {
        wxBusyCursor wait;
        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename());

        (APP->Preferences).ext_SaveLabel = saveFileDialog->GetFilterIndex();

        char path[512];
        strcpy( dir,  saveFileDialog->GetDirectory().ToAscii() );
        strcpy( path, saveFileDialog->GetPath().ToAscii() );
        //flabel = CopyScene(APP->Data.label);
        //flabel = APP->CopyVisibleLabelUnion();
        flabel = APP->CopyDistinctVisibleLabels();

        // Set the extension if missing
        filter = saveFileDialog->GetFilterIndex();
        CheckFileExtension(path,filter);

        // Save
        iftImage *img = SceneToImage(flabel);
        iftWriteImageByExt(img,path);
        iftDestroyImage(&img);
/*
        if (strcasecmp(path + s - 4, ".scn") == 0 || strcasecmp(path + s - 4, ".hdr") == 0 || strcasecmp(path + s - 4, ".img") == 0 || strcasecmp(path + s - 8, ".scn.bz2") == 0 || strcasecmp(path + s - 4, ".nii") == 0 || strcasecmp(path + s - 7, ".nii.gz") == 0){
            flabel->dx = APP->Data.orig->dx;
            flabel->dy = APP->Data.orig->dy;
            flabel->dz = APP->Data.orig->dz;
            if (strcasecmp(path + s - 4, ".hdr") == 0 || strcasecmp(path + s - 4, ".img"))
                WriteAnalyze(flabel, path);
            else WriteScene(flabel, path);
        }
        else
        {
            wxMessageBox(_T("Unsupported extension!"), _T("Error"), wxOK | wxICON_EXCLAMATION, this);
            DestroyScene(&flabel);
            return;
        }
*/
        DestroyScene(&flabel);

        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename() + _T("... Done"));
    }
}

void MyFrame::OnSaveObject(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_SaveObject;
    //Scene *object,*flabel,*orig_i;
    //int p,n,s,filter,maxval,maxval_obj;
    int filter;

    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.scn;*.hdr;*.scn.bz2)|*.scn;*.hdr;*.scn.bz2|"
                                     "Scene files (*.scn)|*.scn|"
                                     "Analyze files (*.hdr)|*.hdr|"
                                     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
                                     "Nifti-1 files (*.nii)|*.nii|"
                                     "Compressed Nifti-1 files (*.nii.gz)|*.nii.gz|"
                                     "All files|*.*"
    );

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    wxFileDialog *saveFileDialog =
            new wxFileDialog ( this,
                               _T("Save Object"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_SAVE | wxFD_OVERWRITE_PROMPT,
                               wxDefaultPosition);

    saveFileDialog->SetFilterIndex((APP->Preferences).ext_SaveObject);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        saveFileDialog->SetDirectory(wxdir);
    }

    SegmObject *obj = APP->GetObjByDialog();

    if (saveFileDialog->ShowModal() == wxID_OK)
    {
        wxBusyCursor wait;
        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename());

        (APP->Preferences).ext_SaveObject = saveFileDialog->GetFilterIndex();

        char path[512];
        //char filename[512];

        //maxval = MaximumValue3(APP->Data.orig);
        strcpy( dir,  saveFileDialog->GetDirectory().ToAscii() );
        strcpy( path, saveFileDialog->GetPath().ToAscii() );

        // Set the extension if missing
        filter = saveFileDialog->GetFilterIndex();
        CheckFileExtension(path,filter);

        // Save
        Scene *scn1 = CreateScene(APP->Data.orig->xsize, APP->Data.orig->ysize, APP->Data.orig->zsize);
        CopyBMap2SceneMask(scn1, obj->mask);
        int p, n = APP->Data.w * APP->Data.h * APP->Data.nframes;
        for (p = 0; p < n; p++)
        {
            if (scn1->data[p] != 0)
                scn1->data[p] = APP->Data.orig->data[p];
        }
        CopyVoxelSizeFromScene(scn1,APP->Data.orig);
        iftImage *img = SceneToImage(scn1);
        iftWriteImageByExt(img,path);
        iftDestroyImage(&img);

        /*
        if (strcasecmp(path + s - 4, ".scn") == 0 || strcasecmp(path + s - 4, ".hdr") == 0 || strcasecmp(path + s - 4, ".img") == 0 || strcasecmp(path + s - 8, ".scn.bz2") == 0 || strcasecmp(path + s - 4, ".nii") == 0 || strcasecmp(path + s - 7, ".nii.gz") == 0)
        {
            Scene *scn1 = CreateScene(APP->Data.orig->xsize, APP->Data.orig->ysize, APP->Data.orig->zsize);
            CloneNiftiHeader( APP->Data.orig, scn1 );
            CopyBMap2SceneMask(scn1, obj->mask);
            int p, n = APP->Data.w * APP->Data.h * APP->Data.nframes;
            for (p = 0; p < n; p++)
            {
                if (scn1->data[p] != 0)
                    scn1->data[p] = APP->Data.orig->data[p];
            }
            scn1->dx = APP->Data.orig->dx;
            scn1->dy = APP->Data.orig->dy;
            scn1->dz = APP->Data.orig->dz;
            if (strcasecmp(path + s - 4, ".hdr") == 0 || strcasecmp(path + s - 4, ".img"))
                WriteAnalyze(scn1, path);
            else WriteScene(scn1, path);
            DestroyScene(&scn1);
        }
        else
        {
            wxMessageBox(_T("Unsupported extension!"), _T("Error"), wxOK | wxICON_EXCLAMATION, this);
            return;
        }
         */


        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename() + _T("... Done"));

        /*
        object = CopyScene(APP->Data.orig);
        n = APP->Data.w*APP->Data.h*APP->Data.nframes;

        for(p=0; p<n; p++){
          if(APP->Data.label->data[p] == 0)
        object->data[p] = 0;
        }
        maxval_obj = MaximumValue3(object);

        flabel = CopyScene(APP->Data.label);

        orig_i = CopyScene(APP->Data.orig);


        s = strlen(path);
        if( s<3 || (strcasecmp(path+s-3, "scn") != 0 && strcasecmp(path+s-3, "hdr") != 0) ){
          filter = saveFileDialog->GetFilterIndex();
          if(filter==0) //.SCN
        strcat(path,".scn");
          else if(filter==1) //.HDR
        strcat(path,".hdr");
          else{
        wxMessageBox(_T("Unable to save volume data. Invalid file format!"), _T("Save Object Error"), wxOK | wxICON_EXCLAMATION, this);
        DestroyScene(&orig_i);
        DestroyScene(&flabel);
        DestroyScene(&object);
        return;
          }
        }

        s = strlen(path);
        if( strcasecmp(path + s - 3, "scn") == 0 ){
          WriteScene(object, path);

          strcpy(filename,path);
          filename[s - 4] = '\0';
          strcat(filename,"_int.scn");
          WriteScene(orig_i, filename);
        }
        else if( strcasecmp(path + s - 3, "hdr") == 0 ){
          WriteScene(object,path);
          //WriteScene(object, (char *)"tmp.scn" );
          //write_analyze((char *)"tmp.scn", path, maxval_obj);

          strcpy(filename,path);
          filename[s - 4] = '\0';
          strcat(filename,"_int.hdr");
          WriteScene(orig_i,filename);
          //WriteScene(orig_i, (char *)"tmp.scn" );
          //write_analyze((char *)"tmp.scn", filename, maxval);
        }
        strcpy(filename,path);
        filename[s - 4] = '\0';
        strcat(filename,"_lb.scn");
        WriteScene(flabel, filename);

        strcpy(filename,path);
        filename[s - 4] = '\0';
        strcat(filename,"_report.txt");

        DestroyScene(&orig_i);
        DestroyScene(&flabel);
        DestroyScene(&object);
        */

    }
}


void MyFrame::OnSaveObjectMask(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_SaveObject;
    //Scene *object,*flabel,*orig_i;
    //int p,n,s,filter,maxval,maxval_obj;
    int filter;

    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.scn;*.hdr;*.scn.bz2)|*.scn;*.hdr;*.scn.bz2|"
                                     "Scene files (*.scn)|*.scn|"
                                     "Analyze files (*.hdr)|*.hdr|"
                                     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
                                     "Nifti-1 files (*.nii)|*.nii|"
                                     "Compressed Nifti-1 files (*.nii.gz)|*.nii.gz|"
                                     "All files|*.*"
    );

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    wxFileDialog *saveFileDialog =
            new wxFileDialog ( this,
                               _T("Save Object"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_SAVE | wxFD_OVERWRITE_PROMPT,
                               wxDefaultPosition);

    saveFileDialog->SetFilterIndex((APP->Preferences).ext_SaveObject);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        saveFileDialog->SetDirectory(wxdir);
    }

    SegmObject *obj = APP->GetObjByDialog();

    if (saveFileDialog->ShowModal() == wxID_OK)
    {
        wxBusyCursor wait;
        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename());

        (APP->Preferences).ext_SaveObject = saveFileDialog->GetFilterIndex();

        char path[512];
        //char filename[512];

        //maxval = MaximumValue3(APP->Data.orig);
        strcpy( dir,  saveFileDialog->GetDirectory().ToAscii() );
        strcpy( path, saveFileDialog->GetPath().ToAscii() );

        // Set the extension if missing
        filter = saveFileDialog->GetFilterIndex();
        CheckFileExtension(path,filter);

        // Save
        Scene *scn1 = CreateScene(APP->Data.orig->xsize, APP->Data.orig->ysize, APP->Data.orig->zsize);
        CopyBMap2SceneMask(scn1, obj->mask);
        CopyVoxelSizeFromScene(scn1,APP->Data.orig);
        iftImage *img = SceneToImage(scn1);
        iftWriteImageByExt(img,path);
        iftDestroyImage(&img);

        /*
        s = strlen(path);
        filter = saveFileDialog->GetFilterIndex();
        if ((s < 4 || strcasecmp(path + s - 4, ".scn") != 0) && filter == 1 )
            strcat(path, ".scn");
        else if ((s < 4 || strcasecmp(path + s - 4, ".hdr") != 0) && filter == 2 )
            strcat(path, ".hdr");
        else if ((s < 8 || strcasecmp(path + s - 8, ".scn.bz2") != 0) && filter == 3 )
            strcat(path, ".scn.bz2");
        else if ((s < 4 || strcasecmp(path + s - 4, ".nii") != 0) && filter == 4 )
            strcat(path, ".nii");
        else if ((s < 7 || strcasecmp(path + s - 7, ".nii.gz") != 0) && filter == 5 )
            strcat(path, ".nii.gz");


        if (strcasecmp(path + s - 4, ".scn") == 0 || strcasecmp(path + s - 4, ".hdr") == 0 || strcasecmp(path + s - 4, ".img") == 0 || strcasecmp(path + s - 8, ".scn.bz2") == 0 || strcasecmp(path + s - 4, ".nii") == 0 || strcasecmp(path + s - 7, ".nii.gz") == 0)
        {
            Scene *scn1 = CreateScene(APP->Data.orig->xsize, APP->Data.orig->ysize, APP->Data.orig->zsize);
            CloneNiftiHeader( APP->Data.orig, scn1 );
            CopyBMap2SceneMask(scn1, obj->mask);
            scn1->dx = APP->Data.orig->dx;
            scn1->dy = APP->Data.orig->dy;
            scn1->dz = APP->Data.orig->dz;

            if (strcasecmp(path + s - 4, ".hdr") == 0 || strcasecmp(path + s - 4, ".img") == 0)
                WriteAnalyze(scn1, path);
            else WriteScene(scn1, path);
            DestroyScene(&scn1);
        }
        else
        {
            wxMessageBox(_T("Unsupported extension!"), _T("Error"), wxOK | wxICON_EXCLAMATION, this);
            return;
        }*/


        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename() + _T("... Done"));

    }
}


void MyFrame::OnSaveObjectMap(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_SaveLabel;
    int filter;

    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.scn;*.hdr;*.scn.bz2)|*.scn;*.hdr;*.scn.bz2|"
                                     "Scene files (*.scn)|*.scn|"
                                     "Analyze files (*.hdr)|*.hdr|"
                                     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
                                     "Nifti-1 files (*.nii)|*.nii|"
                                     "Compressed Nifti-1 files (*.nii.gz)|*.nii.gz|"
                                     "All files|*.*"
    );

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    wxString wxlabelname((APP->Data).labelname, wxConvUTF8);

    wxFileDialog *saveFileDialog =
            new wxFileDialog ( this,
                               _T("Save Object Map"),
                               _T(""),
                               wxlabelname,
                               FILETYPES,
                               wxFD_SAVE | wxFD_OVERWRITE_PROMPT,
                               wxDefaultPosition);

    saveFileDialog->SetFilterIndex((APP->Preferences).ext_SaveObjectMap);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        saveFileDialog->SetDirectory(wxdir);
    }
    if (saveFileDialog->ShowModal() == wxID_OK)
    {
        wxBusyCursor wait;
        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename());

        (APP->Preferences).ext_SaveObjectMap = saveFileDialog->GetFilterIndex();

        char path[512];
        strcpy( dir,  saveFileDialog->GetDirectory().ToAscii() );
        strcpy( path, saveFileDialog->GetPath().ToAscii() );

        // Set the extension if missing
        filter = saveFileDialog->GetFilterIndex();
        CheckFileExtension(path,filter);

        // Save
        iftImage *img = SceneToImage(APP->Data.objmap);
        iftWriteImageByExt(img,path);
        iftDestroyImage(&img);

        /*

        if (strcasecmp(path + s - 4, ".hdr") == 0 || strcasecmp(path + s - 4, ".img") == 0) {
            objmap->dx = APP->Data.orig->dx;
            objmap->dy = APP->Data.orig->dy;
            objmap->dz = APP->Data.orig->dz;
            WriteAnalyze(objmap, path);
        }
        else if (strcasecmp(path + s - 4, ".scn") == 0 || strcasecmp(path + s - 8, ".scn.bz2") == 0 || strcasecmp(path + s - 4, ".nii") == 0 || strcasecmp(path + s - 7, ".nii.gz") == 0){
            objmap->dx = APP->Data.orig->dx;
            objmap->dy = APP->Data.orig->dy;
            objmap->dz = APP->Data.orig->dz;
            WriteScene(objmap, path);
        }
        else
        {
            wxMessageBox(_T("Unsupported extension!"), _T("Error"), wxOK | wxICON_EXCLAMATION, this);
            DestroyScene(&objmap);
            return;
        }

        DestroyScene(&objmap);
         */

        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename() + _T("... Done"));
    }
}


void MyFrame::OnSaveMarkers(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_SaveMarkers;
    int p, n, s, filter, nseeds, id, lb, x, y, z;

    static
    wxChar
            *FILETYPES = _T( "Text files (.TXT)|*.txt|"
                                     "All files|*.*"
    );

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    wxFileDialog *saveFileDialog =
            new wxFileDialog ( this,
                               _T("Save Markers"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_SAVE | wxFD_OVERWRITE_PROMPT,
                               wxDefaultPosition);

    saveFileDialog->SetFilterIndex((APP->Preferences).ext_SaveMarkers);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        saveFileDialog->SetDirectory(wxdir);
    }
    if (saveFileDialog->ShowModal() == wxID_OK) {
        wxBusyCursor wait;
        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename());

        (APP->Preferences).ext_SaveMarkers = saveFileDialog->GetFilterIndex();

        char path[512];
        char *out_path = NULL;
        strcpy(dir, saveFileDialog->GetDirectory().ToAscii());
        strcpy(path, saveFileDialog->GetPath().ToAscii());

        s = strlen(path);
        filter = saveFileDialog->GetFilterIndex();
        if ((s < 4 || !iftEndsWith(path, ".txt")) && filter == 0){
            out_path = iftConcatStrings(2, path, ".txt");
            strcpy(path, out_path);
            iftFree(out_path);
        }

        n = APP->Data.w * APP->Data.h * APP->Data.nframes;
        iftLabeledSet *S = NULL;
        for (p = 0; p < n; p++)
        {
            if (APP->IsSeed(p))
            {
                if (APP->IsMarkedForRemoval(p))
                    lb = -1;
                else
                    lb = APP->GetSeedLabel(p);

                id = APP->GetSeedId(p);

                iftInsertLabeledSetMarkerAndHandicap(&S,p,lb,id,0);
            }
        }

        iftImage *img = SceneToImage(APP->Data.orig);
        iftWriteSeeds(S,img,path);
        iftDestroyLabeledSet(&S);
        iftDestroyImage(&img);

        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename() + _T("... Done"));
    }
}


void MyFrame::OnLoadMarkers(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_LoadMarkers;
    FILE *fp;

    static
    wxChar
            *FILETYPES = _T( "Text files (.TXT)|*.txt|"
                                     "All files|*.*"
    );

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    wxFileDialog *openFileDialog =
            new wxFileDialog ( this,
                               _T("Load Markers"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_OPEN ,
                               wxDefaultPosition);

    openFileDialog->SetFilterIndex((APP->Preferences).ext_LoadMarkers);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        openFileDialog->SetDirectory(wxdir);
    }
    if (openFileDialog->ShowModal() == wxID_OK)
    {
        wxBusyCursor wait;
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename());

        (APP->Preferences).ext_LoadMarkers = openFileDialog->GetFilterIndex();

        char path[512];
        strcpy( dir,  openFileDialog->GetDirectory().ToAscii() );
        strcpy( path, openFileDialog->GetPath().ToAscii() );

        LoadMarkers(path);
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename() + _T("... Done"));
    }
}

void MyFrame::LoadMarkers(char *seeds_path)
{
	int i, p, nseeds, x, y, z, id, lb, handicap;
    iftVoxel u;

	iftImage *img = SceneToImage(APP->Data.orig);
    iftLabeledSet *S = iftReadSeeds(img,seeds_path);

    /*
    fp = fopen(path, "r");
    if (!fp)
    {
        wxMessageBox(_T("Error opening file!"), _T("Error"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }*/

    APP->ResetData();

    //fscanf(fp, " %d \n", &nseeds);

    while (S != NULL){
        p = S->elem;
        lb = S->label;
        id = S->marker;
        handicap = S->handicap;
        u = iftGetVoxelCoord(img,p);
        if (iftValidVoxel(img,u)){
            APP->AddSeed(p, lb, id);

            int cod = APP->GetMarkerValue(id, lb);

            (APP->Data.label)->data[p] = cod;

            if (lb == 0)
                APP->SetLabelColour(cod, NIL);
            else
                APP->SetLabelColour(cod, APP->GetLabelColour(lb));
        }
        S = S->next;
    }
    iftDestroyLabeledSet(&S);
    iftDestroyImage(&img);
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
}

void MyFrame::OnNewProj(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_LoadVolume;
    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.bia;*.scn;*.hdr;*.scn.bz2)|*.bia;*.scn;*.hdr;*.scn.bz2|"
                                     "BIA files (*.bia)|*.bia|"
                                     "Scene files (*.scn)|*.scn|"
                                     "Analyze files (*.hdr)|*.hdr|"
                                     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
                                     "Nifti-1 files (*.nii)|*.nii|"
                                     "Compressed Nifti-1 files (*.nii.gz)|*.nii.gz|"
                                     "All files|*.*"
    );

    if ( !modManager->StopLastActiveProcessingModule() ) return;

    wxFileDialog *openFileDialog =
            new wxFileDialog ( this,
                               _T("Select volume to IMPORT to project..."),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_OPEN ,
                               wxDefaultPosition);

    openFileDialog->SetFilterIndex((APP->Preferences).ext_LoadVolume);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        openFileDialog->SetDirectory(wxdir);
    }
    if (openFileDialog->ShowModal() == wxID_OK)
    {
        if (!APP->Data.loaded)
        {
            APP->DestroyData();
            APP->Clear2DCanvas();
            APP->InitData();
        }

        wxWindowDisabler disableAll;
        wxBusyInfo busy(_T("Please wait, working..."));
        wxBusyCursor wait;
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename());
        wxTheApp->Yield();
        //printf("%s\n",openFileDialog->GetFilename().ToAscii());
        //printf("%s\n",openFileDialog->GetDirectory().ToAscii());
        (APP->Preferences).ext_LoadVolume = openFileDialog->GetFilterIndex();

        char path[512];
        strcpy( dir,  openFileDialog->GetDirectory().ToAscii() );
        strcpy( path, openFileDialog->GetPath().ToAscii() );
        //strcpy(APP->Data.volumename, openFileDialog->GetFilename().ToAscii());
        LoadVolume(path);
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename() + _T("... Done"));

        strcpy(APP->Data.projectfilename, "");
        wxCursor *cross = NULL;
        float zoom = APP->GetZoomLevel();
        cross = CrossCursor(ROUND(zoom));
        //wxCROSS_CURSOR
        APP->Set2DViewCursor(cross, 'x');
        APP->Set2DViewCursor(cross, 'y');
        APP->Set2DViewCursor(cross, 'z');

    }
}


void MyFrame::OnOpenProj(wxCommandEvent &event)
{

    char *dir = (APP->Preferences).dir_LoadVolume;

    static   wxChar
            *FILETYPES = _T( "BIA files (*.bia)|*.bia|"
                                     "All files|*.*"
    );


    if ( !modManager->StopLastActiveProcessingModule() ) return;

    wxFileDialog *openFileDialog =
            new wxFileDialog ( this,
                               _T("Open Project"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_OPEN  | wxFD_OVERWRITE_PROMPT,
                               wxDefaultPosition);

    openFileDialog->SetFilterIndex((APP->Preferences).ext_LoadVolume);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        openFileDialog->SetDirectory(wxdir);
    }
    if (openFileDialog->ShowModal() == wxID_OK)
    {

        if (!APP->Data.loaded)
        {
            APP->DestroyData();
            APP->Clear2DCanvas();
            APP->InitData();
        }

        wxWindowDisabler disableAll;
        wxBusyInfo busy(_T("Please wait, working..."));
        wxBusyCursor wait;
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename());
        wxTheApp->Yield();
        (APP->Preferences).ext_LoadVolume = openFileDialog->GetFilterIndex();

        char path[512];
        strcpy( dir,  openFileDialog->GetDirectory().ToAscii() );
        strcpy( path, openFileDialog->GetPath().ToAscii() );
        OpenProj(path);
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename() + _T("... Done"));
        wxCursor *cross = NULL;
        float zoom = APP->GetZoomLevel();
        cross = CrossCursor(ROUND(zoom));
        //wxCROSS_CURSOR
        APP->Set2DViewCursor(cross, 'x');
        APP->Set2DViewCursor(cross, 'y');
        APP->Set2DViewCursor(cross, 'z');
    }
}


void MyFrame::OpenProj(char *filename)
{

    Scene *scn;
    int modality;
    int *scn_flags = ( int *) calloc( 10, sizeof( int ) );
    if (BIA_t1_v3_ReadScene(&scn, filename, scn_flags) != 0)
    {
        if (BIA_t1_v2_ReadScene(&scn, filename, &APP->Data.oriented, &APP->Data.aligned, &modality) != 0)
        {
            if (BIA_t1_v1_ReadScene(&scn, filename, &APP->Data.oriented) != 0)
            {
                wxMessageBox(_T("Corrupted file!"), _T("Load Volume Error"),
                             wxOK | wxICON_EXCLAMATION, this);
                return;
            }
        }
        else
            APP->Data.modality = (ModalityType) modality;
    }
    else
    {
        APP->Data.oriented = scn_flags[ 0 ];
        APP->Data.aligned = scn_flags[ 1 ];
        APP->Data.modality = ( ModalityType ) scn_flags[ 2 ];
        APP->Data.corrected = scn_flags[ 3 ];
        free(scn_flags);
    }

    //printf("corrected:%d\n",APP->Data.corrected);
    strcpy(APP->Data.projectfilename, filename);
    char *aux = (char *) calloc(1024, sizeof(char));
    sprintf(aux, "VISVA - Volumetric Image Segmentation for Visualization and Analysis - %s", APP->Data.projectfilename);
    wxString wxfilename(aux, wxConvUTF8);
    free(aux);
    SetTitle( wxfilename );

    APP->SetDataVolume(scn);

    BIA_t3_v1_ReadSegmObjs(filename, &(APP->segmobjs));
    APP->DrawSegmObjects();
    BIA_t4_v1_ReadMeasures(filename, &APP->measures);
    BIA_t5_v1_ReadMSP(filename, &APP->Data.msp);

    APP->Data.loaded = 1;


    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    MaximumValue3( APP->Data.orig );
    if ( APP->Data.orig->maxval == 1 )
        APP->Data.modality = BIN_BINARYMASK;
    else if ( APP->Data.modality == UNKNOWN_PROTOCOL )
    {
        //wxMessageBox(_T("The imaging modality was not specified\n on the header of this image.\n"), _T("Warning"), wxOK | wxICON_EXCLAMATION, this);
        APP->ShowImageModalityDialog();
    }

}





void MyFrame::OnSaveProj(wxCommandEvent &event)
{

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is nothing to save!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    if (strcmp(APP->Data.projectfilename, "") == 0)
    {
        char *dir = (APP->Preferences).dir_SaveVolume;
        int s, filter;

        static   wxChar
                *FILETYPES = _T( "BIA files (*.bia)|*.bia|"
                                         "All files|*.*"
        );


        wxFileDialog *saveFileDialog =
                new wxFileDialog ( this,
                                   _T("Save Project"),
                                   _T(""),
                                   _T(""),
                                   FILETYPES,
                                   wxFD_SAVE | wxFD_OVERWRITE_PROMPT,
                                   wxDefaultPosition);

        saveFileDialog->SetFilterIndex((APP->Preferences).ext_SaveVolume);
        if (dir[0] != '\0')
        {
            wxString wxdir(dir, wxConvUTF8);
            saveFileDialog->SetDirectory(wxdir);
        }
        if (saveFileDialog->ShowModal() == wxID_OK)
        {
            wxBusyCursor wait;
            SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename());
            (APP->Preferences).ext_SaveVolume = saveFileDialog->GetFilterIndex();
            strcpy( dir,  saveFileDialog->GetDirectory().ToAscii() );
            strcpy(APP->Data.projectfilename, saveFileDialog->GetPath().ToAscii() );

            // Set the extension if missing
            s = strlen(APP->Data.projectfilename);
            filter = saveFileDialog->GetFilterIndex();
            if ( (s < 3 || strcasecmp(APP->Data.projectfilename + s - 3, "bia") != 0) && filter == 0 )
                strcat(APP->Data.projectfilename, ".bia");

        }
    }
    SaveProj(APP->Data.projectfilename);



    //SetStatusText(_T("Saving file ")+wxString(APP->Data.projectfilename)+_T("... Done"));


}



void MyFrame::OnSaveProjAs(wxCommandEvent &event)
{
    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is nothing to save!"), _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    char *dir = (APP->Preferences).dir_SaveVolume;
    int s, filter;

    static   wxChar
            *FILETYPES = _T( "BIA files (*.bia)|*.bia|"
                                     "All files|*.*"
    );


    wxFileDialog *saveFileDialog =
            new wxFileDialog ( this,
                               _T("Save Project"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_SAVE | wxFD_OVERWRITE_PROMPT,
                               wxDefaultPosition);

    saveFileDialog->SetFilterIndex((APP->Preferences).ext_SaveVolume);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        saveFileDialog->SetDirectory(wxdir);
    }
    if (saveFileDialog->ShowModal() == wxID_OK)
    {
        wxBusyCursor wait;
        SetStatusText(_T("Saving file ") + saveFileDialog->GetFilename());
        (APP->Preferences).ext_SaveVolume = saveFileDialog->GetFilterIndex();
        //strcpy( dir,  saveFileDialog->GetDirectory().ToAscii() );
        strcpy(APP->Data.projectfilename, saveFileDialog->GetPath().ToAscii() );

        // Set the extension if missing
        s = strlen(APP->Data.projectfilename);
        filter = saveFileDialog->GetFilterIndex();
        if ( (s < 3 || strcasecmp(APP->Data.projectfilename + s - 3, "bia") != 0) && filter == 0 )
            strcat(APP->Data.projectfilename, ".bia");

    }

    SaveProj(APP->Data.projectfilename);

    //SetStatusText(_T("Saving file ")+wxString(APP->Data.projectfilename)+_T("... Done"));

}



void MyFrame::SaveProj(char *filename)
{
    BIA_MainHeaderCreate(filename);
    int *scn_flags = (int *) calloc( 4, sizeof( int ) );
    scn_flags[ 0 ] = APP->Data.oriented;
    scn_flags[ 1 ] = APP->Data.aligned;
    scn_flags[ 2 ] = APP->Data.modality;
    scn_flags[ 3 ] = APP->Data.corrected;

    BIA_t1_v3_WriteScene(APP->Data.orig, filename, scn_flags);
    BIA_t3_v1_WriteSegmObjs(filename, APP->segmobjs);
    BIA_t4_v1_WriteMeasures(filename, APP->measures);
    BIA_t5_v1_WriteMSP(filename, APP->Data.msp);
    BIA_MainHeaderUpdate(filename);

    SetStatusText(_T("Project saved. "));

    char *aux = (char *) calloc(1024, sizeof(char));
    sprintf(aux, "VISVA - Volumetric Image Segmentation for Visualization and Analysis - %s", APP->Data.projectfilename);
    wxString wxfilename(aux, wxConvUTF8);
    free(aux);
    free(scn_flags);
    SetTitle( wxfilename );

}






void MyFrame::OnLoadVolume(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_LoadVolume;
    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.bia;*.scn;*.hdr;*.scn.bz2)|*.bia;*.scn;*.hdr;*.scn.bz2|"
                                     "BIA files (*.bia)|*.bia|"
                                     "Scene files (*.scn)|*.scn|"
                                     "Analyze files (*.hdr)|*.hdr|"
                                     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
                                     "Nifti-1 files (*.nii)|*.nii|"
                                     "Compressed Nifti-1 files (*.nii.gz)|*.nii.gz|"
                                     "All files|*.*"
    );


    if ( !modManager->StopLastActiveProcessingModule() )
    {
        printf( "Error: StopLastActiveProcessingModule!\n" );
        return;
    }

    wxFileDialog *openFileDialog =
            new wxFileDialog ( this,
                               _T("Open file"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_OPEN ,
                               wxDefaultPosition);

    openFileDialog->SetFilterIndex((APP->Preferences).ext_LoadVolume);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        openFileDialog->SetDirectory(wxdir);
    }
    if (openFileDialog->ShowModal() == wxID_OK)
    {
        wxWindowDisabler disableAll;
        wxBusyInfo busy(_T("Please wait, working..."));
        wxBusyCursor wait;
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename());
        wxTheApp->Yield();
        //printf("%s\n",openFileDialog->GetFilename().ToAscii());
        //printf("%s\n",openFileDialog->GetDirectory().ToAscii());
        (APP->Preferences).ext_LoadVolume = openFileDialog->GetFilterIndex();

        char path[512];
        strcpy( dir,  openFileDialog->GetDirectory().ToAscii() );
        strcpy( path, openFileDialog->GetPath().ToAscii() );
        //strcpy(APP->Data.volumename, openFileDialog->GetFilename().ToAscii());
        LoadVolume(path);
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename() + _T("... Done"));

        wxCursor *cross = NULL;
        float zoom = APP->GetZoomLevel();
        cross = CrossCursor(ROUND(zoom));
        //wxCROSS_CURSOR
        APP->Set2DViewCursor(cross, 'x');
        APP->Set2DViewCursor(cross, 'y');
        APP->Set2DViewCursor(cross, 'z');
    }
}



void MyFrame::LoadVolume(char *filename)
{
    Scene *scn = NULL;
    iftImage *img = NULL;

    APP->DestroyData();

    MyRemoveDirectory(APP->Data.volumename, filename);


//    char *dir = iftParentDir(filename);
//    img = iftReadImageFolderAsVolume(dir);
//    iftFree(dir);
//
//    scn = APP->iftImageToScene(img);
//    iftDestroyImage(&img);

    if (iftEndsWith(filename, ".bia")) {
        int modality;
        int *scn_flags = ( int *) calloc( 10, sizeof( int ) );
        printf("If 1\n");
        if (BIA_t1_v3_ReadScene(&scn, filename, scn_flags) != 0)
        {
            printf("If 2\n");
            if (BIA_t1_v2_ReadScene(&scn, filename, &APP->Data.oriented, &APP->Data.aligned, &modality) != 0)
            {
                printf("If 3\n");
                if (BIA_t1_v1_ReadScene(&scn, filename, &APP->Data.oriented) != 0)
                {
                    printf("If 4\n");
                    wxMessageBox(_T("Corrupted file!"), _T("Load Volume Error"),
                                 wxOK | wxICON_EXCLAMATION, this);
                    return;
                }
            }
            else
            {
                APP->Data.modality = (ModalityType) modality;
            }
        }
        else
        {
            APP->Data.oriented = scn_flags[ 0 ];
            APP->Data.aligned = scn_flags[ 1 ];
            APP->Data.modality = ( ModalityType ) scn_flags[ 2 ];
            APP->Data.corrected = scn_flags[ 3 ];
            free(scn_flags);
        }
    }
    /* Changed by Samuka Martins - Using the new functions to read an Analyze in libIFT */
    else if (iftRegexMatch(filename, "^.+\\.(nii|nii.gz|hdr|scn|scn.bz2)$")) {
        iftImage *img = iftReadImageByExt(filename);
        
        if (img == NULL)
        {
            wxMessageBox(_T("Corrupted file!"), _T("Load Volume Error"),
                         wxOK | wxICON_EXCLAMATION, this);
            return;
        }

        // scn = APP->iftImageToScene(img);
        scn = ImageToScene(img);
        iftDestroyImage(&img);
    }
    else if (iftEndsWith(filename, ".npy")) {
        iftFImage *fimg = iftReadFImage(filename);
        iftImage   *img = iftFImageToImage(fimg, 255);
        iftDestroyFImage(&fimg);

        scn = CreateScene(img->xsize, img->ysize, img->zsize);
        free(scn->data);
        scn->dx = img->dx;
        scn->dy = img->dy;
        scn->dz = img->dz;
        scn->data = img->val;
        img->val = NULL;
        iftDestroyImage(&img);
    }
    else
    {
        wxMessageBox(_T("Unable to load volume data. Invalid file format!"), _T("Load Volume Error"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    char *aux = (char *) calloc(1024, sizeof(char));
    if (strcmp(APP->Data.projectfilename, "") == 0)
        sprintf(aux, "VISVA - Volumetric Image Segmentation for Visualization and Analysis - %s (Untitled Project)", filename);
    else
        sprintf(aux, "VISVA - Volumetric Image Segmentation for Visualization and Analysis - %s", APP->Data.projectfilename);
    wxString wxfilename(aux, wxConvUTF8);
    free(aux);
    SetTitle( wxfilename );


    //scn = APP->iftImageToScene(img);
    //iftImageForest *fst = iftCreateImageForest(iftCreateImage(100, 100, 100), iftSpheric(1));

    APP->SetDataVolume(scn);
    APP->Data.loaded = 1;
    APP->Data.oriented = 0;
    APP->Data.modality = T1_WEIGHTED; /*UNKNOWN_PROTOCOL (default, if set, dialog box will open)*/

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);

    if ( APP->Data.orig->maxval == 1 )
        APP->Data.modality = BIN_BINARYMASK;
    else if ( APP->Data.modality == UNKNOWN_PROTOCOL )
    {
        //wxMessageBox(_T("The imaging modality was not specified\n on the header of this image.\n"),_T("Warning"),wxOK | wxICON_EXCLAMATION, this);
        APP->ShowImageModalityDialog();
    }
}


void MyFrame::OnLoadScnGradient(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_LoadVolume;
    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.sgr;*.sgr.bz2)|*.sgr;*.sgr.bz2|"
                                     "Scene files (*.sgr)|*.sgr|"
                                     "Compressed Scene files (*.sgr.bz2)|*.sgr.bz2|"
                                     "All files|*.*"
    );

    if ( !modManager->StopLastActiveProcessingModule() ) return;

    wxFileDialog *openFileDialog =
            new wxFileDialog ( this,
                               _T("Open gradient"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_OPEN ,
                               wxDefaultPosition);

    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        openFileDialog->SetDirectory(wxdir);
    }
    if (openFileDialog->ShowModal() == wxID_OK)
    {
        wxWindowDisabler disableAll;
        wxBusyInfo busy(_T("Please wait, working..."));
        wxBusyCursor wait;
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename());
        wxTheApp->Yield();

        char path[512];
        strcpy( dir,  openFileDialog->GetDirectory().ToAscii() );
        strcpy( path, openFileDialog->GetPath().ToAscii() );
        LoadScnGradient(path);
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename() + _T("... Done"));

        wxCursor *cross = NULL;
        float zoom = APP->GetZoomLevel();
        cross = CrossCursor(ROUND(zoom));
        //wxCROSS_CURSOR
        APP->Set2DViewCursor(cross, 'x');
        APP->Set2DViewCursor(cross, 'y');
        APP->Set2DViewCursor(cross, 'z');
    }
}


void MyFrame::LoadScnGradient(char *filename)
{
    ScnGradient *grad = NULL;
    Scene *scn = NULL;
    int s;

    APP->DestroyData();

//    MyRemoveDirectory(APP->Data.volumename, filename);

    s = strlen(filename);
    if ( strcasecmp(filename + s - 3, "scn") == 0 )
        grad = ReadScnGradient( filename );
    else if ( strcasecmp(filename + s - 3, "bz2") == 0 )
    {
        grad = ReadCompressedScnGradient(filename);
        if (grad == NULL)
        {
            wxMessageBox(_T("Corrupted file!"), _T("Load Gradient Error"),
                         wxOK | wxICON_EXCLAMATION, this);
            return;
        }
    }
    else
    {
        wxMessageBox(_T("Unable to load gradient data. Invalid file format!"),
                     _T("Load Gradient Error"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    char *aux = (char *) calloc(1024, sizeof(char));
    if (strcmp(APP->Data.projectfilename, "") == 0)
        sprintf(aux, "VISVA - Volumetric Image Segmentation for Visualization and Analysis %s (Untitled Project)", filename);
    else
        sprintf(aux, "VISVA - Volumetric Image Segmentation for Visualization and Analysis %s", APP->Data.projectfilename);
    wxString wxfilename(aux, wxConvUTF8);
    free(aux);
    SetTitle( wxfilename );

    ComputeScnGradientMagnitude(grad);
    ScnGradientMaximumMag(grad);
    scn = CopyScene(grad->mag);
    APP->SetDataVolume(scn);
    APP->Data.loaded = 1;
    APP->Data.oriented = 0;
    APP->Data.modality = UNKNOWN_PROTOCOL;
    DestroyScnGradient(&(APP->Data.grad));
    APP->Data.grad = grad;

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
}

void MyFrame::OnLoadArcWeight(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_LoadVolume;
    static
    wxChar
            *FILETYPES = _T( "Supported Formats (*.scn)|*.scn|"
                                     "All files|*.*"
    );

    if ( !modManager->StopLastActiveProcessingModule() ) return;

    wxFileDialog *openFileDialog =
            new wxFileDialog ( this,
                               _T("Open ArcWeight"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_OPEN ,
                               wxDefaultPosition);

    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        openFileDialog->SetDirectory(wxdir);
    }
    if (openFileDialog->ShowModal() == wxID_OK)
    {
        wxWindowDisabler disableAll;
        wxBusyInfo busy(_T("Please wait, working..."));
        wxBusyCursor wait;
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename());
        wxTheApp->Yield();

        char path[512];
        strcpy( dir,  openFileDialog->GetDirectory().ToAscii() );
        strcpy( path, openFileDialog->GetPath().ToAscii() );
        LoadArcWeight(path);
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename() + _T("... Done"));

        wxCursor *cross = NULL;
        float zoom = APP->GetZoomLevel();
        cross = CrossCursor(ROUND(zoom));
        //wxCROSS_CURSOR
        APP->Set2DViewCursor(cross, 'x');
        APP->Set2DViewCursor(cross, 'y');
        APP->Set2DViewCursor(cross, 'z');
    }
}

void MyFrame::LoadArcWeight(char *filename)
{
    Scene *scn = NULL;

    if (iftRegexMatch(filename, "^.+\\.(nii|nii.gz|hdr|scn|scn.bz2)$")) {
        iftImage *img = iftReadImageByExt(filename);
        scn = ImageToScene(img);
        iftDestroyImage(&img);
    }
    else
    {
        wxMessageBox(_T("Unable to load gradient data. Invalid file format!"),
                     _T("Load Gradient Error"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    char *aux = (char *) calloc(1024, sizeof(char));
    if (strcmp(APP->Data.projectfilename, "") == 0)
        sprintf(aux, "VISVA - Volumetric Image Segmentation for Visualization and Analysis %s (Untitled Project)", filename);
    else
        sprintf(aux, "VISVA - Volumetric Image Segmentation for Visualization and Analysis %s", APP->Data.projectfilename);
    wxString wxfilename(aux, wxConvUTF8);
    free(aux);
    SetTitle( wxfilename );

    //Put the arc in the APP->Data.arcweight variable. Cross your fingers
    if (APP->Data.arcw)
        DestroyScene(&(APP->Data.arcw));
    // APP->Data.arcw = APP->iftImageToScene(arc);
    APP->Data.arcw = scn;

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
}


void MyFrame::OnLoadLabel(wxCommandEvent &event)
{
    char *dir = (APP->Preferences).dir_LoadLabel;
    static
    wxChar
            *FILETYPES = _T( "Scene files (*.scn)|*.scn|"
                                     "Analyze files (*.hdr)|*.hdr|"
                                     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
                                     "Nifti-1 files (*.nii)|*.nii|"
                                     "Compressed Nifti-1 files (*.nii.gz)|*.nii.gz|"
                                     "All files|*.*"
    );

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded!"),
                     _T("No Data"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    if ( !modManager->StopLastActiveProcessingModule() ) return;

    wxFileDialog *openFileDialog =
            new wxFileDialog ( this,
                               _T("Load Label"),
                               _T(""),
                               _T(""),
                               FILETYPES,
                               wxFD_OPEN ,
                               wxDefaultPosition);

    openFileDialog->SetFilterIndex((APP->Preferences).ext_LoadLabel);
    if (dir[0] != '\0')
    {
        wxString wxdir(dir, wxConvUTF8);
        openFileDialog->SetDirectory(wxdir);
    }
    if (openFileDialog->ShowModal() == wxID_OK)
    {
        wxWindowDisabler disableAll;
        wxBusyInfo busy(_T("Please wait, working..."));
        wxBusyCursor wait;
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename());
        wxTheApp->Yield();
        //printf("%s\n",openFileDialog->GetFilename().ToAscii());
        //printf("%s\n",openFileDialog->GetDirectory().ToAscii());
        (APP->Preferences).ext_LoadLabel = openFileDialog->GetFilterIndex();

        char path[512];
        strcpy( dir,  openFileDialog->GetDirectory().ToAscii() );
        strcpy( path, openFileDialog->GetPath().ToAscii() );
        //strcpy(APP->Data.volumename, openFileDialog->GetFilename().ToAscii());
        LoadLabel(path, NULL, 0);
        SetStatusText(_T("Loading ") + openFileDialog->GetFilename() + _T("... Done"));
        strcpy(APP->Data.labelname, openFileDialog->GetFilename().ToAscii());
    }
}


void MyFrame::LoadLabel(char *filename, char *name, int color)
{
    Scene *label;
    int p, n, lb, l, Lmax;
    char oname[512];
    int *hist = NULL;
    if (name != NULL) strcpy(oname, name);

    if (!APP->Data.loaded)
        return;

    /* Changed by Samuka Martins - Using the new functions to read an Analyze in libIFT */
    if (iftRegexMatch(filename, "^.+\\.(nii|nii.gz|hdr|scn|scn.bz2)$")) {
        iftImage *img = iftReadImageByExt(filename);
        label = ImageToScene(img);
        iftDestroyImage(&img);
    }
    else
    {
        wxMessageBox(_T("Unable to load volume data. Invalid file format!"),
                     _T("Load Label Error"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    if (label->xsize != APP->Data.w ||
        label->ysize != APP->Data.h ||
        label->zsize != APP->Data.nframes)
    {
        DestroyScene(&label);
        wxMessageBox(_T("Incompatible volume data."),
                     _T("Load Label Error"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    //----------------------------------------------

    // if(name==NULL){
    //   int r = APP->ShowNewObjDialog(&color, oname);
    //   if(r) return;
    // }

    // APP->DelAllSeeds();
    // n = APP->Data.w*APP->Data.h*APP->Data.nframes;
    // for(p=0; p<n; p++){
    //   lb = label->data[p];
    //   (APP->Data.label)->data[p] = lb;
    // }
    // APP->SetLabelColour(0, NIL);
    // APP->SetLabelColour(1, color);
    // DestroyScene(&label);
    // APP->AddObj(oname, color);
    //----------------------------------------------

    APP->DelAllSeeds();
    n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    Lmax = MaximumValue3(label);
    hist = AllocIntArray(Lmax + 1);
    for (p = 0; p < n; p++)
    {
        lb = label->data[p];
        hist[lb]++;
    }
    
    iftColorTable *ct = iftCategoricalColorTable(Lmax);

    for (l = 1; l <= Lmax; l++)
    {
        if (hist[l] == 0) continue;

        /** CHANGED BY Samuel - Now, if the mask is binary, visva will generate a random color for it */
        /***********************************************************************/
        // if (name == NULL && Lmax == 1)
        // {
        //     printf("********************************************\n");
        //     exit(-1);
        //     if (APP->ShowNewObjDialog(&color, oname))
        //         continue;
        // }
        // else
        // {
            // sprintf(oname, "Label%02d", l);
            // color = randomColor(); //0xFFFF00;
        // }

        // GAMBIARRA - do not allow 0-values in the channels in order to make the color visualization better
        iftColor RGB = iftYCbCrtoRGB(ct->color[l-1], 255);
        sprintf(oname, "Label%02d", l);
        // int R = ct->color[l-1].val[0] == 0 ? 127 : ct->color[l-1].val[0];
        // int G = ct->color[l-1].val[1] == 0 ? 127 : ct->color[l-1].val[1];
        // int B = ct->color[l-1].val[2] == 0 ? 127 : ct->color[l-1].val[2];

        int R = RGB.val[0] == 0 ? 127 : RGB.val[0];
        int G = RGB.val[1] == 0 ? 127 : RGB.val[1];
        int B = RGB.val[2] == 0 ? 127 : RGB.val[2];
        color = triplet(R,G,B);
        /***********************************************************************/


        SetScene(APP->Data.label, 0);
        for (p = 0; p < n; p++)
        {
            lb = label->data[p];
            if (l == lb)
                (APP->Data.label)->data[p] = lb;
        }
        APP->SetLabelColour(0, NIL);
        APP->SetLabelColour(l, color);
        APP->AddObj(oname, color);
    }
    iftDestroyColorTable(&ct);

    for (l = 1; l <= Lmax; l++)
    {
        sprintf(oname, "Label%02d", l);
        APP->SetObjVisibility(oname, true);
    }

    free(hist);
    DestroyScene(&label);
    //----------------------------------------------
    APP->DrawSegmObjects();
}


void MyFrame::OnQuit(wxCommandEvent &WXUNUSED(event))
{
Close(TRUE);
}


void MyFrame::OnAbout(wxCommandEvent &WXUNUSED(event))
{
static const char *lic_unicamp = "RESTRICTED FOR RESEARCH AND EDUCATIONAL USAGE\n"\
                                     "IN UNICAMP FACILITIES.\n"\
                                     "Terms of use: this release of VISVA is available for research\n"\
                                     "and educational usage inside Unicamp only. The authors grant\n"\
                                     "permission to copy and install this software in computers\n"\
                                     "inside the campus. Users are not authorized to copy, sell, trade\n"\
                                     "or redistribute this software to/with third parties. The source\n"\
                                     "code, if made available, is for the convenience of compiling on\n"\
                                     "multiple platforms only. Derivative works and redistributions of\n"\
                                     "the source are not allowed at all.";

char z[2048];

sprintf(z, "VISVA version 1.0\nEdition/Distribution: unrestricted\n\n");

strcat(z,
"Volumetric Image Segmentation for Visualization and Analysis\n\n"\
           "(C) 2005-2009 Paulo A.V. Miranda and Guilherme C. S. Ruppert and Felipe P.G. Bergo and Alexandre X. Falcao and Fabio Augusto Menocci Cappabianco\n"\
           "Developed by Paulo A.V. Miranda and Guilherme C. S. Ruppert and Felipe P.G. Bergo and Alexandre X. Falcao and Fabio Augusto Menocci Cappabianco\n"\
           "at the Institute of Computing, Unicamp, Campinas, Brazil.\n\n"\
           "Website: http://www.ic.unicamp.br/~afalcao\n"\
           "Contact: pavmbr@yahoo.com.br or guilhermeruppert@gmail.com or bergo@seul.org or afalcao@ic.unicamp.br or fcappabianco@gmail.com\n\n");

strcat(z, lic_unicamp);

wxString wxz(z, wxConvUTF8);
wxMessageBox(wxz, _T("About BIA"), wxOK | wxICON_INFORMATION, this);
}



void MyFrame::OnVolumeInformation(wxCommandEvent &WXUNUSED(event))
{
char z[1024];
char textline[512];
float dx, dy, dz, volume;
int n, bpp, levels, Imax, Imin;

if (!APP->Data.loaded)
{
wxMessageBox(_T("There is no volume loaded."),
        _T("Volume Information"), wxOK | wxICON_EXCLAMATION, this);
}
else
{
sprintf(z, "Filename: %s\n\n", APP->Data.volumename);
sprintf(textline, "Imaging modality: %s\n\n", ModalityNames[APP->Data.modality]);
strcat(z, textline);

dx = (APP->Data.orig)->dx;
dy = (APP->Data.orig)->dy;
dz = (APP->Data.orig)->dz;
sprintf(textline, "Voxel size: %.6f x %.6f x %.6f = %.6f units^3\n\n", dx, dy, dz, dx * dy * dz);
strcat(z, textline);

n = APP->Data.w * APP->Data.h * APP->Data.nframes;
volume = (float)n * dx * dy * dz;
sprintf(textline, "Volume size: %d x %d x %d = %d voxels\n", APP->Data.w, APP->Data.h, APP->Data.nframes, n);
strcat(z, textline);
sprintf(textline, "\t\t(%.2f units^3)\n\n", volume);
strcat(z, textline);

Imax = MaximumValue3(APP->Data.orig);
Imin = MinimumValue3(APP->Data.orig);
bpp  = GetRadiometricRes3(APP->Data.orig);
levels = (int)powf(2.0, (float)bpp);
sprintf(textline, "Radiometric resolution: %d levels (%d bits)\n", levels, bpp);
strcat(z, textline);

sprintf(textline, "Maximum Value: %d, Minimum Value: %d\n", Imax, Imin);
strcat(z, textline);

wxString wxz(z, wxConvUTF8);
wxMessageBox(wxz, _T("Volume Information"), wxOK | wxICON_INFORMATION, this);
}
}


void MyFrame::OnParamEstimation(wxCommandEvent &WXUNUSED(event))
{
char z[1024];
char textline[512];
MRI_Info info;
AdjRel *A;
int p;

if (!APP->Data.loaded)
{
wxMessageBox(_T("There is no volume loaded."),
        _T("Parameters Estimation"), wxOK | wxICON_EXCLAMATION, this);
}
else
{
APP->Busy((char *)"Please wait, computing MRI information...");
info = EstimateMRI_Information(APP->Data.orig);

sprintf(z, "Aligned: %d\n\n", APP->Data.aligned);
sprintf(textline, "Tcsf: %d + %d\n", info.Tcsf, info.Ecsf);
strcat(z, textline);
sprintf(textline, "Tgm:  %d\n", info.Tgm);
strcat(z, textline);
sprintf(textline, "Twm:  %d\n", info.Twm);
strcat(z, textline);
sprintf(textline, "Tup:  %d\n", info.Tup);
strcat(z, textline);
sprintf(textline, "COG: (%d, %d, %d)\n", info.COG.x, info.COG.y, info.COG.z);
strcat(z, textline);
p = VoxelAddress(APP->Data.orig, info.COG.x, info.COG.y, info.COG.z);
A = Circular(4.0);
APP->AddSeedsInBrush(p, 1, 1, A, 'x');
APP->AddSeedsInBrush(p, 1, 1, A, 'y');
APP->AddSeedsInBrush(p, 1, 1, A, 'z');
DestroyAdjRel(&A);
APP->Set2DViewSlice(info.COG);

APP->Unbusy();
APP->Refresh2DCanvas();
wxString wxz(z, wxConvUTF8);
wxMessageBox(wxz, _T("Parameters Estimation"), wxOK | wxICON_INFORMATION, this);
}
}


void MyFrame::OnSelectView(wxCommandEvent &event)
{
    //printf("event  id: %d\n",event.GetId());
    int viewtype;
    int w, h, min;
    viewtype = event.GetId();

    if (viewtype == ID_View0) this->UniformView();

    else
    {
        min = splitter->GetMinimumPaneSize();
        splitter->GetSize(&w, &h);
        if (viewtype < ID_View3)
        {
            splitter->SetSashPosition(h - min - 1, true);

            min = topWindow->GetMinimumPaneSize();
            topWindow->GetSize(&w, &h);
            if (viewtype == ID_View1)
                topWindow->SetSashPosition(w - min - 1, true);
            else
                topWindow->SetSashPosition(min, true);
        }

        else
        {
            splitter->SetSashPosition(min, true);

            min = bottomWindow->GetMinimumPaneSize();
            bottomWindow->GetSize(&w, &h);
            if (viewtype == ID_View3)
                bottomWindow->SetSashPosition(w - min - 1, true);
            else
                bottomWindow->SetSashPosition(min, true);
        }
    }
}


void MyFrame::OnPreprocessing(wxCommandEvent &event)
{

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded."),
                     _T("Preprocessing"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    modManager->ShowModuleWindow(Module::PREPROC);
}


void MyFrame::OnSegmentation(wxCommandEvent &event)
{

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded."),
                     _T("Segmentation"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    modManager->ShowModuleWindow(Module::SEGMENTATION);
}


void MyFrame::OnAnalysis(wxCommandEvent &event)
{

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded."),
                     _T("Analysis"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    modManager->ShowModuleWindow(Module::ANALYSIS);
}


void MyFrame::OnMeasurements(wxCommandEvent &event)
{

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded."),
                     _T("Analysis"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    if (APP->Data.oriented == 0 )
    {
        int n = modManager->GetNumberModules(Module::PREPROC);
        if (n == 0) Error((char *)"BIAClass", (char *)"Setorientation not loaded");
        ((PreProcModule *) modManager->GetModule(Module::PREPROC, 1))->Start();
        wxTheApp->Yield();
        if ( APP->Data.oriented == 0 )
            return;
    }


    MeasurementsDialog           *Window;
    Window = new MeasurementsDialog();

    if (Window->ShowModal() == wxID_OK)
    {

        // Measures that uses symmetry
        if (Window->GetM1() == 1 || Window->GetM2() == 1)
        {

            Scene *msp;
            if ( APP->Data.aligned == 0 )
            {
                APP->Busy((char *)"Computing MSP...");
                APP->StatusMessage((char *)"Computing MSP...");
                msp = BIA_MSP_Align(APP->Data.orig, NULL, 2, 0, APP->segmobjs, &(APP->Data.msp));
            }
            else
                msp = CopyScene(APP->Data.orig);

            APP->Busy((char *)"Registering hemispheres...");
            APP->StatusMessage((char *)"Registering hemispheres...");
            float T[4][4];
            float *best_theta;
            Scene *reg = SelfRegisterAxial(msp, T, &best_theta);
            DestroyScene(&msp);

            if (Window->GetM1() == 1)
            {
                APP->Busy((char *)"Computing M1...");
                APP->StatusMessage((char *)"Computing M1...");
                ComputeM1(APP->measures, reg);
            }

            if (Window->GetM2() == 1)
            {
                APP->Busy((char *)"Computing M2...");
                APP->StatusMessage((char *)"Computing M2...");
                ComputeM2(APP->measures, reg);
            }

            DestroyScene(&reg);

        }

        APP->StatusMessage((char *)"Done");
        APP->Unbusy();

        Window = new MeasurementsDialog();
        if (Window->ShowModal() == wxID_OK);

    }


}













































void MyFrame::OnGroupStudy(wxCommandEvent &event)
{


    wxMessageBox(_T("To be done."), _T("Ops..."), wxOK | wxICON_EXCLAMATION, this);

    /*
    float T[4][4];
    float *best_theta;
    Scene *res = SelfRegisterAxial(APP->Data.orig,T,&best_theta);
    WriteScene(res,"selfreg.scn");

    wxMessageBox(_T("Done!!!"), _T("Ops..."), wxOK | wxICON_EXCLAMATION, this);
    */

}



void MyFrame::OnRegistration(wxCommandEvent &event)
{

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded."),
                     _T("Preprocessing"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    modManager->ShowModuleWindow(Module::REGISTRATION);
}


void MyFrame::On3DVisualization(wxCommandEvent &event)
{

    if (!APP->Data.loaded)
    {
        wxMessageBox(_T("There is no volume loaded."),
                     _T("3D Visualization"), wxOK | wxICON_EXCLAMATION, this);
        return;
    }

    modManager->ShowModuleWindow(Module::VIEW3D);
}

//--------------------------------------------------------

void app_selective_refresh(Voxel v)
{
    Voxel Cut;

    if (!APP->Data.loaded) return;

    Cut = APP->Get2DViewSlice();

    /*
    if(Cut.x == v.x){
      app_draw_frame(1);
      Views[1]->canvas->Refresh();
    }
    if(Cut.y == v.y){
      app_draw_frame(2);
      Views[2]->canvas->Refresh();
    }
    if(Cut.z == v.z){
      app_draw_frame(0);
      Views[0]->canvas->Refresh();
    }
    */
    APP->Refresh2DCanvas();
}


//----------------------------------------

/*
ObjectDialog :: ObjectDialog(wxWindow *parent)
  : BaseDialog(parent, (char *)"Object Dialog"){
  SegmObject *segm;
  wxColour wxcolor;
  int i,n,id;

  wxScrolledWindow *swind = new wxScrolledWindow(this, -1, wxDefaultPosition, wxDefaultSize, wxHSCROLL | wxVSCROLL, _T("scrolledWindow"));

  sizer = new wxBoxSizer(wxVERTICAL);

  wxBitmap *bmeye   = new wxBitmap(eye_xpm);
  wxBitmap *bmneye  = new wxBitmap(noteye_xpm);
  wxBitmap *bmtrash = new wxBitmap(trash_xpm);

  n = this->n = APP->GetNumberOfObjs();
  if(n>0){
    v_but_color = (AlphaColourButton **)malloc(sizeof(AlphaColourButton *)*n);
    v_but_eye = (BitmapToggleButton **)malloc(sizeof(BitmapToggleButton *)*n);
    v_but_trash = (wxBitmapButton **)malloc(sizeof(wxBitmapButton *)*n);
  }
  for(i=0; i<n; i++){
    segm = APP->GetObjByIndex(i);

    wxPanel *bkg = new wxPanel(swind, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL, _T("panel_bkg"));
    if(i%2==0)
      SetColor(&wxcolor, 0xffffff);
    else
      SetColor(&wxcolor, 0xffffdd);
    bkg->SetBackgroundColour(wxcolor);

    id = APP->idManager->AllocID();
    v_but_color[i] = new AlphaColourButton(bkg, wxID_ANY);
    v_but_eye[i] = new BitmapToggleButton(bkg, wxID_ANY,
                      bmneye, bmeye);
    v_but_trash[i] = new wxBitmapButton(bkg, id, *bmtrash, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("trash"));

    v_but_color[i]->SetValue(segm->color);
    v_but_color[i]->SetAlpha(segm->alpha);
    v_but_eye[i]->SetValue(segm->visibility);

    Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
         wxCommandEventHandler(ObjectDialog::OnDelete),
         NULL, NULL );

    wxString *wxstr = new wxString(segm->name, wxConvUTF8);
    wxStaticText *tname = new wxStaticText(bkg, -1, *wxstr, wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);

    hsizer->Add(v_but_eye[i],   0, wxALIGN_LEFT|wxEXPAND);
    hsizer->Add(v_but_trash[i], 0, wxALIGN_RIGHT|wxEXPAND);
    hsizer->Add(v_but_color[i], 0, wxALIGN_LEFT|wxEXPAND);
    hsizer->AddSpacer(10);
    hsizer->Add(tname,          1, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);

    bkg->SetSizer(hsizer, true);
    hsizer->SetSizeHints(bkg);
    hsizer->Layout();

    sizer->Add(bkg, 0, wxEXPAND);
  }

  swind->SetSizer(sizer, true);
  sizer->SetSizeHints(swind);
  sizer->Layout();

  this->AddPanel((wxPanel *)swind);
}
*/


ObjectDialog :: ObjectDialog(wxWindow *parent)
        : BaseDialog(parent, (char *)"Object Dialog")
{
    SegmObject *segm;
    wxColour wxcolor;
    int i, n, id, id2;
    int w, h, wm = 200;
    wxSize size(350, 420);
    //wxSize size(220, 120);
    wxPanel *bkg;

    wxScrolledWindow *swind = new wxScrolledWindow(this, -1,
                                                   wxDefaultPosition,
                                                   wxDefaultSize,
                                                   wxHSCROLL | wxVSCROLL,
                                                   _T("scrolledWindow"));

    swind->SetMinSize(size);
    wxBitmap *bmeye   = new wxBitmap(eye_xpm);
    wxBitmap *bmneye  = new wxBitmap(noteye_xpm);
    wxBitmap *bmtrash = new wxBitmap(trash_xpm);

    n = this->n = APP->GetNumberOfObjs();

    if (n > 0)
    {
        v_panel_bkg = (wxPanel **)malloc(sizeof(wxPanel *)*n);
        v_but_color = (AlphaColourButton **)malloc(sizeof(AlphaColourButton *)*n);
        v_but_eye = (BitmapToggleButton **)malloc(sizeof(BitmapToggleButton *)*n);
        v_but_trash = (wxBitmapButton **)malloc(sizeof(wxBitmapButton *)*n);
        v_obj_name = (wxTextCtrl **) malloc(sizeof(wxTextCtrl*)*n);
    }
    else
    {
        v_panel_bkg = NULL;
        v_but_color = NULL;
        v_but_eye   = NULL;
        v_but_trash = NULL;
    }

    swind->SetScrollRate(5, 5);
    swind->Scroll(0, 0);
    swind->SetVirtualSize(200, 10 + 40 * n);

    for (i = 0; i < n; i++)
    {
        segm = APP->GetObjByIndex(i);

        size.SetHeight(wxDefaultCoord);
        size.SetWidth(200);
        bkg = new wxPanel(swind, wxID_ANY,
                          wxPoint(10, 10 + 40 * i), //wxDefaultPosition,
                          wxDefaultSize, wxTAB_TRAVERSAL, _T("panel_bkg"));
        bkg->SetMinSize(size);

        if (i % 2 == 0)
            SetColor(&wxcolor, 0xffffff);
        else
            SetColor(&wxcolor, 0xffffdd);
        bkg->SetBackgroundColour(wxcolor);

        v_panel_bkg[i] = bkg;
        v_but_color[i] = new AlphaColourButton(bkg, wxID_ANY);
        id2 = APP->idManager->AllocID();
        v_but_eye[i] = new BitmapToggleButton(bkg, id2, bmneye, bmeye);
        id = APP->idManager->AllocID();
        v_but_trash[i] = new wxBitmapButton(bkg, id, *bmtrash, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("trash"));

        v_but_color[i]->SetValue(segm->color);
        v_but_color[i]->SetAlpha(segm->alpha);
        v_but_eye[i]->SetValue(segm->visibility);

        Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(ObjectDialog::OnDelete),
                 NULL, NULL );
        Connect( id2, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(ObjectDialog::OnChange),
                 NULL, NULL );

        id = APP->idManager->AllocID();

        wxString wxstr = wxString::FromAscii(segm->name);
        v_obj_name[i] = new wxTextCtrl(bkg, id, wxstr, wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, wxDefaultValidator, _T("obj_name"));
        //wxString *wxstr = new wxString(segm->name, wxConvUTF8);
        //wxStaticText *tname = new wxStaticText(bkg, -1, *wxstr, wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
        size.SetHeight(wxDefaultCoord);
        size.SetWidth(200);
        //size.SetWidth(120);
        v_obj_name[i]->SetMinSize(size);
        wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);

        hsizer->Add(v_but_eye[i],   0, wxALIGN_LEFT | wxEXPAND);
        hsizer->Add(v_but_trash[i], 0, wxALIGN_RIGHT | wxEXPAND);
        hsizer->Add(v_but_color[i], 0, wxALIGN_LEFT | wxEXPAND);
        hsizer->AddSpacer(10);
        hsizer->Add(v_obj_name[i] ,1, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);

        bkg->SetSizer(hsizer, true);
        hsizer->SetSizeHints(bkg);
        hsizer->Layout();

        bkg->GetSize(&w, &h);
        if (w > wm) wm = w;
    }
    for (i = 0; i < n; i++)
    {
        size.SetHeight(wxDefaultCoord);
        size.SetWidth(wm);
        (v_panel_bkg[i])->SetSize(size);
    }
    swind->SetVirtualSize(wm, 10 + MAX(40 * n, 40));


    wxPaintDC paintDC(swind);
    swind->DoPrepareDC(paintDC);

    delete bmeye;
    delete bmneye;
    delete bmtrash;

    this->AddPanel((wxPanel *)swind);
}


ObjectDialog :: ~ObjectDialog()
{
    int i, n, id, id2;

    n = this->n;
    for (i = 0; i < n; i++)
    {
        id = v_but_trash[i]->GetId();
        id2 = v_but_eye[i]->GetId();
        Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(ObjectDialog::OnDelete),
                    NULL, NULL );
        Disconnect( id2, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(ObjectDialog::OnChange),
                    NULL, NULL );
        APP->idManager->FreeID(id);
    }
    if (n > 0)
    {
        if (v_panel_bkg != NULL) free(v_panel_bkg);
        if (v_but_color != NULL) free(v_but_color);
        if (v_but_eye != NULL)   free(v_but_eye);
        if (v_but_trash != NULL) free(v_but_trash);
        if (v_obj_name != NULL)   free(v_obj_name);
        v_panel_bkg = NULL;
        v_but_color = NULL;
        v_but_eye   = NULL;
        v_but_trash = NULL;
        v_obj_name  = NULL;
    }
}


void ObjectDialog :: OnDelete(wxCommandEvent &event)
{
    SegmObject *segm;
    char msg[1024];
    int i, n, id;

    id = event.GetId();

    n = this->n;
    for (i = 0; i < n; i++)
    {
        if (id == v_but_trash[i]->GetId())
            break;
    }
    if (i < n)
    {
        segm = APP->GetObjByIndex(i);
        sprintf(msg, "Do you really wish to delete the %s?", segm->name);

        wxString wxmsg(msg, wxConvUTF8);
        wxMessageDialog *dialog = new wxMessageDialog(this, wxmsg, _T("Delete Confirmation"), wxYES_NO | wxICON_QUESTION, wxDefaultPosition);
        if (dialog->ShowModal() == wxID_YES)
        {
            APP->DelObjByIndex(i);
            APP->DrawSegmObjects();
            BaseDialog::OnOk(event);
        }
    }
}


void ObjectDialog :: OnCancel(wxCommandEvent &event)
{
    BaseDialog::OnCancel(event);
}

void ObjectDialog :: OnOk(wxCommandEvent &event)
{
    SegmObject *segm;
    int i, n;
    wxString wxstr;

    n = APP->GetNumberOfObjs();
    for (i = 0; i < n; i++)
    {
        segm = APP->GetObjByIndex(i);
        wxstr = v_obj_name[i]->GetValue();
        strcpy(segm->name, wxstr.mb_str());
        segm->color = v_but_color[i]->GetValue();
        segm->alpha = v_but_color[i]->GetAlpha();
        segm->visibility = v_but_eye[i]->GetValue();
    }
    APP->DrawSegmObjects();
    BaseDialog::OnOk(event);
}


void ObjectDialog :: OnChange(wxCommandEvent &event)
{
    SegmObject *segm;
    int i, n;
    n = APP->GetNumberOfObjs();
    for (i = 0; i < n; i++)
    {
        segm = APP->GetObjByIndex(i);
        segm->color = v_but_color[i]->GetValue();
        segm->alpha = v_but_color[i]->GetAlpha();
        segm->visibility = v_but_eye[i]->GetValue();
    }
    APP->DrawSegmObjects();
}




//----------------------------------------
ModalityPickerDialog::ModalityPickerDialog(wxWindow *parent,
                                           wxWindowID id,
char *title)
: BaseDialog(parent, title)
{
int i, n = UNKNOWN_PROTOCOL + 1;
wxString *wxchoices;
BasePanel  *panel;
wxStaticText *tText = NULL;
wxStaticText *tModality = NULL;

panel = new BasePanel(this);
panel->Show(true);

tText = new wxStaticText(panel, -1,
                         _T("\n  Please select the correct modality.  \n"),
                         wxDefaultPosition, wxDefaultSize,
                         wxALIGN_LEFT, _T("staticText1"));
wxBoxSizer *hbsText = new wxBoxSizer(wxHORIZONTAL);
hbsText->Add(tText, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
panel->sizer->Add(hbsText, 0, wxEXPAND);


wxchoices = new wxString[n];
for (i = 0; i < n; i++)
wxchoices[i] = wxString::FromAscii(ModalityNames[i]);

tModality = new wxStaticText(panel, -1, _T("Modality:"),
                             wxDefaultPosition, wxDefaultSize,
                             wxALIGN_LEFT, _T("staticText2"));
chModality = new wxChoice(panel, id,
                          wxDefaultPosition, wxDefaultSize,
                          n, wxchoices, 0, wxDefaultValidator,
                          _T("choice0"));
chModality->SetSelection(0);

wxBoxSizer *hbsModality = new wxBoxSizer(wxHORIZONTAL);
hbsModality->Add(tModality,  0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
hbsModality->Add(chModality, 1, wxALIGN_LEFT | wxEXPAND);
panel->sizer->Add(hbsModality, 0, wxEXPAND);

panel->sizer->SetSizeHints(panel);
panel->sizer->Layout();

this->AddPanel(panel);
}


ModalityPickerDialog::~ModalityPickerDialog() {}


ModalityType ModalityPickerDialog::GetModality()
{
    int modality;
    modality = chModality->GetSelection();
    return (ModalityType)modality;
}


//----------------------------------------

ObjectPickerDialog::ObjectPickerDialog(wxWindow *parent,
                                       wxWindowID id,
char *title,
const char *choices[],
int n,
bool readonly,
bool getcolor,
bool getname)
: BaseDialog(parent, title)
{
long style;
int i;
wxString *wxchoices;
wxString wxstr;
BasePanel  *panel;

panel = new BasePanel(this);
panel->Show(true);

if (getcolor)
{
wxStaticText *tColor = new wxStaticText(panel, -1, _T("Color"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText1"));
butColor = new ColourButton(panel, id);
wxBoxSizer *hbsColor = new wxBoxSizer(wxHORIZONTAL);
hbsColor->Add(tColor, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
hbsColor->Add(butColor, 0, wxALIGN_LEFT);
panel->sizer->Add(hbsColor, 0, wxEXPAND);
}
else
butColor = NULL;

if (getname)
{
if (n > 0)
{
wxchoices = new wxString[n];
for (i = 0; i < n; i++)
wxchoices[i] = wxString::FromAscii(choices[i]);
wxstr = wxString::FromAscii(choices[0]);
}
else
wxchoices = NULL;

if (readonly)
{
style = wxCB_READONLY;
if (n > 0) wxstr = wxString::FromAscii(choices[0]);
}
else
{
style = wxCB_DROPDOWN;
wxstr = _T("");
}

wxStaticText *tName = new wxStaticText(panel, -1, _T("Name"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText2"));
texName = new wxComboBox(panel, id, wxstr, wxDefaultPosition, wxDefaultSize, n, wxchoices, style, wxDefaultValidator, _T("comboBox"));
if (readonly && n > 0) texName->SetSelection(0);
wxBoxSizer *hbsName = new wxBoxSizer(wxHORIZONTAL);
hbsName->Add(tName, 0, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
hbsName->Add(texName, 1, wxALIGN_LEFT | wxEXPAND);
panel->sizer->Add(hbsName, 0, wxEXPAND);
}
else
texName = NULL;

panel->sizer->SetSizeHints(panel);
panel->sizer->Layout();

this->AddPanel(panel);
}


ObjectPickerDialog::~ObjectPickerDialog() {}


void ObjectPickerDialog::GetName(char *name)
{
    if (texName != NULL)
    {
        wxString wxstr = texName->GetValue();
        //strcpy(name, wxstr.mb_str());
        strcpy(name, wxstr.ToAscii());
    }
    else
        name[0] = '\0';
}


int  ObjectPickerDialog::GetColor()
{
    if (butColor != NULL)
        return butColor->GetValue();
    else
        return NIL;
}


//-------------------------------------------


// PopUpView :: PopUpView(Scene *scn, char *title)
//   : BaseDialog(APP->Window, title){
//   View2DModule*view2D;
//   wxSize size(200, 200);
//   int n;

//   this->view_scn = scn;
//   n = modManager->GetNumberModules(Module::VIEW2D);
//   if(n==0) Error((char *)"PopUpView",(char *)"View2DModule not loaded");
//   view2D = (View2DModule*)modManager->GetModule(Module::VIEW2D, 0);

//   wxPanel *opanel = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNO_BORDER, _T("panel0"));
//   this->id_ori = APP->idManager->AllocID();
//   wxString oriChoices[3];
//   oriChoices[0] = _T("Axial");
//   oriChoices[1] = _T("Coronal");
//   oriChoices[2] = _T("Sagittal");
//   chOrientation = new wxChoice(opanel, id_ori, wxDefaultPosition, wxDefaultSize, 3, oriChoices, 0, wxDefaultValidator, _T("choice0"));
//   chOrientation->SetSelection(0);
//   this->current = 0;

//   wxSizer *osizer = new wxBoxSizer(wxHORIZONTAL);
//   opanel->SetSizer(osizer, true);
//   osizer->Add(chOrientation, 0, wxALIGN_RIGHT|wxALIGN_CENTER_VERTICAL);
//   osizer->SetSizeHints(opanel);
//   osizer->Layout();

//   sizer->Prepend(opanel, 0, wxBOTTOM);
//   sizer->PrependSpacer(2);
//   sizer->SetSizeHints(this);

//   this->view[0] = view2D->GetCustomViewPanel(this, scn, 'z');
//   this->view[1] = view2D->GetCustomViewPanel(this, scn, 'y');
//   this->view[2] = view2D->GetCustomViewPanel(this, scn, 'x');
//   //this->AddPanel(this->view[this->current]);
//   this->AddPanel(this->view[0]);
//   this->AddPanel(this->view[1]);
//   this->AddPanel(this->view[2]);
//   sizer->Hide(this->view[1], true);
//   sizer->Hide(this->view[2], true);
//   sizer->Layout();
//   this->SetMinSize(size);

//   Connect( id_ori, wxEVT_COMMAND_CHOICE_SELECTED,
//     wxCommandEventHandler(PopUpView::OnChangeSliceOrientation),
//     NULL, NULL );

//   this->ShowWindow();
// }


// PopUpView::~PopUpView(){
//   View2DModule*view2D;
//   int n;

//   n = modManager->GetNumberModules(Module::VIEW2D);
//   if(n==0) Error((char *)"PopUpView",(char *)"View2DModule not loaded");
//   view2D = (View2DModule*)modManager->GetModule(Module::VIEW2D, 0);
//   view2D->DelViewPanel(&(this->view[0]));
//   view2D->DelViewPanel(&(this->view[1]));
//   view2D->DelViewPanel(&(this->view[2]));

//   Disconnect( id_ori, wxEVT_COMMAND_CHOICE_SELECTED,
//        wxCommandEventHandler(PopUpView::OnChangeSliceOrientation),
//        NULL, NULL );
//   APP->idManager->FreeID(id_ori);
// }


// void PopUpView::OnChangeSliceOrientation(wxCommandEvent& event){
//   wxSize size = this->GetSize();

//   sizer->Hide(this->view[this->current], true);
//   //this->DetachPanel(this->view[this->current]);

//   this->current = chOrientation->GetSelection();
//   //this->AddPanel(this->view[this->current]);
//   sizer->Show(this->view[this->current], true, true);
//   sizer->Layout();
//   this->SetSize(size);
// }





MeasurementsDialog::MeasurementsDialog()
        : wxDialog(APP->Window, wxID_ANY, _T("Measurements"), wxDefaultPosition, wxSize(380, 390), wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("Measures"))
{


    wxBoxSizer *sizer = new  wxBoxSizer(wxVERTICAL);

    wxStaticText *tmsg1 = new wxStaticText(this, wxID_ANY, _T("Current Measures:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));


    wxGrid *grid = new wxGrid(this, wxID_ANY,
                              wxDefaultPosition,
                              wxSize(219, 180),    //wxDefaultSize,
                              wxWANTS_CHARS,
                              _T("StatsGrid"));

    if (APP->measures->n > 10 ) grid->CreateGrid(APP->measures->n, 1, wxGrid::wxGridSelectCells);
    else
    {
        grid->CreateGrid(10, 1, wxGrid::wxGridSelectCells);
        int i;
        for (i = 0; i < 10; i++)
        {
            wxString wxstr(" ", wxConvUTF8);
            grid->SetRowLabelValue(i, wxstr);
            grid->SetReadOnly( i, 0, true );
        }
    }
    grid->SetDefaultCellAlignment(wxALIGN_CENTRE, wxALIGN_CENTRE);
    grid->SetRowLabelSize(100);
    grid->SetColSize(0, 100);
    grid->SetColLabelValue(0, _T("Value"));

    int i;
    for (i = 0; i < APP->measures->n; i++)
    {
        wxString wxstr(APP->measures->measure[i].name, wxConvUTF8);
        grid->SetRowLabelValue(i, wxstr);

        grid->SetReadOnly( i, 0, true );
        char str[100];
        sprintf(str, "%.5f", APP->measures->measure[i].value);
        wxString wxstr2(str, wxConvUTF8);
        grid->SetCellValue(i, 0, wxstr2);
    }

    wxStaticText *tmsg = new wxStaticText(this, wxID_ANY, _T("Check all the measures you want to (re)calculate:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    m1 = new wxCheckBox(this, wxID_ANY, _T("M1"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    m2 = new wxCheckBox(this, wxID_ANY, _T("M2"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    //m3 = new wxCheckBox(this,wxID_ANY,_T("M2"),wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    m1->SetValue(1);
    m2->SetValue(1);
    //m3->SetValue(1);

    wxStaticLine *sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));
    wxStaticLine *sline2 = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));

    wxSizer *sbutton = this->CreateButtonSizer(wxOK | wxCANCEL);

    sizer->AddSpacer(25);

    sizer->Add(tmsg1,    0, wxCENTER);
    sizer->Add(grid,    0, wxCENTER);

    sizer->AddSpacer(25);
    sizer->Add(sline2,    0, wxEXPAND);
    sizer->AddSpacer(20);

    sizer->Add(tmsg,    0, wxCENTER);
    sizer->Add(m1,    0, wxCENTER);
    sizer->Add(m2,    0, wxCENTER);
    //sizer->Add(m3,    0, wxCENTER);

    sizer->AddSpacer(5);
    sizer->Add(sline,    0, wxEXPAND);
    sizer->AddSpacer(5);

    sizer->Add(sbutton,    0, wxCENTER);
    this->SetSizer(sizer, true);
    //sizer->Layout();
    //sizer->SetSizeHints(this);

    //nb->AddPage(sizer2, _T("Compute"),true,-1);
    //nb->AddPage(sizer, _T("Measures"),false,-1);


}


MeasurementsDialog::~MeasurementsDialog()
{

}

int MeasurementsDialog::GetM1()
{
    return m1->GetValue();
}

int MeasurementsDialog::GetM2()
{
    return m2->GetValue();
}

// int MeasurementsDialog::GetM3()
// {
//   return m3->GetValue();
// }




