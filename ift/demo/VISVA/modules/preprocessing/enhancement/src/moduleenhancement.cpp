
#include "moduleenhancement.h"
#include "enhancementdialog.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Enhancement{

    ModuleEnhancement :: ModuleEnhancement()
            : PreProcModule(){
        SetName((char *)"Object Enhancement");
        SetAuthor((char *)"Paulo A.V. Miranda");
        SoftwareVersion ver(2,0,0);
        SetVersion(ver);

        optDialog = NULL;
        this->gradi = NULL;
    }


    ModuleEnhancement :: ~ModuleEnhancement(){
    }


    void ModuleEnhancement :: Start(){
        this->markerID = 2;
        APP->EnableObjWindow(false);
        APP->ResetData();

        this->active = true;
        APP->SetLabelColour(0, NIL);
        APP->SetLabelColour(1, 0xFFFF00);

        this->AllocData();

        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true,1.0);
    }


    bool ModuleEnhancement :: Stop(){
        static const char *title = {"Keep enhancement?"};
        static const char *msg = {
                "You are about to leave the enhancement module.\nSave changes?"};

        if(!this->active) return true;

        wxString wxtitle(title, wxConvUTF8);
        wxString wxmsg(msg, wxConvUTF8);

        wxMessageDialog dialog(APP->Window,
                               wxmsg, wxtitle,
                               wxYES_NO | wxICON_QUESTION,
                               wxDefaultPosition);

        if(dialog.ShowModal() == wxID_YES)
            this->Finish();
        else{
            SetScene(APP->Data.arcw, 0);
            (APP->Data.arcw)->maxval = 0;
            this->FreeData();
        }

        return true;
    }


    void ModuleEnhancement :: Finish(){
        int p,n;

        n = APP->Data.w*APP->Data.h*APP->Data.nframes;
        p = 0;
        this->FreeData();
    }


    void ModuleEnhancement :: Reset(){
        static const char *title = "Reset enhancement?";
        static const char *msg = "Current enhancement will be lost.\nAre you sure you want to reset?";

        wxString wxtitle(title, wxConvUTF8);
        wxString wxmsg(msg, wxConvUTF8);

        wxMessageDialog dialog(APP->Window,
                               wxmsg, wxtitle,
                               wxYES_NO | wxICON_QUESTION,
                               wxDefaultPosition);

        if(dialog.ShowModal() == wxID_YES){
            APP->ResetData();
            this->markerID = 2;

            SetScene(APP->Data.arcw, 0);
            (APP->Data.arcw)->maxval = 0;

            if(APP->Data.objmap != NULL)
                DestroyScene(&(APP->Data.objmap));
            APP->Data.objmap = NULL;

            APP->Refresh2DCanvas();
            APP->Refresh3DCanvas(true, 1.0);
        }
    }


    typedef struct _argfuzzymap{
        Subgraph *sge;
        Subgraph *sgi;
        Scene *objmap;
        float scn_max;
        int begin;
        int end;
    } ArgFuzzyMap;


    void *ThreadFuzzyMap(void *arg){
        ArgFuzzyMap *p_arg;
        float *tfeat; // temporary feature vector.
        float weight,Vi,Ve,cst,scn_max;
        Subgraph *sge=NULL,*sgi=NULL;
        Scene *objmap=NULL;
        int i,j,p;

        p_arg   = (ArgFuzzyMap *)arg;
        objmap  = p_arg->objmap;
        scn_max = p_arg->scn_max;
        sge     = p_arg->sge;
        sgi     = p_arg->sgi;

        tfeat = AllocFloatArray(sgi->nfeats);

        for(p=p_arg->begin; p<=p_arg->end; p++){
            tfeat[0] = (float)(APP->Data.orig)->data[p]/scn_max;

            Ve = FLT_MAX;
            for(i=0; i<sge->nnodes; i++){
                j = sge->ordered_list_of_nodes[i];

                if(Ve <= sge->node[j].pathval)
                    break;
                weight = opf_ArcWeight(sge->node[j].feat, tfeat, sge->nfeats);
                cst = MAX(weight, sge->node[j].pathval);
                if(cst<Ve) Ve = cst;
            }

            Vi = FLT_MAX;
            for(i=0; i<sgi->nnodes; i++){
                j = sgi->ordered_list_of_nodes[i];

                if(Vi <= sgi->node[j].pathval)
                    break;
                weight = opf_ArcWeight(sgi->node[j].feat, tfeat, sgi->nfeats);
                cst = MAX(weight, sgi->node[j].pathval);
                if(cst<Vi) Vi = cst;
            }

            Vi += 0.000001;
            Ve += 0.000001;
            objmap->data[p] = ROUND(10000*Ve/(Vi+Ve));
        }
        free(tfeat);
        return NULL;
    }

    void RandomizeIntArray(int *array, int n){
        int i,j,tmp;
        for(i=0; i<n; i++){
            j = RandomInteger(0, n-1);
            //Swap i & j:
            tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
        }
    }


    void ModuleEnhancement :: Run(){
        timer tic;
        Set *Si=NULL,*Se=NULL,*tmp=NULL;
        Subgraph *sg=NULL,*sge=NULL,*sgi=NULL;
        int nnodes,ni,ne,nni,nne;
        int nsamples,nsamples_i,nsamples_e;
        int i,j,p,ii,ie,Gmax;
        int *mapping=NULL,*sortition=NULL;
        float scn_max;
        Scene *objmap=NULL;
        ArgFuzzyMap arg;
        ArgFuzzyMap args[8];
        int nprocs;
        int first,last,nelems,de;
        pthread_t thread_id[8];
        int iret[8];

        APP->Busy((char *)"Please wait, working...");
        APP->StatusMessage((char *)"Please wait - Computation in progress...");

        opf_ArcWeight=opf_EuclDist;

        nprocs = GetNumberOfProcessors();
        printf("nprocs: %d\n",nprocs);
        if(nprocs>=8) nprocs = 8;

        Si = APP->CopyInternalSeeds();
        Se = APP->CopyExternalSeeds();

        ni = GetSetSize(Si);
        ne = GetSetSize(Se);
        //printf(" ni %d, ne %d\n",ni,ne);


        StartTimer(&tic);

        if(ni>0 && ne>0){
            //*******CREATE SUBGRAPH*****************
            nsamples = MIN(250, ni+ne);
            nsamples_i = ROUND(nsamples*((float)ni/(float)(ni+ne)));
            nsamples_e = ROUND(nsamples*((float)ne/(float)(ni+ne)));

            nsamples_i = MIN(MAX(nsamples_i, 1), ni);
            nsamples_e = MIN(MAX(nsamples_e, 1), ne);
            nsamples = nsamples_i + nsamples_e;

            nnodes = nsamples;
            sg = CreateSubgraph(nnodes);
            //printf("Create nnodes: %d\n",sg->nnodes);
            sg->nfeats  = 1;
            sg->nlabels = 2;
            scn_max = (float)MaximumValue3(APP->Data.orig);

            sortition = AllocIntArray(ne);
            for(i=0; i<ne; i++){
                if(i<nsamples_e) sortition[i] = 1;
                else             sortition[i] = 0;
            }
            RandomizeIntArray(sortition, ne);

            i=0; j=0;
            tmp = Se;
            while(tmp!=NULL){
                p = tmp->elem;
                if(sortition[j]>0){
                    sg->node[i].truelabel = 0;
                    sg->node[i].pixel = p;
                    sg->node[i].feat  = AllocFloatArray(sg->nfeats);
                    sg->node[i].feat[0] = (float)(APP->Data.orig)->data[p]/scn_max;
                    i++;
                }
                tmp = tmp->next;
                j++;
            }
            free(sortition);

            sortition = AllocIntArray(ni);
            for(i=0; i<ni; i++){
                if(i<nsamples_i) sortition[i] = 1;
                else             sortition[i] = 0;
            }
            RandomizeIntArray(sortition, ni);

            i=nsamples_e; j=0;
            tmp = Si;
            while(tmp!=NULL){
                p = tmp->elem;
                if(sortition[j]>0){
                    sg->node[i].truelabel = 1;
                    sg->node[i].pixel = p;
                    sg->node[i].feat  = AllocFloatArray(sg->nfeats);
                    sg->node[i].feat[0] = (float)(APP->Data.orig)->data[p]/scn_max;
                    i++;
                }
                tmp = tmp->next;
                j++;
            }
            free(sortition);

            //*******OPF TRAINING*****************
            opf_OPFTraining(sg);
            //printf("Training nnodes: %d\n",sg->nnodes);

            //*******SPLIT SUBGRAPH***************
            nni = 0;
            nne = 0;
            for(i=0; i<sg->nnodes; i++){
                if(sg->node[i].truelabel == 0){ nne++; }
                else if(sg->node[i].truelabel == 1){ nni++; }
            }

            //printf("After: ni %d, ne %d\n",nni,nne);
            sge = CreateSubgraph(nne);
            sge->nfeats  = 1;
            sge->nlabels = 2;
            sgi = CreateSubgraph(nni);
            sgi->nfeats  = 1;
            sgi->nlabels = 2;
            nni = 0;
            nne = 0;
            mapping = AllocIntArray(sg->nnodes);
            for(i=0; i<sg->nnodes; i++){
                if(sg->node[i].truelabel == 0){
                    CopySNode(&(sge->node[nne]), &(sg->node[i]), sg->nfeats);
                    mapping[i] = nne;
                    nne++;
                }
                else if(sg->node[i].truelabel == 1){
                    CopySNode(&(sgi->node[nni]), &(sg->node[i]), sg->nfeats);
                    mapping[i] = nni;
                    nni++;
                }
            }
            //fixing ordered_list_of_nodes:
            ii = 0;
            ie = 0;
            for(i=0; i<sg->nnodes; i++){
                j = sg->ordered_list_of_nodes[i];
                if(sg->node[j].truelabel == 0){
                    sge->ordered_list_of_nodes[ie] = mapping[j];
                    ie++;
                }
                else if(sg->node[j].truelabel == 1){
                    sgi->ordered_list_of_nodes[ii] = mapping[j];
                    ii++;
                }
            }
            //printf("ii %d, ie %d\n",ii,ie);

            //*******FUZZY MEMBERSHIP***************
            objmap = CreateScene(APP->Data.w, APP->Data.h, APP->Data.nframes);
            objmap->dx = (APP->Data.orig)->dx;
            objmap->dy = (APP->Data.orig)->dy;
            objmap->dz = (APP->Data.orig)->dz;

            arg.sge     = sge;
            arg.sgi     = sgi;
            arg.objmap  = objmap;
            arg.scn_max = scn_max;

            first  = 0;
            last   = objmap->n-1;
            nelems = last-first+1;
            de     = nelems/nprocs;
            arg.begin = NIL;
            arg.end   = first-1;
            for(i=0; i<nprocs; i++){
                args[i] = arg;

                args[i].begin = arg.end+1;
                if(i<nprocs-1) args[i].end = args[i].begin+(de-1);
                else           args[i].end = last;

                //Create independent threads each of which will execute function
                iret[i] = pthread_create(&thread_id[i], NULL,
                                         ThreadFuzzyMap,
                                         (void*)&args[i]);
                arg = args[i];
            }

            //Wait till threads are complete before main continues.
            for(i=0; i<nprocs; i++)
                pthread_join(thread_id[i], NULL);


            if(APP->Data.objmap!=NULL)
                DestroyScene(&(APP->Data.objmap));

            //AdjRel3 *A;
            //A = Spheric(1.7);
            //APP->Data.objmap = MedianFilter3(objmap, A);
            //DestroyAdjRel3(&A);
            APP->Data.objmap = FastOptGaussianBlur3(objmap);
            MaximumValue3(APP->Data.objmap);

            DestroyScene(&objmap);

            //*******GRADIENT***************
            if(APP->Data.arcw!=NULL) DestroyScene(&(APP->Data.arcw));

            APP->Data.arcw = SphericalGradient(APP->Data.objmap, 1.0);
            Gmax = MaximumValue3(APP->Data.arcw);

            WeightedMean3inplace(APP->Data.arcw, this->gradi, 0.5,
                                 0, Gmax,
                                 0, (this->gradi)->maxval,
                                 0, MAX(Gmax,(this->gradi)->maxval));
            MaximumValue3(APP->Data.arcw);

            DestroySubgraph(&sge);
            DestroySubgraph(&sgi);
            DestroySubgraph(&sg);
            free(mapping);
        }

        printf("\nEnhancement Time: ");
        StopTimer(&tic);

        DestroySet(&Si);
        DestroySet(&Se);

        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);

        APP->StatusMessage((char *)"Done");
        APP->Unbusy();

        WriteScene(APP->Data.arcw, "arcweight.scn");
        WriteScene(APP->Data.objmap, "objmap.scn");
    }


    void ModuleEnhancement :: AllocData(){
        int x,y,w,h;
        timer tic;

        APP->ResetData();
        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true,1.0);

        if(optDialog!=NULL) delete optDialog;
        EnhancementDialog *dialog = new EnhancementDialog(APP->Window, this);
        optDialog = (BaseDialog*)dialog;
        optDialog->Show(true);
        APP->Window->GetPosition(&x, &y);
        APP->Window->GetSize(&w, &h);
        optDialog->Move(MAX(x-20,0), h/2);

        //--------------------
        if(this->gradi!=NULL)
            DestroyScene(&this->gradi);

        APP->Busy((char *)"Please wait, computing gradient...");
        APP->StatusMessage((char *)"Please wait - Computing Gradient...");

        StartTimer(&tic);
        this->gradi = BrainGrad3(APP->Data.orig);
        MaximumValue3(this->gradi);

        printf("\nGradient Time: ");
        StopTimer(&tic);

        APP->StatusMessage((char *)"Done");
        APP->Unbusy();
        //---------------------
    }


    void ModuleEnhancement :: FreeData(){
        APP->SetDefaultInteractionHandler();
        APP->EnableObjWindow(true);
        APP->ResetData();
        APP->DrawSegmObjects();
        this->active = false;

        optDialog->Show(false);
        if(optDialog!=NULL)
            optDialog->Destroy(); //delete optDialog;
        optDialog = NULL;

        if(this->gradi!=NULL)
            DestroyScene(&this->gradi);
        this->gradi = NULL;
    }


    AdjRel *ModuleEnhancement :: GetBrush(){
        EnhancementDialog *iopt = (EnhancementDialog *)optDialog;
        return iopt->GetBrush();
    }

    wxCursor *ModuleEnhancement :: GetBrushCursor(int zoom){
        EnhancementDialog *iopt = (EnhancementDialog *)optDialog;
        return iopt->GetBrushCursor(zoom);
    }

    void ModuleEnhancement :: NextBrush(){
        EnhancementDialog *iopt = (EnhancementDialog *)optDialog;
        iopt->NextBrush();
    }

    void ModuleEnhancement :: PrevBrush(){
        EnhancementDialog *iopt = (EnhancementDialog *)optDialog;
        iopt->PrevBrush();
    }

    int ModuleEnhancement :: GetCodeID(int cod){
        return (cod>>8); //(int)label/256;
    }

    int ModuleEnhancement :: GetCodeLabel(int cod){
        return (cod & 0xff);  //(label % 256);
    }

    int ModuleEnhancement :: GetCodeValue(int id, int lb){
        return ((id<<8) | lb);  //(id*256 + (int)lb);
    }

} //end Enhancement namespace




