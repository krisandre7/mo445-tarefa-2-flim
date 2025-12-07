
#include "slicecanvas.h"

namespace SliceView{

    SliceCanvas :: SliceCanvas(wxWindow *parent)
            : Canvas(parent){
        this->axis = 'z';
        this->highlight = HIGHLIGHT_FILL;
        this->data = DATA_ORIG;
        this->marker = MARKER_ON;
        this->drawmarker = true;
        this->custom_scn = NULL;
        defaulthandler = new Basic::NavigatorHandler();
        SetInteractionHandler(defaulthandler);
        this->drawhandler = NULL;
    }


    SliceCanvas :: ~SliceCanvas(){
        delete defaulthandler;
    }


    void SliceCanvas :: SetSliceAxis(char axis){
        switch(axis){
            case 'x':
            case 'X':
                this->axis = 'x';
                break;
            case 'y':
            case 'Y':
                this->axis = 'y';
                break;
            case 'z':
            case 'Z':
                this->axis = 'z';
                break;
            default:
                this->axis = 'z';
        }
    }


    char SliceCanvas :: GetSliceAxis(){
        return this->axis;
    }

    Voxel SliceCanvas :: GetCutVoxel(){
        SliceView *sview;
        sview = (SliceView *)this->GetParent();
        return sview->GetCutVoxel();
    }


    void SliceCanvas :: OnMouseEvent(wxMouseEvent& event){
        int p,x,y;
        Voxel out;
        SliceView *sview;

        event.GetPosition(&x, &y);
        p = Canvas2voxel(x, y, &out);
        sview = (SliceView *)this->GetParent();
        if(p>=0)
            sview->ShowCoordinate(&out);
        else
            sview->ShowCoordinate(NULL);
        if(handler!=NULL && p>=0)
            handler->OnMouseEvent(event, p);
    }



    int SliceCanvas :: Slice2voxel(int x, int y, Voxel *out) {
        Voxel Cut = GetCutVoxel();
        if (!APP->Data.loaded) return -1;

        if(this->axis == 'z'){
            out->x = x;
            out->y = y;
            out->z = Cut.z;
        }
        else if(this->axis == 'x'){
            out->x = Cut.x;
            out->y = y;
            out->z = x;
        }
        else if(this->axis == 'y'){
            out->x = x;
            out->y = Cut.y;
            out->z = y;
        }

        if( !ValidVoxel(APP->Data.orig, out->x, out->y, out->z) ) return -1;

        return VoxelAddress(APP->Data.orig, out->x, out->y, out->z);
    }


    int SliceCanvas :: Canvas2voxel(int x, int y, Voxel *out) {
        SliceView *sview;
        int scx,scy,xUnit,yUnit;
        int sx,sy,w,h,rw,rh;
        float fx,fy;

        if (!APP->Data.loaded) return -1;

        sview = (SliceView *)this->GetParent();
        GetViewStart(&scx, &scy);
        GetScrollPixelsPerUnit(&xUnit, &yUnit);
        scx *= xUnit;
        scy *= yUnit;
        x += scx;
        y += scy;

        fx = (float)x / this->zoom;
        fy = (float)y / this->zoom;
        x = ROUND(fx);
        y = ROUND(fy);

        rw = this->imageWidth;
        rh = this->imageHeight;

        w = rw; h = rh;
        if(sview->ntimes%2==1){
            w = rh; h = rw;
        }

        RotateBackNTimes(&sx, &sy, x, y, rw, rh, sview->ntimes);
        // Só processa o evento caso o mouse esteja
        //dentro da imagem.
        if(sx<0 || sy<0 || sx>=w || sy>=h) return -1;

        return this->Slice2voxel(sx,sy,out);
    }


    int  SliceCanvas :: Canvas2Address(int x, int y){
        Voxel out;
        return Canvas2voxel(x, y, &out);
    }


    void SliceCanvas :: SetInteractionHandler(InteractionHandler *handler){
        if(handler==NULL)
            this->handler = defaulthandler;
        else
            this->handler = handler;
    }


    void SliceCanvas :: SetCustomData(Scene *scn){
        if(scn==NULL) return;
        MaximumValue3(scn);
        this->custom_scn = scn;
        this->data = DATA_CUSTOM;
        this->marker = MARKER_OFF;
        this->highlight = HIGHLIGHT_OFF;
    }


    void SliceCanvas :: Set2DViewOptions(HighlightType highlight,
                                         DataType data,
                                         MarkerType marker){
        if(this->data==DATA_CUSTOM) return;
        if(data==DATA_CUSTOM) return;
        this->highlight = highlight;
        this->data = data;
        this->marker = marker;
        this->Draw();
    }


    void SliceCanvas :: FindImageSize(int *w, int *h){
        switch(this->axis){
            case 'z':
                *w = APP->Data.w;
                *h = APP->Data.h;
                break;
            case 'x':
                *w = APP->Data.nframes;
                *h = APP->Data.h;
                break;
            case 'y':
                *w = APP->Data.w;
                *h = APP->Data.nframes;
                break;
        }
    }


    void SliceCanvas :: Zoomin(){
        Canvas::Zoomin();
        this->DrawOverlay();
    }

    void SliceCanvas :: Zoomout(){
        Canvas::Zoomout();
        this->DrawOverlay();
    }


    void SliceCanvas :: SetZoomLevel(float zoom){
        Canvas::SetZoomLevel(zoom);
        this->DrawOverlay();
    }


    void SliceCanvas :: RotateBackNTimes(int *j, int *i,
                                         int rj, int ri,
                                         int rw, int rh,
                                         int ntimes){
        ntimes %= 4;
        switch(ntimes){
            case 0:
                *j = rj;
                *i = ri;
                break;
            case 1:
                *j = ri;
                *i = (rw-1) - rj;
                break;
            case 2:
                *j = (rw-1) - rj;
                *i = (rh-1) - ri;
                break;
            case 3:
                *j = (rh-1) - ri;
                *i = rj;
                break;
        }
    }


    void SliceCanvas :: RotateNTimes(int *rj, int *ri,
                                     int j, int i,
                                     int w, int h,
                                     int ntimes){
        ntimes %= 4;
        switch(ntimes){
            case 0:
                *rj = j;
                *ri = i;
                break;
            case 1:
                *ri = j;
                *rj = (h-1) - i;
                break;
            case 2:
                *rj = (w-1) - j;
                *ri = (h-1) - i;
                break;
            case 3:
                *rj = i;
                *ri = (w-1) - j;
                break;
        }
    }


    void SliceCanvas :: Draw(){
        SliceView *sview;
        Scene *scn=NULL;
        CImage *cimg=NULL;
        bool isseed;
        int maxval,val;
        int w,h,i,j,p,q,tmp;
        int rw,rh,ri,rj;
        int lb,color,ntimes;
        float alpha;
        Voxel v, Cut = GetCutVoxel();

        if(!APP->Data.loaded) return;

        sview = (SliceView *)this->GetParent();
        if(Cut.x>=APP->Data.w ||
           Cut.x<0 ||
           Cut.y>=APP->Data.h ||
           Cut.y<0 ||
           Cut.z>=APP->Data.nframes ||
           Cut.z<0){
            Cut.x = APP->Data.w/2;
            Cut.y = APP->Data.h/2;
            Cut.z = APP->Data.nframes/2;
        }

        if(this->drawhandler!=NULL)
            drawhandler->OnRefresh2D(this->axis);

        if(this->data == DATA_ORIG)
            scn = APP->Data.orig;
        else if(this->data == DATA_ARCW)
            scn = APP->Data.arcw;
        else if(this->data == DATA_OBJMAP)
            scn = APP->Data.objmap;
        else if(this->data == DATA_GRAD){
            this->DrawScnGradient();
            return;
        }
        else if(this->data == DATA_CUSTOM)
            scn = this->custom_scn;
        else
            scn = APP->Data.orig;

        if(scn!=NULL)
            maxval = SceneImax(scn); //MaximumValue3(scn);

        this->FindImageSize(&w, &h);
        sview->ntimes %= 4;
        ntimes = sview->ntimes;
        rw = w; rh = h;
        if(ntimes%2==1){
            rw = h; rh = w;
        }

        cimg  = CreateCImage(rw, rh);

        if(scn!=NULL){
            // paint original image
            for(ri=0; ri<rh; ri++){
                for(rj=0; rj<rw; rj++){
                    RotateBackNTimes(&j, &i, rj, ri, rw, rh, ntimes);

                    p = Slice2voxel(j, i, &v);
                    lb = GetVoxel(APP->Data.label, p);

                    if(this->marker==MARKER_ON)
                        isseed = APP->IsSeed(p);
                    else
                        isseed = false;

                    q = rj + ri*rw;
                    val = GetVoxel(scn, p);

                    if(isseed){
                        color = APP->GetLabelColour(lb);
                        if(color==NIL) color = 0xffffff;
                        /*
                          color = APP->GetLabelColour(APP->GetSeedLabel(p));

                          if(APP->IsMarkedForRemoval(p)){
                          if(color==NIL)
                          color = 0x0000aa;
                          else
                          color = inverseColor(color);
                          }
                          else if(color==NIL)
                          color = 0xffffff;
                        */
                    }
                    else if(APP->GetLabelColour(lb)==NIL){
                        if(this->data == DATA_OBJ)
                            color = 0x000000;
                        else
                            color = xgray(IntegerNormalize(val,0,maxval,0,255));
                    }
                    else{
                        if(this->highlight!=HIGHLIGHT_FILL &&
                           this->highlight!=HIGHLIGHT_INSIDE)
                            color = xgray(IntegerNormalize(val,0,maxval,0,255));
                        else{
                            alpha = (((float)APP->GetLabelAlpha(lb))/255.0);
                            tmp = xgray(IntegerNormalize(val,0,maxval,0,255));
                            color = mergeColorsRGB(tmp,APP->GetLabelColour(lb),alpha);
                        }
                    }
                    cimg->C[0]->val[q] = t0(color);
                    cimg->C[1]->val[q] = t1(color);
                    cimg->C[2]->val[q] = t2(color);
                }
            }
        }

        this->DrawCImage(cimg);
        DestroyCImage(&cimg);

        this->DrawOverlay();
    }


    void SliceCanvas :: DrawScnGradient(){
        SliceView *sview;
        CImage *cimg=NULL;
        ScnGradient *grad;
        int H,S,V;
        int gw,gh,gmax;
        int w,h,i,j,p,q;
        int rw,rh,ri,rj;
        int color,ntimes;
        float theta;
        Voxel v, Cut = GetCutVoxel();

        grad = APP->Data.grad;
        if(!APP->Data.loaded) return;

        sview = (SliceView *)this->GetParent();
        if(Cut.x>=APP->Data.w ||
           Cut.x<0 ||
           Cut.y>=APP->Data.h ||
           Cut.y<0 ||
           Cut.z>=APP->Data.nframes ||
           Cut.z<0){
            Cut.x = APP->Data.w/2;
            Cut.y = APP->Data.h/2;
            Cut.z = APP->Data.nframes/2;
        }

        this->FindImageSize(&w, &h);
        sview->ntimes %= 4;
        ntimes = sview->ntimes;
        rw = w; rh = h;
        if(ntimes%2==1){
            rw = h; rh = w;
        }

        cimg = CreateCImage(rw, rh);

        if(APP->Data.grad!=NULL){
            // paint original image
            gmax = SceneImax(grad->mag);
            for(ri=0; ri<rh; ri++){
                for(rj=0; rj<rw; rj++){
                    RotateBackNTimes(&j, &i, rj, ri, rw, rh, ntimes);

                    p = Slice2voxel(j, i, &v);
                    q = rj + ri*rw;

                    switch(this->axis){
                        case 'z':
                            gw = GetVoxel(grad->Gx, p);
                            gh = GetVoxel(grad->Gy, p);
                            break;
                        case 'x':
                            gw = GetVoxel(grad->Gz, p);
                            gh = GetVoxel(grad->Gy, p);
                            break;
                        case 'y':
                            gw = GetVoxel(grad->Gx, p);
                            gh = GetVoxel(grad->Gz, p);
                            break;
                    }
                    theta = atan2f(-gh, gw);
                    theta -= (ntimes*PI/2.0);
                    if(theta<0.0)    theta += 2.0*PI;
                    if(theta>2.0*PI) theta -= 2.0*PI;
                    H = ROUND(255.*(theta/(2.0*PI)));
                    S = 255;
                    V = xgray(IntegerNormalize(GetVoxel(grad->mag, p),
                                               0,gmax,0,255));
                    color = HSV2RGB(triplet(H,S,V));

                    cimg->C[0]->val[q] = t0(color);
                    cimg->C[1]->val[q] = t1(color);
                    cimg->C[2]->val[q] = t2(color);
                }
            }
        }
        this->DrawCImage(cimg);
        DestroyCImage(&cimg);

        this->DrawOverlay();
    }


    void SliceCanvas :: SetRefreshHandler(RefreshHandler *handler){
        this->drawhandler = handler;
    }


    void SliceCanvas :: ChangeDrawMarker(){
        if( this->drawmarker == true )
            this->drawmarker = false;
        else
            this->drawmarker = true;
    }

    void SliceCanvas :: DrawOverlay(){
        SliceView *sview;
        int w,h,i,j,k,p,q;
        int rw,rh,ri,rj;
        int lb,color,ntimes;
        Pixel u,ru;
        Voxel v, Cut = GetCutVoxel();
        AdjRel *A=NULL;
        float zoom = this->GetZoomLevel();
        int izoom = (zoom<0.99)?(1):(ROUND(zoom));
        wxCoord x=0,y=0;
        wxColour colour;
        SetColor(&colour, 0xFF0000);
        wxPen pen(colour, 1, wxSOLID);

        if(!APP->Data.loaded) return;

        sview = (SliceView *)this->GetParent();
        this->FindImageSize(&w, &h);
        sview->ntimes %= 4;
        ntimes = sview->ntimes;
        rw = w; rh = h;
        if(ntimes%2==1){
            rw = h; rh = w;
        }

        /* paint object borders */
        if(this->highlight!=HIGHLIGHT_OFF &&
           this->highlight!=HIGHLIGHT_INSIDE){
            A = FastCircular(1.0);
            for(ri=0; ri<rh; ri++){
                for(rj=0; rj<rw; rj++){
                    RotateBackNTimes(&j, &i, rj, ri, rw, rh, ntimes);

                    p = Slice2voxel(j, i, &v);
                    lb = GetVoxel(APP->Data.label, p);

                    color = APP->GetLabelColour(lb);
                    if(color==NIL) continue;

                    x = ROUND(rj*zoom);
                    y = ROUND(ri*zoom);
                    for(k=1; k<A->n; k++){
                        ru.x = rj + A->dx[k];
                        ru.y = ri + A->dy[k];

                        if( ru.x>=0 && ru.x<rw && ru.y>=0 && ru.y<rh ){
                            RotateBackNTimes(&(u.x), &(u.y), ru.x, ru.y, rw, rh, ntimes);

                            q = Slice2voxel(u.x, u.y, &v);
                            lb = GetVoxel(APP->Data.label, q);
                            if(color != APP->GetLabelColour(lb)){
                                SetColor(&colour, color);
                                pen.SetColour(colour);
                                ibuf->SetPen(pen);

                                if(A->dx[k]== 1)
                                    ibuf->DrawLine(x+izoom-1, y,
                                                   x+izoom-1, y+izoom);
                                else if(A->dx[k]==-1)
                                    ibuf->DrawLine(x, y,
                                                   x, y+izoom);
                                else if(A->dy[k]== 1)
                                    ibuf->DrawLine(x, y+izoom-1,
                                                   x+izoom, y+izoom-1);
                                else if(A->dy[k]==-1)
                                    ibuf->DrawLine(x, y,
                                                   x+izoom, y);
                            }
                        }
                    }
                }
            }
            DestroyAdjRel(&A);
        }

        wxCoord x1=0,y1=0;
        wxCoord x2=0,y2=0;

        if( this->drawmarker == true ) {
            SetColor(&colour, 0x00FF00);
            pen.SetColour(colour);
            ibuf->SetPen(pen);

            if(this->axis == 'z'){
                RotateNTimes(&rj, &ri, Cut.x, 0,   w, h, ntimes);
                x1 = ROUND(rj*zoom); y1 = ROUND(ri*zoom);
                RotateNTimes(&rj, &ri, Cut.x, h-1, w, h, ntimes);
                x2 = ROUND(rj*zoom); y2 = ROUND(ri*zoom);
                if(x1==x2){ x1 += izoom/2; x2 += izoom/2; }
                if(y1==y2){ y1 += izoom/2; y2 += izoom/2; }
                ibuf->DrawLine(x1, y1, x2, y2);

                RotateNTimes(&rj, &ri, 0, Cut.y,   w, h, ntimes);
                x1 = ROUND(rj*zoom); y1 = ROUND(ri*zoom);
                RotateNTimes(&rj, &ri, w-1, Cut.y, w, h, ntimes);
                x2 = ROUND(rj*zoom); y2 = ROUND(ri*zoom);
                if(x1==x2){ x1 += izoom/2; x2 += izoom/2; }
                if(y1==y2){ y1 += izoom/2; y2 += izoom/2; }
                ibuf->DrawLine(x1, y1, x2, y2);
            }
            else if(this->axis == 'x'){
                RotateNTimes(&rj, &ri, Cut.z, 0,   w, h, ntimes);
                x1 = ROUND(rj*zoom); y1 = ROUND(ri*zoom);
                RotateNTimes(&rj, &ri, Cut.z, h-1, w, h, ntimes);
                x2 = ROUND(rj*zoom); y2 = ROUND(ri*zoom);
                if(x1==x2){ x1 += izoom/2; x2 += izoom/2; }
                if(y1==y2){ y1 += izoom/2; y2 += izoom/2; }
                ibuf->DrawLine(x1, y1, x2, y2);

                RotateNTimes(&rj, &ri, 0, Cut.y,   w, h, ntimes);
                x1 = ROUND(rj*zoom); y1 = ROUND(ri*zoom);
                RotateNTimes(&rj, &ri, w-1, Cut.y, w, h, ntimes);
                x2 = ROUND(rj*zoom); y2 = ROUND(ri*zoom);
                if(x1==x2){ x1 += izoom/2; x2 += izoom/2; }
                if(y1==y2){ y1 += izoom/2; y2 += izoom/2; }
                ibuf->DrawLine(x1, y1, x2, y2);
            }
            else if(this->axis == 'y'){
                RotateNTimes(&rj, &ri, Cut.x, 0,   w, h, ntimes);
                x1 = ROUND(rj*zoom); y1 = ROUND(ri*zoom);
                RotateNTimes(&rj, &ri, Cut.x, h-1, w, h, ntimes);
                x2 = ROUND(rj*zoom); y2 = ROUND(ri*zoom);
                if(x1==x2){ x1 += izoom/2; x2 += izoom/2; }
                if(y1==y2){ y1 += izoom/2; y2 += izoom/2; }
                ibuf->DrawLine(x1, y1, x2, y2);

                RotateNTimes(&rj, &ri, 0, Cut.z,   w, h, ntimes);
                x1 = ROUND(rj*zoom); y1 = ROUND(ri*zoom);
                RotateNTimes(&rj, &ri, w-1, Cut.z, w, h, ntimes);
                x2 = ROUND(rj*zoom); y2 = ROUND(ri*zoom);
                if(x1==x2){ x1 += izoom/2; x2 += izoom/2; }
                if(y1==y2){ y1 += izoom/2; y2 += izoom/2; }
                ibuf->DrawLine(x1, y1, x2, y2);
            }
        }
        this->Refresh();
    }



    int SliceCanvas :: xgray(int value) {
        float v,width,level,l1,l2;
        int B,C;

        APP->GetBriContr(&B, &C);
        v = (float) value;
        level = (1.0 - (float)B/100.0) * 255.0;
        width = (1.0 - (float)C/100.0) * 255.0;
        l1 = level - width/2.0;
        l2 = level + width/2.0;
        if(l1<0)   l1 = 0.0;
        if(l2>255) l2 = 255.0;

        if(value < l1) v = 0.0;
        else if(value >= l2) v = 255.0;
        else{
            v = (value - l1)/(l2-l1);
            v *= 255.0;
        }

        return(gray((int)v));
    }

} //end SliceView namespace
