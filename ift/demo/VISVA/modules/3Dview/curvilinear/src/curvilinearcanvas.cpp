
#include "curvilinearcanvas.h"

namespace Curvilinear
{

void MyDistTrans3(Scene *bin, int *dmap, int *BinCount)
{
    Scene *Dx = NULL, *Dy = NULL, *Dz = NULL, *cont, *cost;
    Queue *Q = NULL;
    int i, p, q, n, xysize, rnk;
    Voxel u, v;
    int *sq = NULL, tmp = INT_MAX, dx, dy, dz;
    AdjRel3 *A6 = Spheric(1.0);

    n  = MAX(bin->xsize, MAX(bin->ysize, bin->zsize));
    sq = AllocIntArray(n);
    for (i = 0; i < n; i++)
        sq[i] = i * i;

    cont = GetBorder3(bin, A6);
    cost = CreateScene(bin->xsize, bin->ysize, bin->zsize);
    Dx = CreateScene(bin->xsize, bin->ysize, bin->zsize);
    Dy = CreateScene(bin->xsize, bin->ysize, bin->zsize);
    Dz = CreateScene(bin->xsize, bin->ysize, bin->zsize);
    n  = bin->xsize * bin->ysize * bin->zsize;
    Q = CreateQueue(2 * (bin->xsize + bin->ysize + bin->zsize) + 3, n);

    for (p = 0; p < n; p++)
    {
        if (bin->data[p] != 0)
        {
            if (cont->data[p] > 0)
            {
                cost->data[p] = 0;
                InsertQueue(Q, cost->data[p] % Q->C.nbuckets, p);
            }
            else
                cost->data[p] = INT_MAX;
        }
        else
        {
            if (cost->data[p] != INT_MIN)
                cost->data[p] = 0;
        }
    }
    DestroyScene(&cont);

    xysize = cost->xsize * cost->ysize;
    rnk = 0;
    while (!EmptyQueue(Q))
    {
        p = RemoveQueue(Q);
        dmap[rnk] = p;
        rnk++;
        BinCount[cost->data[p]]++;

        u.x = (p % xysize) % cost->xsize;
        u.y = (p % xysize) / cost->xsize;
        u.z = p / xysize;
        for (i = 1; i < A6->n; i++)
        {
            v.x = u.x + A6->dx[i];
            v.y = u.y + A6->dy[i];
            v.z = u.z + A6->dz[i];

            if (ValidVoxel(cost, v.x, v.y, v.z))
            {
                q = VoxelAddress(cost, v.x, v.y, v.z);
                if (cost->data[p] < cost->data[q])
                {
                    dx  = Dx->data[p] + abs(A6->dx[i]);
                    dy  = Dy->data[p] + abs(A6->dy[i]);
                    dz  = Dz->data[p] + abs(A6->dz[i]);
                    tmp = sq[dx] + sq[dy] + sq[dz];
                    if (tmp < cost->data[q])
                    {
                        if (cost->data[q] == INT_MAX)
                            InsertQueue(Q, tmp % Q->C.nbuckets, q);
                        else
                            UpdateQueue(Q, q, cost->data[q] % Q->C.nbuckets, tmp % Q->C.nbuckets);
                        cost->data[q] = tmp;
                        Dx->data[q] = dx;
                        Dy->data[q] = dy;
                        Dz->data[q] = dz;
                    }
                }
            }

        }
    }

    DestroyQueue(&Q);
    DestroyAdjRel3(&A6);
    free(sq);
    DestroyScene(&Dx);
    DestroyScene(&Dy);
    DestroyScene(&Dz);
    DestroyScene(&cost);
}


CurvilinearCanvas :: CurvilinearCanvas(wxWindow *parent)
    : RenderCanvas(parent)
{
    light.x =  0;
    light.y =  0;
    light.z = -1;
    ka = 0.20;
    kd = 0.60;
    ks = 0.40;
    sre = 10;
    zgamma = 0.25;
    zstretch = 1.25;
    vxbuf = NULL;
    defaulthandler = new CurvilinearHandler(this);
    SetInteractionHandler(defaulthandler);
}


void CurvilinearCanvas :: SetInteractionHandler(InteractionHandler *handler)
{
    if (handler == NULL)
        this->handler = defaulthandler;
    else
        this->handler = handler;
}

int CurvilinearCanvas :: GetSurfaceVoxel(int p)
{
    return this->vxbuf[p];
}


CImage *CurvilinearCanvas :: Render2CImage(bool dataChanged,
        int skip)
{
    Scene *bin, *bin2, *label;
    Scene *closed = NULL;
    static int *dmap = NULL;
    static int *BinCount = NULL;
    static int prevdist = NIL;
    int p, n, D, mapi, Dmax, w, h, nframes;
    int *bmap = NULL;
    int maplen = 0;
    int distance;
    timer tic, toc;
    float totaltime, fdist;
    float dx = (APP->Data.orig)->dx;
    CurvilinearView *view;
    CImage *cimg;

    if (!APP->Data.loaded) return NULL;
    if (skip <= 0) skip = 1;

    w = APP->Data.w;
    h = APP->Data.h;
    nframes = APP->Data.nframes;
    D = w * w + h * h + nframes * nframes;
    n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    view = (CurvilinearView *)this->GetParent();
    distance = view->GetDistance();

    if ( dataChanged || dmap == NULL )
    {

        APP->Busy((char *)"Please wait, working...");
        APP->StatusMessage((char *)"Please wait - Computation in progress...");

        gettimeofday(&tic, NULL);

        if (dmap != NULL) free(dmap);
        if (BinCount != NULL) free(BinCount);

        dmap = (int *)malloc(sizeof(int) * n);
        BinCount = (int *)malloc(sizeof(int) * D);
        for (p = 0; p < D; p++)
            BinCount[p] = 0;

        label = APP->Data.label;
        bin = CreateScene(w, h, nframes);
        for (p = 0; p < n; p++)
        {
            if (APP->GetLabelColour(label->data[p]) != NIL)
                bin->data[p] = 1;
            else
                bin->data[p] = 0;
        }

        bin2 = AddFrame3(bin, 20, 0);
        DestroyScene(&bin);
        closed = CloseBin3(bin2, 20.0);
        DestroyScene(&bin2);
        bin2 = RemFrame3(closed, 20);
        DestroyScene(&closed);
        closed = bin2;

        MyDistTrans3(closed, dmap, BinCount);
        DestroyScene(&closed);

        Dmax = 0;
        for (p = 0; p < D; p++)
            if (BinCount[p] > 0)
                Dmax = p;
        Dmax = ROUND(sqrt((float)Dmax));
        if (Dmax <= 0) Dmax = 1;
        view->sDist->SetRange(0, ROUND(dx * Dmax));
        view->sDist->SetValue(0);

        gettimeofday(&toc, NULL);
        totaltime = (toc.tv_sec - tic.tv_sec) * 1000.0 + (toc.tv_usec - tic.tv_usec) * 0.001;
        printf("\nCloseBin3 Time: %f ms\n", totaltime);
    }

    mapi = p = 0;
    fdist = (float)distance;
    while (p < D)
    {
        if (sqrt((float)p) >= fdist - 0.5)
            break;
        else
            mapi += BinCount[p];
        p++;
    }
    maplen = 0;
    while (p < D)
    {
        if (sqrt((float)p) >= fdist + 0.5)
            break;
        else
            maplen += BinCount[p];
        p++;
    }
    bmap = (int *)malloc(sizeof(int) * maplen);
    memcpy(bmap, dmap + mapi, sizeof(int)*maplen);

    if (distance != prevdist)
        dataChanged = true;
    prevdist = distance;

    cimg = render_map(bmap, maplen, APP->Data.orig,
                      dataChanged, skip);
    free(bmap);

    APP->StatusMessage((char *)"");
    APP->Unbusy();
    return cimg;
}

void CurvilinearCanvas :: drawRender(bool dataChanged,
                                     int skip)
{
    CImage *cimg_l, *cimg_r, *cimg;
    Matrix *brot;
    float delta = 3.7 * (PI / 180.0); //Toe-in angle.
    CurvilinearView *view;
    bool is3D = false;

    if (!APP->Data.loaded) return;
    if (skip <= 0) skip = 1;

    view = (CurvilinearView *)this->GetParent();
    is3D = view->but3D->GetValue();

    if (is3D)
    {
        brot = CopyMatrix(rot);
        RotateY(delta / 2.0);
        cimg_l = Render2CImage(dataChanged, skip);
        //WriteCImage(cimg_l, "left.ppm");

        RotateY(-delta);
        cimg_r = Render2CImage(dataChanged, skip);
        //WriteCImage(cimg_r, "right.ppm");

        DestroyMatrix(&rot);
        rot = brot;
        this->rotChanged = true;
        cimg = Render2CImage(dataChanged, skip);
        //WriteCImage(cimg, "normal.ppm");

        DestroyCImage(&cimg);
        cimg = Anaglyph3D(cimg_l, cimg_r);
        DestroyCImage(&cimg_l);
        DestroyCImage(&cimg_r);
    }
    else
    {
        cimg = Render2CImage(dataChanged, skip);
    }

    DrawCImage(cimg);
    Refresh();

    DestroyCImage(&cimg);
}

CImage *CurvilinearCanvas :: render_map(int *bmap, int maplen,
                                        Scene *vol,
                                        bool dataChanged,
                                        int skip)
{
    int w, h, w2, h2, W2, H2, D2, WH, W, H, D;
    int i, j, k, m, n, p, q;
    Scene *label;
    static float *xb = NULL, *yb = NULL, *normals = NULL, *zbuf = NULL;
    static int prevskip = NIL;
    float fa, fb;
    Point A, B;
    Vector V1, V2;
    float diag, dz, kz, z, Y;
    int ca;
    int *pm;
    int splatsz;
    int px, py, pz;
    CImage *dest;
    float R[4][4];
    int maxval, c;

    if (skip <= 0) skip = 1;

    label = APP->Data.label;

    R[0][0] = rot->val[0]; R[0][1] = rot->val[1]; R[0][2] = rot->val[2]; R[0][3] = 0.0;
    R[1][0] = rot->val[3]; R[1][1] = rot->val[4]; R[1][2] = rot->val[5]; R[1][3] = 0.0;
    R[2][0] = rot->val[6]; R[2][1] = rot->val[7]; R[2][2] = rot->val[8]; R[2][3] = 0.0;
    R[3][0] = 0.0;         R[3][1] = 0.0;         R[3][2] = 0.0;         R[3][3] = 1.0;

    W = vol->xsize;
    H = vol->ysize;
    D = vol->zsize;

    maxval = MaximumValue3(vol);

    w = MAX(MAX(W, H), D);
    h = w;
    w2 = w / 2;
    h2 = h / 2;
    dest = CreateCImage(w, h);

    W2 = W / 2;
    H2 = H / 2;
    D2 = D / 2;
    WH = W * H;

    VectorNormalize(&light);

    if (this->rotChanged || dataChanged ||
            vxbuf == NULL || prevskip != skip)
    {

        if (vxbuf != NULL)   free(vxbuf);
        if (zbuf != NULL)    free(zbuf);
        if (xb != NULL)      free(xb);
        if (yb != NULL)      free(yb);
        if (normals != NULL) free(normals);

        /* init z-buffer and float (x,y) buffers (for normals) */
        vxbuf = (int *) malloc( w * h * sizeof(int) );
        zbuf = (float *) malloc( w * h * sizeof(float) );
        xb = (float *) malloc( w * h * sizeof(float) );
        yb = (float *) malloc( w * h * sizeof(float) );
        normals = (float *) malloc( w * h * sizeof(float) );

        for (i = 0; i < w * h; i++)
        {
            zbuf[i] = FLT_MAX;
            xb[i] = 0.0;
            yb[i] = 0.0;
            normals[i] = 0.0;
            vxbuf[i] = NIL;
        }

        /* make projection */
        pm = bmap;
        for (i = 0; i < maplen; i += skip, pm += skip)
        {

            n = *pm;

            pz = n / WH;
            py = (n % WH) / W;
            px = (n % WH) % W;

            A.x = px - W2;
            A.y = py - H2;
            A.z = pz - D2;

            V1.x = A.x;
            V1.y = A.y;
            V1.z = A.z;
            V2 = VectorRotate(V1, R);
            B.x = V2.x;
            B.y = V2.y;
            B.z = V2.z;

            px = w2 + (int)B.x;
            py = h2 + (int)B.y;

            if (px < 0 || py < 0 || px >= w || py >= h) continue;

            p = px + py * w;

            if (B.z < zbuf[p])
            {
                zbuf[p] = B.z;
                xb[p] = B.x;
                yb[p] = B.y;
                vxbuf[p] = n;
            }
        }

        /* forward-only 2x2 splatting */

        splatsz = 2;
        if (skip > 2) splatsz = 8;

        for (j = h - splatsz; j >= 0; j--)
            for (i = w - splatsz; i >= 0; i--)
            {
                p = i + j * w;
                if (zbuf[p] != FLT_MAX)
                {
                    fa = zbuf[p];
                    for (k = 0; k < splatsz; k++)
                        for (m = 0; m < splatsz; m++)
                        {
                            fb = zbuf[q = p + k + m * w];
                            if (fa < fb)
                            {
                                zbuf[q] = fa;
                                xb[q] = xb[p] + k;
                                yb[q] = yb[p] + m;
                                vxbuf[q] = vxbuf[p];
                            }
                        }
                }
            }

        /* compute normals */

        for (j = 2; j < h - 2; j++)
            for (i = 2; i < w - 2; i++)
                if (zbuf[i + j * w] != FLT_MAX)
                    render_calc_normal(i, j, w, xb, yb, zbuf, normals);

    }

    /* render the monster */
    A.x = W;
    A.y = H;
    A.z = D;
    diag = VectorMagnitude((Vector *)&A) / 2.0;

    for (j = 0; j < h; j++)
    {
        for (i = 0; i < w; i++)
        {
            p = i + j * w;

            if (zbuf[p] != FLT_MAX)
            {
                z = zbuf[p];
                dz = (diag - z);
                kz = (dz * dz) / (4.0 * diag * diag);

                kz = zgamma + zstretch * kz;

                fa = normals[p];
                fb = phong_specular(fa, sre);

                c = xgray(IntegerNormalize(GetVoxel(vol, vxbuf[p]), 0, maxval, 0, 255));

                ca = RGB2YCbCr(c);
                Y = (float) t0(ca);
                Y /= 255.0;

                //Y = ka + kz * ( kd * Y * fa + ks * Y * fb);

                Y *= 255.0;
                if (Y > 255.0) Y = 255.0;
                if (Y < 0.0) Y = 0.0;
                ca = triplet((int) Y, t1(ca), t2(ca));
                ca = YCbCr2RGB(ca);

                dest->C[0]->val[p] = t0(ca);
                dest->C[1]->val[p] = t1(ca);
                dest->C[2]->val[p] = t2(ca);
            }
        }
    }
    this->rotChanged = false;
    prevskip = skip;

    return dest;
}



void CurvilinearCanvas :: render_calc_normal(int x, int y, int rw, float *xb, float *yb, float *zb, float *out)
{
    int nx[8] = {  0,  1,  1,  1,  0, -1, -1, -1 };
    int ny[8] = { -1, -1,  0,  1,  1,  1,  0, -1 };

    int qx[8] = {  1,  2,  2,  1, -1, -2, -2, -1 };
    int qy[8] = { -2, -1,  1,  2,  2,  1, -1, -2 };

    unsigned int i;
    int k, nv, a;
    float est;

    Vector p[3], w[3], normal = {0.0, 0.0, 0.0};

    nv = 0;
    k  = 1;
    for (i = 0; i < 8; i++)
    {

        a = x + y * rw;
        if (zb[a] == FLT_MAX) continue;
        p[0].x = xb[a];   p[0].y = yb[a];   p[0].z = zb[a];

        a = (x + nx[(i + k) % 8]) + (y + ny[(i + k) % 8]) * rw;
        if (zb[a] == FLT_MAX) continue;
        p[1].x = xb[a];   p[1].y = yb[a];   p[1].z = zb[a];

        a = (x + nx[i]) + (y + ny[i]) * rw;
        if (zb[a] == FLT_MAX) continue;
        p[2].x = xb[a];   p[2].y = yb[a];   p[2].z = zb[a];

        w[0].x = p[1].x - p[0].x;
        w[0].y = p[1].y - p[0].y;
        w[0].z = p[1].z - p[0].z;

        w[1].x = p[2].x - p[0].x;
        w[1].y = p[2].y - p[0].y;
        w[1].z = p[2].z - p[0].z;

        w[2] = VectorProd(w[0], w[1]);
        VectorNormalize(&w[2]);

        ++nv;
        normal.x += w[2].x;
        normal.y += w[2].y;
        normal.z += w[2].z;
    }

    for (i = 0; i < 8; i++)
    {

        a = x + y * rw;
        if (zb[a] == FLT_MAX) continue;
        p[0].x = xb[a];   p[0].y = yb[a];   p[0].z = zb[a];

        a = (x + qx[(i + k) % 8]) + (y + qy[(i + k) % 8]) * rw;
        if (zb[a] == FLT_MAX) continue;
        p[1].x = xb[a];   p[1].y = yb[a];   p[1].z = zb[a];

        a = (x + qx[i]) + (y + qy[i]) * rw;
        if (zb[a] == FLT_MAX) continue;
        p[2].x = xb[a];   p[2].y = yb[a];   p[2].z = zb[a];

        w[0].x = p[1].x - p[0].x;
        w[0].y = p[1].y - p[0].y;
        w[0].z = p[1].z - p[0].z;

        w[1].x = p[2].x - p[0].x;
        w[1].y = p[2].y - p[0].y;
        w[1].z = p[2].z - p[0].z;

        w[2] = VectorProd(w[0], w[1]);
        VectorNormalize(&w[2]);

        ++nv;
        normal.x += w[2].x;
        normal.y += w[2].y;
        normal.z += w[2].z;
    }

    if (!nv)
    {
        est = 1.0;
    }
    else
    {
        VectorNormalize(&normal);
        est = ScalarProd(light, normal);
        if (est < 0.0) est = 0.0;
    }

    a = x + rw * y;
    out[a] = est;
}


float CurvilinearCanvas :: phong_specular(float angcos, int n)
{
    float a, r;

    a = acos(angcos);
    if (a > M_PI / 4.0)
        return 0.0;

    a = cos(2.0 * a);
    r = a;
    while (n != 1)
    {
        r *= a;
        --n;
    }
    return r;
}


int CurvilinearCanvas :: xgray(int value)
{
    float v, width, level, l1, l2;
    int B, C;

    APP->GetBriContr(&B, &C);
    v = (float) value;
    level = (1.0 - (float)B / 100.0) * 255.0;
    width = (1.0 - (float)C / 100.0) * 255.0;
    l1 = level - width / 2.0;
    l2 = level + width / 2.0;
    if (l1 < 0)   l1 = 0.0;
    if (l2 > 255) l2 = 255.0;

    if (value < l1) v = 0.0;
    else if (value >= l2) v = 255.0;
    else
    {
        v = (value - l1) / (l2 - l1);
        v *= 255.0;
    }

    return (gray((int)v));
}


CImage *CurvilinearCanvas :: Anaglyph3D(CImage *cimg_l, CImage *cimg_r)
{
    CImage *cimg;
    int j, i, p;
    if (cimg_l->C[0]->ncols != cimg_r->C[0]->ncols ||
            cimg_l->C[0]->nrows != cimg_r->C[0]->nrows)
    {
        Error((char *)"Incompatible images", (char *)"Anaglyph3D");
    }
    cimg = CreateCImage(cimg_l->C[0]->ncols, cimg_l->C[0]->nrows);
    for (i = 0; i < cimg_l->C[0]->nrows; i++)
    {
        for (j = 0; j < cimg_l->C[0]->ncols; j++)
        {
            p = j + i * cimg_l->C[0]->ncols;
            (cimg->C[0])->val[p] = (cimg_l->C[0])->val[p];
            (cimg->C[1])->val[p] = (cimg_r->C[1])->val[p];
            (cimg->C[2])->val[p] = (cimg_r->C[2])->val[p];
        }
    }
    return cimg;
}


} //end Curvilinear namespace

