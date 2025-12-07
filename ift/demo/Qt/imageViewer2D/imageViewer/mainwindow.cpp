#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->label2DImageArea->setStyleSheet("border: 1px solid black");
    ui->label3DImageArea->setStyleSheet("border: 1px solid black");
    ui->label2DImageArea->setVisible(false);
    ui->label3DImageArea->setVisible(false);
    ui->doubleSpinBoxThetaX->setVisible(false);
    ui->doubleSpinBoxThetaY->setVisible(false);

    connect(ui->label3DImageArea,SIGNAL(mousePress(const QPoint&)),
            this, SLOT(clickEvent3dArea(const QPoint&)));
    connect(ui->label3DImageArea,SIGNAL(mouseHolding(const QPoint&)),
            this, SLOT(pressedEvent3dArea(const QPoint&)));
    connect(ui->label3DImageArea,SIGNAL(mouseRelease(const QPoint&)),
            this, SLOT(releasedEvent3dArea(const QPoint&)));

}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::on_actionOpen_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), QString(),
                                                    tr("png images (*.png);;C++ Files (*.cpp *.h)"));

    if (!fileName.isEmpty()) {
        if(image2D!= NULL){
            iftDestroyImage(&image2D);
            iftDestroyImage(&image3D);

        }
        image2D = iftReadImagePNG(fileName.toLatin1().data());
        image3D = iftLiftImage2(image2D,255);
        ui->label2DImageArea->setGeometry(ui->label2DImageArea->x(),
                                          ui->label2DImageArea->y(),
                                          image2D->xsize,
                                          image2D->ysize);

        Qimage2D = image2LabelArea(image2D);
        displayQimageOnLabel(Qimage2D,ui->label2DImageArea);
        lastThetaX = 0;
        lastThetaY = 0;
        imageRendering(image3D,lastThetaX,lastThetaY,&image3DProjection);

        ui->label3DImageArea->setGeometry(ui->label2DImageArea->x()+image2D->xsize+1,
                                          ui->label2DImageArea->y(),
                                          image3DProjection->xsize,
                                          image3DProjection->ysize);
        Qimage3D = image2LabelArea(image3DProjection);
        displayQimageOnLabel(Qimage3D,ui->label3DImageArea);
        ui->doubleSpinBoxThetaX->setGeometry(ui->label3DImageArea->x()+image3DProjection->xsize+1,
                                             ui->label3DImageArea->y(),
                                             66,23);
        ui->doubleSpinBoxThetaY->setGeometry(ui->doubleSpinBoxThetaX->x(),
                                             ui->doubleSpinBoxThetaX->y()+23+1,
                                             66,23);
        ui->label2DImageArea->setVisible(true);
        ui->label3DImageArea->setVisible(true);
        ui->doubleSpinBoxThetaX->setVisible(true);
        ui->doubleSpinBoxThetaY->setVisible(true);
    }

}

QImage* MainWindow::image2LabelArea(iftImage *image){
    QImage* Qimage = new QImage(image->xsize,image->ysize,QImage::Format_RGB32);
    QRgb value = qRgb(0,0,0);
    int k = 0;
    int R,G,B;

    if(image->Cb == NULL){
        for(int y=0; y < image->ysize; y++){
            for(int x=0; x < image->xsize; x++){
                value = qRgb(image->val[k],image->val[k],image->val[k]);
                Qimage->setPixel(x,y,value);
                k++;
            }
        }
    }else{
        iftColor RGB, YCbCR;
        for(int y=0; y < image->ysize; y++){
            for(int x=0; x < image->xsize; x++){
                YCbCR.val[0] = image->val[k];
                YCbCR.val[1] = image->Cb[k];
                YCbCR.val[2] = image->Cr[k];
                RGB = iftYCbCrtoRGB(YCbCR,255);
                R = RGB.val[0];
                G = RGB.val[1];
                B = RGB.val[2];
                value = qRgb(R,G,B);
                Qimage->setPixel(x,y,value);
                k++;
            }
        }
    }
    return Qimage;
}

void MainWindow::displayQimageOnLabel(QImage *image,QLabel *imageArea){
    imageArea->setPixmap(QPixmap::fromImage((*image)));
}

void MainWindow::imageRendering(iftImage* image3D,
                                     float thetax_degree,
                                     float thetay_degree,
                                     iftImage** outputImage){
    float cubeDiagonal;
    //iftImage *outputImage = NULL;
    int cubeDiagonal_voxels;
    iftMatrix *Ry = iftRotationMatrix(IFT_AXIS_Y,thetay_degree);
    iftMatrix *Rx = iftRotationMatrix(IFT_AXIS_X,thetax_degree);
    cubeDiagonal = sqrtf((image3D->xsize*image3D->xsize)+
                         (image3D->ysize*image3D->ysize)+
                         (image3D->zsize*image3D->zsize));
    cubeDiagonal_voxels = (int)cubeDiagonal;
    if((*outputImage) != NULL){
        if((*outputImage)->n != cubeDiagonal_voxels*cubeDiagonal_voxels){
            iftDestroyImage(outputImage);
            *outputImage = iftCreateImage(cubeDiagonal_voxels,cubeDiagonal_voxels,1);
            if(image3D->Cb != NULL){
                (*outputImage)->Cb = iftAllocUShortArray((*outputImage)->n);
                (*outputImage)->Cr = iftAllocUShortArray((*outputImage)->n);
                for (int p=0; p < (*outputImage)->n; p++) {
                    (*outputImage)->Cb[p] = 128;
                    (*outputImage)->Cr[p] = 128;
                }
            }
        }
    }else{
        *outputImage = iftCreateImage(cubeDiagonal_voxels,cubeDiagonal_voxels,1);
        if(image3D->Cb != NULL){
            (*outputImage)->Cb = iftAllocUShortArray((*outputImage)->n);
            (*outputImage)->Cr = iftAllocUShortArray((*outputImage)->n);
            for (int p=0; p < (*outputImage)->n; p++) {
                (*outputImage)->Cb[p] = 128;
                (*outputImage)->Cr[p] = 128;
            }
        }
    }

    iftMatrix* pc_dash = iftIdentityMatrix(4);
    iftMatrix* pc = iftIdentityMatrix(4);

    pc_dash->val[3] = -cubeDiagonal/2.;
    pc_dash->val[7] = -cubeDiagonal/2.;
    pc_dash->val[11] = -cubeDiagonal/2.;

    pc->val[3] = image3D->xsize/2.;
    pc->val[7] = image3D->ysize/2.;
    pc->val[11] = image3D->zsize/2.;


    iftMatrix*aux = iftMultMatrices(Ry,pc_dash);
    iftMatrix*aux2 = iftMultMatrices(Rx,aux);
    iftMatrix*T_inv = iftMultMatrices(pc,aux2);

    iftMatrix *nj = iftCreateMatrix(4,6);
    iftMatrix *cj = iftCreateMatrix(4,6);
    iftMatrix *n = iftCreateMatrix(4,1);

    //-z normal
    nj->val[0] = 0;
    nj->val[1] = 0;
    nj->val[2] = -1;
    nj->val[3] = 0;
    //z normal
    nj->val[4] = 0;
    nj->val[5] = 0;
    nj->val[6] = 1;
    nj->val[7] = 0;
    //-y normal
    nj->val[8] = 0;
    nj->val[9] = -1;
    nj->val[10] = 0;
    nj->val[11] = 0;
    //y normal
    nj->val[12] = 0;
    nj->val[13] = 1;
    nj->val[14] = 0;
    nj->val[15] = 0;
    //-x normal
    nj->val[16] = -1;
    nj->val[17] = 0;
    nj->val[18] = 0;
    nj->val[19] = 0;
    //x normal
    nj->val[20] = -1;
    nj->val[21] = 0;
    nj->val[22] = 0;
    nj->val[23] = 0;
    //-z normal
    cj->val[0] = image3D->xsize/2.;
    cj->val[1] = image3D->ysize/2.;
    cj->val[2] = 0;
    cj->val[3] = 1;
    //z normal
    cj->val[4] = image3D->xsize/2.;
    cj->val[5] = image3D->ysize/2.;
    cj->val[6] = (image3D->zsize);
    cj->val[7] = 1;
    //-y normal
    cj->val[8] = image3D->xsize/2.;
    cj->val[9] = 0;
    cj->val[10] = image3D->zsize/2.;
    cj->val[11] = 1;
    //y normal
    cj->val[12] = image3D->xsize/2.;
    cj->val[13] = image3D->ysize;
    cj->val[14] = image3D->zsize/2.;
    cj->val[15] = 1;
    //-x normal
    cj->val[16] = 0;
    cj->val[17] = image3D->ysize/2.;
    cj->val[18] = image3D->zsize/2.;
    cj->val[19] = 1 ;
    //x normal
    cj->val[20] = image3D->xsize;
    cj->val[21] = image3D->ysize/2.;
    cj->val[22] = image3D->zsize/2.;
    cj->val[23] = 1;

    n->val[0] = 0;
    n->val[1] = 0;
    n->val[2] = 1;
    n->val[3] = 0;


    iftMatrix* p0 = iftCreateMatrix(4,1);
    iftMatrix* p0_t = iftCreateMatrix(4,1);
    iftMatrix* p1 = iftCreateMatrix(4,1);
    iftMatrix* pn = iftCreateMatrix(4,1);
    iftMatrix* n_t = iftCreateMatrix(4,1);
    iftMatrix *vec_aux = iftCreateMatrix(4,1);
    iftMatrix* pointAux = iftCreateMatrix(4,1);

    float lambdaMax;
    float lambdaMin;
    int k;
    float inn;
    float inn2;
    float lambda;
    float MinFloat = std::numeric_limits<float>::min();
    float MaxFloat = std::numeric_limits<float>::max();
    iftTransposeMatrixInPlace(T_inv);
    iftMultMatricesInplace(n,T_inv,&n_t);

    int pixelIndex = 0;
    int v,u,i;
    //#pragma omp parallel for private(v,u,p0,lambdaMax,lambdaMin,k,i,inn,inn2,lambda,pointAux,p1,pn,pixelIndex)
    for (v = 0; v < cubeDiagonal_voxels; ++v) {
        for (u = 0; u < cubeDiagonal_voxels; ++u) {
            (*outputImage)->val[pixelIndex] = 0;
            if((*outputImage)->Cb != NULL){
                (*outputImage)->Cb[pixelIndex] = 128;
                (*outputImage)->Cr[pixelIndex] = 128;
            }
            p0->val[0] = u;
            p0->val[1] = v;
            p0->val[2] = -cubeDiagonal/2;
            p0->val[3] = 1;

            lambdaMax = MinFloat;
            lambdaMin = MaxFloat;
            iftMultMatricesInplace(p0,T_inv,&p0_t);
            k = 0;

            for (i=0; i<6; i++){
                k = i*4;
                vec_aux->val[0] = cj->val[k+0] - p0_t->val[0];
                vec_aux->val[1] = cj->val[k+1] - p0_t->val[1];
                vec_aux->val[2] = cj->val[k+2] - p0_t->val[2];

                inn = nj->val[k+0]*vec_aux->val[0];
                inn += nj->val[k+1]*vec_aux->val[1];
                inn += nj->val[k+2]*vec_aux->val[2];

                inn2 = nj->val[k+0]*n_t->val[0];
                inn2 += nj->val[k+1]*n_t->val[1];
                inn2 += nj->val[k+2]*n_t->val[2];
                inn2 = inn2 + 0.0000001;
                lambda = inn/inn2;

                pointAux->val[0] = (int)(p0_t->val[0] + n_t->val[0]*lambda);
                pointAux->val[1] = (int)(p0_t->val[1] + n_t->val[1]*lambda);
                pointAux->val[2] = (int)(p0_t->val[2] + n_t->val[2]*lambda);
                pointAux->val[3] = 1;

                if(pointAux->val[0] <= image3D->xsize && pointAux->val[1] <= image3D->ysize && pointAux->val[2] <= image3D->zsize){
                    if(pointAux->val[0] >= 0 && pointAux->val[1] >= 0 && pointAux->val[2] >= 0){
                        if(lambda>lambdaMax){
                            lambdaMax = lambda;
                        }
                        if(lambda < lambdaMin){
                            lambdaMin = lambda;
                        }

                    }
                }

            }
            if(lambdaMax - lambdaMin > 0.01){
                p1->val[0] = (int)(p0_t->val[0] + n_t->val[0]*lambdaMin);
                p1->val[1] = (int)(p0_t->val[1] + n_t->val[1]*lambdaMin);
                p1->val[2] = (int)(p0_t->val[2] + n_t->val[2]*lambdaMin);
                p1->val[3] = 1;

                pn->val[0] = (int)(p0_t->val[0] + n_t->val[0]*lambdaMax);
                pn->val[1] = (int)(p0_t->val[1] + n_t->val[1]*lambdaMax);
                pn->val[2] = (int)(p0_t->val[2] + n_t->val[2]*lambdaMax);
                pn->val[3] = 1;

                computeBrightValue(p0_t,p1,pn,image3D,*outputImage,
                                   pixelIndex,cubeDiagonal,255,lambdaMin);
            }
            pixelIndex++;
        }
    }
    iftDestroyMatrix(&T_inv);
    iftDestroyMatrix(&cj);
    iftDestroyMatrix(&nj);
    iftDestroyMatrix(&aux);
    iftDestroyMatrix(&aux2);
    iftDestroyMatrix(&Rx);
    iftDestroyMatrix(&Ry);
    iftDestroyMatrix(&pc_dash);
    iftDestroyMatrix(&pc);
    iftDestroyMatrix(&pointAux);
    iftDestroyMatrix(&vec_aux);
    iftDestroyMatrix(&n);
    iftDestroyMatrix(&n_t);
    iftDestroyMatrix(&p1);
    iftDestroyMatrix(&pn);
    iftDestroyMatrix(&p0);
    iftDestroyMatrix(&p0_t);
}

void MainWindow::computeBrightValue(iftMatrix *p0, iftMatrix *p1,iftMatrix *pn,
                                    iftImage *image3D,iftImage *outputImage,
                                    int pixelIndex, float cubeDiagonal,
                                    int imageMaximumValue, float lambdaMin){

    iftMatrix* delta = iftSubtractMatrices(pn,p1);
    float deltaZ = fabs(delta->val[2]);
    float deltaY = fabs(delta->val[1]);
    float deltaX = fabs(delta->val[0]);
    float a1,a2,b1,b2;
    int x,y,z;
    int step;
    int index3D_image;
    float depthObject;
    float depthCube;
    float depthEnvironment;
    float fact1;
    float fact2 = ((cubeDiagonal-255)/cubeDiagonal);
    iftVoxel voxel;
    float factor = (cubeDiagonal/2)/(lambdaMin);
    factor = (factor>1)?1:factor;

    p1->val[0] = (p1->val[0] >= image3D->xsize) ? image3D->xsize-1 : p1->val[0];
    p1->val[1] = (p1->val[1] >= image3D->ysize) ? image3D->ysize-1 : p1->val[1];
    p1->val[2] = (p1->val[2] >= image3D->zsize) ? image3D->zsize-1 : p1->val[2];
    pn->val[0] = (pn->val[0] >= image3D->xsize) ? image3D->xsize-1 : pn->val[0];
    pn->val[1] = (pn->val[1] >= image3D->ysize) ? image3D->ysize-1 : pn->val[1];
    pn->val[2] = (pn->val[2] >= image3D->zsize) ? image3D->zsize-1 : pn->val[2];

    if(deltaZ >= deltaY && deltaZ >= deltaX){
        a1 = (pn->val[0]-p1->val[0])/(pn->val[2]-p1->val[2] + 0.00001);
        b1 = p1->val[0] - a1*p1->val[2];
        a2 = (pn->val[1]-p1->val[1])/(pn->val[2]-p1->val[2] + 0.00001);
        b2 = p1->val[1] - a2*p1->val[2];

        if(delta->val[2] >= 0){
            step = 1;
        }else{
            step = -1;
        }
        z = p1->val[2];
        for(int l=0;l < deltaZ;l++){
            x = a1*z + b1;
            y = a2*z + b2;
            voxel.x = x;
            voxel.y = y;
            voxel.z = z;
            index3D_image = iftGetVoxelIndex(image3D,voxel);
            if (image3D->val[index3D_image] > 0){
                depthEnvironment = computeDiagonal(p1->val[0]-p0->val[0],p1->val[1]-p0->val[1],p1->val[2]-p0->val[2]);
                depthObject = computeDiagonal(x-p1->val[0],y-p1->val[1],z-p1->val[2]);
                fact1 = depthObject/cubeDiagonal;
                fact1 = (1-fact1)*255;
                //fact1 = (1-fact1);
                //fact2 = depthEnvironment/factor;
                //fact2 = (fact2 > 255)?255:fact2;
                outputImage->val[pixelIndex] = (factor*fact1);
                if(outputImage->Cb != NULL){
                    outputImage->Cb[pixelIndex] = image3D->Cb[index3D_image];
                    outputImage->Cr[pixelIndex] = image3D->Cr[index3D_image];
                }
                break;
            }
            z += step;
        }
    }else if(deltaY >= deltaZ && deltaY >= deltaX){
        a1 = (pn->val[0]-p1->val[0])/(pn->val[1]-p1->val[1] + 0.00001);
        b1 = p1->val[0] - a1*p1->val[1];
        a2 = (pn->val[2]-p1->val[2])/(pn->val[1]-p1->val[1] + 0.00001);
        b2 = p1->val[2] - a2*p1->val[1];

        if(delta->val[1] >= 0){
            step = 1;
        }else{
            step = -1;
        }
        y = p1->val[1];
        for(int l=0;l < deltaY;l++){
            x = a1*y + b1;
            z = a2*y + b2;
            voxel.x = x;
            voxel.y = y;
            voxel.z = z;
            index3D_image = iftGetVoxelIndex(image3D,voxel);
            if (image3D->val[index3D_image] > 0){
                depthEnvironment = computeDiagonal(p1->val[0]-p0->val[0],p1->val[1]-p0->val[1],p1->val[2]-p0->val[2]);
                depthObject = computeDiagonal(x-p1->val[0],y-p1->val[1],z-p1->val[2]);
                fact1 = depthObject/cubeDiagonal;
                fact1 = (1-fact1)*255;
                //fact1 = (1-fact1);
                //fact2 = depthEnvironment/factor;
                //fact2 = (fact2 > 255)?255:fact2;
                outputImage->val[pixelIndex] = (factor*fact1);
                if(outputImage->Cb != NULL){
                    outputImage->Cb[pixelIndex] = image3D->Cb[index3D_image];
                    outputImage->Cr[pixelIndex] = image3D->Cr[index3D_image];
                }
                break;
            }
            y += step;
        }
    }else if(deltaX >= deltaZ && deltaX>=deltaY){
        a1 = (pn->val[1]-p1->val[1])/(pn->val[0]-p1->val[0] + 0.00001);
        b1 = p1->val[1] - a1*p1->val[0];
        a2 = (pn->val[2]-p1->val[2])/(pn->val[0]-p1->val[0] + 0.00001);
        b2 = p1->val[2] - a2*p1->val[0];

        if(delta->val[0] >= 0){
            step = 1;
        }else{
            step = -1;
        }
        x = p1->val[0];
        for(int l=0;l < deltaX;l++){
            y = a1*x + b1;
            z = a2*x + b2;
            voxel.x = x;
            voxel.y = y;
            voxel.z = z;
            index3D_image = iftGetVoxelIndex(image3D,voxel);
            if (image3D->val[index3D_image] > 0){
                depthEnvironment = computeDiagonal(p1->val[0]-p0->val[0],p1->val[1]-p0->val[1],p1->val[2]-p0->val[2]);
                depthObject = computeDiagonal(x-p1->val[0],y-p1->val[1],z-p1->val[2]);
                fact1 = depthObject/cubeDiagonal;
                fact1 = (1-fact1)*255;
                //fact1 = (1-fact1);
                //fact2 = depthEnvironment/factor;
                //fact2 = (fact2 > 255)?255:fact2;
                outputImage->val[pixelIndex] = (factor*fact1);
                if(outputImage->Cb != NULL){
                    outputImage->Cb[pixelIndex] = image3D->Cb[index3D_image];
                    outputImage->Cr[pixelIndex] = image3D->Cr[index3D_image];
                }
                break;
            }
            x += step;
        }
    }
    iftDestroyMatrix(&delta);
}




void MainWindow::on_doubleSpinBoxThetaX_valueChanged(double arg1)
{
    if(!isMouseEvent){
        imageRendering(image3D,arg1, ui->doubleSpinBoxThetaY->value(),&image3DProjection);
        Qimage3D = image2LabelArea(image3DProjection);
        displayQimageOnLabel(Qimage3D,ui->label3DImageArea);
        lastThetaX = arg1;
    }
}

void MainWindow::on_doubleSpinBoxThetaY_valueChanged(double arg1)
{
    if(!isMouseEvent){
        imageRendering(image3D,ui->doubleSpinBoxThetaX->value(),arg1,&image3DProjection);
        Qimage3D = image2LabelArea(image3DProjection);
        displayQimageOnLabel(Qimage3D,ui->label3DImageArea);
        lastThetaY = arg1;
    }
}


float MainWindow::computeDiagonal(float x,float y,float z){
    return sqrtf(x*x + y*y + z*z);
}


void MainWindow::clickEvent3dArea(const QPoint& pos){
    mouseX_start = pos.x()-(ui->label3DImageArea->width()/2.);
    mouseY_start = pos.y()-(ui->label3DImageArea->height()/2.);
    isMouseEvent = true;
}



void MainWindow::pressedEvent3dArea(const QPoint& pos){
    mouseX_Delta = pos.x()-(ui->label3DImageArea->width()/2.) - mouseX_start;
    mouseY_Delta = pos.y()-(ui->label3DImageArea->height()/2.) - mouseY_start;
    thetaXaccumulated = -mouseY_Delta/(ui->label3DImageArea->height()/2.)*90.0;
    thetaYaccumulated = mouseX_Delta/(ui->label3DImageArea->width()/2.)*90.0;

    if(renderIterator == 0){
        //fprintf(stderr,"theta: %f",ThetaXaccumulated);
        imageRendering(image3D,
                      lastThetaX+thetaXaccumulated,
                      lastThetaY+thetaYaccumulated,
                       &image3DProjection);
        Qimage3D = image2LabelArea(image3DProjection);
        displayQimageOnLabel(Qimage3D,ui->label3DImageArea);
        ui->doubleSpinBoxThetaX->setValue(lastThetaX+thetaXaccumulated);
        ui->doubleSpinBoxThetaY->setValue(lastThetaY+thetaYaccumulated);

    }
    renderIterator++;
    if(renderIterator == 5){
        renderIterator = 0;
    }
}

void MainWindow::releasedEvent3dArea(const QPoint& pos){
    mouseX_Delta = pos.x()-(ui->label3DImageArea->width()/2.) - mouseX_start;
    mouseY_Delta = pos.y()-(ui->label3DImageArea->height()/2.) - mouseY_start;
    thetaXaccumulated = -mouseY_Delta/(ui->label3DImageArea->height()/2.)*90.0;
    thetaYaccumulated = mouseX_Delta/(ui->label3DImageArea->width()/2.)*90.0;

    imageRendering(image3D,
                  lastThetaX+thetaXaccumulated,
                  lastThetaY+thetaYaccumulated,
                  &image3DProjection);
    Qimage3D = image2LabelArea(image3DProjection);
    displayQimageOnLabel(Qimage3D,ui->label3DImageArea);

    ui->doubleSpinBoxThetaX->setValue(lastThetaX+thetaXaccumulated);
    ui->doubleSpinBoxThetaY->setValue(lastThetaY+thetaYaccumulated);

    lastThetaX = lastThetaX+thetaXaccumulated;
    lastThetaY = lastThetaY+thetaYaccumulated;
    isMouseEvent = false;
}
