#include "interpolation.h"
#include "image.h"
#include "common.h"

int valueInterpol(Point p, Image *img){

    float d_q12_q2, d_q12_q1, d_q34_q3, d_q34_q4, d_q1_q2, d_q3_q4,d_q12_q34;
    float d_p_q12,d_p_q34;
    float I_q12,I_q34;
    int I_p,I_q1,I_q2,I_q3, I_q4;
    int x,X,y,Y; //X,Y maior inteiro proximo, x,y:menor inteiro proximo

    x=(int)p.x;
    X=(int)p.x+1;
    y=(int)p.y;
    Y=(int)p.y+1;

    // point is in pixel center
    if((x==p.x) && (y==p.y))
        return img->val[img->tbrow[y]+x];
    
    // point has 2 neighboors in x direction
    if(y==p.y){
      I_q1=img->val[img->tbrow[y]+x];
      I_q2=img->val[img->tbrow[y]+X];
      d_q12_q1=p.x-x; 
      d_q12_q2=X-p.x;
      d_q1_q2=X-x;
      I_q12=(d_q12_q2*I_q1 + d_q12_q1*I_q2)/d_q1_q2;
      return ((int)I_q12);
    }
    
    // point has 2 neighboors in y direction
    if (x==p.x){
      I_q1=img->val[img->tbrow[y]+x];
      I_q2=img->val[img->tbrow[Y]+x];
      d_q12_q1=p.y-y; 
      d_q12_q2=Y-p.y;
      d_q1_q2=Y-y;
      I_q12=(d_q12_q2*I_q1 + d_q12_q1*I_q2)/d_q1_q2;
      return ((int)I_q12);;
    }
    
    //if point has 4 neighboors
    I_q1=img->val[img->tbrow[y]+x]; I_q2=img->val[img->tbrow[y]+X];
    I_q3=img->val[img->tbrow[Y]+x]; I_q4=img->val[img->tbrow[Y]+X];

    d_q12_q1=d_q34_q3=p.x-x; 
    d_q34_q4=d_q12_q2=X-p.x;

    d_p_q12=p.y-y;
    d_p_q34=Y-p.y;

    d_q1_q2=d_q3_q4=X-x;
    d_q12_q34=Y-y;

    I_q12=(d_q12_q2*I_q1 + d_q12_q1*I_q2)/d_q1_q2;
    I_q34=(d_q34_q4*I_q3 + d_q34_q3*I_q4)/d_q3_q4;
    I_p= (int)((d_p_q34*I_q12 + d_p_q12*I_q34)/d_q12_q34);

    if(I_p<0){

        printf("Negativo: %d  \nPonto[%f][%f]  Vizinhos[%d][%d]  [%d][%d]  [%d][%d]  [%d][%d]\n",I_p,p.x,p.y,x,y,X,y,x,Y,X,Y);
    }
    return I_p;
}

