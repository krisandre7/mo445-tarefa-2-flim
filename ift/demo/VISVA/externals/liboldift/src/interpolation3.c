#include "interpolation3.h"
#include "scene.h"
#include "common.h"
#include "interpolation.h"

int valueInterpol3(Point p, Scene *scn){

  float d_q12_q2, d_q12_q1, d_q34_q3, d_q34_q4, d_q1_q2, d_q3_q4,d_q12_q34;
  float d_q56_q6, d_q56_q5, d_q78_q7, d_q78_q8, d_q5_q6, d_q7_q8,d_q56_q78;
  float d_p1_q12,d_p1_q34,d_p2_q56,d_p2_q78,d_p1_p2;
  float d_p_p1,d_p_p2,d_p_q12,d_p_q34;
  float I_q12,I_q34,I_q56,I_q78;
  float I_p1,I_p2;
  int I_p,I_q1,I_q2,I_q3, I_q4,I_q5,I_q6,I_q7,I_q8;
  int x,X,y,Y,z,Z; //X,Y maior inteiro proximo, x,y:menor inteiro proximo

  x=(int)p.x;
  X=(int)p.x+1;
  y=(int)p.y;
  Y=(int)p.y+1;
  z=(int)p.z;
  Z=(int)p.z+1;

  //if point is in the center of the slice
  if(z==p.z){
     if((x==p.x) && (y==p.y))
        return scn->data[scn->tbz[z]+scn->tby[y]+x];
    
    // point has 2 neighboors in x direction
    if(y==p.y){
      I_q1=scn->data[scn->tbz[z]+scn->tby[y]+x];
      I_q2=scn->data[scn->tbz[z]+scn->tby[y]+X];
      d_q12_q1=p.x-x; 
      d_q12_q2=X-p.x;
      d_q1_q2=X-x;
      I_q12=(d_q12_q2*I_q1 + d_q12_q1*I_q2)/d_q1_q2;
      return ((int)I_q12);
    }
    
    // point has 2 neighboors in y direction
    if (x==p.x){
      I_q1=scn->data[scn->tbz[z]+scn->tby[y]+x];
      I_q2=scn->data[scn->tbz[z]+scn->tby[Y]+x];
      d_q12_q1=p.y-y; 
      d_q12_q2=Y-p.y;
      d_q1_q2=Y-y;
      I_q12=(d_q12_q2*I_q1 + d_q12_q1*I_q2)/d_q1_q2;
      return ((int)I_q12);;
    }
    
    //if point has 4 neighboors
    I_q1=scn->data[scn->tbz[z]+scn->tby[y]+x]; I_q2=scn->data[scn->tbz[z]+scn->tby[y]+X];
    I_q3=scn->data[scn->tbz[z]+scn->tby[Y]+x]; I_q4=scn->data[scn->tbz[z]+scn->tby[Y]+X];

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

        printf("Negativo: %d  \nPonto[%f][%f]  Vizinhos[%d][%d]  [%d][%d] [%d][%d] [%d][%d] \n",I_p,p.x,p.y,x,y,X,y,x,Y,X,Y);
    }
    return I_p;
  }
  
  //if point has 2 neighbors in z direction
  if(y==p.y && x==p.x && z!=p.z){
      I_q1=scn->data[scn->tbz[(int)z]+ scn->tby[(int)y]+(int)x];
      I_q2=scn->data[scn->tbz[(int)Z]+ scn->tby[(int)y]+(int)x];
      d_q12_q1 =p.z-z;
      d_q12_q2 =Z-p.z;
      d_q1_q2=Z-z;
      I_q12=(d_q12_q2*I_q1 + d_q12_q1*I_q2)/d_q1_q2;
      return((int)I_q12);
  }
  
 
  
 //if point has the 8 neighbors 
 I_q1=scn->data[scn->tbz[(int)z]+ scn->tby[(int)y]+(int)x]; I_q2=scn->data[scn->tbz[(int)z]+ scn->tby[(int)y]+(int)X];
 I_q3=scn->data[scn->tbz[(int)z]+ scn->tby[(int)Y]+(int)x]; I_q4=scn->data[scn->tbz[(int)z]+ scn->tby[(int)Y]+(int)X];

 I_q5=scn->data[scn->tbz[(int)Z]+ scn->tby[(int)y]+(int)x]; I_q6=scn->data[scn->tbz[(int)Z]+ scn->tby[(int)y]+(int)X];
 I_q7=scn->data[scn->tbz[(int)Z]+ scn->tby[(int)Y]+(int)x]; I_q8=scn->data[scn->tbz[(int)Z]+ scn->tby[(int)Y]+(int)X];
	
		
d_q12_q1 = d_q34_q3 = d_q56_q5 = d_q78_q7 = p.x-x; 
d_q34_q4 = d_q12_q2 = d_q56_q6 = d_q78_q8 = X-p.x;

d_p1_q12 = d_p2_q56 = p.y-y;
d_p1_q34 = d_p2_q78 = Y-p.y;
	
d_q1_q2 = d_q3_q4 = d_q5_q6 = d_q7_q8 = X-x;
d_q12_q34 = d_q56_q78= Y-y;
	
I_q12=(d_q12_q2*I_q1 + d_q12_q1*I_q2)/d_q1_q2;
I_q34=(d_q34_q4*I_q3 + d_q34_q3*I_q4)/d_q3_q4;
I_p1=(d_p1_q34*I_q12 + d_p1_q12*I_q34)/d_q12_q34;
	
I_q56=(d_q56_q6*I_q5 + d_q56_q5*I_q6)/d_q5_q6;
I_q78=(d_q78_q8*I_q7 + d_q78_q7*I_q8)/d_q7_q8;
I_p2=(d_p2_q78*I_q56 + d_p2_q56*I_q78)/d_q56_q78;
	
d_p_p1=p.z-z;
d_p_p2=Z-p.z;
d_p1_p2=Z-z;
	
I_p= (int)((d_p_p2*I_p1 + d_p_p1*I_p2)/d_p1_p2);
if(I_p<0){
    printf("Negativo: %d  \nPonto[%f][%f]  Vizinhos[%d][%d]  [%d][%d] [%d][%d] [%d][%d]\n",I_p,p.x,p.y,x,y,X,y,x,Y,X,Y);
}

return I_p;

}

