#include "prscene.h"

/*** Compute voxels in the border of the objects and the size of the
     voxel list ***/

Scene *PRSObjectBorder(Scene *label, int *nvoxels)
{
  Scene   *border=CreateScene(label->xsize,label->ysize,label->zsize);
  AdjRel3 *A=Spheric(1.0);
  Voxel u,v;
  int p,q,i;  

  *nvoxels = 0;
  for (u.z=0; u.z < label->zsize; u.z++){
    for (u.y=0; u.y < label->ysize; u.y++){
      for (u.x=0; u.x < label->xsize; u.x++){
	p = u.x + label->tby[u.y] + label->tbz[u.z];
	if (label->data[p]>0){ // p belongs to an object 
	  for (i=1; i < A->n; i++){
	    v.x = u.x + A->dx[i];
	    v.y = u.y + A->dy[i];
	    v.z = u.z + A->dz[i];
	    if (ValidVoxel(label,v.x,v.y,v.z)){
	      q = v.x + label->tby[v.y] + label->tbz[v.z];
	      if (label->data[p] != label->data[q]){// p belongs to its border
		border->data[p]=1;
		*nvoxels=*nvoxels+1;
		break;
	      }
	    }else {
	      border->data[p]=1;
	      *nvoxels=*nvoxels+1;
	      break;
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);

  return(border);
}


/*** Create voxel list with attributes ***/

Voxel_attr *PRSCreateVoxelList(Scene *grey, Scene *label, int *nvoxels)
{
  Voxel_attr *voxel;
  Voxel u;
  int p,i;
  AdjRel3 *A=Spheric(5.0),*A26=Spheric(1.8);
  AdjVxl  *V=AdjVoxels(grey,A);
  Vector normal;
  Scene *border;
  Scene *dist=DistTrans3(label,A26,BOTH,0);

  /* Compute voxels for visualization */

  border = PRSObjectBorder(label,nvoxels);

  voxel  = (Voxel_attr *) calloc(*nvoxels,sizeof(Voxel_attr));

  i = 0;
  for (u.z=0; u.z < border->zsize; u.z++){
    for (u.y=0; u.y < border->ysize; u.y++){
      for (u.x=0; u.x < border->xsize; u.x++){
	  p = u.x + border->tby[u.y] + border->tbz[u.z];
	  if (border->data[p]==1){ /* p belongs to the voxel list */
	    voxel[i].x = (ushort)(u.x);
	    voxel[i].y = (ushort)(u.y);
	    voxel[i].z = (ushort)(u.z);
	    voxel[i].object_index = (uchar) label->data[p];
	    // compute normal index
	    Gradient3(grey, &u, &normal, A, V);
	    //DistGradient3(dist, &u, &normal, A, V);
	    VectorNormalize(&normal);
	    voxel[i].normal_index = GetNormalIndex(&normal);
	    i++;
	  }
      }
    }
  }

  DestroyScene(&dist);
  DestroyScene(&border);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
  return(voxel);
}

Obj_attr *PRSCreateObjectTable()
{
  Obj_attr *otable=(Obj_attr *)calloc(256,sizeof(Obj_attr));
  int i;
  // Set default opacity and visibility
    for (i=1; i < 256; i++) {
      otable[i].opac  = 255;
      otable[i].visib = 1;
    }
    otable[0].opac  = 0; // background
    otable[0].visib = 0; 
    // Set color table: Karina, coloque aqui uma tabela de cores 
    for (i=1; i < 256; i++) {
      otable[i].R = otable[i].G = otable[i].B = 255;
    }
    return(otable);
}

/***  Create a PRScene ***/

PRScene *CreatePRScene(Scene *grey, Scene *label) {
  PRScene *prs=(PRScene *)calloc(1,sizeof(PRScene));
  
  // Create voxel list with attributes
  
  prs->voxel = PRSCreateVoxelList(grey,label,&(prs->nvoxels));
  prs->xsize = label->xsize;
  prs->ysize = label->ysize;
  prs->zsize = label->zsize;
  
  // Compute diagonal
  
  prs->diagonal = (int)(sqrt(prs->xsize*prs->xsize + prs->ysize*prs->ysize + prs->zsize*prs->zsize)+0.5); 
  
  // Allocate lookup tables of rendering parameters
  prs->lkt_a1   = AllocFloatArray(1.5*prs->diagonal);
  prs->lkt_a2   = AllocFloatArray(1.5*prs->diagonal);
  prs->lkt_A2b1 = AllocFloatArray(1.5*prs->diagonal);
  prs->lkt_C2b1 = AllocFloatArray(1.5*prs->diagonal);
  prs->lkt_B2b2 = AllocFloatArray(1.5*prs->diagonal);
  prs->lkt_D2b2 = AllocFloatArray(1.5*prs->diagonal); 
  
  //Create lookup tables that accelerate shading
  prs->normal_table = CreateNormalTable();
  prs->object_table = PRSCreateObjectTable();
  prs->nobjects = MaximumValue3(label);
  prs->depth_shading_table = AllocIntArray(prs->diagonal);
  prs->d_min = 0;
  prs->d_max = prs->diagonal-1;
  
  //Allocate space for index buffer, depth buffer, and final image
  
  prs->index_buffer = AllocIntArray(prs->diagonal*prs->diagonal);
  prs->depth_buffer = AllocFloatArray(prs->diagonal*prs->diagonal);
  prs->cimg = CreateCImage(prs->diagonal,prs->diagonal);
  
  // Initialize rendering parameters
  
  SimplerUpdatePRSceneView(prs,0,0);
  
  return(prs);
}

/***  Destroy a PRScene ***/
void DestroyPRScene(PRScene **prs)
{
  PRScene *paux=*prs;

  if (paux != NULL) {
    DestroyCImage(&(paux->cimg));
    free(paux->depth_buffer);
    free(paux->index_buffer);
    free(paux->depth_shading_table);
    free(paux->object_table);
    free(paux->normal_table);
    free(paux->lkt_B2b2);
    free(paux->lkt_D2b2);
    free(paux->lkt_A2b1);
    free(paux->lkt_C2b1);
    free(paux->lkt_a2);
    free(paux->lkt_a1);
    free(paux);
    *prs = NULL;
  }
}

/*** Update all parameters that depend on thetaX and thetaY ***/

void UpdatePRSceneView(PRScene *prs, float thetaX, float thetaY)
{
  float Rx[4][4],Ry[4][4];
  float Raux[4][4], Rnew[4][4];
  float A1,B1,C1,D1,E1,F1,Delta,A2,B2,C2,D2, min=0;
  float b11=0, b21=0, b12=0, b22=0, b13=0, b23=0, b14=0, b24=0; 
  float a1=0,a2=0;
  double trash;
  int i, j, n;

  // reset viewing buffers

  n = prs->diagonal*prs->diagonal; 
  for (i=0; i < n; i++) {
    prs->cimg->C[0]->val[i]=prs->cimg->C[1]->val[i]=
      prs->cimg->C[2]->val[i]=0;
    prs->depth_buffer[i]=INT_MAX;
    prs->index_buffer[i]=NIL;
  }

  // update rendering parameters: Karina: colocar copia do update como
  // reset com a primeira parte do if. Retirar esta parte deste
  // funcao. Trocar update por reset em createprscene.

  if ((thetaX==0.0)&&(thetaY==0.0)){
    prs->n.x = 0; 
    prs->n.y = 0; 
    prs->n.z = 1;
    prs->paxis = 'z';
    prs->p1.x = -prs->diagonal/2.0; 
    prs->p1.y = -prs->diagonal/2.0; 
    prs->p1.z = -prs->diagonal/2.0;
    prs->p2.x =  prs->diagonal/2.0; 
    prs->p2.y = -prs->diagonal/2.0; 
    prs->p2.z = -prs->diagonal/2.0;
    prs->p3.x = -prs->diagonal/2.0; 
    prs->p3.y =  prs->diagonal/2.0; 
    prs->p3.z = -prs->diagonal/2.0;
    prs->p4.x =  prs->diagonal/2.0; 
    prs->p4.y =  prs->diagonal/2.0; 
    prs->p4.z = -prs->diagonal/2.0;
    prs->R[0][0] = 1; prs->R[0][1] = 0; prs->R[0][2] = 0; prs->R[0][3] = 0;
    prs->R[1][0] = 0; prs->R[1][1] = 1; prs->R[1][2] = 0; prs->R[1][3] = 0;
    prs->R[2][0] = 0; prs->R[2][1] = 0; prs->R[2][2] = 1; prs->R[2][3] = 0;
    prs->R[3][0] = 0; prs->R[3][1] = 0; prs->R[3][2] = 0; prs->R[3][3] = 1;
  } else {

    thetaX = (float)(360.*modf(thetaX/360.,&trash));
    if (thetaX < 0.0)
      thetaX += 360.0;
    thetaY = (float)(360.*modf(thetaY/360.,&trash));
    if (thetaY < 0.0)
      thetaY += 360.0;
  
    thetaX = thetaX*PI/180.;
    thetaY = thetaY*PI/180.;

    // create Rx matrix
    Rx[0][3] = Rx[1][3] = Rx[2][3] = Rx[3][2] = Rx[3][1] = Rx[3][0] = 0;
    Rx[3][3] = 1;
    Rx[0][0] = 1;
    Rx[0][1] = Rx[0][2] = Rx[1][0] = Rx[2][0] = 0;
    Rx[1][1] = Rx[2][2] = cos(thetaX);
    Rx[1][2] = sin(thetaX);
    Rx[2][1] = -sin(thetaX);
    // create Ry matrix
    Ry[0][3] = Ry[1][3] = Ry[2][3] = Ry[3][2] = Ry[3][1] = Ry[3][0] = 0;
    Ry[3][3] = 1;
    Ry[1][1] = 1;
    Ry[0][1] = Ry[1][0] = Ry[2][1] = Ry[1][2] = 0;
    Ry[0][0] = Ry[2][2] = cos(thetaY);
    Ry[0][2] = -sin(thetaY);
    Ry[2][0] = sin(thetaY);
    MultMatrices(Rx,Ry,Raux); // Raux = Rx*Ry
    MultMatrices(Raux,prs->R,Rnew); // Rnew = Raux*prs->R
    // Update prs->R
    for(i=0; i<4; i++)
      for(j=0; j<4; j++)
	prs->R[i][j]=Rnew[i][j];
    // Update n, p1, p2, p3, p4
    //prs->n  = (Vector) RotatePoint(prs->R,(Point)prs->n);
	prs->n  = RotatePoint(prs->R,prs->n);
	trash   = sqrt(prs->n.x*prs->n.x + prs->n.y*prs->n.y + prs->n.z*prs->n.z);
    prs->n.x /= trash;
    prs->n.y /= trash;
    prs->n.z /= trash;

    prs->paxis = PAxis(prs->n);
    prs->p1 = RotatePoint(prs->R,prs->p1);
    prs->p2 = RotatePoint(prs->R,prs->p2);
    prs->p3 = RotatePoint(prs->R,prs->p3);
    prs->p4 = RotatePoint(prs->R,prs->p4);    
  }

  switch(prs->paxis) {
    
  case 'x':   
    min = -prs->xsize/2.0;
    a1  = prs->n.z / prs->n.x;
    a2  = prs->n.y / prs->n.x;
    
    b11 = -a1*prs->p1.x + prs->p1.z;
    b12 = -a1*prs->p2.x + prs->p2.z;
    b13 = -a1*prs->p3.x + prs->p3.z;
    b14 = -a1*prs->p4.x + prs->p4.z;
    b21 = -a2*prs->p1.x + prs->p1.y;
    b22 = -a2*prs->p2.x + prs->p2.y;
    b23 = -a2*prs->p3.x + prs->p3.y;
    b24 = -a2*prs->p4.x + prs->p4.y;
    break;
    
  case 'y':
    min = -prs->ysize/2.0;
    a1  = prs->n.x / prs->n.y;
    a2  = prs->n.z / prs->n.y;
    
    b11 = -a1*prs->p1.y + prs->p1.x;
    b12 = -a1*prs->p2.y + prs->p2.x;
    b13 = -a1*prs->p3.y + prs->p3.x;
    b14 = -a1*prs->p4.y + prs->p4.x;
    b21 = -a2*prs->p1.y + prs->p1.z;
    b22 = -a2*prs->p2.y + prs->p2.z;
    b23 = -a2*prs->p3.y + prs->p3.z;
    b24 = -a2*prs->p4.y + prs->p4.z;
    break;
    
  case 'z':
    min = -prs->zsize/2.0;
    a1 = prs->n.x / prs->n.z;
    a2 = prs->n.y / prs->n.z;

    b11 = -a1*prs->p1.z + prs->p1.x;
    b12 = -a1*prs->p2.z + prs->p2.x;
    b13 = -a1*prs->p3.z + prs->p3.x;
    b14 = -a1*prs->p4.z + prs->p4.x;
    b21 = -a2*prs->p1.z + prs->p1.y;
    b22 = -a2*prs->p2.z + prs->p2.y;
    b23 = -a2*prs->p3.z + prs->p3.y;
    b24 = -a2*prs->p4.z + prs->p4.y;

    break;
  }
  
  // update origin of the parametric plane
  
  if ((b11 <= b12) && (b11 <= b13) && (b11 <= b14))
    prs->b1_min = b11;
  else 
    if ((b12 <= b11) && (b12 <= b13) && (b12 <= b14))
      prs->b1_min = b12;
    else 
      if ((b13 <= b11) && (b13 <= b12) && (b13 <= b14))
        prs->b1_min = b13;
      else 
	if ((b14 <= b11) && (b14 <= b12) && (b14 <= b13))
	  prs->b1_min = b14;
  
  if ((b21 <= b22) && (b21 <= b23) && (b21 <= b24))
      prs->b2_min = b21;
  else 
    if ((b22 <= b21) && (b22 <= b23) && (b22 <= b24))
      prs->b2_min = b22;
    else 
      if ((b23 <= b21) && (b23 <= b22) && (b23 <= b24))
        prs->b2_min = b23;
      else 
	if ((b24 <= b21) && (b24 <= b22) && (b24 <= b23))
        prs->b2_min = b24;
  
  
  // update lookup tables that accelerate projection
  /*
  printf("%c\n",prs->paxis);
  printf("b11 %f b21 %f b12 %f b22 %f b13 %f b23 %f b14 %f b24 %f\n",b11,b21,b12,b22,b13,b23,b14,b24);
  printf("b1min %f b2min %f\n",prs->b1_min,prs->b2_min);
  */

  A1 = (b12 - b11)/(prs->diagonal-1.0);
  B1 = (b13 - b11)/(prs->diagonal-1.0);
  C1 = (b22 - b21)/(prs->diagonal-1.0);
  D1 = (b23 - b21)/(prs->diagonal-1.0);
  E1 = b11;
  F1 = b21;

  Delta = A1*D1 - B1*C1;
  A2 =  D1/Delta;
  B2 = -B1/Delta;
  C2 = -C1/Delta;
  D2 =  A1/Delta;

  prs->E2 = (B1*F1-D1*E1)/Delta;
  prs->F2 = (C1*E1-A1*F1)/Delta;

  for(i=0; i<1.5*prs->diagonal; i++) {
    prs->lkt_a1[i]   = (a1*(i+min));
    prs->lkt_a2[i]   = (a2*(i+min));
    prs->lkt_A2b1[i] = A2*(i+prs->b1_min);
    prs->lkt_C2b1[i] = C2*(i+prs->b1_min);
    prs->lkt_B2b2[i] = B2*(i+prs->b2_min);
    prs->lkt_D2b2[i] = D2*(i+prs->b2_min);
  }
}

/***  Update rendering parameters that affect display ***/
void UpdatePRSceneDisplay(PRScene *prs) {

/*** Not implemented yet ***/

    return;
}


/***  Rendering ***/

void RenderPRScene (PRScene *prs, float thetaX, float thetaY) {

  SimplerUpdatePRSceneView(prs, thetaX, thetaY);
  SimplerPRSProject(prs);
  PRSShading(prs);
}


/***  Projection ***/

void PRSProject (PRScene *prs) {

    int i, j, k, *tbrow=prs->cimg->C[0]->tbrow;
    float b1, b2, voxel_depth;
    int b1_ind, b2_ind;
    int p;
    Vector normal;

    prs->d_min = INT_MAX;
    prs->d_max = INT_MIN;
    prs->i_min = INT_MAX;
    prs->i_max = INT_MIN;
    prs->j_min = INT_MAX;
    prs->j_max = INT_MIN;

    switch(prs->paxis) {

    case 'x':

      for(k=0; k<prs->nvoxels; k++) { 
	
	normal.x = prs->normal_table[prs->voxel[k].normal_index].x;
	normal.y = prs->normal_table[prs->voxel[k].normal_index].y;
	normal.z = prs->normal_table[prs->voxel[k].normal_index].z;

	if ((normal.x*prs->n.x + normal.y*prs->n.y + normal.z*prs->n.z) < 0){
	
	b1 = (float)prs->voxel[k].z - prs->zsize/2.0 - prs->lkt_a1[(int)prs->voxel[k].x];
	b2 = (float)prs->voxel[k].y - prs->ysize/2.0 - prs->lkt_a2[(int)prs->voxel[k].x];
		
	if (prs->n.x > 0)
	  voxel_depth = prs->voxel[k].x;
	else
	  voxel_depth = prs->xsize - 1 - prs->voxel[k].x;
	
	if (voxel_depth < prs->d_min) prs->d_min = voxel_depth;
	if (voxel_depth > prs->d_max) prs->d_max = voxel_depth;	    
	
	b1_ind = (int)(b1 - prs->b1_min );
	b2_ind = (int)(b2 - prs->b2_min );

	i = (int)(prs->lkt_A2b1[b1_ind] + prs->lkt_B2b2[b2_ind] + prs->E2);
	j = (int)(prs->lkt_C2b1[b1_ind] + prs->lkt_D2b2[b2_ind] + prs->F2);
	
	p = i+tbrow[j];

	if (voxel_depth < prs->depth_buffer[p]) {

	  prs->index_buffer[p] = k; 
	  prs->depth_buffer[p] = voxel_depth;

	  if (i < prs->i_min) prs->i_min = i;
	  if (i > prs->i_max) prs->i_max = i;
	  if (j < prs->j_min) prs->j_min = j;
	  if (j > prs->j_max) prs->j_max = j;
	}
	}
      }
      break;

      case'y':

      for(k=0; k<prs->nvoxels; k++) { 
	
	normal.x = prs->normal_table[prs->voxel[k].normal_index].x;
	normal.y = prs->normal_table[prs->voxel[k].normal_index].y;
	normal.z = prs->normal_table[prs->voxel[k].normal_index].z;

	if ((normal.x*prs->n.x + normal.y*prs->n.y + normal.z*prs->n.z) < 0){
	
	b1 = (float)prs->voxel[k].x - prs->xsize/2.0 - prs->lkt_a1[(int)prs->voxel[k].y];
	b2 = (float)prs->voxel[k].z - prs->zsize/2.0 - prs->lkt_a2[(int)prs->voxel[k].y];

	if (prs->n.y > 0)
	  voxel_depth = prs->voxel[k].y;
	else
	  voxel_depth = prs->ysize - 1 - prs->voxel[k].y;
	

	if (voxel_depth < prs->d_min) prs->d_min = voxel_depth;
	if (voxel_depth > prs->d_max) prs->d_max = voxel_depth;	    
	
	b1_ind = (int)(b1 - prs->b1_min );
	b2_ind = (int)(b2 - prs->b2_min );
       
	i = (int)(prs->lkt_A2b1[b1_ind] + prs->lkt_B2b2[b2_ind] + prs->E2);
	j = (int)(prs->lkt_C2b1[b1_ind] + prs->lkt_D2b2[b2_ind] + prs->F2);
	
	p = i+tbrow[j];

	if (voxel_depth < prs->depth_buffer[p]) {

	  prs->index_buffer[p] = k; 
	  prs->depth_buffer[p] = voxel_depth;

	  if (i < prs->i_min) prs->i_min = i;
	  if (i > prs->i_max) prs->i_max = i;
	  if (j < prs->j_min) prs->j_min = j;
	  if (j > prs->j_max) prs->j_max = j;
	}
      }
    }
      break;

    case 'z':

      for(k=0; k<prs->nvoxels; k++) {
	
	normal.x = prs->normal_table[prs->voxel[k].normal_index].x;
	normal.y = prs->normal_table[prs->voxel[k].normal_index].y;
	normal.z = prs->normal_table[prs->voxel[k].normal_index].z;

	if ((normal.x*prs->n.x + normal.y*prs->n.y + normal.z*prs->n.z) < 0){
	
	b1 = (float)prs->voxel[k].x - prs->xsize/2.0 - prs->lkt_a1[(int)prs->voxel[k].z];
	b2 = (float)prs->voxel[k].y - prs->ysize/2.0 - prs->lkt_a2[(int)prs->voxel[k].z];
		
	if (prs->n.z > 0)
	  voxel_depth = prs->voxel[k].z;
	else
	  voxel_depth = prs->zsize - 1 - prs->voxel[k].z;
	
	if (voxel_depth < prs->d_min) prs->d_min = voxel_depth;
	if (voxel_depth > prs->d_max) prs->d_max = voxel_depth;	    
	
	b1_ind = (int)(b1 - prs->b1_min );
	b2_ind = (int)(b2 - prs->b2_min );

	i = (int)(prs->lkt_A2b1[b1_ind] + prs->lkt_B2b2[b2_ind] + prs->E2);
	j = (int)(prs->lkt_C2b1[b1_ind] + prs->lkt_D2b2[b2_ind] + prs->F2);
	
	p = i+tbrow[j];

	if (voxel_depth < prs->depth_buffer[p]) {

	  prs->index_buffer[p] = k; 
	  prs->depth_buffer[p] = voxel_depth;

	  if (i < prs->i_min) prs->i_min = i;
	  if (i > prs->i_max) prs->i_max = i;
	  if (j < prs->j_min) prs->j_min = j;
	  if (j > prs->j_max) prs->j_max = j;
	}
	}
       }
      break;
    }
}

/***  Shading ***/

// Atualizar para buscar informacoes no objeto

void PRSShading (PRScene *prs) {

    int i,j,l,p,ind;
    int color,*tbrow=prs->cimg->C[0]->tbrow;
    float ks=0.2,kd=0.7,ka=0.1*255.0,costheta,cos2theta,ns=2.0,pow;
    float diff,spec;
    Vector normal;

    for (i=0; i < prs->d_min; i++) 
      prs->depth_shading_table[i]=0;
    for (i=prs->d_min; i <= prs->d_max; i++) 
      prs->depth_shading_table[i]=MIN((int)(255*((prs->d_max-(float)i)/(prs->d_max-prs->d_min))),255);
    for (i=prs->d_max+1; i < prs->diagonal; i++) 
      prs->depth_shading_table[i]=255;


    /* shading */
    
    for(j=prs->j_min; j <= prs->j_max; j++) { 
      for(i=prs->i_min; i<= prs->i_max; i++) { 

	p = i+tbrow[j];

	if(prs->index_buffer[p] != -1) {
	  color = prs->depth_shading_table[(int)prs->depth_buffer[p]];
	  
	  normal.x = 
	    prs->normal_table[prs->voxel[prs->index_buffer[p]].normal_index].x;
	  normal.y = 
	    prs->normal_table[prs->voxel[prs->index_buffer[p]].normal_index].y;
	  normal.z = 
	    prs->normal_table[prs->voxel[prs->index_buffer[p]].normal_index].z;

	  costheta = -(prs->n.x*normal.x + prs->n.y*normal.y + prs->n.z*normal.z);
	  
	  diff = kd*costheta;
	  cos2theta = 2*costheta - 1;
	  
	  if (cos2theta < 0){
	    spec = 0;
	  }else {
	    pow = 1.;
	    for (l=1; l <= ns; l++)
	      pow = pow*cos2theta;
	    spec = ks*pow;
	  }
	  
	  color = (int)(color*(diff+spec) + ka);

	  if (color < 0) 
	    color = 0;
	  else 
	    if (color > 255) color=255;
	  
	  prs->cimg->C[0]->val[p] = prs->cimg->C[1]->val[p] = prs->cimg->C[2]->val[p] = color;	  
	}
      }
    }


    /* splatting */

    for(j=prs->j_min; j <= prs->j_max; j++) { 
      for(i=prs->i_min; i<= prs->i_max; i++) { 

	p = i+tbrow[j];
	
	color = prs->cimg->C[0]->val[p];

	ind = p-1;
	if (ind>=0){
	  if (prs->depth_buffer[ind]>prs->depth_buffer[p]){
	    prs->depth_buffer[ind] = prs->depth_buffer[p];
	    prs->cimg->C[0]->val[ind] = color;
	    prs->cimg->C[1]->val[ind] = color;
	    prs->cimg->C[2]->val[ind] = color;
	  }
	}

	ind = p+1;
	if (ind<prs->diagonal){
	  if (prs->depth_buffer[ind]>prs->depth_buffer[p]){
	    prs->depth_buffer[ind] = prs->depth_buffer[p];
	    prs->cimg->C[0]->val[ind] = color;
	    prs->cimg->C[1]->val[ind] = color;
	    prs->cimg->C[2]->val[ind] = color;
	  }
	}

	ind = p-prs->diagonal;	  
	if (ind>=0){
	  if (prs->depth_buffer[ind]>prs->depth_buffer[p]){
	    prs->depth_buffer[ind] = prs->depth_buffer[p];
	    prs->cimg->C[0]->val[ind] = color;
	    prs->cimg->C[1]->val[ind] = color;
	    prs->cimg->C[2]->val[ind] = color;	    
	  }
	}

	ind = p+prs->diagonal;	
	if (ind<prs->diagonal){
	  if (prs->depth_buffer[ind]>prs->depth_buffer[p]){
	    prs->depth_buffer[ind] = prs->depth_buffer[p];
	    prs->cimg->C[0]->val[ind] = color;
	    prs->cimg->C[1]->val[ind] = color;
	    prs->cimg->C[2]->val[ind] = color;
	  }
	}
	
	ind = p-prs->diagonal-1;
	if (ind>=0){
	  if (prs->depth_buffer[ind]>prs->depth_buffer[p]){
	    prs->depth_buffer[ind] = prs->depth_buffer[p];
	    prs->cimg->C[0]->val[ind] = color;
	    prs->cimg->C[1]->val[ind] = color;
	    prs->cimg->C[2]->val[ind] = color;
	  }
	}
	 
	ind = p+prs->diagonal-1; 
	if (ind<prs->diagonal){
	  if (prs->depth_buffer[ind]>prs->depth_buffer[p]){
	    prs->depth_buffer[ind] = prs->depth_buffer[p];
	    prs->cimg->C[0]->val[ind] = color;
	    prs->cimg->C[1]->val[ind] = color;
	    prs->cimg->C[2]->val[ind] = color;
	  }
	}
	ind = p-prs->diagonal+1;
	if (ind>=0){
	  if (prs->depth_buffer[ind]>prs->depth_buffer[p]){
	    prs->depth_buffer[ind] = prs->depth_buffer[p];
	    prs->cimg->C[0]->val[ind] = color;
	    prs->cimg->C[1]->val[ind] = color;
	    prs->cimg->C[2]->val[ind] = color;
	  }
	}
	  
	ind = p+prs->diagonal+1;
	if (ind<prs->diagonal){
	  if (prs->depth_buffer[ind]>prs->depth_buffer[p]){
	    prs->depth_buffer[ind] = prs->depth_buffer[p];
	    prs->cimg->C[0]->val[ind] = color;
	    prs->cimg->C[1]->val[ind] = color;
	    prs->cimg->C[2]->val[ind] = color;
	  }
	}
      }
    }
}

void SimplerUpdatePRSceneView(PRScene *prs, float thetaX, float thetaY)
{
  float Rx[4][4],Ry[4][4];
  float Raux[4][4], Rnew[4][4];
  double trash;
  int i, j, n;

  // reset viewing buffers

  n = prs->diagonal*prs->diagonal; 
  for (i=0; i < n; i++) {
    prs->cimg->C[0]->val[i]=prs->cimg->C[1]->val[i]=
      prs->cimg->C[2]->val[i]=0;
    prs->depth_buffer[i]=INT_MAX;
    prs->index_buffer[i]=NIL;
  }

  // update rendering parameters: Karina: colocar copia do update como
  // reset com a primeira parte do if. Retirar esta parte deste
  // funcao. Trocar update por reset em createprscene.

  if ((thetaX==0.0)&&(thetaY==0.0)){
    prs->n.x = 0; 
    prs->n.y = 0; 
    prs->n.z = 1;
    prs->paxis = 'z';
    prs->p1.x = -prs->diagonal/2.0; 
    prs->p1.y = -prs->diagonal/2.0; 
    prs->p1.z = -prs->diagonal/2.0;
    prs->p2.x =  prs->diagonal/2.0; 
    prs->p2.y = -prs->diagonal/2.0; 
    prs->p2.z = -prs->diagonal/2.0;
    prs->p3.x = -prs->diagonal/2.0; 
    prs->p3.y =  prs->diagonal/2.0; 
    prs->p3.z = -prs->diagonal/2.0;
    prs->p4.x =  prs->diagonal/2.0; 
    prs->p4.y =  prs->diagonal/2.0; 
    prs->p4.z = -prs->diagonal/2.0;
    prs->R[0][0] = 1; prs->R[0][1] = 0; prs->R[0][2] = 0; prs->R[0][3] = 0;
    prs->R[1][0] = 0; prs->R[1][1] = 1; prs->R[1][2] = 0; prs->R[1][3] = 0;
    prs->R[2][0] = 0; prs->R[2][1] = 0; prs->R[2][2] = 1; prs->R[2][3] = 0;
    prs->R[3][0] = 0; prs->R[3][1] = 0; prs->R[3][2] = 0; prs->R[3][3] = 1;
  } else {

    thetaX = (float)(360.*modf(thetaX/360.,&trash));
    if (thetaX < 0.0)
      thetaX += 360.0;
    thetaY = (float)(360.*modf(thetaY/360.,&trash));
    if (thetaY < 0.0)
      thetaY += 360.0;
  
    thetaX = thetaX*PI/180.;
    thetaY = thetaY*PI/180.;

    // create Rx matrix
    Rx[0][3] = Rx[1][3] = Rx[2][3] = Rx[3][2] = Rx[3][1] = Rx[3][0] = 0;
    Rx[3][3] = 1;
    Rx[0][0] = 1;
    Rx[0][1] = Rx[0][2] = Rx[1][0] = Rx[2][0] = 0;
    Rx[1][1] = Rx[2][2] = cos(thetaX);
    Rx[1][2] = sin(thetaX);
    Rx[2][1] = -sin(thetaX);
    // create Ry matrix
    Ry[0][3] = Ry[1][3] = Ry[2][3] = Ry[3][2] = Ry[3][1] = Ry[3][0] = 0;
    Ry[3][3] = 1;
    Ry[1][1] = 1;
    Ry[0][1] = Ry[1][0] = Ry[2][1] = Ry[1][2] = 0;
    Ry[0][0] = Ry[2][2] = cos(thetaY);
    Ry[0][2] = -sin(thetaY);
    Ry[2][0] = sin(thetaY);
    MultMatrices(Rx,Ry,Raux); // Raux = Rx*Ry
    MultMatrices(Raux,prs->R,Rnew); // Rnew = Raux*prs->R
    // Update prs->R
    for(i=0; i<4; i++)
      for(j=0; j<4; j++)
	prs->R[i][j]=Rnew[i][j];
    // Update n, p1, p2, p3, p4
    prs->n  = RotatePoint(prs->R,prs->n);
    trash   = sqrt(prs->n.x*prs->n.x + prs->n.y*prs->n.y + prs->n.z*prs->n.z);
    prs->n.x /= trash;
    prs->n.y /= trash;
    prs->n.z /= trash;
  }
}

void SimplerPRSProject (PRScene *prs) {

    int i, j, k, *tbrow=prs->cimg->C[0]->tbrow;
    float voxel_depth;
    int p;
    Vector normal;

    prs->d_min = INT_MAX;
    prs->d_max = INT_MIN;
    prs->i_min = INT_MAX;
    prs->i_max = INT_MIN;
    prs->j_min = INT_MAX;
    prs->j_max = INT_MIN;

    for(k=0; k<prs->nvoxels; k++) { 
	
      normal.x = prs->normal_table[prs->voxel[k].normal_index].x;
      normal.y = prs->normal_table[prs->voxel[k].normal_index].y;
      normal.z = prs->normal_table[prs->voxel[k].normal_index].z;

      if ((normal.x*prs->n.x + normal.y*prs->n.y + normal.z*prs->n.z) < 0){
	
	voxel_depth = prs->R[0][2]*(prs->voxel[k].x-prs->xsize/2.0) + 
	  prs->R[1][2]*(prs->voxel[k].y-prs->ysize/2.0) + 
	  prs->R[2][2]*(prs->voxel[k].z-prs->zsize/2.0) + prs->diagonal/2.0;  
	
	i = (int)(prs->R[0][0]*(prs->voxel[k].x-prs->xsize/2.0) + 
	  prs->R[1][0]*(prs->voxel[k].y-prs->ysize/2.0) + 
	  prs->R[2][0]*(prs->voxel[k].z-prs->zsize/2.0) + prs->diagonal/2.0 + 0.5);  
	
	j = (int)(prs->R[0][1]*(prs->voxel[k].x-prs->xsize/2.0) + 
	  prs->R[1][1]*(prs->voxel[k].y-prs->ysize/2.0) + 
	  prs->R[2][1]*(prs->voxel[k].z-prs->zsize/2.0) + prs->diagonal/2.0 + 0.5);  

	if (voxel_depth < prs->d_min) prs->d_min = voxel_depth;
	if (voxel_depth > prs->d_max) prs->d_max = voxel_depth;	    

	p = i+tbrow[j];

	if (voxel_depth < prs->depth_buffer[p]) {

	  prs->index_buffer[p] = k; 
	  prs->depth_buffer[p] = voxel_depth;

	  if (i < prs->i_min) prs->i_min = i;
	  if (i > prs->i_max) prs->i_max = i;
	  if (j < prs->j_min) prs->j_min = j;
	  if (j > prs->j_max) prs->j_max = j;
	}
      }
    }
}
