CImage *CSWShellRendering(Shell *shell,Context *cxt) {

  CImage *cimg=NULL;
  FBuffer *shear=NULL,*warp=NULL;
  Voxel p,q,c;
  Pixel d;
  Vector viewer;
  AdjPxl *fprint=NULL;

  int u,v,ut,vt,uw,vw,is,iw,iw_p,i,i_p,i1,i2,j,n;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,alpha,opac,nopac[256],k; 
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  warp = CreateFBuffer(cxt->width, cxt->height);
  InitFBuffer(shear,1.0);
  InitFBuffer(warp,1.0);
  InitZBuffer(cxt->zbuff);

  cimg=CreateCImage(cxt->width, cxt->height);
  fprint = AdjPixels(cimg->C[0],cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  switch(cxt->PAxis) {

  case 'x':
    
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Mzx->tbrow[p.x] + p.z;
        
        if (shell->Mzx->val[i] >= 0) {
          
          if (cxt->dy == 1) {              
            i1 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i2 = shell->Mzx->val[i]-1;            
          } else {            
            i2 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
          for (i = i1; i != i2; i += cxt->dy) {
	    // shear
            p.y = shell->pointer[i].y;

	    if (cxt->ymin <= p.y && p.y <= cxt->ymax) {

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	      
	      u = (int) (q.y + cxt->Su[p.x]); 
	      v = (int) (q.z + cxt->Sv[p.x]); 
	    
	      is = u + ut + shear->tbv[v + vt];
	    
	      if (shear->val[is] > 0.05) {
		
		// normal
		i_p = shell->pointer[i].i;
		
		if (shell->voxel[i_p].visible) {		
		
		  l = shell->voxel[i_p].obj;
		  if (l) {

		    n = shell->voxel[i_p].normal;	      
		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;
		  
		    if (cos_a > 0) {
		    
		      // shading
		    
		      w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt; k = cxt->depth - w;

		      idist = 1.0 - w/(float)cxt->depth;
		    
		      cos_2a = 2*cos_a*cos_a - 1.0;
		    
		      if (cos_2a < 0.) {
			pow = 0.;
			spec = 0.;
		      } else {
			pow = 1.;
			for (j=0; j < cxt->obj[l].ns; j++) 
			  pow = pow * cos_2a;
			spec = cxt->obj[l].spec * pow;
		      }
		    
		      opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		      amb  = cxt->obj[l].amb;
		      diff = cxt->obj[l].diff * cos_a;
		    
		      shading = (amb + idist * (diff + spec));
		    
		      //warp	      
		      uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		    
		      vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		    
		      iw = warp->tbv[vw + d.y] + uw + d.x;
		    
		      // splatting
		      for (j=0; j < fprint->n; j++){		
			iw_p = iw + fprint->dp[j];
			if (warp->val[iw_p] > .05) {
			  alpha = cxt->footprint->val[j] * opac;
			  AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			  warp->val[iw_p] *= (1.0-alpha);
			  if (cxt->zbuff->voxel[iw_p] == NIL) {
			    cxt->zbuff->voxel[iw_p] = i_p;
			    cxt->zbuff->object[iw_p] = l;

			    if (cxt->zbuff->dist[iw_p] > k)
			      cxt->zbuff->dist[iw_p] = k;
			  }
			}
		      }
		      shear->val[is]=warp->val[iw];
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    break;
    
  case 'y': 
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
            p.x = shell->voxel[i].x;

	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = (int) (q.z + cxt->Su[p.y]); 
	      v = (int) (q.x + cxt->Sv[p.y]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {
		  l = shell->voxel[i_p].obj;
		  if (l) {
		    n = shell->voxel[i_p].normal;

		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;

		    if (cos_a > 0) {

		      // shading

		      w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt; k = cxt->depth - w;
		      idist = 1.0 - w/(float)cxt->depth;

		      cos_2a = 2*cos_a*cos_a - 1.0;
		
		      if (cos_2a < 0.) {
			pow = 0.;
			spec = 0.;
		      } else {
			pow = 1.;
			for (j=0; j < cxt->obj[l].ns; j++) 
			  pow = pow * cos_2a;
			spec = cxt->obj[l].spec * pow;
		      }

		      opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		      amb  = cxt->obj[l].amb;
		      diff = cxt->obj[l].diff * cos_a;
		
		      shading =  (amb + idist * (diff + spec));
		
		      // warp	      
		      uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		      vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		
		      iw = warp->tbv[vw + d.y] + uw + d.x;
		
		      // splatting
		      for (j=0; j < fprint->n; j++){		
			iw_p = iw + fprint->dp[j];
			if (warp->val[iw_p] > .05) {
			  alpha = cxt->footprint->val[j] * opac;
			  AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			  warp->val[iw_p] *= (1.0-alpha);
			  if (cxt->zbuff->voxel[iw_p] == NIL) {
			    cxt->zbuff->voxel[iw_p] = i_p;
			    cxt->zbuff->object[iw_p] = l;
			    if (cxt->zbuff->dist[iw_p] > k)
		              cxt->zbuff->dist[iw_p] = k;
			  }
			}
		      }
		      shear->val[is]=warp->val[iw];
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    break;

  case 'z': 
    
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) 
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
            p.x = shell->voxel[i].x;

	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 

	      u = (int) (q.x + cxt->Su[p.z]); 
	      v = (int) (q.y + cxt->Sv[p.z]); 

	      is = u + ut + shear->tbv[v + vt];
	    
	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {
		  l = shell->voxel[i_p].obj;
		  if (l) {
		    n = shell->voxel[i_p].normal;

		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;

		    if (cos_a > 0) {

		      // shading

		      w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt; k = cxt->depth - w;
		      idist = 1.0 - w/(float)cxt->depth;

		      cos_2a = 2*cos_a*cos_a - 1.0;
		
		      if (cos_2a < 0.) {
			pow = 0.;
			spec = 0.;
		      } else {
			pow = 1.;
			for (j=0; j < cxt->obj[l].ns; j++) 
			  pow = pow * cos_2a;
			spec = cxt->obj[l].spec * pow;
		      }
		
		      opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac; 
		      amb  = cxt->obj[l].amb;
		      diff = cxt->obj[l].diff * cos_a;
		
		      shading =  (amb + idist * (diff + spec));
		
		      // warp
		      uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		      vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		
		      iw = warp->tbv[vw + d.y] + uw + d.x;
		
		      // splatting
		
		      for (j=0; j < fprint->n; j++){		
			iw_p = iw + fprint->dp[j];
			if (warp->val[iw_p] > .05) {
			  alpha = cxt->footprint->val[j] * opac;
			  AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			  warp->val[iw_p] *= (1.0-alpha);
			  if (cxt->zbuff->voxel[iw_p] == NIL) {
			    cxt->zbuff->voxel[iw_p] = i_p;
			    cxt->zbuff->object[iw_p] = l;
			    if (cxt->zbuff->dist[iw_p] > k)
		              cxt->zbuff->dist[iw_p] = k;
			  }
			}
		      }	      
		      shear->val[is]=warp->val[iw];
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    break;

      
  default: 
    Error(MSG1,"CSWShellRendering");
    break;
    
  }			
  DestroyFBuffer(&shear);
  DestroyFBuffer(&warp);
  DestroyAdjPxl(&fprint);
  return(cimg);
}

void    ShellRenderingDrawCut(Image *img, Shell *shell,Context *cxt, FBuffer *cumopac, char axis)
{

  int ut,vt,u,v,ww,iw,iw_p,i,i_p,i1,i2,j;
  float w,wt,k; 
  uchar l;

  Voxel p,q,c;
  Pixel d;
  AdjPxl *fprint=NULL;
  fprint = AdjPixels(img,cxt->footprint->adj);    

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  switch(axis) {

  case 'x':
    
    p.x = cxt->xi;
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      i = shell->Mzx->tbrow[p.x] + p.z;
      
      if (shell->Mzx->val[i] >= 0) {
	
	if (cxt->dy == 1) {              
	  i1 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Mzx->val[i]-1;            
	} else {            
	  i2 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Mzx->val[i]-1;
	}
	i2 += cxt->dy;          
	for (i = i1; i != i2; i += cxt->dy) {
	  
	  p.y = shell->pointer[i].y;

	  if (cxt->ymin <= p.y && p.y <= cxt->ymax) {

	    i_p = shell->pointer[i].i;	    

	    if (shell->voxel[i_p].visible) {

	      q.x = p.x - c.x;	    
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	        	      
	      l = shell->voxel[i_p].obj;

	      if (l) {

		u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
		v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;	     	      
		ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		w = ww + wt;k = cxt->depth - w;
		// splatting
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];

		  if (cxt->zbuff->dist[iw_p] > k) {
		    cumopac->val[iw_p] = 0.0;
		    img->val[iw_p] = shell->body[i_p].val;
		    cxt->zbuff->voxel[iw_p] = i_p;
		    cxt->zbuff->object[iw_p] = l;
		    cxt->zbuff->dist[iw_p] = k;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
 
    break;

  case 'y': 
    
    p.y = cxt->yi;
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      i = shell->Myz->tbrow[p.z] + p.y;
        
      if (shell->Myz->val[i] >= 0) {
          
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i2 += cxt->dx;          
	for (i = i1; i != i2; i += cxt->dx) {

	  // shear
	  p.x = shell->voxel[i].x;

	  if (cxt->xmin <= p.x && p.x <= cxt->xmax) {

	    i_p = i;	      

	    if (shell->voxel[i_p].visible) {

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    

	      l = shell->voxel[i_p].obj;

	      if (l) {
	      
		u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
		v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;	     	      
		ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		w = ww + wt;k = cxt->depth - w;

		// splatting
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];

		  if (cxt->zbuff->dist[iw_p] > k) {
		    cumopac->val[iw_p] = 0.0;
		    img->val[iw_p] = shell->body[i_p].val;
		    cxt->zbuff->voxel[iw_p] = i_p;
		    cxt->zbuff->object[iw_p] = l;
		    cxt->zbuff->dist[iw_p] = k;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  
    break;

  case 'z':
    
    p.z = cxt->zi;
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
      
      i = shell->Myz->tbrow[p.z] + p.y;
      
      if (shell->Myz->val[i] >= 0) {
	
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i2 += cxt->dx;          
	for (i = i1; i != i2; i += cxt->dx) {
	  
	  p.x = shell->voxel[i].x;
	  
	  if (cxt->xmin <= p.x && p.x <= cxt->xmax) {        	   

	    i_p = i;
	    
	    if (shell->voxel[i_p].visible) {
	      
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 	      

	      l = shell->voxel[i_p].obj;
	      
	      if (l) {
		
		u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
		v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;	     	      
		ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		w = ww + wt;k = cxt->depth - w;

		// splatting
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];

		  if (cxt->zbuff->dist[iw_p] > k) {
		    cumopac->val[iw_p] = 0.0;
		    img->val[iw_p] = shell->body[i_p].val;
		    cxt->zbuff->voxel[iw_p] = i_p;
		    cxt->zbuff->object[iw_p] = l;
		    cxt->zbuff->dist[iw_p] = k;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    break;

  default: 
    Error(MSG1,"ShellRenderingDrawCut");
    break;

  }  
  DestroyAdjPxl(&fprint);
}

void    ShellRenderingDrawCuts(Image *img, Shell *shell,Context *cxt, FBuffer *cumopac, char axis)
{
}


void SWShellRenderingDrawCut(Image *img, Shell *shell,Context *cxt, FBuffer *shear, FBuffer *warp, char axis) 
{

  int u,v,ut,vt,uw,vw,ww,is,iw,iw_p,i,i_p,i1,i2,j;
  float w,wt,k; 
  uchar l;

  Voxel p,q,c;
  Pixel d;
  AdjPxl *fprint=NULL;
  fprint = AdjPixels(img,cxt->footprint->adj);    

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  switch(axis) {

  case 'x':
    
    p.x = cxt->xi;
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      i = shell->Mzx->tbrow[p.x] + p.z;
      
      if (shell->Mzx->val[i] >= 0) {
	
	if (cxt->dy == 1) {              
	  i1 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Mzx->val[i]-1;            
	} else {            
	  i2 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Mzx->val[i]-1;
	}
	i2 += cxt->dy;          
	for (i = i1; i != i2; i += cxt->dy) {
	  
	  p.y = shell->pointer[i].y;
	  if (cxt->ymin <= p.y && p.y <= cxt->ymax) {

	    i_p = shell->pointer[i].i;

	    if (shell->voxel[i_p].visible) {

	      // shear
	    
	      q.x = p.x - c.x;	    
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = (int) (q.y + cxt->Su[p.x]); 
	      v = (int) (q.z + cxt->Sv[p.x]); 
	    	    
	      is = u + ut+ shear->tbv[v + vt];
	    
	      if (shear->val[is] > 0.05) {
	      
		l = shell->voxel[i_p].obj;

		if (l) {

		  // warp
		  uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		  vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
	      
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;	     
	      
		  ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
	      
		  w = ww + wt;k = cxt->depth - w;
		  // splatting
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;		
		  for (j=0; j < fprint->n; j++){		
		    iw_p = iw + fprint->dp[j];
		    if (cxt->zbuff->dist[iw_p] > k) {
		      warp->val[iw_p] = 0.0;
		      img->val[iw_p] = shell->body[i_p].val;
		      cxt->zbuff->voxel[iw_p] = i_p;
		      cxt->zbuff->object[iw_p] = l;
		      cxt->zbuff->dist[iw_p] = k;
		    }
		  }
		  shear->val[is]=warp->val[iw];
		}      
	      }
	    }
	  }
	}
      }
    }
    break;

  case 'y': 
    
    p.y = cxt->yi;
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      i = shell->Myz->tbrow[p.z] + p.y;
        
      if (shell->Myz->val[i] >= 0) {
          
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i2 += cxt->dx;          
	for (i = i1; i != i2; i += cxt->dx) {

	  p.x = shell->voxel[i].x;

	  if (cxt->xmin <= p.x && p.x <= cxt->xmax) {

	    i_p = i;
	  
	    if (shell->voxel[i_p].visible) {

	      // shear	    
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = (int) (q.z + cxt->Su[p.y]); 
	      v = (int) (q.x + cxt->Sv[p.y]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {

		l = shell->voxel[i_p].obj;

		if (l) {

		  // warp
		  uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		  vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
	       
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;	     
	       
		  ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
	       
		  w = ww + wt;k = cxt->depth - w;

		  // splatting
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;		
		  for (j=0; j < fprint->n; j++){		
		    iw_p = iw + fprint->dp[j];
		    if (cxt->zbuff->dist[iw_p] > k) {
		      warp->val[iw_p] = 0.0;
		      img->val[iw_p] = shell->body[i_p].val;
		      cxt->zbuff->voxel[iw_p] = i_p;
		      cxt->zbuff->object[iw_p] = l;		      
		      cxt->zbuff->dist[iw_p] = k;
		    }
		  }
		  shear->val[is]=warp->val[iw];
		}      
	      }
	    }
	  }
	}
      }
    }
    break;

  case 'z':
   
    p.z = cxt->zi;
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
      i = shell->Myz->tbrow[p.z] + p.y;
      
      if (shell->Myz->val[i] >= 0) {
	
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i2 += cxt->dx;          
	for (i = i1; i != i2; i += cxt->dx) {
	  
	  p.x = shell->voxel[i].x;

	  if (cxt->xmin <= p.x && p.x <= cxt->xmax) {        	   

	    i_p = i;
	  
	    if (shell->voxel[i_p].visible) {

	      // shear
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 

	      u = (int) (q.x + cxt->Su[p.z]); 
	      v = (int) (q.y + cxt->Sv[p.z]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      

		l = shell->voxel[i_p].obj;

		if (l) {

		  // warp
		  uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		  vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
	      
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;	     
	      
		  ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
	      
		  w = ww + wt;k = cxt->depth - w;
		  // splatting
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;		
		  for (j=0; j < fprint->n; j++){		
		    iw_p = iw + fprint->dp[j];
		    if (cxt->zbuff->dist[iw_p] > k) {
		      warp->val[iw_p] = 0.0;
		      img->val[iw_p] = shell->body[i_p].val;
		      cxt->zbuff->voxel[iw_p] = i_p;
		      cxt->zbuff->object[iw_p] = l;
		      cxt->zbuff->dist[iw_p] = k;
		    }
		  }
		  shear->val[is]=warp->val[iw];
		}      
	      }
	    }
	  }
	}
      }
    }
    break;

  default: 
    Error(MSG1,"CSWShellRenderingDrawCut");
    break;

  }  
  DestroyAdjPxl(&fprint);
}

void ShellRenderingDrawCutPlane(Image *img, Shell *shell,Context *cxt, FBuffer *cumopac) 
{
  int i,j,n,u,v,ut,vt,l;
  float w,wt;
  Voxel s,c;
  Pixel p;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  ut = cxt->width/2;
  vt = cxt->height/2;
  wt = (float)cxt->depth/2.;

  n = cxt->width * cxt->height;
  for (p.y=0;p.y<cxt->height;p.y++) {
    for (p.x=0;p.x<cxt->width;p.x++) {
      u = p.x - ut;
      v = p.y - vt;      
      i = cxt->zbuff->tbv[p.y] + p.x;
      w = wt - cxt->zbuff->dist[i];
      s.x = (int) (cxt->IR[0][0] * u + cxt->IR[0][1] * v + cxt->IR[0][2] * w) + c.x;
      s.y = (int) (cxt->IR[1][0] * u + cxt->IR[1][1] * v + cxt->IR[1][2] * w) + c.y;
      s.z = (int) (cxt->IR[2][0] * u + cxt->IR[2][1] * v + cxt->IR[2][2] * w) + c.z;
      if (cxt->xmin <= s.x && s.x <= cxt->xmax && \
	  cxt->zmin <= s.z && s.z <= cxt->zmax && \
	  cxt->ymin <= s.y && s.y <= cxt->ymax) {
	j = VoxelExist(shell,&s);
	if (j>0) {
	  l = shell->voxel[j].obj;
	  if (l) {
	    img->val[i] = shell->body[j].val;
	    cxt->zbuff->voxel[i] = j;
	    cxt->zbuff->object[i] = l;
	    cumopac->val[i] = 0.0;
	  }
	}
      }
    }
  }
}

void    ShellRenderingDrawSceneCutPlane(Image *img, Shell *shell,Context *cxt, FBuffer *cumopac) {

  int i,j,n,u,v,ut,vt,l;
  float w,wt;
  Voxel s,c;
  Pixel p;
  Image *label=NULL,*tmp=NULL;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  ut = cxt->width/2;
  vt = cxt->height/2;
  wt = (float)cxt->depth/2.;

  tmp = CreateImage(cxt->width,cxt->height);

  n = cxt->width * cxt->height;
  for (p.y=0;p.y<cxt->height;p.y++) {
    for (p.x=0;p.x<cxt->width;p.x++) {
      u = p.x - ut;
      v = p.y - vt;      
      i = cxt->zbuff->tbv[p.y] + p.x;
      w = wt - cxt->zbuff->dist[i];
      s.x = (int) (cxt->IR[0][0] * u + cxt->IR[0][1] * v + cxt->IR[0][2] * w) + c.x;
      s.y = (int) (cxt->IR[1][0] * u + cxt->IR[1][1] * v + cxt->IR[1][2] * w) + c.y;
      s.z = (int) (cxt->IR[2][0] * u + cxt->IR[2][1] * v + cxt->IR[2][2] * w) + c.z;

      if (cxt->xmin <= s.x && s.x <= cxt->xmax && \
	  cxt->zmin <= s.z && s.z <= cxt->zmax && \
	  cxt->ymin <= s.y && s.y <= cxt->ymax) {

	j = shell->scn->tbz[s.z] + shell->scn->tby[s.y] + s.x;
	l = shell->scn->data[j];
	if (l) {
	  img->val[i] = l;
	  cxt->zbuff->voxel[i] = j;
	}
      }
      
      j = VoxelExist(shell,&s);
      if (j>0) {
	l = shell->voxel[j].obj;
	if (l && shell->voxel[j].visible) {
	  tmp->val[i] = shell->voxel[j].obj;
	}
      }      
    }
  }
  label = FillSlice(tmp);
  DestroyImage(&tmp);
  for (i=0;i<n;i++) {
    if (img->val[i]) {
      cxt->zbuff->object[i] = label->val[i];
    }
    if (!label->val[i]) {
      cxt->zbuff->voxel[i] = NIL;
    } else {
      cumopac->val[i] = 0.0;
    }

      
  }
  DestroyImage(&label);
}


Image *ShellRenderingDrawObjectCut(Shell *shell,Context *cxt,char axis) 
{

  int ut,vt,u,v,iw,iw_p,i,i_p,i1,i2,j;
  int xi,yi,zi,xf,yf,zf;
  float w,wt; 
  uchar l;
  Image *img=NULL,*label=NULL;
  Voxel p,q,c;
  Pixel d;
  AdjPxl *fprint=NULL;


  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  if (cxt->dx > 0) {xi = 0;xf = cxt->xsize-1;} else {xf = 0;xi = cxt->xsize-1;}
  if (cxt->dy > 0) {yi = 0;yf = cxt->ysize-1;} else {yf = 0;yi = cxt->ysize-1;}
  if (cxt->dz > 0) {zi = 0;zf = cxt->zsize-1;} else {zf = 0;zi = cxt->zsize-1;}

  xf += cxt->dx;
  yf += cxt->dy;
  zf += cxt->dz;

  img = CreateImage(cxt->width,cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);    
  
  switch(axis) {

  case 'x':
    
    p.x = cxt->xi;
    for (p.z = zi; p.z != zf; p.z += cxt->dz) {
      i = shell->Mzx->tbrow[p.x] + p.z;
      
      if (shell->Mzx->val[i] >= 0) {
	
	if (cxt->dy == 1) {              
	  i1 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Mzx->val[i]-1;            
	} else {            
	  i2 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Mzx->val[i]-1;
	}
	i2 += cxt->dy;          
	for (i = i1; i != i2; i += cxt->dy) {
	  
	  p.y = shell->pointer[i].y;

	  i_p = shell->pointer[i].i;	    

	  if (shell->voxel[i_p].visible) {
	    
	    q.x = p.x - c.x;	    
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	    
	    l = shell->voxel[i_p].obj;
	    
	    if (l) {
	      
	      u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
	      v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
	      w=(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
	      iw = cxt->zbuff->tbv[v + d.y] + u + d.x;	     	      
	      // splatting
	      for (j=0; j < fprint->n; j++){		
		iw_p = iw + fprint->dp[j];
		if (cxt->zbuff->dist[iw_p] > cxt->depth - w) {
		  img->val[iw_p] = l;
		}
	      }
	    }
	  }
	}
      }
    }
    break;

  case 'y': 
    
    p.y = cxt->yi;
    for (p.z = zi; p.z != zf; p.z += cxt->dz) {
      i = shell->Myz->tbrow[p.z] + p.y;
        
      if (shell->Myz->val[i] >= 0) {
          
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i2 += cxt->dx;          
	for (i = i1; i != i2; i += cxt->dx) {

	  // shear
	  p.x = shell->voxel[i].x;

	  i_p = i;	      
	  
	  if (shell->voxel[i_p].visible) {
	    
	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	    	    
	    l = shell->voxel[i_p].obj;
	    
	    if (l) {
	      
	      u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
	      v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
	      w=(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
	      iw = cxt->zbuff->tbv[v + d.y] + u + d.x;
	      
	      // splatting
	      for (j=0; j < fprint->n; j++){		
		iw_p = iw + fprint->dp[j];
		if (cxt->zbuff->dist[iw_p] > cxt->depth - w) {
		  img->val[iw_p] = l;
		}
	      }
	    }
	  }
	}
      }
    }
  
    break;

  case 'z':
    
    p.z = cxt->zi;
    for (p.y = yi; p.y != yf; p.y += cxt->dy) {
      
      i = shell->Myz->tbrow[p.z] + p.y;
      
      if (shell->Myz->val[i] >= 0) {
	
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i2 += cxt->dx;          
	for (i = i1; i != i2; i += cxt->dx) {
	  
	  p.x = shell->voxel[i].x;
	  
	  i_p = i;
	  
	  if (shell->voxel[i_p].visible) {
	    
	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z; 	      
	    
	    l = shell->voxel[i_p].obj;
	    
	    if (l) {
	      
	      u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
	      v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
	      w=(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
	      iw = cxt->zbuff->tbv[v + d.y] + u + d.x;
	      // splatting
	      
	      for (j=0; j < fprint->n; j++){		
		iw_p = iw + fprint->dp[j];
		if (cxt->zbuff->dist[iw_p] > cxt->depth - w) {
		  img->val[iw_p] = l;
		}
	      }
	    }
	  }
	}
      }
    }
    break;

  default: 
    Error(MSG1,"ShellRenderingDrawCut");
    break;

  }  
  label = FillSlice(img);
  DestroyImage(&img);
  DestroyAdjPxl(&fprint);
  return(label);
}

void ISWShellRenderingDrawSceneCut(Image *img, Shell *shell,Context *cxt, FBuffer *shear, FBuffer *warp, char axis) 
{

  int u,v,ut,vt,uw,vw,i,j,k;
  float wt,Su,Sv; 
  Image *lx=NULL,*ly=NULL,*lz=NULL;
  Voxel p,q,c;
  Pixel d;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  lx = ShellRenderingDrawObjectCut(shell,cxt,'x');
  ly = ShellRenderingDrawObjectCut(shell,cxt,'y');
  lz = ShellRenderingDrawObjectCut(shell,cxt,'z');

  switch(cxt->PAxis) {

  case 'x':

    Su = fabs(cxt->ISu[cxt->ki]);
    Sv = fabs(cxt->ISv[cxt->ki]);

    for (v=0; v < cxt->zbuff->vsize; v++)
      for (u=0; u < cxt->zbuff->usize; u++){
	j   = u+cxt->zbuff->tbv[v];
	uw = u - d.x;
	vw = v - d.y;
	q.x = uw*cxt->IW[0][0] + vw*cxt->IW[0][1] + c.y;
	q.y = uw*cxt->IW[1][0] + vw*cxt->IW[1][1] + c.z;	
	for (k=cxt->ki; k != cxt->kf; k += cxt->dk) {
	  p.y = ROUND(q.x + cxt->ISu[k]); 
	  p.z = ROUND(q.y + cxt->ISv[k]);
	  p.x = k;	    
	  if (ValidVoxel(shell->scn,p.x,p.y,p.z)) {
	    i   = p.x + shell->scn->tby[p.y] + shell->scn->tbz[p.z];
	    if (lz->val[j]) {
	      if ((p.x >= cxt->xmin)&&(p.x < cxt->xmax)&&
		  (p.y >= cxt->ymin)&&(p.y < cxt->ymax)&&
		  (p.z == cxt->zi)){
		img->val[j] = shell->scn->data[i];
		cxt->zbuff->object[j] = (uchar)lz->val[j];
		cxt->zbuff->voxel[j] = i;
		warp->val[j] = 0.0;
	      }
	    }
	    if (lx->val[j]) {
	      if ((p.z >= cxt->zmin)&&(p.z < cxt->zmax)&&
		  (p.y >= cxt->ymin)&&(p.y < cxt->ymax)&&
		  (p.x == cxt->xi)){
		img->val[j] = shell->scn->data[i];
		cxt->zbuff->object[j] = (uchar)lx->val[j];
		cxt->zbuff->voxel[j] = i;
		warp->val[j] = 0.0;
	      }
	    }
	    if (ly->val[j]) {
	      if ((p.z >= cxt->zmin)&&(p.z < cxt->zmax)&&
		  (p.x >= cxt->xmin)&&(p.x < cxt->xmax)&&
		  (p.y == cxt->yi)){
		img->val[j] = shell->scn->data[i];
		cxt->zbuff->object[j] = (uchar)ly->val[j];
		cxt->zbuff->voxel[j] = i;
		warp->val[j] = 0.0;
	      }
	    }
	  }
	}
      }
    break;

  case 'y': 

    Su = fabs(cxt->ISu[cxt->ki]);
    Sv = fabs(cxt->ISv[cxt->ki]);

    for (v=0; v < cxt->zbuff->vsize; v++)
      for (u=0; u < cxt->zbuff->usize; u++){
	j   = u+cxt->zbuff->tbv[v];
	uw = u - d.x;
	vw = v - d.y;
	q.x = uw*cxt->IW[0][0] + vw*cxt->IW[0][1] + c.z;
	q.y = uw*cxt->IW[1][0] + vw*cxt->IW[1][1] + c.x;
	for (k=cxt->ki; k != cxt->kf; k += cxt->dk) {
	  p.z = ROUND(q.x + cxt->ISu[k]); 
	  p.x = ROUND(q.y + cxt->ISv[k]); 
	  p.y = k;
	  if (ValidVoxel(shell->scn,p.x,p.y,p.z)) {
	    i   = p.x + shell->scn->tby[p.y] + shell->scn->tbz[p.z];
	    if (lz->val[j]) {
	      if ((p.x >= cxt->xmin)&&(p.x < cxt->xmax)&&
		  (p.y >= cxt->ymin)&&(p.y < cxt->ymax)&&
		  (p.z == cxt->zi)){
		img->val[j] = shell->scn->data[i];
		cxt->zbuff->object[j] = (uchar)lz->val[j];
		cxt->zbuff->voxel[j] = i;
		warp->val[j] = 0.0;
	      }
	    }
	    if (lx->val[j]) {
	      if ((p.z >= cxt->zmin)&&(p.z < cxt->zmax)&&
		  (p.y >= cxt->ymin)&&(p.y < cxt->ymax)&&
		  (p.x == cxt->xi)){
		img->val[j] = shell->scn->data[i];
		cxt->zbuff->object[j] = (uchar)lx->val[j];
		cxt->zbuff->voxel[j] = i;
		warp->val[j] = 0.0;	       
	      }
	    }
	    if (ly->val[j]) {
	      if ((p.z >= cxt->zmin)&&(p.z < cxt->zmax)&&
		  (p.x >= cxt->xmin)&&(p.x < cxt->xmax)&&
		  (p.y == cxt->yi)){
		  img->val[j] = shell->scn->data[i];
		  cxt->zbuff->object[j] = (uchar)ly->val[j];
		  cxt->zbuff->voxel[j] = i;
		  warp->val[j] = 0.0;
	      }
	    }
	  }
	}
      }
    break;

  case 'z':

    Su = fabs(cxt->ISu[cxt->ki]);
    Sv = fabs(cxt->ISv[cxt->ki]);

    for (v=0; v < cxt->zbuff->vsize; v++)
      for (u=0; u < cxt->zbuff->usize; u++){
	j   = u+cxt->zbuff->tbv[v];
	uw = u - d.x;
	vw = v - d.y;
	q.x = uw*cxt->IW[0][0] + vw*cxt->IW[0][1] + c.x;
	q.y = uw*cxt->IW[1][0] + vw*cxt->IW[1][1] + c.y;	
	for (k=cxt->ki; k != cxt->kf; k += cxt->dk) {
	  p.x = ROUND(q.x + cxt->ISu[k]); 
	  p.y = ROUND(q.y + cxt->ISv[k]); 
	  p.z = k;
	  if (ValidVoxel(shell->scn,p.x,p.y,p.z)) {
	    i   = p.x + shell->scn->tby[p.y] + shell->scn->tbz[p.z];
	    if (lz->val[j]) {
	      if ((p.x >= cxt->xmin)&&(p.x < cxt->xmax)&&\
		  (p.y >= cxt->ymin)&&(p.y < cxt->ymax)&&\
		  (p.z == cxt->zi)){
		img->val[j] = shell->scn->data[i];
		cxt->zbuff->object[j] = (uchar)lz->val[j];
		cxt->zbuff->voxel[j] = i;
		warp->val[j] = 0.0;
	      } 
	    }
	    if (lx->val[j]) {
	      if ((p.z >= cxt->zmin)&&(p.z < cxt->zmax)&&\
		  (p.y >= cxt->ymin)&&(p.y < cxt->ymax)&&\
		  (p.x == cxt->xi)){
		img->val[j] = shell->scn->data[i];
 		cxt->zbuff->object[j] = (uchar)lx->val[j];
		cxt->zbuff->voxel[j] = i;
		warp->val[j] = 0.0;
	      }
	    }
	    if (ly->val[j]) {
	      if ((p.z >= cxt->zmin)&&(p.z < cxt->zmax)&&\
		  (p.x >= cxt->xmin)&&(p.x < cxt->xmax)&&\
		  (p.y == cxt->yi)){
		img->val[j] = shell->scn->data[i];
		cxt->zbuff->object[j] = (uchar)ly->val[j];
		cxt->zbuff->voxel[j] = i;
		warp->val[j] = 0.0;
	      }
	    }
	  }
	}
      }    
    break;
    
  default: 
    Error(MSG1,"CSWShellRenderingDrawSceneCut");
    break;

  }
  DestroyImage(&lx);
  DestroyImage(&ly);
  DestroyImage(&lz);
}


void SWShellRenderingDrawSceneCut(Image *img, Shell *shell,Context *cxt, FBuffer *shear, FBuffer *warp, char axis) 
{

  int u,v,ut,vt,uw,vw,ww,is,iw,iw_p,i,i_p,i1,i2,j,index;
  float w,wt; 
  uchar l;

  Voxel p,q,c;
  Pixel d,e;
  AdjPxl *fprint=NULL;
  fprint = AdjPixels(img,cxt->footprint->adj);    

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  switch(axis) {

  case 'x':
    
    p.x = cxt->xi;
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      i = shell->Mzx->tbrow[p.x] + p.z;
      
      if (shell->Mzx->val[i] >= 0) {
	
	if (cxt->dy == 1) {              
	  i1 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Mzx->val[i]-1;            
	} else {            
	  i2 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Mzx->val[i]-1;
	}
	i = i1;
	i_p = shell->pointer[i].i;
	e.y = shell->pointer[i2].y + cxt->dy;
	for (p.y = shell->pointer[i1].y; p.y != e.y; p.y += cxt->dy) {
	  index = shell->scn->tbz[p.z] + shell->scn->tby[p.y] + p.x;	  
	  if (cxt->ymin <= p.y && p.y < cxt->ymax) {
	    if (shell->scn->data[index]) {
	      i_p = shell->pointer[i].i;
	      if (shell->pointer[i].y == p.y)
		i += cxt->dy;
	    }
	    if (shell->voxel[i_p].visible) {

	      // shear
	    
	      q.x = p.x - c.x;	    
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = (int) (q.y + cxt->Su[p.x]); 
	      v = (int) (q.z + cxt->Sv[p.x]); 
	    	    
	      is = u + ut+ shear->tbv[v + vt];
	    
	      if (shear->val[is] > 0.05) {
	      
		l = shell->voxel[i_p].obj;

		if (l) {

		  // warp
		  uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		  vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
	      
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;	     
	      
		  ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
	      
		  w = ww + wt;
		  // splatting
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;		
		  for (j=0; j < fprint->n; j++){		
		    iw_p = iw + fprint->dp[j];
		    if (cxt->zbuff->dist[iw_p] > cxt->depth - w) {
		      warp->val[iw_p] = 0.0;		      
		      img->val[iw_p] = shell->scn->data[index];
		      cxt->zbuff->voxel[iw_p] = i_p;
		      cxt->zbuff->object[iw_p] = l;
		      cxt->zbuff->dist[iw_p] = cxt->depth - w;
		    }
		  }
		  shear->val[is]=warp->val[iw];
		}      
     	      }
	    }
	  }
	}
      }
    }
    break;

  case 'y': 
    
    p.y = cxt->yi;
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      i = shell->Myz->tbrow[p.z] + p.y;
        
      if (shell->Myz->val[i] >= 0) {
          
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i = i1;
	i_p = i;
	e.x = shell->voxel[i2].x + cxt->dx;
	for (p.x = shell->voxel[i1].x; p.x != e.x; p.x += cxt->dx) {
	  index = shell->scn->tbz[p.z] + shell->scn->tby[p.y] + p.x;	  
	  if (cxt->xmin <= p.x && p.x < cxt->xmax) {
	    if (shell->scn->data[index]) {
	      i_p = i;
	      if (shell->voxel[i].x == p.x)
		i += cxt->dx;
	    }
	    if (shell->voxel[i_p].visible) {

	      // shear	    
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = (int) (q.z + cxt->Su[p.y]); 
	      v = (int) (q.x + cxt->Sv[p.y]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {

		l = shell->voxel[i_p].obj;

		if (l) {

		  // warp
		  uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		  vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
	       
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;	     
	       
		  ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
	       
		  w = ww + wt;

		  // splatting
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;		
		  for (j=0; j < fprint->n; j++){		
		    iw_p = iw + fprint->dp[j];
		    
		    if (cxt->zbuff->dist[iw_p] > cxt->depth - w) {
		      warp->val[iw_p] = 0.0;		      
		      img->val[iw_p] = shell->scn->data[index];
		      cxt->zbuff->voxel[iw_p] = i_p;
		      cxt->zbuff->object[iw_p] = l;		      
		      cxt->zbuff->dist[iw_p] = cxt->depth - w;
		    }
		  }
		  shear->val[is]=warp->val[iw];
		}      
	      }
	    }
	  }
	}
      }
    }
    break;

  case 'z':
   
    p.z = cxt->zi;
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
      i = shell->Myz->tbrow[p.z] + p.y;
      
      if (shell->Myz->val[i] >= 0) {
	
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i = i1;
	i_p = i;
	e.x = shell->voxel[i2].x + cxt->dx;
	for (p.x = shell->voxel[i1].x; p.x != e.x; p.x += cxt->dx) {
	  index = shell->scn->tbz[p.z] + shell->scn->tby[p.y] + p.x;	  
	  if (cxt->xmin <= p.x && p.x < cxt->xmax) {
	    if (shell->scn->data[index]){
	      i_p = i;
	      if (shell->voxel[i].x == p.x)
		i += cxt->dx;
	    }

	    if (shell->voxel[i_p].visible) {

	      // shear
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 

	      u = (int) (q.x + cxt->Su[p.z]); 
	      v = (int) (q.y + cxt->Sv[p.z]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		l = shell->voxel[i_p].obj;

		if (l) {

		  // warp
		  uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		  vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
	      
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;	     
	      
		  ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
	      
		  w = ww + wt;
		  // splatting
		  iw = cxt->zbuff->tbv[vw + d.y] + uw + d.x;		
		  for (j=0; j < fprint->n; j++){		
		    iw_p = iw + fprint->dp[j];
		    if (cxt->zbuff->dist[iw_p] > cxt->depth - w) {
		      warp->val[iw_p] = 0.0;
		      img->val[iw_p] = shell->scn->data[index];
		      cxt->zbuff->voxel[iw_p] = i_p;
		      cxt->zbuff->object[iw_p] = l;
		      cxt->zbuff->dist[iw_p] = cxt->depth - w;
		    }
		  }
		  shear->val[is]=warp->val[iw];
		}      
	      }
	    }
	  }
	}
      }
    }
    break;

  default: 
    Error(MSG1,"CSWShellRenderingDrawCut");
    break;
  }  

  
  DestroyAdjPxl(&fprint);
}


void ShellRenderingDrawSceneCut(Image *img, Shell *shell,Context *cxt, FBuffer *cumopac, char axis)
{
  
  int ut,vt,u,v,ww,iw,iw_p,i,i_p,i1,i2,j,index;
  float w,wt; 
  uchar l;

  Voxel p,q,c;
  Pixel d,e;
  AdjPxl *fprint=NULL;
  fprint = AdjPixels(img,cxt->footprint->adj);    

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  switch(axis) {

  case 'x':
    
    p.x = cxt->xi;
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      i = shell->Mzx->tbrow[p.x] + p.z;
      
      if (shell->Mzx->val[i] >= 0) {
	
	if (cxt->dy == 1) {              
	  i1 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Mzx->val[i]-1;            
	} else {            
	  i2 = shell->Mzx->val[i];
	  i++;
	  while (shell->Mzx->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Mzx->val[i]-1;
	}
	i = i1;
	e.y = shell->pointer[i2].y + cxt->dy;
	for (p.y = shell->pointer[i1].y; p.y != e.y; p.y += cxt->dy) {
	  index = shell->scn->tbz[p.z] + shell->scn->tby[p.y] + p.x;	  
	  if (cxt->ymin <= p.y && p.y <= cxt->ymax && shell->scn->data[index]){
	    i_p = shell->pointer[i].i;
/* 	    if (shell->pointer[i].y == p.y)  */
/* 	      i += cxt->dy; */

/* 	    if (shell->voxel[i_p].visible) { */

	      q.x = p.x - c.x;	    
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	        	      
	      l = shell->voxel[i_p].obj;

	      if (l) {

		u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
		v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;	     	      
		ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		w = ww + wt;
		// splatting
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (cxt->zbuff->dist[iw_p] > cxt->depth - w) {
		    cumopac->val[iw_p] = 0.0; 
		    img->val[iw_p] = shell->scn->data[index];
		    cxt->zbuff->voxel[iw_p] = i_p;
		    cxt->zbuff->object[iw_p] = l;
              	    cxt->zbuff->dist[iw_p] = cxt->depth - w;		  
		  }
		}
		// }
	    }
	  }
	}
      }
    }
 
    break;

  case 'y': 
    
    p.y = cxt->yi;
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      i = shell->Myz->tbrow[p.z] + p.y;
        
      if (shell->Myz->val[i] >= 0) {
          
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i = i1;
	e.x = shell->voxel[i2].x + cxt->dx;
	for (p.x = shell->voxel[i1].x; p.x != e.x; p.x += cxt->dx) {
	  index = shell->scn->tbz[p.z] + shell->scn->tby[p.y] + p.x;	  
	  if (cxt->xmin <= p.x && p.x <= cxt->xmax && shell->scn->data[index]){
	    i_p = i;
/* 	    if (shell->voxel[i].x == p.x)  */
/* 	      i += cxt->dx; */

/* 	    if (shell->voxel[i_p].visible) { */

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      l = shell->voxel[i_p].obj;

	      if (l) {
	      
		u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
		v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;	     	      
		ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		w = ww + wt;

		// splatting
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (cxt->zbuff->dist[iw_p] > cxt->depth - w) {
		    cumopac->val[iw_p] = 0.0;
		    img->val[iw_p] = shell->scn->data[index];
		    cxt->zbuff->voxel[iw_p] = i_p;
		    cxt->zbuff->object[iw_p] = l;
		    cxt->zbuff->dist[iw_p] = cxt->depth - w;		  
		  }
		}
		// }
	    }
	  }
	}
      }
    }
  
    break;

  case 'z':
    
    p.z = cxt->zi;
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
      
      i = shell->Myz->tbrow[p.z] + p.y;
      
      if (shell->Myz->val[i] >= 0) {
	
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i = i1;
	e.x = shell->voxel[i2].x + cxt->dx;
	for (p.x = shell->voxel[i1].x; p.x != e.x; p.x += cxt->dx) {
	  index = shell->scn->tbz[p.z] + shell->scn->tby[p.y] + p.x;
	  if (cxt->xmin <= p.x && p.x <= cxt->xmax && shell->scn->data[index]){
	    i_p = i;
/* 	    if (shell->voxel[i].x == p.x)  */
/* 	      i += cxt->dx; */

/* 	    if (shell->voxel[i_p].visible) { */
	      
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 	      

	      l = shell->voxel[i_p].obj;

	      if (l) {
		
		u=(int)(cxt->R[0][0]*q.x + cxt->R[0][1]*q.y + cxt->R[0][2]*q.z);
		v=(int)(cxt->R[1][0]*q.x + cxt->R[1][1]*q.y + cxt->R[1][2]*q.z);
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;	     	      
		ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		w = ww + wt;

		// splatting
		iw = cxt->zbuff->tbv[v + d.y] + u + d.x;
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (cxt->zbuff->dist[iw_p] > cxt->depth - w){
		    cumopac->val[iw_p] = 0.0;
		    img->val[iw_p] = shell->scn->data[index];
		    cxt->zbuff->voxel[iw_p] = i_p;
		    cxt->zbuff->object[iw_p] = l;
		    cxt->zbuff->dist[iw_p] = cxt->depth - w;		 
		  }
		}
		//}
	    }
	  }
	}
      }
    }
    break;

  default: 
    Error(MSG1,"ShellRenderingDrawCut");
    break;

  }  
  DestroyAdjPxl(&fprint);
}

Image *SWShellRenderingCut(Shell *shell,Context *cxt, Curve *cs) {

  Image *img=NULL,*tmp=NULL;
  FBuffer *shear=NULL,*warp=NULL;
  Voxel p,q,c;
  Pixel d;
  Vector viewer;
  AdjPxl *fprint=NULL;

  int u,v,ut,vt,uw,vw,is,iw,iw_p,i,i_p,i1,i2,j,n;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,nopac[256],opac,k;
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  warp = CreateFBuffer(cxt->width, cxt->height);
  InitFBuffer(shear,1.0);
  InitFBuffer(warp,1.0);
  InitZBuffer(cxt->zbuff);
  
  img=CreateImage(cxt->width, cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  switch(cxt->PAxis) {

  case 'x':

    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'x');
      ShellRenderingDrawCut(img,shell,cxt,warp,'y');
      ShellRenderingDrawCut(img,shell,cxt,warp,'z');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      DestroyImage(&img);
      img = tmp;
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'x');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'y');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'z');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      DestroyImage(&img);
      img = tmp;
    }
    
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Mzx->tbrow[p.x] + p.z;
        
        if (shell->Mzx->val[i] >= 0) {
          
          if (cxt->dy == 1) {              
            i1 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i2 = shell->Mzx->val[i]-1;            
          } else {            
            i2 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
          for (i = i1; i != i2; i += cxt->dy) {

	    // shear
	    i_p = shell->pointer[i].i;
	    l = shell->voxel[i_p].obj;

	    if (l && shell->voxel[i_p].visible) {

	      p.y = shell->pointer[i].y;
	      if (cxt->ymin <= p.y && p.y <= cxt->ymax) {
		q.x = p.x - c.x;
		q.y = p.y - c.y;
		q.z = p.z - c.z;

		u = (int) (q.y + cxt->Su[p.x]); 
		v = (int) (q.z + cxt->Sv[p.x]); 

		is = u + ut + shear->tbv[v + vt];

		if (shear->val[is] > 0.05) {
	    
		  // normal

		  n = shell->voxel[i_p].normal;

		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;

		  if (cos_a > 0.0) {

		    // shading


		    w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
                    k = cxt->depth - w;
		    idist = 1.0 - w/(float)cxt->depth;

		    cos_2a = 2*cos_a*cos_a - 1.0;
	      
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }

		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac; 
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;

		    shading =  (float)255.* opac * (amb + idist * (diff + spec));

		    //warp	      
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
	      
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		    iw = warp->tbv[vw + d.y] + uw + d.x;
	      
		    // splatting
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			if (cxt->zbuff->dist[iw_p] > k) {
			  img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
			  warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
			  if (cxt->zbuff->voxel[iw_p] == NIL) {
			    cxt->zbuff->voxel[iw_p] = i_p;
			    cxt->zbuff->object[iw_p] = l;
			    cxt->zbuff->dist[iw_p] = k;
			  }
			}
		      }
		    }
		    shear->val[is]=warp->val[iw];
		  }
		}
	      }
	    }
	  }	 	 
	}
      }

    break;
    
  case 'y': 

    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'y');
      ShellRenderingDrawCut(img,shell,cxt,warp,'z');
      ShellRenderingDrawCut(img,shell,cxt,warp,'x');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      DestroyImage(&img);
      img = tmp;
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'y');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'z');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'x');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      DestroyImage(&img);
      img = tmp;
    }
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    i_p = i;
	    l = shell->voxel[i_p].obj;

	    if (l && shell->voxel[i_p].visible) {

	      p.x = shell->voxel[i].x;
	      if (cxt->xmin <= p.x && p.x <= cxt->xmax) { 	   
		q.x = p.x - c.x;
		q.y = p.y - c.y;
		q.z = p.z - c.z;
	    
		u = (int) (q.z + cxt->Su[p.y]); 
		v = (int) (q.x + cxt->Sv[p.y]); 

		is = u + ut+ shear->tbv[v + vt];

		if (shear->val[is] > 0.05) {
	      
		  // normal

		  n = shell->voxel[i_p].normal;

		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;

		  if (cos_a > 0.0) {

		    // shading

		    w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
                    k = cxt->depth - w;
		    idist = 1.0 - w/(float)cxt->depth;

		    cos_2a = 2*cos_a*cos_a - 1.0;
		
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }

		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;
		
		    shading = (float)255.* opac * (amb + idist * (diff + spec));
		
		    // warp	      
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		
		    iw = warp->tbv[vw + d.y] + uw + d.x;
		
		    // splatting
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			if (cxt->zbuff->dist[iw_p] > k) {
			  img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
			  warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
			  if (cxt->zbuff->voxel[iw_p] == NIL) {
			    cxt->zbuff->voxel[iw_p] = i_p;
			    cxt->zbuff->object[iw_p] = l;
			    cxt->zbuff->dist[iw_p] = k;
			  }
			}
		      }
		    }
		    shear->val[is]=warp->val[iw];
		  }
		}
	      }
	    }
	  }
	}
      }
    break;

  case 'z': 

    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'z');
      ShellRenderingDrawCut(img,shell,cxt,warp,'x');
      ShellRenderingDrawCut(img,shell,cxt,warp,'y');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      DestroyImage(&img);
      img = tmp;
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'z');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'x');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'y');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      DestroyImage(&img);
      img = tmp;
    }
    
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) 
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    i_p = i;
	    l = shell->voxel[i_p].obj;

	    if (l && shell->voxel[i_p].visible) {

	      p.x = shell->voxel[i].x;
	      if (cxt->xmin <= p.x && p.x <= cxt->xmax) { 
		q.x = p.x - c.x;
		q.y = p.y - c.y;
		q.z = p.z - c.z; 

		u = (int) (q.x + cxt->Su[p.z]); 
		v = (int) (q.y + cxt->Sv[p.z]); 

		is = u + ut + shear->tbv[v + vt];
	    
		if (shear->val[is] > 0.05) {
	      
		  // normal
		  n = shell->voxel[i_p].normal;

		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;

		  if (cos_a > 0.0) {

		    // shading

		    w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
                    k = cxt->depth - w;
		    idist = 1.0 - w/(float)cxt->depth;

		    cos_2a = 2*cos_a*cos_a - 1.0;
		
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }
		
		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;
		
		    shading =  (float)255.* opac * (amb + idist * (diff + spec));
		
		    // warp
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		
		    iw = warp->tbv[vw + d.y] + uw + d.x;
		
		    // splatting
		
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			if (cxt->zbuff->dist[iw_p] > k) {
			  img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
			  warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
			  if (cxt->zbuff->voxel[iw_p] == NIL) {
			    cxt->zbuff->voxel[iw_p] = i_p;
			    cxt->zbuff->object[iw_p] = l;
			    cxt->zbuff->dist[iw_p] = k;
			  }
			}
		      }
		    }	      
		    shear->val[is]=warp->val[iw];
		  }
		}
	      }
	    }
	  }
	}
      }
    break;
      
  default: 
    Error(MSG1,"SWShellRendering");
    break;
    
  }			
  DestroyFBuffer(&shear);
  DestroyFBuffer(&warp);
  DestroyAdjPxl(&fprint);
  return(img);
}

CImage *CSWShellRenderingCut(Shell *shell,Context *cxt, Curve *cs) 
{

  CImage *cimg=NULL;
  Image *img=NULL,*tmp=NULL;
  FBuffer *shear=NULL,*warp=NULL;
  Voxel p,q,c;
  Pixel d;
  Vector viewer;
  AdjPxl *fprint=NULL;


  int u,v,ut,vt,uw,vw,ww,is,iw,iw_p,i,i_p,i1,i2,j,n;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,alpha,nopac[256],opac,k; 
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  warp = CreateFBuffer(cxt->width, cxt->height);

  InitFBuffer(shear,1.0);
  InitFBuffer(warp,1.0);
  InitZBuffer(cxt->zbuff);

  img = CreateImage(cxt->width,cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  switch(cxt->PAxis) {

  case 'x':


    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'x');
      ShellRenderingDrawCut(img,shell,cxt,warp,'y');
      ShellRenderingDrawCut(img,shell,cxt,warp,'z');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else if (shell->scn) {
      ISWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'x');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,img);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    } 
      
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) {
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
	i = shell->Mzx->tbrow[p.x] + p.z;
        
	if (shell->Mzx->val[i] >= 0) {
          
	  if (cxt->dy == 1) {              
	    i1 = shell->Mzx->val[i];
	    i++;
	    while (shell->Mzx->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Mzx->val[i]-1;            
	  } else {            
	    i2 = shell->Mzx->val[i];
	    i++;
	    while (shell->Mzx->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
	  for (i = i1; i != i2; i += cxt->dy) {

	    // shear
	    p.y = shell->pointer[i].y;
	    if (cxt->ymin <= p.y && p.y <= cxt->ymax) {
		  
	      q.x = p.x - c.x;	    
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;

	      u = (int) (q.y + cxt->Su[p.x]); 
	      v = (int) (q.z + cxt->Sv[p.x]); 

	      is = u + ut + shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
		
		// normal
		i_p = shell->pointer[i].i;
		
		if (shell->voxel[i_p].visible) {		
		
		  l = shell->voxel[i_p].obj;

		  if (l) {

		  n = shell->voxel[i_p].normal;
		  
		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;
		  
		  if (cos_a > 0) {
		       
		    //warp	      
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		    
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		    iw = warp->tbv[vw + d.y] + uw + d.x;

		    ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;		  
		    w = ww + wt;k = cxt->depth - w;

		    // shading

		    idist = 1.0 - w/(float)cxt->depth;
		    
		    cos_2a = 2*cos_a*cos_a - 1.0;
		    
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }
		    
		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;
		    
		    shading = (amb + idist * (diff + spec));
		    
		    // splatting
		    
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			alpha = cxt->footprint->val[j] * opac;
			AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			warp->val[iw_p] *= (1.0-alpha);
			if (cxt->zbuff->voxel[iw_p] == NIL) {
			  cxt->zbuff->voxel[iw_p] = i_p;
			  cxt->zbuff->object[iw_p] = l;
			  if (cxt->zbuff->dist[iw_p] > k)
 		            cxt->zbuff->dist[iw_p] = k;
			}
		      }
		    }
		    shear->val[is]=warp->val[iw];
		  }
		} 		     
	      }
	    }
	  }
	}
      }
    }
    }
    break;
    
  case 'y': 


    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'y');
      ShellRenderingDrawCut(img,shell,cxt,warp,'z');
      ShellRenderingDrawCut(img,shell,cxt,warp,'x');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else if (shell->scn) {
      ISWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'y');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,img);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    }
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
	i = shell->Myz->tbrow[p.z] + p.y;
        
	if (shell->Myz->val[i] >= 0) {
          
	  if (cxt->dx == 1) {              
	    i1 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Myz->val[i]-1;            
	  } else {            
	    i2 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
	  for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    p.x = shell->voxel[i].x;
	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {        	   
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = (int) (q.z + cxt->Su[p.y]); 
	      v = (int) (q.x + cxt->Sv[p.y]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {

		  l = shell->voxel[i_p].obj;

		  if (l) {

		  n = shell->voxel[i_p].normal;

		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;

		  if (cos_a > 0) {
		    
		    // warp	      
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		    iw = warp->tbv[vw + d.y] + uw + d.x;
		  
		    ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;

		    w = ww + wt;k = cxt->depth - w;

		    // shading
		   
		    idist = 1.0 - w/(float)cxt->depth;

		    cos_2a = 2*cos_a*cos_a - 1.0;
		
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }

		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;
		
		    shading =  (amb + idist * (diff + spec));
		
		    // splatting
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			alpha = cxt->footprint->val[j] * opac;
			AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			warp->val[iw_p] *= (1.0-alpha);
			if (cxt->zbuff->voxel[iw_p] == NIL) {
			  cxt->zbuff->voxel[iw_p] = i_p;
			  cxt->zbuff->object[iw_p] = l;
			  if (cxt->zbuff->dist[iw_p] > k)
		            cxt->zbuff->dist[iw_p] = k;
			}
		      }
		    }
		    shear->val[is]=warp->val[iw];
		  }
		}
	      }
	    }
	  }
	}
	}
      }
    }
    break;
    
  case 'z': 

    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'z');
      ShellRenderingDrawCut(img,shell,cxt,warp,'x');
      ShellRenderingDrawCut(img,shell,cxt,warp,'y');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,img);
    } else if (shell->scn) {
      ISWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'z');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    }

    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
	i = shell->Myz->tbrow[p.z] + p.y;
        
	if (shell->Myz->val[i] >= 0) {
          
	  if (cxt->dx == 1) {              
	    i1 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Myz->val[i]-1;            
	  } else {            
	    i2 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
	  for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    p.x = shell->voxel[i].x;
	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {        	   
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 

	      u = (int) (q.x + cxt->Su[p.z]); 
	      v = (int) (q.y + cxt->Sv[p.z]); 

	      is = u + ut + shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {

		  l = shell->voxel[i_p].obj;

		  if (l) {

		  n = shell->voxel[i_p].normal;

		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;

		  if (cos_a > 0) {

		    // warp
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		    iw = warp->tbv[vw + d.y] + uw + d.x;

		    ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		  
		    w = ww + wt;k = cxt->depth - w;

		    // shading

		    idist = 1.0 - w/(float)cxt->depth;

		    cos_2a = 2*cos_a*cos_a - 1.0;
		
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }
		
		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;
		
		    shading =  (amb + idist * (diff + spec));
				
		    // splatting
		    iw = warp->tbv[vw + d.y] + uw + d.x;		
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			alpha = cxt->footprint->val[j] * opac;
			AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			warp->val[iw_p] *= (1.0-alpha);
			if (cxt->zbuff->voxel[iw_p] == NIL) {
			  cxt->zbuff->voxel[iw_p] = i_p;
			  cxt->zbuff->object[iw_p] = l;
			  if (cxt->zbuff->dist[iw_p] > k)
		            cxt->zbuff->dist[iw_p] = k;
			}
		      }
		    }	      
		    shear->val[is]=warp->val[iw];
		  } 
		}
	      }
	    }
	  }
	}
      }
    }
    }
    break;
      
  default: 
    Error(MSG1,"CSWShellRenderingCut");
    break;
  }		
  DestroyImage(&img);  
  DestroyImage(&tmp);
  DestroyFBuffer(&shear);
  DestroyFBuffer(&warp);
  DestroyAdjPxl(&fprint);
  return(cimg);
}

CImage *FastCSWShellRenderingCut(Shell *shell,Context *cxt, Curve *cs) 
{

  CImage *cimg=NULL;
  Image *img=NULL,*tmp=NULL;
  FBuffer *shear=NULL,*warp=NULL;
  Voxel p,q,c;
  Pixel d;
  Vector viewer;
  AdjPxl *fprint=NULL;


  int u,v,ut,vt,uw,vw,ww,is,iw,iw_p,i,i_p,i1,i2,j,n;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,alpha,nopac[256],opac,k; 
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  warp = CreateFBuffer(cxt->width, cxt->height);

  InitFBuffer(shear,1.0);
  InitFBuffer(warp,1.0);
  InitZBuffer(cxt->zbuff);

  img = CreateImage(cxt->width,cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  switch(cxt->PAxis) {

  case 'x':


    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'x');
      ShellRenderingDrawCut(img,shell,cxt,warp,'y');
      ShellRenderingDrawCut(img,shell,cxt,warp,'z');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'x');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'y');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'z');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    } 
      
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) {
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
	i = shell->Mzx->tbrow[p.x] + p.z;
        
	if (shell->Mzx->val[i] >= 0) {
          
	  if (cxt->dy == 1) {              
	    i1 = shell->Mzx->val[i];
	    i++;
	    while (shell->Mzx->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Mzx->val[i]-1;            
	  } else {            
	    i2 = shell->Mzx->val[i];
	    i++;
	    while (shell->Mzx->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
	  for (i = i1; i != i2; i += cxt->dy) {

	    // shear
	    p.y = shell->pointer[i].y;
	    if (cxt->ymin <= p.y && p.y <= cxt->ymax) {
		  
	      q.x = p.x - c.x;	    
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;

	      u = (int) (q.y + cxt->Su[p.x]); 
	      v = (int) (q.z + cxt->Sv[p.x]); 

	      is = u + ut + shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
		
		// normal
		i_p = shell->pointer[i].i;
		
		if (shell->voxel[i_p].visible) {		
		
		  l = shell->voxel[i_p].obj;

		  if (l) {

		  n = shell->voxel[i_p].normal;
		  
		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;
		  
		  if (cos_a > 0) {
		       
		    //warp	      
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		    
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		    iw = warp->tbv[vw + d.y] + uw + d.x;

		    ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;		  
		    w = ww + wt;k = cxt->depth - w;

		    // shading

		    idist = 1.0 - w/(float)cxt->depth;
		    
		    cos_2a = 2*cos_a*cos_a - 1.0;
		    
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }
		    
		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;
		    
		    shading = (amb + idist * (diff + spec));
		    
		    // splatting
		    
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			alpha = cxt->footprint->val[j] * opac;
			AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			warp->val[iw_p] *= (1.0-alpha);
			if (cxt->zbuff->voxel[iw_p] == NIL) {
			  cxt->zbuff->voxel[iw_p] = i_p;
			  cxt->zbuff->object[iw_p] = l;
			  if (cxt->zbuff->dist[iw_p] > k)
		  cxt->zbuff->dist[iw_p] = k;
			}
		      }
		    }
		    shear->val[is]=warp->val[iw];
		  }
		} 		     
	      }
	    }
	  }
	}
      }
    }
    }
    break;
    
  case 'y': 


    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'y');
      ShellRenderingDrawCut(img,shell,cxt,warp,'z');
      ShellRenderingDrawCut(img,shell,cxt,warp,'x');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'y');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'z');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'x');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    }
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
	i = shell->Myz->tbrow[p.z] + p.y;
        
	if (shell->Myz->val[i] >= 0) {
          
	  if (cxt->dx == 1) {              
	    i1 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Myz->val[i]-1;            
	  } else {            
	    i2 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
	  for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    p.x = shell->voxel[i].x;
	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {        	   
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = (int) (q.z + cxt->Su[p.y]); 
	      v = (int) (q.x + cxt->Sv[p.y]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {

		  l = shell->voxel[i_p].obj;

		  if (l) {

		  n = shell->voxel[i_p].normal;

		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;

		  if (cos_a > 0) {
		    
		    // warp	      
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		    iw = warp->tbv[vw + d.y] + uw + d.x;
		  
		    ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;

		    w = ww + wt;k = cxt->depth - w;

		    // shading
		   
		    idist = 1.0 - w/(float)cxt->depth;

		    cos_2a = 2*cos_a*cos_a - 1.0;
		
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }

		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;
		
		    shading =  (amb + idist * (diff + spec));
		
		    // splatting
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			alpha = cxt->footprint->val[j] * opac;
			AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			warp->val[iw_p] *= (1.0-alpha);
			if (cxt->zbuff->voxel[iw_p] == NIL) {
			  cxt->zbuff->voxel[iw_p] = i_p;
			  cxt->zbuff->object[iw_p] = l;
			  if (cxt->zbuff->dist[iw_p] > k)
		  cxt->zbuff->dist[iw_p] = k;
			}
		      }
		    }
		    shear->val[is]=warp->val[iw];
		  }
		}
	      }
	    }
	  }
	}
	}
      }
    }
    break;
    
  case 'z': 

    if (shell->body) {
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'z');
      ShellRenderingDrawCut(img,shell,cxt,warp,'x');
      ShellRenderingDrawCut(img,shell,cxt,warp,'y');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'z');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'x');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'y');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    }

    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
	i = shell->Myz->tbrow[p.z] + p.y;
        
	if (shell->Myz->val[i] >= 0) {
          
	  if (cxt->dx == 1) {              
	    i1 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Myz->val[i]-1;            
	  } else {            
	    i2 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
	  for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    p.x = shell->voxel[i].x;
	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {        	   
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 

	      u = (int) (q.x + cxt->Su[p.z]); 
	      v = (int) (q.y + cxt->Sv[p.z]); 

	      is = u + ut + shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {

		  l = shell->voxel[i_p].obj;

		  if (l) {

		  n = shell->voxel[i_p].normal;

		  cos_a = viewer.x * shell->normaltable[n].x +\
		    viewer.y * shell->normaltable[n].y +\
		    viewer.z * shell->normaltable[n].z;

		  if (cos_a > 0) {

		    // warp
		    uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		    vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		    iw = warp->tbv[vw + d.y] + uw + d.x;

		    ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		  
		    w = ww + wt;k = cxt->depth - w;

		    // shading

		    idist = 1.0 - w/(float)cxt->depth;

		    cos_2a = 2*cos_a*cos_a - 1.0;
		
		    if (cos_2a < 0.) {
		      pow = 0.;
		      spec = 0.;
		    } else {
		      pow = 1.;
		      for (j=0; j < cxt->obj[l].ns; j++) 
			pow = pow * cos_2a;
		      spec = cxt->obj[l].spec * pow;
		    }
		
		    opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		    amb  = cxt->obj[l].amb;
		    diff = cxt->obj[l].diff * cos_a;
		
		    shading =  (amb + idist * (diff + spec));
				
		    // splatting
		    iw = warp->tbv[vw + d.y] + uw + d.x;		
		    for (j=0; j < fprint->n; j++){		
		      iw_p = iw + fprint->dp[j];
		      if (warp->val[iw_p] > .05) {
			alpha = cxt->footprint->val[j] * opac;
			AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			warp->val[iw_p] *= (1.0-alpha);
			if (cxt->zbuff->voxel[iw_p] == NIL) {
			  cxt->zbuff->voxel[iw_p] = i_p;
			  cxt->zbuff->object[iw_p] = l;
			  if (cxt->zbuff->dist[iw_p] > k)
		  cxt->zbuff->dist[iw_p] = k;
			}
		      }
		    }	      
		    shear->val[is]=warp->val[iw];
		  } 
		}
	      }
	    }
	  }
	}
      }
    }
    }
    break;
      
  default: 
    Error(MSG1,"CSWShellRenderingCut");
    break;
  }		
  DestroyImage(&img);  
  DestroyImage(&tmp);
  DestroyFBuffer(&shear);
  DestroyFBuffer(&warp);
  DestroyAdjPxl(&fprint);
  return(cimg);
}

CImage *CSWShellRenderingCutPlane (Shell *shell,Context *cxt, Curve *cs, Plane *pl) 
{

  CImage *cimg=NULL;
  Image *img=NULL,*tmp=NULL;
  FBuffer *shear=NULL,*warp=NULL;
  Voxel p,q,c;
  Pixel d;
  Vector viewer;
  AdjPxl *fprint=NULL;

  int u,v,ut,vt,uw,vw,ww,is,iw,iw_p,i,i_p,i1,i2,j,n;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,alpha,nopac[256],opac,k; 
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  warp = CreateFBuffer(cxt->width, cxt->height);

  InitFBuffer(shear,1.0);
  InitFBuffer(warp,1.0);
  InitZBuffer(cxt->zbuff);

  SetPlaneZBuffer(cxt->zbuff, pl);

  img = CreateImage(cxt->width,cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  switch(cxt->PAxis) {

  case 'x':

    if (shell->body) {
      ShellRenderingDrawCutPlane(img,shell,cxt,warp);
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'x');
      ShellRenderingDrawCut(img,shell,cxt,warp,'y');
      ShellRenderingDrawCut(img,shell,cxt,warp,'z');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'x');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'y');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'z');
      ShellRenderingDrawSceneCutPlane(img,shell,cxt,warp);
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    }
      
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) {
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
	i = shell->Mzx->tbrow[p.x] + p.z;
        
	if (shell->Mzx->val[i] >= 0) {
          
	  if (cxt->dy == 1) {              
	    i1 = shell->Mzx->val[i];
	    i++;
	    while (shell->Mzx->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Mzx->val[i]-1;            
	  } else {            
	    i2 = shell->Mzx->val[i];
	    i++;
	    while (shell->Mzx->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
	  for (i = i1; i != i2; i += cxt->dy) {

	    // shear
	    p.y = shell->pointer[i].y;
	    if (cxt->ymin <= p.y && p.y <= cxt->ymax) {
		  
	      q.x = p.x - c.x;	    
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;

	      u = (int) (q.y + cxt->Su[p.x]); 
	      v = (int) (q.z + cxt->Sv[p.x]); 

	      is = u + ut + shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
		
		// normal
		i_p = shell->pointer[i].i;
		
		if (shell->voxel[i_p].visible) {		
		
		  l = shell->voxel[i_p].obj;

		  if (l) {

		    n = shell->voxel[i_p].normal;
		    
		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;		    
		    
		    if (cos_a > 0) {
		      
		      //warp	      
		      uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		      
		      vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		      
		      iw = warp->tbv[vw + d.y] + uw + d.x;
		      
		      ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;		  
		      w = ww + wt;k = cxt->depth - w;
		      
		      if (cxt->zbuff->dist[iw] > k) {

			// shading
			
			idist = 1.0 - w/(float)cxt->depth;
			
			cos_2a = 2*cos_a*cos_a - 1.0;
			
			if (cos_2a < 0.) {
			  pow = 0.;
			  spec = 0.;
			} else {
			  pow = 1.;
			  for (j=0; j < cxt->obj[l].ns; j++) 
			    pow = pow * cos_2a;
			  spec = cxt->obj[l].spec * pow;
			}
			
			opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
			amb  = cxt->obj[l].amb;
			diff = cxt->obj[l].diff * cos_a;
		    
			shading = (amb + idist * (diff + spec));
			
			// splatting
			
			for (j=0; j < fprint->n; j++){		
			  iw_p = iw + fprint->dp[j];
			  if (warp->val[iw_p] > .05) {
			    if (cxt->zbuff->dist[iw_p] > k) {
			      alpha = cxt->footprint->val[j] * opac;
			      AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			      warp->val[iw_p] *= (1.0-alpha);
			      if (cxt->zbuff->voxel[iw_p] == NIL) {
				cxt->zbuff->voxel[iw_p] = i_p;
				cxt->zbuff->object[iw_p] = l;
				cxt->zbuff->dist[iw_p] = k;
			      }
			    }
			  }
			}			  
			shear->val[is]=warp->val[iw];		      
		      } 
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    break;
    
  case 'y': 

    if (shell->body) {
      ShellRenderingDrawCutPlane(img,shell,cxt,warp);
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'y');
      ShellRenderingDrawCut(img,shell,cxt,warp,'z');
      ShellRenderingDrawCut(img,shell,cxt,warp,'x');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'y');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'z');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'x');
      ShellRenderingDrawSceneCutPlane(img,shell,cxt,warp);
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    }
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
	i = shell->Myz->tbrow[p.z] + p.y;
        
	if (shell->Myz->val[i] >= 0) {
          
	  if (cxt->dx == 1) {              
	    i1 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Myz->val[i]-1;            
	  } else {            
	    i2 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
	  for (i = i1; i != i2; i += cxt->dx) {
	    
	    // shear
	    p.x = shell->voxel[i].x;
	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {        	   
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	      
	      u = (int) (q.z + cxt->Su[p.y]); 
	      v = (int) (q.x + cxt->Sv[p.y]); 
	      
	      is = u + ut+ shear->tbv[v + vt];
	      
	      if (shear->val[is] > 0.05) {
		
		// normal
		i_p = i;
		
		if (shell->voxel[i_p].visible) {
		  
		  l = shell->voxel[i_p].obj;
		  
		  if (l) {
		    
		    n = shell->voxel[i_p].normal;
		    
		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;
		      
		    if (cos_a > 0) {
			
		      // warp	      
			uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		      
		      vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		      
		      iw = warp->tbv[vw + d.y] + uw + d.x;
		      
		      ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		      
		      w = ww + wt;k = cxt->depth - w;

		      if (cxt->zbuff->dist[iw] > k) {

			// shading
			
			idist = 1.0 - w/(float)cxt->depth;
			
			cos_2a = 2*cos_a*cos_a - 1.0;
			
			if (cos_2a < 0.) {
			  pow = 0.;
			  spec = 0.;
			} else {
			  pow = 1.;
			  for (j=0; j < cxt->obj[l].ns; j++) 
			    pow = pow * cos_2a;
			  spec = cxt->obj[l].spec * pow;
			}
			
			opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
			amb  = cxt->obj[l].amb;
			diff = cxt->obj[l].diff * cos_a;
			
			shading =  (amb + idist * (diff + spec));
			
			// splatting
			for (j=0; j < fprint->n; j++){		
			  iw_p = iw + fprint->dp[j];
			  if (warp->val[iw_p] > .05) {
  			    if (cxt->zbuff->dist[iw_p] > k) {
			      alpha = cxt->footprint->val[j] * opac;
			      AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			      warp->val[iw_p] *= (1.0-alpha);

			      if (cxt->zbuff->voxel[iw_p] == NIL) {
				cxt->zbuff->voxel[iw_p] = i_p;
				cxt->zbuff->object[iw_p] = l;
				cxt->zbuff->dist[iw_p] = k;
			      }
			    }
			  }
			}
			shear->val[is]=warp->val[iw];
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    break;
    
  case 'z': 

    if (shell->body) {
      ShellRenderingDrawCutPlane(img,shell,cxt,warp);
      SWShellRenderingDrawCut(img,shell,cxt,shear,warp,'z');
      ShellRenderingDrawCut(img,shell,cxt,warp,'x');
      ShellRenderingDrawCut(img,shell,cxt,warp,'y');
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else if (shell->scn) {
      SWShellRenderingDrawSceneCut(img,shell,cxt,shear,warp,'z');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'x');
      ShellRenderingDrawSceneCut(img,shell,cxt,warp,'y');
      ShellRenderingDrawSceneCutPlane(img,shell,cxt,warp);
      tmp =  LinearStretch(img,(int)cs->X[0],(int)cs->X[1],(int)cs->Y[0],(int)cs->Y[1]);
      cimg = Colorize(cxt,tmp);
    } else {
      cimg = CreateCImage(cxt->width,cxt->height);
    }

    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
	i = shell->Myz->tbrow[p.z] + p.y;
        
	if (shell->Myz->val[i] >= 0) {
          
	  if (cxt->dx == 1) {              
	    i1 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i2 = shell->Myz->val[i]-1;            
	  } else {            
	    i2 = shell->Myz->val[i];
	    i++;
	    while (shell->Myz->val[i] < 0) {
	      i++;
	    } 
	    i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
	  for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    p.x = shell->voxel[i].x;
	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {        	   
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 

	      u = (int) (q.x + cxt->Su[p.z]); 
	      v = (int) (q.y + cxt->Sv[p.z]); 

	      is = u + ut + shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {

		  l = shell->voxel[i_p].obj;

		  if (l) {

		    n = shell->voxel[i_p].normal;

		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;
		    
		    if (cos_a > 0) {
		      
		      // warp
		      uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		      
		      vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		      
		      iw = warp->tbv[vw + d.y] + uw + d.x;
		      
		      ww = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z;
		      
		      w = ww + wt;k = cxt->depth - w;

		      if (cxt->zbuff->dist[iw] > k) {
		      
			// shading

			cos_2a = 2*cos_a*cos_a - 1.0;
		      
			idist = 1.0 - w/(float)cxt->depth;

			if (cos_2a < 0.) {
			  pow = 0.;
			  spec = 0.;
			} else {
			  pow = 1.;
			  for (j=0; j < cxt->obj[l].ns; j++) 
			    pow = pow * cos_2a;
			  spec = cxt->obj[l].spec * pow;
			}
		
			opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
			amb  = cxt->obj[l].amb;
			diff = cxt->obj[l].diff * cos_a;
		
			shading =  (amb + idist * (diff + spec));
			
			// splatting
			iw = warp->tbv[vw + d.y] + uw + d.x;		
			for (j=0; j < fprint->n; j++){		
			  iw_p = iw + fprint->dp[j];
			  if (warp->val[iw_p] > .05) {
			    if (cxt->zbuff->dist[iw_p] > k) {
			      alpha = cxt->footprint->val[j] * opac;
			      AccVoxelColor(cimg,cxt,iw_p,shading,warp->val[iw_p]*alpha,l);
			      warp->val[iw_p] *= (1.0-alpha);
			      if (cxt->zbuff->voxel[iw_p] == NIL) {
				cxt->zbuff->voxel[iw_p] = i_p;
				cxt->zbuff->object[iw_p] = l;
				cxt->zbuff->dist[iw_p] = k;
			      }
			    }
			  }
			}	      
			shear->val[is]=warp->val[iw];
		      } 
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    break;
      
  default: 
    Error(MSG1,"CSWShellRenderingCutPlane");
    break;
  }
  DestroyImage(&img);  
  DestroyImage(&tmp);
  DestroyFBuffer(&shear);
  DestroyFBuffer(&warp);
  DestroyAdjPxl(&fprint);
  return(cimg);
}


void EWAKernel(Kernel *K,float vx, float vy, float E, float R, float s, float f)
{
  float z,e,r_xy,R2,r2,z2,e2;
  int i;

  R2 = R*R;
  for (i=0;i<K->adj->n;i++) {
    z = (-K->adj->dx[i] * vx) + (-K->adj->dy[i] * vy);
    r2 = K->adj->dx[i] * K->adj->dx[i] + K->adj->dy[i] * K->adj->dy[i];
    z2 = z*z;    
    r_xy = r2 - z2;
    e = r_xy/E;
    e2 = e*e;
    K->val[i] = s*exp (-f*(e2 + z2)/R2);
  }
}

Image *EWASWShellRendering(Shell *shell,Context *cxt) {

  Image *img=NULL;
  FBuffer *shear=NULL,*warp=NULL;
  Voxel p,q,c;
  Pixel d;
  Vector viewer;
  AdjPxl *fprint=NULL;

  int u,v,ut,vt,uw,vw,is,iw,iw_p,i,i_p,i1,i2,j,n;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,nopac[256],opac;
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  warp = CreateFBuffer(cxt->width, cxt->height);
  InitFBuffer(shear,1.0);
  InitFBuffer(warp,1.0);

  img=CreateImage(cxt->width, cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  switch(cxt->PAxis) {

  case 'x':
    
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Mzx->tbrow[p.x] + p.z;
        
        if (shell->Mzx->val[i] >= 0) {
          
          if (cxt->dy == 1) {              
            i1 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i2 = shell->Mzx->val[i]-1;            
          } else {            
            i2 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
          for (i = i1; i != i2; i += cxt->dy) {

	    // shear
            p.y = shell->pointer[i].y;

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;

	    u = (int) (q.y + cxt->Su[p.x]); 
	    v = (int) (q.z + cxt->Sv[p.x]); 

	    is = u + ut + shear->tbv[v + vt];

	    if (shear->val[is] > 0.05) {
	    
	      // normal
	      i_p = shell->pointer[i].i;
	      n = shell->voxel[i_p].normal;

	      cos_a = viewer.x * shell->normaltable[n].x +\
		viewer.y * shell->normaltable[n].y +\
		viewer.z * shell->normaltable[n].z;

	      if (cos_a > 0) {

		// shading
		l = shell->voxel[i_p].obj;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		idist = 1.0 - w/(float)cxt->depth;

		cos_2a = 2*cos_a*cos_a - 1.0;
	      
		if (cos_2a < 0.) {
		  pow = 0.;
		  spec = 0.;
		} else {
		  pow = 1.;
		  for (j=0; j < cxt->obj[l].ns; j++) 
		    pow = pow * cos_2a;
		  spec = cxt->obj[l].spec * pow;
		}

		opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		amb  = cxt->obj[l].amb;
		diff = cxt->obj[l].diff * cos_a;

		shading =  opac * (amb + idist * (diff + spec));

		//warp	      
		uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
	      
		vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		iw = warp->tbv[vw + d.y] + uw + d.x;
	      
		// splatting
		EWAKernel(cxt->footprint,shell->normaltable[n].z,shell->normaltable[n].y, shell->normaltable[n].x, 2.5, 1.0,1.0);
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (warp->val[iw_p] > .05) {                
		    img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
		    warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
		  }
		}
		shear->val[is]=warp->val[iw];
	      }
	    }
	  }
	}
      }

    break;
    
  case 'y': 
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
            p.x = shell->voxel[i].x;
	   
	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	    
	    u = (int) (q.z + cxt->Su[p.y]); 
	    v = (int) (q.x + cxt->Sv[p.y]); 

	    is = u + ut+ shear->tbv[v + vt];

	    if (shear->val[is] > 0.05) {
	      
	      // normal
	      i_p = i;
	      n = shell->voxel[i_p].normal;

	      cos_a = viewer.x * shell->normaltable[n].x +\
		viewer.y * shell->normaltable[n].y +\
		viewer.z * shell->normaltable[n].z;

	      if (cos_a > 0) {

		// shading
		l = shell->voxel[i_p].obj;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		idist = 1.0 - w/(float)cxt->depth;

		cos_2a = 2*cos_a*cos_a - 1.0;
		
		if (cos_2a < 0.) {
		  pow = 0.;
		  spec = 0.;
		} else {
		  pow = 1.;
		  for (j=0; j < cxt->obj[l].ns; j++) 
		    pow = pow * cos_2a;
		  spec = cxt->obj[l].spec * pow;
		}

		opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		amb  = cxt->obj[l].amb;
		diff = cxt->obj[l].diff * cos_a;
		
		shading =  opac * (amb + idist * (diff + spec));
		
		// warp	      
		uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		
		iw = warp->tbv[vw + d.y] + uw + d.x;
		
		// splatting
		EWAKernel(cxt->footprint,shell->normaltable[n].y,shell->normaltable[n].z, shell->normaltable[n].x, 2.5, 1.0,1.0);
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (warp->val[iw_p] > .05) {                
		    img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
 		    warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
		  }
		}
		shear->val[is]=warp->val[iw];
	      }
	    }
	  }
	}
      }
    break;

  case 'z': 
    
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) 
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
            p.x = shell->voxel[i].x;

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z; 

	    u = (int) (q.x + cxt->Su[p.z]); 
	    v = (int) (q.y + cxt->Sv[p.z]); 

	    is = u + ut + shear->tbv[v + vt];
	    
	    if (shear->val[is] > 0.05) {
	      
	      // normal
	      i_p = i;
	      n = shell->voxel[i_p].normal;

	      cos_a = viewer.x * shell->normaltable[n].x +\
		viewer.y * shell->normaltable[n].y +\
		viewer.z * shell->normaltable[n].z;

	      if (cos_a > 0) {

		// shading
		l = shell->voxel[i_p].obj;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		idist = 1.0 - w/(float)cxt->depth;

		cos_2a = 2*cos_a*cos_a - 1.0;
		
		if (cos_2a < 0.) {
		  pow = 0.;
		  spec = 0.;
		} else {
		  pow = 1.;
		  for (j=0; j < cxt->obj[l].ns; j++) 
		    pow = pow * cos_2a;
		  spec = cxt->obj[l].spec * pow;
		}
		
		opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		amb  = cxt->obj[l].amb;
		diff = cxt->obj[l].diff * cos_a;
		
		shading =  opac * (amb + idist * (diff + spec));
		
		// warp
		uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		
		iw = warp->tbv[vw + d.y] + uw + d.x;
		
		// splatting
		EWAKernel(cxt->footprint,shell->normaltable[n].x,shell->normaltable[n].y, shell->normaltable[n].z, 2.5, 1.0,1.0);
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (warp->val[iw_p] > .05) {                
		    img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
		    warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
		  }
		}	      
		shear->val[is]=warp->val[iw];
	      }
	    }
	  }
	}
      }
    break;
      
  default: 
    Error(MSG1,"SWShellRendering");
    break;
    
  }			
  DestroyFBuffer(&shear);
  DestroyFBuffer(&warp);
  DestroyAdjPxl(&fprint);
  return(img);
}

Image *DirectVolumeRendering(Shell *shell,Context *cxt) {

  Image *img=NULL;
  FBuffer *shear=NULL,*warp=NULL;
  Voxel p,q,c;
  Pixel d;
  Vector viewer;
  AdjPxl *fprint=NULL;

  int u,v,ut,vt,uw,vw,is,iw,iw_p,i,i_p,i1,i2,j,n;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,nopac[256],opac;
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  warp = CreateFBuffer(cxt->width, cxt->height);
  InitFBuffer(shear,1.0);
  InitFBuffer(warp,1.0);
  
  img=CreateImage(cxt->width, cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  switch(cxt->PAxis) {

  case 'x':
    
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Mzx->tbrow[p.x] + p.z;
        
        if (shell->Mzx->val[i] >= 0) {
          
          if (cxt->dy == 1) {              
            i1 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i2 = shell->Mzx->val[i]-1;            
          } else {            
            i2 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
          for (i = i1; i != i2; i += cxt->dy) {

	    // shear

	    i_p = shell->pointer[i].i;

            if (shell->voxel[i_p].opac > 0.05) {
 
	      p.y = shell->pointer[i].y;

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;

	      u = (int) (q.y + cxt->Su[p.x]); 
	      v = (int) (q.z + cxt->Sv[p.x]); 

	      is = u + ut + shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	    
		// normal

		n = shell->voxel[i_p].normal;

		cos_a = fabs(viewer.x * shell->normaltable[n].x +\
			     viewer.y * shell->normaltable[n].y +\
			     viewer.z * shell->normaltable[n].z);
	      

		// shading
		l = shell->voxel[i_p].obj;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		idist = 1.0 - w/(float)cxt->depth;
	      
		cos_2a = 2*cos_a*cos_a - 1.0;
	      
		if (cos_2a < 0.) {
		  pow = 0.;
		  spec = 0.;
		} else {
		  pow = 1.;
		  for (j=0; j < cxt->obj[l].ns; j++) 
		    pow = pow * cos_2a;
		  spec = cxt->obj[l].spec * pow;
		}

		opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		amb  = cxt->obj[l].amb;
		diff = cxt->obj[l].diff * cos_a;

		shading =  255. * opac * (amb + idist * (diff + spec));

		//warp	      
		uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
	      
		vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);

		iw = warp->tbv[vw + d.y] + uw + d.x;
	      
		// splatting
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (warp->val[iw_p] > .05) {                
		    img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
		    warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
		  }
		}
		shear->val[is]=warp->val[iw];
	      }
	    }
	  }
	}
      }
  

    break;
    
  case 'y': 
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    i_p = i;

            if (shell->voxel[i_p].opac > 0.05) {

	      p.x = shell->voxel[i].x;
	   
	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = (int) (q.z + cxt->Su[p.y]); 
	      v = (int) (q.x + cxt->Sv[p.y]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		// normal
		n = shell->voxel[i_p].normal;

		cos_a = fabs(viewer.x * shell->normaltable[n].x +\
			     viewer.y * shell->normaltable[n].y +\
			     viewer.z * shell->normaltable[n].z);

		// shading
		l = shell->voxel[i_p].obj;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		idist = 1.0 - w/(float)cxt->depth;

		cos_2a = 2*cos_a*cos_a - 1.0;
		
		if (cos_2a < 0.) {
		  pow = 0.;
		  spec = 0.;
		} else {
		  pow = 1.;
		  for (j=0; j < cxt->obj[l].ns; j++) 
		    pow = pow * cos_2a;
		  spec = cxt->obj[l].spec * pow;
		}

		opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		amb  = cxt->obj[l].amb;
		diff = cxt->obj[l].diff * cos_a;
		
		shading =  opac * (amb + idist * (diff + spec));
		
		// warp	      
		uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		
		iw = warp->tbv[vw + d.y] + uw + d.x;
		
		// splatting
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (warp->val[iw_p] > .05) {                
		    img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
		    warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
		  }
		}
		shear->val[is]=warp->val[iw];
	      }
	    }
	  }
	}
      }

    break;

  case 'z': 
    
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) 
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
	    i_p = i;

            if (shell->voxel[i_p].opac > 0.05) {

	      p.x = shell->voxel[i].x;

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 

	      u = (int) (q.x + cxt->Su[p.z]); 
	      v = (int) (q.y + cxt->Sv[p.z]); 

	      is = u + ut + shear->tbv[v + vt];
	    
	      if (shear->val[is] > 0.05) {
	      
		// normal
		n = shell->voxel[i_p].normal;

		cos_a = fabs(viewer.x * shell->normaltable[n].x +\
			     viewer.y * shell->normaltable[n].y +\
			     viewer.z * shell->normaltable[n].z);


		// shading
		l = shell->voxel[i_p].obj;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		idist = 1.0 - w/(float)cxt->depth;

		cos_2a = 2*cos_a*cos_a - 1.0;
		
		if (cos_2a < 0.) {
		  pow = 0.;
		  spec = 0.;
		} else {
		  pow = 1.;
		  for (j=0; j < cxt->obj[l].ns; j++) 
		    pow = pow * cos_2a;
		  spec = cxt->obj[l].spec * pow;
		}

		opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		amb  = cxt->obj[l].amb;
		diff = cxt->obj[l].diff * cos_a;
		
		shading =  opac * (amb + idist * (diff + spec));
		
		// warp
		uw = (int)(cxt->W[0][0] * u + cxt->W[0][1] * v);
		
		vw = (int)(cxt->W[1][0] * u + cxt->W[1][1] * v);
		
		iw = warp->tbv[vw + d.y] + uw + d.x;
		
		// splatting
		
		for (j=0; j < fprint->n; j++){		
		  iw_p = iw + fprint->dp[j];
		  if (warp->val[iw_p] > .05) {                
		    img->val[iw_p] += (int)(shading * warp->val[iw_p] * cxt->footprint->val[j]);
		    warp->val[iw_p] *= (1.0- (opac * cxt->footprint->val[j]));
		  }
		}	      
		shear->val[is]=warp->val[iw];
	      }
	    }
	  }
	}
      }

    break;
      
  default: 
    Error(MSG1,"DirectVolumeRendering");
    break;
    
  }			
  DestroyFBuffer(&shear);
  DestroyFBuffer(&warp);
  DestroyAdjPxl(&fprint);
  return(img);
}

