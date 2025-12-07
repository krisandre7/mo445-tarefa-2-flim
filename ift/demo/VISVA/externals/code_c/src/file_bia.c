
#include <oldift.h>
#include "arraylist.h"
#include "segmobject.h"
#include "measures.h"
#include "file_bia.h"

//############################################################
//############## COMMON FUNCTIONS ############################
//############################################################

int BIA_GetFileSize(char *filename) 
{
  FILE *fp = fopen(filename,"r");
  if (fp==NULL) return -1;
  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);
  fclose(fp);
  return size;
}


int BIA_MemcpyFromFile(char *buffer, char *filename, int offset, int size)
{
  FILE *fp = fopen(filename,"rb");
  if (fp==NULL) return 1;
  if (fseek(fp,offset,SEEK_SET)!=0) return 1;
  if (fread(buffer,size,1,fp)!=1) return 1;
  fclose(fp);
  return 0;
}


int BIA_ChunkRead(char filename[], uint file_offset, BIA_ChunkHeader *hdr, char **data) {
  FILE *fp;
  int n;
  fp = fopen(filename,"rb");
  if (fp==NULL) return 1;
  fseek(fp,file_offset,SEEK_SET);
  n = fread(hdr,sizeof(BIA_ChunkHeader),1,fp);
  if (n!=1) { fclose(fp); return 1; }
  *data = (char *) malloc(hdr->size - sizeof(BIA_ChunkHeader));
  if (*data==NULL) { fclose(fp); return 1; }
  n = fread(*data,hdr->size-sizeof(BIA_ChunkHeader),1,fp);
  if (n!=1) { fclose(fp); return 1; }
  fclose(fp);
  return 0;
}

int BIA_PrintInfo(char filename[]) 
{

  BIA_ChunkHeader hdr;  
  char *data;
  int curr_pos;
  int filesize=BIA_GetFileSize(filename);

  BIA_MainHeaderPrintInfo(filename);
  if (filesize<0) return 1;
  curr_pos = sizeof(BIA_MainHeader);
  int counter=1;
  while (curr_pos<filesize-1) {
    if (BIA_ChunkRead(filename, curr_pos, &hdr, &data)) return 1;
    uint n,i;
    printf("CHUNK #%d\n",counter++);
    printf("  Size = %d\n",hdr.size);
    printf("  Type = %d\n",hdr.type);
    printf("  Version = %d\n",hdr.version);
    printf("  Compression = %d\n",hdr.compression);
    printf("  Checksum = %d\n",hdr.checksum);
    printf("  Data = ");
    n = hdr.size-sizeof(BIA_ChunkHeader);
    if (n>30) n=30;
    for (i=0;i<n;i++) printf("%c",data[i]);
    if (n<hdr.size-sizeof(BIA_ChunkHeader)) printf("... (continues) ");
    printf("\n");

    free(data);  
    curr_pos+=hdr.size;
  }
  return 0;
}  
 
 



//############################################################
//############## MAIN HEADER FUNCTIONS #######################
//############################################################

int BIA_MainHeaderPrintInfo(char filename[]) {
  BIA_MainHeader hdr;
  int n; 
  char *c;
  n = BIA_MainHeaderRead(filename,&hdr);
  if (n!=0) return n;
  c = (char *)&hdr.idstring;
  printf("MAIN HEADER INFO - File %s\n",filename);
  printf("  Identification String = %c%c%c%c\n",c[0],c[1],c[2],c[3]);
  printf("  Version = %d\n",hdr.version);
  printf("  Size = %d\n",hdr.size);
  printf("  Num. of Chunks = %d\n",hdr.numofchunks);
  printf("  Creation Time = %s",ctime((time_t *)&hdr.ctime));
  printf("  Modification Time = %s",ctime((time_t *)&hdr.mtime));
  printf("  Checksum = %d\n",hdr.checksum);
  return 0;
}


int BIA_MainHeaderCreate(char filename[]) {
  BIA_MainHeader hdr;
  FILE *fp;
  int n;
  hdr.idstring=557926722;
  hdr.version=1;
  hdr.size=24;
  hdr.numofchunks=0;
  hdr.ctime=time(NULL);
  hdr.mtime=time(NULL);
  hdr.checksum=0;
  fp = fopen(filename,"wb");
  if (fp==NULL) return 1;
  n = fwrite(&hdr,sizeof(BIA_MainHeader),1,fp);
  if (n!=1) return 1;
  fclose(fp);
  return 0;
}


int BIA_MainHeaderUpdate(char filename[]) {
  BIA_MainHeader hdr;
  int n;
  n = BIA_MainHeaderRead(filename,&hdr);
  if (n!=0) return 1;
  hdr.version=1;
  hdr.size=BIA_GetFileSize(filename);
  // count chunks
  int count=0,curr_pos=0;
  curr_pos = sizeof(BIA_MainHeader);
  while (curr_pos<hdr.size-1) {
    uint size;
    if (BIA_MemcpyFromFile((char*)&size,filename,curr_pos,sizeof(uint))==0)
      count++;
    curr_pos+=size;
  }
  hdr.numofchunks=count;
  hdr.mtime=time(NULL);
  hdr.checksum=0;
  FILE *fp;
  fp = fopen(filename,"r+b");
  if (fp==NULL) return 1;
  n = fwrite(&hdr,sizeof(BIA_MainHeader),1,fp);
  fclose(fp);
  if (n!=1) return 1;
  return 0;
}


int BIA_MainHeaderRead(char filename[], BIA_MainHeader *hdr) {
  FILE *fp;
  int n;
  fp = fopen(filename,"rb");
  if (fp==NULL) return 1;
  n = fread(hdr,sizeof(BIA_MainHeader),1,fp);
  fclose(fp);
  if (n!=1) return 1;
  return 0;
}



//############################################################
//############### CHUNK FUNCTIONS ############################
//############################################################


int BIA_ChunkAppend(char filename[], uint size, uint type, uint version, uint compression, char *data)  {
  int n;
  BIA_ChunkHeader hdr;
  hdr.size=size+sizeof(BIA_ChunkHeader);
  hdr.type=type;
  hdr.version=version;
  hdr.compression=compression; // no compression yet, has to be 0
  hdr.checksum=0; // no integrity check yet
  FILE *fp = fopen(filename,"ab");
  if (fp==NULL) return 1;
  n=fwrite(&hdr,sizeof(BIA_ChunkHeader),1,fp);
  if (n!=1)  { fclose(fp); return 1; }
  n=fwrite(data,size,1,fp);
  if (n!=1)  { fclose(fp); return 1; }
  fclose(fp);
  return 0;
}


int BIA_ChunkHeaderRead(char filename[], uint file_offset, BIA_ChunkHeader *hdr) {
  FILE *fp;
  int n;
  fp = fopen(filename,"rb");
  if (fp==NULL) return 1;
  fseek(fp,file_offset,SEEK_SET);
  n = fread(hdr,sizeof(BIA_ChunkHeader),1,fp);
  if (n!=1) { fclose(fp); return 1; }
  fclose(fp);
  return 0;
}




// return 1 if file not found
// return 2 if type not found
int BIA_FindChunkByType(char filename[], int type, BIA_ChunkHeader *hdr, char **data) 
{
  BIA_ChunkHeader hdr2;  
  int curr_pos;
  int filesize=BIA_GetFileSize(filename);
  int found = 0;

  if (filesize<0) return 1;

  curr_pos = sizeof(BIA_MainHeader);
  while (!found && curr_pos<filesize-1) {
    if (BIA_ChunkHeaderRead(filename, curr_pos, &hdr2)) return 1;

    if (hdr2.type==type) {
      if (BIA_ChunkRead(filename, curr_pos, hdr, data)) return 1;
      return 0;
    }
    curr_pos+=hdr2.size;
  }
  return 2;
}  







//############################################################
//############## VOLUME FUNCTIONS ############################
//############################################################


int BIA_t1_v1_WriteScene(Scene *scn, char *filename, int oriented) {
  char *data;
  //int nbytes;
  BIA_Chunk_t1_v1_hdr *hdr;
  ushort *imgdata;
  uint n = scn->xsize*scn->ysize*scn->zsize; 
  //int Imax = MaximumValue3(scn);
  data = (char *) malloc(sizeof(BIA_Chunk_t1_v1_hdr)+(n*2)); // 16 bits only
  hdr=(BIA_Chunk_t1_v1_hdr *) data;
  imgdata=(ushort *) (data+sizeof(BIA_Chunk_t1_v1_hdr));
  hdr->xsize=scn->xsize;
  hdr->ysize=scn->ysize;
  hdr->zsize=scn->zsize;
  hdr->bpv=2;
  hdr->dx=scn->dx;
  hdr->dy=scn->dy;
  hdr->dz=scn->dz;
  hdr->orientation=oriented;
  hdr->compression=0;
  hdr->datasize=n*2;
  hdr->optsize=0;
  int i;
  for (i=0;i<n;i++) 
    imgdata[i]=(ushort) scn->data[i];
  BIA_ChunkAppend(filename, sizeof(BIA_Chunk_t1_v1_hdr)+(n*2), 1, 1, 0, data);
  free(data);
  return 0;
}

int BIA_t1_v1_ReadScene(Scene **scn, char *filename, int *oriented) {
  printf("BIA_t1_v1_ReadScene\n");
  BIA_ChunkHeader chdr;
  char *data;
  BIA_ChunkRead(filename,sizeof(BIA_MainHeader),&chdr,&data);
  if (chdr.type!=1 || chdr.version!=1) return 1;
  BIA_Chunk_t1_v1_hdr *hdr;
  ushort *imgdata;
  hdr=(BIA_Chunk_t1_v1_hdr *) data;
  imgdata=(ushort *) (data+sizeof(BIA_Chunk_t1_v1_hdr));
  *scn = CreateScene(hdr->xsize,hdr->ysize,hdr->zsize);
  (*scn)->dx=hdr->dx;
  (*scn)->dy=hdr->dy;
  (*scn)->dz=hdr->dz;
  if (hdr->orientation==1) (*oriented)=1;
  else (*oriented)=0;
  int i,n;
  n = hdr->xsize*hdr->ysize*hdr->zsize; 
  for (i=0;i<n;i++) 
    (*scn)->data[i]= imgdata[i];
  free(data);
  return 0;
}

int BIA_t1_v2_WriteScene(Scene *scn, char *filename, int oriented, int aligned, int modality) {
  char *data;
  //int nbytes;
  BIA_Chunk_t1_v2_hdr *hdr;
  ushort *imgdata;
  uint n = scn->xsize*scn->ysize*scn->zsize; 
  //int Imax = MaximumValue3(scn);
  data = (char *) malloc(sizeof(BIA_Chunk_t1_v2_hdr)+(n*2)); // 16 bits only
  hdr=(BIA_Chunk_t1_v2_hdr *) data;
  imgdata=(ushort *) (data+sizeof(BIA_Chunk_t1_v2_hdr));
  hdr->xsize=scn->xsize;
  hdr->ysize=scn->ysize;
  hdr->zsize=scn->zsize;
  hdr->bpv=2;
  //hdr->version = 2;
  hdr->dx=scn->dx;
  hdr->dy=scn->dy;
  hdr->dz=scn->dz;
  hdr->orientation=oriented;
  hdr->aligned=aligned;
  hdr->modality=modality;
  hdr->compression=0;
  hdr->datasize=n*2;
  hdr->optsize=0;
  int i;
  for (i=0;i<n;i++) 
    imgdata[i]=(ushort) scn->data[i];
  BIA_ChunkAppend(filename, sizeof(BIA_Chunk_t1_v2_hdr)+(n*2), 1, 2, 0, data);
  free(data);
  return 0;
}

int BIA_t1_v2_ReadScene(Scene **scn, char *filename, int *oriented, int *aligned, int *modality) {
  printf("BIA_t1_v2_ReadScene\n");
  BIA_ChunkHeader chdr;
  char *data;
  BIA_ChunkRead(filename,sizeof(BIA_MainHeader),&chdr,&data);
  if (chdr.type!=1 || chdr.version!=2) return 1;
  BIA_Chunk_t1_v2_hdr *hdr;
  ushort *imgdata;
  hdr=(BIA_Chunk_t1_v2_hdr *) data;
  imgdata=(ushort *) (data+sizeof(BIA_Chunk_t1_v2_hdr));
  *scn = CreateScene(hdr->xsize,hdr->ysize,hdr->zsize);
  (*scn)->dx=hdr->dx;
  (*scn)->dy=hdr->dy;
  (*scn)->dz=hdr->dz;
  if (hdr->orientation==1) (*oriented)=1;
  else (*oriented)=0;
  if (hdr->aligned==1) (*aligned)=1;
  else (*aligned)=0;
  *modality = hdr->modality;
  int i,n;
  n = hdr->xsize*hdr->ysize*hdr->zsize; 
  for (i=0;i<n;i++) 
    (*scn)->data[i]= imgdata[i];
  free(data);
  return 0;
}

int BIA_t1_v3_WriteScene(Scene *scn, char *filename, int *scn_flags) {
  char *data;
  //int nbytes;
  BIA_Chunk_t1_v3_hdr *hdr;
  ushort *imgdata;
  uint n = scn->xsize*scn->ysize*scn->zsize; 
  //int Imax = MaximumValue3(scn);
  data = (char *) malloc(sizeof(BIA_Chunk_t1_v3_hdr)+(n*2)); // 16 bits only
  hdr=(BIA_Chunk_t1_v3_hdr *) data;
  imgdata=(ushort *) (data+sizeof(BIA_Chunk_t1_v3_hdr));
  hdr->xsize=scn->xsize;
  hdr->ysize=scn->ysize;
  hdr->zsize=scn->zsize;
  hdr->bpv=2;
  //hdr->version = 2;
  hdr->dx=scn->dx;
  hdr->dy=scn->dy;
  hdr->dz=scn->dz;
  hdr->orientation = scn_flags[ 0 ];
  hdr->aligned = scn_flags[ 1 ];
  hdr->modality = scn_flags[ 2 ];
  hdr->corrected = scn_flags[ 3 ];
  hdr->compression=0;
  hdr->datasize=n*2;
  hdr->optsize=0;
  int i;
  for (i=0;i<n;i++) 
    imgdata[i]=(ushort) scn->data[i];
  BIA_ChunkAppend(filename, sizeof(BIA_Chunk_t1_v3_hdr)+(n*2), 1, 3, 0, data);
  free(data);
  return 0;
}

int BIA_t1_v3_ReadScene(Scene **scn, char *filename, int *scn_flags) {
  printf("BIA_t1_v3_ReadScene\n");
  BIA_ChunkHeader chdr;
  char *data;
  BIA_ChunkRead(filename,sizeof(BIA_MainHeader),&chdr,&data);
  if (chdr.type!=1 || chdr.version!=3)
    return 1;
  BIA_Chunk_t1_v3_hdr *hdr;
  ushort *imgdata;
  hdr=(BIA_Chunk_t1_v3_hdr *) data;
  imgdata=(ushort *) (data+sizeof(BIA_Chunk_t1_v3_hdr));
  *scn = CreateScene(hdr->xsize,hdr->ysize,hdr->zsize);
  (*scn)->dx=hdr->dx;
  (*scn)->dy=hdr->dy;
  (*scn)->dz=hdr->dz;
  scn_flags[ 0 ] = hdr->orientation;
  scn_flags[ 1 ] = hdr->aligned;
  scn_flags[ 2 ] = hdr->modality;
  scn_flags[ 3 ] = hdr->corrected;
  int i,n;
  n = hdr->xsize*hdr->ysize*hdr->zsize; 
  for (i=0;i<n;i++) 
    (*scn)->data[i]= imgdata[i];
  free(data);
  return 0;
}



//############################################################
//############### LABEL FUNCTIONS ############################
//############################################################


int BIA_t3_v1_WriteSegmObjs(char *filename, ArrayList *segmobjs) 
{

  int n = segmobjs->n;
  SegmObject *obj=NULL;
  BIA_SegmObjs_t3_v1_hdr hdr;
  uint *numoflabels;
  char *data;
  int curr_pos=0;

  if (n==0) {
    data = (char *) malloc(4);
    numoflabels = (uint *)data;
    *numoflabels = n;
    curr_pos+=sizeof(uint);
  }
  if (n>0) {
    obj = (SegmObject *)GetArrayListElement(segmobjs, 0); // just to get VN
    data = (char *) malloc(4+(sizeof(BIA_SegmObjs_t3_v1_hdr)+obj->mask->VN)*n);
    numoflabels = (uint *)data;
    *numoflabels = n;
    curr_pos+=sizeof(uint);
    int i;
    for (i=0; i < n; i++) {
      obj = (SegmObject *)GetArrayListElement(segmobjs, i);
      hdr.color = obj->color;
      hdr.alpha = obj->alpha;
      hdr.visibility = obj->visibility;
      strcpy(hdr.labelname, obj->name);
      hdr.bmap_n = obj->mask->N;
      hdr.bmap_vn = obj->mask->VN;
      memcpy(data+curr_pos,&hdr,sizeof(BIA_SegmObjs_t3_v1_hdr));
      curr_pos+=sizeof(BIA_SegmObjs_t3_v1_hdr);
      memcpy(data+curr_pos,obj->mask->data,sizeof(char)*hdr.bmap_vn);
      curr_pos+=hdr.bmap_vn;
    }
    if (!BIA_ChunkAppend(filename, curr_pos, 3, 1, 0, data)) return 1;
    free(data);    
  }

  return 0;
}


// return 1 if filenot found
// retutn 2 if segmobjs not found
int BIA_t3_v1_ReadSegmObjs(char *filename, ArrayList **segmobjs) 
{
  char *data;
  BIA_ChunkHeader chunk_hdr;
  uint numoflabels;
  int curr_pos=0;

  int n;
  n = BIA_FindChunkByType(filename,3,&chunk_hdr,&data);
  if (n==1) return 1;
  if (n==2) return 2;
  
  memcpy(&numoflabels,data,sizeof(uint));
  curr_pos+=sizeof(uint);
  
  //*segmobjs = (SegmObject *) CreateArrayList(32);

  int counter=0;
  BIA_SegmObjs_t3_v1_hdr hdr;
  while (counter<numoflabels && curr_pos<(chunk_hdr.size-sizeof(BIA_ChunkHeader)-4)) {
    memcpy(&hdr,data+curr_pos,sizeof(BIA_SegmObjs_t3_v1_hdr));
    curr_pos+=sizeof(BIA_SegmObjs_t3_v1_hdr);
    SegmObject *obj = CreateSegmObject(hdr.labelname,hdr.color);
    obj->alpha=hdr.alpha;
    obj->visibility = hdr.visibility;
    obj->mask = BMapNew(hdr.bmap_n);
    memcpy(obj->mask->data,data+curr_pos,hdr.bmap_vn);
    curr_pos+=hdr.bmap_vn;
    AddArrayListElement(*segmobjs, (void *)obj);
  }
  free(data);
  return 0;
}



//############################################################
//############### LABEL FUNCTIONS ############################
//############################################################


int BIA_t4_v1_WriteMeasures(char *filename, Measures *M) 
{

  if (M==NULL || M->n==0) return 0;
    
  int size=sizeof(M->n);
  int i;
  for (i=0;i<M->n;i++) {
    size += sizeof(short int);
    size += strlen(M->measure[i].name);
    size += sizeof(float);
  }

  char *data = (char *) malloc(size);
  int curr_pos=0;
  // store n
  memcpy(data+curr_pos,&M->n,sizeof(int));
  curr_pos += sizeof(int);
  // store the measures
  short int len;
  for (i=0;i<M->n;i++) {
    len = strlen(M->measure[i].name);
    // save lenght of the name
    memcpy(data+curr_pos,&len,sizeof(short int));
    curr_pos += sizeof(short int);
    // save name
    memcpy(data+curr_pos,M->measure[i].name,len);
    curr_pos += len;
    // save value
    memcpy(data+curr_pos,&M->measure[i].value,sizeof(float));
    curr_pos += sizeof(float);
  }
  if (BIA_ChunkAppend(filename, curr_pos, 4, 1, 0, data)!=0) { 
    free(data); 
    return 1; 
  }
  free(data);
  return 0;
}






// return 1 if filenot found
int BIA_t4_v1_ReadMeasures(char *filename, Measures **M) 
{
  char *data;
  BIA_ChunkHeader chunk_hdr;
  int x = BIA_FindChunkByType(filename,4,&chunk_hdr,&data);
  if (x!=0) return 1;

  int curr_pos=0;
  if (*M!=NULL) DestroyMeasures(M);
  *M = CreateMeasures();
  int N;
  memcpy(&N,data+curr_pos,sizeof(int)); 
  curr_pos += sizeof(int);
  char name[30];
  short int len;
  int i;
  float value;
  for (i=0;i<N;i++) {
    memcpy(&len,data+curr_pos,sizeof(short int)); 
    curr_pos += sizeof(short int);
    memcpy(name,data+curr_pos,len); 
    name[len]='\0';
    curr_pos += len;
    memcpy(&value,data+curr_pos,sizeof(float));  
    curr_pos += sizeof(float);
    AddMeasure(*M,name,value);
  }
  free(data);
  
  return 0;
}

//############################################################
//################# MSP FUNCTIONS ############################
//############################################################


int BIA_t5_v1_WriteMSP(char *filename, Plane *MSP) {
  if (MSP==NULL) return 0;
    
  int size = sizeof(Plane);
  char *data = (char *) malloc(size);
  
  memcpy(data,MSP,sizeof(Plane));
  if (BIA_ChunkAppend(filename, size, 5, 1, 0, data)!=0) { 
    free(data); 
    return 1; 
  }
  free(data);
  return 0;
}

int BIA_t5_v1_ReadMSP(char *filename, Plane **MSP) {
  char *data;
  BIA_ChunkHeader chunk_hdr;
  if( BIA_FindChunkByType( filename,5,&chunk_hdr,&data ) != 0 )
    return 1;

  if (*MSP != NULL) DestroyPlane(MSP);
  *MSP = (Plane *) calloc(1,sizeof(Plane));
  memcpy(*MSP,data,sizeof(Plane)); 
  free(data);
  
  return 0;
}
