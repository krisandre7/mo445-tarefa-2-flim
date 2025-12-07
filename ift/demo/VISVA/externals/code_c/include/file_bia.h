


typedef struct {
  uint idstring;  // has to be 557926722 ("BIA!")
  uint version;
  uint size;
  uint numofchunks;
  uint ctime;
  uint mtime;
  uint checksum;
} BIA_MainHeader; // 28 Bytes



typedef struct {
  uint size;
  uint type;
  uint version;
  uint compression;
  uint checksum;
} BIA_ChunkHeader; // 20 bytes


typedef struct {
  uint xsize;
  uint ysize;
  uint zsize;
  uint bpv; //bytes per voxel
  double dx;
  double dy;
  double dz;
  uint orientation;
  uint compression;
  uint datasize;
  uint optsize;
} BIA_Chunk_t1_v1_hdr; // 56 Bytes

typedef struct {
  uint xsize;
  uint ysize;
  uint zsize;
  uint bpv; //bytes per voxel
  double dx;
  double dy;
  double dz;
  uint orientation;
  uint aligned; // Version 2 field!
  uint modality;
  uint compression;
  uint datasize;
  uint optsize;
} BIA_Chunk_t1_v2_hdr; // 64 Bytes

typedef struct {
  uint xsize;
  uint ysize;
  uint zsize;
  uint bpv; //bytes per voxel
  double dx;
  double dy;
  double dz;
  uint orientation;
  uint aligned; // Version 2 field!
  uint corrected; // Version 3 field!
  uint modality;
  uint compression;
  uint datasize;
  uint optsize;
} BIA_Chunk_t1_v3_hdr; // 68 Bytes


typedef struct {
  uint color;
  uint alpha;
  uint visibility;
  char labelname[1024];
  uint bmap_n;
  uint bmap_vn;
} BIA_SegmObjs_t3_v1_hdr;


int BIA_PrintInfo(char filename[]);


int BIA_MainHeaderPrintInfo(char filename[]);
int BIA_MainHeaderCreate(char filename[]);
int BIA_MainHeaderUpdate(char filename[]);
int BIA_MainHeaderRead(char filename[], BIA_MainHeader *hdr);



int BIA_t1_v1_WriteScene(Scene *scn, char *filename, int oriented);
int BIA_t1_v1_ReadScene(Scene **scn, char *filename, int *oriented);
int BIA_t1_v2_WriteScene(Scene *scn, char *filename, int oriented, int aligned, int modality);
int BIA_t1_v2_ReadScene(Scene **scn, char *filename, int *oriented, int *aligned, int *modality);
int BIA_t1_v3_WriteScene(Scene *scn, char *filename, int *scn_flags);
int BIA_t1_v3_ReadScene(Scene **scn, char *filename, int *scn_flags);
int BIA_t3_v1_WriteSegmObjs(char *filename, ArrayList *segmobjs);
int BIA_t3_v1_ReadSegmObjs(char *filename, ArrayList **segmobjs);


int BIA_t4_v1_WriteMeasures(char *filename, Measures *M);
int BIA_t4_v1_ReadMeasures(char *filename, Measures **M);

int BIA_t5_v1_WriteMSP(char *filename, Plane *MSP);
int BIA_t5_v1_ReadMSP(char *filename, Plane **MSP);
