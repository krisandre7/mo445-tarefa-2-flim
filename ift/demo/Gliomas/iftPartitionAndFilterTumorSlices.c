/**
 * @brief Partition 3D image and filtering based on tumor size
 * @author Matheus Cerqueira
 * @date Feb, 2024
 *
 * Image Coordinates
 *     z
 *    /
 *   /
 *  /___________ x
 * |
 * |
 * |
 * y
 *
 *
 */



#include "ift.h"

#define POV 0
#define MAXB16V 65535

iftDict *iftGetArguments(int argc, const char **argv);
void iftGetRequiredArgs( iftDict * args,  char **input_dir_path,  char **output_dir_path, 
                         char **gt_dir_path,  int *stride,  int *min_size,  int *axis);
void iftValidateRequiredArgs(const char *input_dir_path, const char *output_dir_path, 
                        const char *gt_dir_path,   int stride,   int min_size,   int axis);
iftImage *GetPOVSliceAxial(iftImage *img, int z, int neuroradio_pov);
iftImage *GetPOVSliceSagittal(iftImage *img, int x, int neuroradio_pov);
iftImage *GetPOVSliceCoronal(iftImage *img, int y, int neuroradio_pov);



int checkTumorSize(iftImage * slice, int min_size){

  int c_size = 0;
  for(int p=0; p<slice->n; p++){
    if(slice->val[p]>0){
      c_size++;
    }
  }

  if (c_size < min_size){
    return 0;
  }
  return 1;

}


int main(int argc, const char **argv)
{
    timer       *t1=NULL,*t2=NULL;

    /*--------------------------------------------------------*/

    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/

    iftDict  *args       = NULL;
    char *input_dir_path = NULL;
    char *gt_dir_path = NULL;
    char *output_dir_path = NULL;
    int stride,min_size, axis;
    char filename[200], ext[10], *basename;

    args = iftGetArguments(argc, argv);
    iftGetRequiredArgs(args, &input_dir_path, &output_dir_path, &gt_dir_path, &stride, &min_size, &axis);

    char output_dir_path_gt[100];
    char output_dir_path_or[100];
    sprintf(output_dir_path_gt, "%s/%s", output_dir_path, "gt");
    sprintf(output_dir_path_or, "%s/%s", output_dir_path, "orig");


    iftFileSet *fs_orig = iftLoadFileSetFromDirOrCSV(input_dir_path, 1, 1);
    int nimages          = fs_orig->n;

    sprintf(ext, "%s", iftFileExt(fs_orig->files[0]->path));

    // for each image
    for (int i = 0; i < nimages; i++) {
        iftImage * img = iftReadImageByExt(fs_orig->files[i]->path);
        basename = iftFilename(fs_orig->files[i]->path, ext);  

        sprintf(filename, "%s/%s%s", gt_dir_path, basename,ext);
        iftImage * label_img = iftReadImageByExt(filename);

        iftImage * tmp;
        tmp = iftNormalize(img, 0, 255);
        iftDestroyImage(&img);
        img = tmp;


        tmp = iftNormalize(label_img, 0, 255);
        iftDestroyImage(&label_img);
        label_img = tmp;

        int slice_max = img->zsize;
        // getSliceInfo(img, &slice_max, fun_ptr);
        int c_slice=0;
        //para cada slice na orientação pedida
        while(c_slice < slice_max){

            //pede slices
            iftImage *i_slc = GetPOVSliceAxial(img, c_slice, POV);
            iftImage *l_slc = GetPOVSliceAxial(label_img, c_slice, POV);

            tmp = iftAreaOpen(l_slc, 10, NULL);
            iftDestroyImage(&l_slc);
            l_slc = tmp;


            // verifica se satisfaz tamanho mínimo de tumor
            if(checkTumorSize(l_slc, min_size)){

              tmp = iftCloseBin(l_slc, 20);
              iftDestroyImage(&l_slc);
              l_slc = tmp;


              //salva imagens
              sprintf(filename, "%s/%s_a%d_s%04d%s", output_dir_path_or, basename,axis,c_slice,".png");
              iftWriteImageByExt(i_slc, filename);

              sprintf(filename, "%s/%s_a%d_s%04d%s", output_dir_path_gt, basename,axis,c_slice,".png");
              iftWriteImageByExt(l_slc, filename);
              break;
            }

            iftDestroyImage(&i_slc);
            iftDestroyImage(&l_slc);

  
            c_slice +=stride;
        }
        iftDestroyImage(&img);
        iftDestroyImage(&label_img);
        
        break;
    }

   
    iftFree(input_dir_path);
    iftFree(output_dir_path);
    iftDestroyFileSet(&fs_orig);


    t2     = iftToc();
    fprintf(stdout,"Image slices extracted in %f ms\n",iftCompTime(t1,t2));

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return(0);

}

iftDict *iftGetArguments(int argc, const char **argv) {
    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-dir", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input image folder"},
            {.short_name = "-m", .long_name = "--mask-dir", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="GT folder"},
            {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Output folder"},
            {.short_name = "-s", .long_name = "--stride", .has_arg=true, .required=true, .arg_type = IFT_LONG_TYPE, .help="Stride size (int)"},
            {.short_name = "", .long_name = "--min-size", .has_arg=true, .required=true, .arg_type = IFT_LONG_TYPE, .help="Min tumor size in voxels"},
            {.short_name = "-a", .long_name = "--axis", .has_arg=true, .required=true, .arg_type = IFT_LONG_TYPE, .help="Axis {0,1,2}"},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    char program_description[IFT_STR_DEFAULT_SIZE] = "convert 3d images to slices while filtering by tumor size";

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);

    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(iftDict * args,  char **input_dir_path,  char **output_dir_path, 
                         char **gt_dir_path,  int *stride,  int *min_size,  int *axis) {

    *input_dir_path = iftGetStrValFromDict("--input-dir", args);
    *output_dir_path = iftGetStrValFromDict("--output-dir", args);
    *gt_dir_path = iftGetStrValFromDict("--mask-dir", args);
    *stride = (int)iftGetLongValFromDict("--stride", args);
    *min_size = (int)iftGetLongValFromDict("--min-size", args);
    *axis = (int)iftGetLongValFromDict("--axis", args);
    

    iftValidateRequiredArgs(*input_dir_path, *output_dir_path, *gt_dir_path, *stride, *min_size, *axis);

    puts("-----------------------");
    printf("- Input Dir: \"%s\"\n", *input_dir_path);
    printf("- GT Folder: \"%s\"\n", *gt_dir_path);
    printf("- Output Slices Folder: \"%s\"\n", *output_dir_path);
    printf("- Stride: \"%d\"\n", *stride);
    printf("- Min Size: \"%d\"\n", *min_size);
    printf("- Axis: \"%d\"\n", *axis);
    puts("-----------------------");
}

void iftValidateRequiredArgs(const char *input_dir_path, const char *output_dir_path, 
                        const char *gt_dir_path,   int stride,   int min_size,   int axis){

    char filename[200];

    if (!iftDirExists(input_dir_path)) {
        iftError("Invalid Input Image's Pathname: \"%s\"", "iftValidateRequiredArgs", input_dir_path);
    }

    // OUTPUT FOLDER
    if (!iftDirExists(output_dir_path)) {
        printf("Creating the output dir :  %s \n", output_dir_path);
        iftMakeDir(output_dir_path);

        sprintf(filename, "%s/%s",output_dir_path,"orig");
        iftMakeDir(filename);
        sprintf(filename, "%s/%s",output_dir_path,"gt");
        iftMakeDir(filename);
    }else{
      sprintf(filename, "%s/%s",output_dir_path,"orig");
      if(!iftDirExists(filename))
        iftMakeDir(filename);
      
      sprintf(filename, "%s/%s",output_dir_path,"gt");
      if(!iftDirExists(filename))
        iftMakeDir(filename);
    }

    // GT FOLDER
    if (!iftDirExists(gt_dir_path)) {
        iftError("Invalid Mask Image's Pathname: \"%s\"", "iftValidateRequiredArgs", gt_dir_path);
    }

    if (!(stride>0)){
        iftError("Invalid Stride: \"%d\"", "iftValidateRequiredArgs", stride);
    }

    if (!(min_size>0)){
        iftError("Invalid Min tumor size: \"%d\"", "iftValidateRequiredArgs", min_size);
    }

    
    if (!(axis>=0 && axis<=2)){
        iftError("Invalid Axis: \"%d\" it should be {0,1,2}" , "iftValidateRequiredArgs", axis);
    }

}


iftImage *GetPOVSliceAxial(iftImage *img, int z, int neuroradio_pov)
{
  iftImage *slc = iftCreateImage(img->xsize,img->ysize,1);
  iftVoxel  u;
  int       q=0;
  
  if(neuroradio_pov==0){
    u.z = z;
    for (u.y = 0; u.y < img->ysize; u.y++)
      for (u.x = 0; u.x < img->xsize; u.x++){
        int p = iftGetVoxelIndex(img,u);      
        slc->val[q] = img->val[p];
        q++;
      }
  }else{
    u.z = img->zsize - z -1;
    for (u.y = 0; u.y < img->ysize; u.y++)
      for (u.x = img->xsize-1; u.x >=0 ; u.x--){
        int p = iftGetVoxelIndex(img,u);      
        slc->val[q] = img->val[p];
        q++;
      }
  }

  return(slc);
}


iftImage *GetPOVSliceSagittal(iftImage *img, int x, int neuroradio_pov)
{
  iftImage *slc = iftCreateImage(img->ysize,img->zsize,1);
  iftVoxel  u;
  int       q=0;
  
  if(neuroradio_pov==0){
    u.x = img->xsize - x -1;
    for (u.z = img->zsize-1; u.z >= 0; u.z--)
      for (u.y = 0; u.y < img->ysize; u.y++){
        int p = iftGetVoxelIndex(img,u);      
        slc->val[q] = img->val[p];
        q++;
      }
  }else{
    u.x =  x;
    for (u.z = img->zsize-1; u.z >=0; u.z--)
      for (u.y = img->ysize-1; u.y >=0; u.y--){
        int p = iftGetVoxelIndex(img,u);      
        slc->val[q] = img->val[p];
        q++;
      }
  }

  return(slc);
}


iftImage *GetPOVSliceCoronal(iftImage *img, int y, int neuroradio_pov)
{
  iftImage *slc = iftCreateImage(img->xsize,img->zsize,1);
  iftVoxel  u;
  int       q=0;
  
  if(neuroradio_pov==0){
    u.y = y;
    for (u.z = img->zsize-1; u.z >=0; u.z--)
      for (u.x = 0; u.x < img->xsize; u.x++){
        int p = iftGetVoxelIndex(img,u);      
        slc->val[q] = img->val[p];
        q++;
      }
  }else{
    u.y = img->ysize - y-1;
    for (u.z = img->zsize-1; u.z >= 0; u.z--)
      for (u.x = img->xsize-1; u.x >=0 ; u.x--){
        int p = iftGetVoxelIndex(img,u);      
        slc->val[q] = img->val[p];
        q++;
      }
  }

  return(slc);
}
