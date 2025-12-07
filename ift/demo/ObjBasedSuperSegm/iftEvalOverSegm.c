#include <ift.h>

/**********************************************************************
                            DEFINITIONS
**********************************************************************/
#define PROG_DESC \
        "Evaluates the over-segmented result from a algorithm considering the most "\
        " classical metrics, such as Boundary Recall and Under-Segmentation error.\n"\
        "It assumes that the inputs are label images if no border flag is set."
#define IMG_DESC \
        "Input segmented image (label or border)."
#define GT_NAME_DESC \
        "Input ground-truth name sans ext"
#define GT_DIR_DESC \
        "Input ground-truth directory"
#define GT_FLAG_DESC \
        "Flag indicating whether the ground-truth is a border image."
#define IMG_FLAG_DESC \
        "Flag indicating whether the image is a border image."
#define CSV_FLAG_DESC \
        "Flag indicating whether the output is printed in a CSV format"

/**********************************************************************
                              METHODS
**********************************************************************/
iftDict *iftGetArgs
(int argc, const char *argv[]);

void iftGetAndValReqArgs
(  iftDict *args, char **img_path, char **gt_path, char **gt_name);

void iftGetAndValOptArgs
(  iftDict *args, bool *gt_border, bool *img_border, bool *csv_flag);

void iftObtainSegms
(char* path, bool is_border, iftImage** border, iftImage** label);

/**********************************************************************
										MAIN
**********************************************************************/
int main(int argc, const char *argv[]) 
{
  bool csv_flag, is_gt_border, is_img_border;
  int nSp;
  float br, ue;
  char *img_path, *gt_path, *gt_name;
  char regex[512];
	iftDict *args;
	iftImage  *img_border, *img_label;
  iftDir* gt_dir;
	
	args = iftGetArgs(argc, argv);
	iftGetAndValReqArgs(args, &img_path, &gt_path, &gt_name);
	iftGetAndValOptArgs(args, &is_gt_border, &is_img_border, &csv_flag);

  iftObtainSegms(img_path, is_img_border, &img_border, &img_label);

  if(!is_img_border && iftMinimumValue(img_label) == 0)
  {
    iftAddScalarInPlace(img_label, 1);
  }

  nSp = iftMaximumValue(img_label) - iftMinimumValue(img_label) + 1;
  
  br = 0.0; ue = 0.0;
  
  sprintf(regex,"%s*",gt_name);
  gt_dir = iftLoadFilesFromDirByRegex(gt_path, regex);
  for(int i = 0; i < gt_dir->nfiles; i++)
  {
    iftImage *gt_border, *gt_label;
    
    
    iftObtainSegms(gt_dir->files[i]->path, is_gt_border, &gt_border, &gt_label);    
    
    br += iftBoundaryRecall(gt_border, img_border, 2.0);
    ue += iftUnderSegmentation(gt_label, img_label);
  
    iftDestroyImage(&gt_border);
    iftDestroyImage(&gt_label);
  }
  
  br /= gt_dir->nfiles;
  ue /= gt_dir->nfiles;

  // Print metrics
	if( !csv_flag )
	{
    printf("Number of Superpixels = %d\n", nSp);
  	printf("Boundary Recall = %f \n", br);
	  printf("Under-segmentation error = %f \n", ue);
	}
	else printf("%d,%f,%f\n", nSp, br, ue);
  
  // Free
  iftDestroyImage(&img_border);
  iftDestroyImage(&img_label);
  iftDestroyDict(&args);
  iftDestroyDir(&gt_dir);
  iftFree(img_path);
  iftFree(gt_path);
  iftFree(gt_name);

  return(0);
}

/**********************************************************************
										METHODS
**********************************************************************/
iftDict *iftGetArgs
(int argc, const char *argv[])
{
	int n_opts;
	iftCmdLineParser *parser;
	iftDict *args;

	char descr[2048] = PROG_DESC;
	iftCmdLineOpt cmd[] = {
		{.short_name = "-i", .long_name = "--input-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
			.required = true, .help = IMG_DESC},
		{.short_name = "-gd", .long_name = "--gt-dir", .has_arg = true, .arg_type = IFT_STR_TYPE,
			.required = true, .help = GT_DIR_DESC},
    {.short_name = "-gn", .long_name = "--gt-name", .has_arg = true, .arg_type = IFT_STR_TYPE,
      .required = true, .help = GT_NAME_DESC},
		{.short_name = "-gb", .long_name = "--gt-border", .has_arg = false,
			.required = false, .help = GT_FLAG_DESC},
		{.short_name = "-ib", .long_name = "--img-border", .has_arg = false,
			.required = false, .help = IMG_FLAG_DESC},
		{.short_name = "-c", .long_name = "--csv-fmt", .has_arg = false, 
			.required = false, .help = CSV_FLAG_DESC}
	};
	n_opts = sizeof(cmd) / sizeof(iftCmdLineOpt);

	// Parser Setup
	parser = iftCreateCmdLineParser(descr, n_opts, cmd);
	args = iftParseCmdLine(argc, argv, parser);

	// Free
	iftDestroyCmdLineParser(&parser);

	return args;
}

void iftGetAndValReqArgs
(  iftDict *args, char **img_path, char **gt_path, char **gt_name)
{
	// Input Image
	// Get
	*img_path	= iftGetStrValFromDict("--input-img", args);

	// Validate
  if (!iftIsImageFile(*img_path)) 
  {
  	iftError("Invalid input image: \"%s\"", "iftGetAndValReqArgs", *img_path);
  } 
  if (!iftFileExists(*img_path))
  {
  	iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *img_path);
  } 
  
	// Get
  *gt_path = iftGetStrValFromDict("--gt-dir", args);

  // Validate
  if (!iftDirExists(*gt_path))
  {
  	iftError("Non-existent dir: \"%s\"", "iftGetAndValReqArgs", *gt_path);
  } 
  
  *gt_name = iftGetStrValFromDict("--gt-name", args);
}

void iftGetAndValOptArgs
(  iftDict *args, bool *gt_border, bool *img_border, bool *csv_flag)
{
	// Is GT a border image?
  // Get
  *gt_border = false;
 
	// Validate
	if( iftDictContainKey("--gt-border", args, NULL)){ *gt_border = true; }
  
  // Is the input a border image?
  // Get
  *img_border = false;
 
	// Validate
	if( iftDictContainKey("--img-border", args, NULL)){ *img_border = true; }
	
	// Print CSV format?
  // Get
  *csv_flag = false;
 
	// Validate
	if( iftDictContainKey("--csv-fmt", args, NULL)){ *csv_flag = true; }
}

void iftObtainSegms
(char* path, bool is_border, iftImage** border, iftImage** label)
{
	iftImage  *tmp;

  if( !is_border ) 
  { 
		*label = iftReadImageByExt(path);
		*border = iftBorderImage(*label,1); 
	}
  else 
  {
		*border = iftReadImageByExt(path);
  	tmp = iftCopyImage(*border);
    *label = iftBorderImageToLabelImage(tmp);

    //Free
    iftDestroyImage(&tmp);
  }
}
