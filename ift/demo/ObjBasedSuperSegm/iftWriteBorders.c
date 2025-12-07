#include <ift.h>

/**********************************************************************
                              DEFINITIONS
**********************************************************************/
#define PROG_DESC \
        "Writes the contours of a segmentation result (label or border) on a "\
        "copy of the image given.\nIt is possible to select the thickness "\
        "(by the adjacency relation radius) and the color of the borders."
#define IMG_DESC \
        "Input image"
#define SEGM_DESC \
        "Segmented image"
#define BORD_DESC \
        "Is a border image?"
#define THICK_DESC \
        "Thickness of the borders (x >= 0)"
#define COLOR_RED_DESC \
        "Percentage of RED color (0.0 <= x <= 1.0)"
#define COLOR_BLUE_DESC \
        "Percentage of BLUE color (0.0 <= x <= 1.0)"
#define COLOR_GREEN_DESC \
        "Percentage of GREEN color (0.0 <= x <= 1.0)"
#define OUT_DESC \
        "Output image"
        
/**********************************************************************
                                METHODS
**********************************************************************/
iftDict *iftGetArgs(int argc, const char *argv[]);

void iftGetAndValReqArgs(  iftDict *args, char **img_path, char **segm_path, bool* is_label,  char **out_path);

void iftGetAndValOptArgs(  iftDict *args, double *thick, double *r, double *g, double *b);

/**********************************************************************
																MAIN
**********************************************************************/
int main(int argc, const char *argv[]) 
{
	int normvalue, red, green, blue;
  double thick, r_perc, g_perc, b_perc;
  bool is_label;
  char *img_path, *segm_path, *out_path;
  iftImage  *img, *segm, *overlay;
  iftColor RGB, YCbCr;
  iftAdjRel *A;
  iftDict* args;

  // Obtaining user inputs
  args = iftGetArgs(argc, argv);

  iftGetAndValReqArgs(args, &img_path, &segm_path, &is_label, &out_path);
  iftGetAndValOptArgs(args, &thick, &r_perc, &g_perc, &b_perc);

  img  = iftReadImageByExt(img_path);
  segm = iftReadImageByExt(segm_path);

  normvalue = iftNormalizationValue(iftMaximumValue(img)); 

  A = iftCircular(thick);
  
  iftImage *added = iftAddScalar(segm,1);

  if( is_label ) overlay = iftBorderImage(added,0);
  else {
  	iftBorderImageToLabelImage(added);
    overlay = iftBorderImage(added,0);
  }
  
  iftDestroyImage(&added);

  red = (int)(normvalue * r_perc);
  green = (int)(normvalue * g_perc);
  blue = (int)(normvalue * b_perc);

  RGB = iftRGBColor(red, green, blue);
  YCbCr = iftRGBtoYCbCr(RGB, normvalue);
  
  iftDrawBorders(img,overlay,A,YCbCr,A);

  iftWriteImageByExt(img, out_path);

  // Free
  iftDestroyImage(&img);
  iftDestroyImage(&segm);
  iftDestroyImage(&overlay);
  iftDestroyAdjRel(&A);
  iftDestroyDict(&args);

  return 0;
}

/**********************************************************************
																METHODS
**********************************************************************/

iftDict *iftGetArgs(int argc, const char *argv[]) 
{
	char descr[2048] = PROG_DESC;
	int n_opts;
	iftCmdLineParser *parser;
	iftDict *args;

	iftCmdLineOpt cmd[] = {
		{.short_name = "-i", .long_name = "--input-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
						.required = true, .help = IMG_DESC},
		{.short_name = "-s", .long_name = "--segm-img", .has_arg = true, .arg_type = IFT_STR_TYPE,
						.required = true, .help = SEGM_DESC},
		{.short_name = "-b", .long_name = "--border-img", .has_arg = false,
						.required = false, .help = BORD_DESC},
		{.short_name = "-t", .long_name = "--thickness", .has_arg = true, .arg_type=IFT_DBL_TYPE,
						.required = false, .help = THICK_DESC},
		{.short_name = "-cr", .long_name = "--color-red", .has_arg = true, .arg_type=IFT_DBL_TYPE,
						.required = false, .help = COLOR_RED_DESC},
		{.short_name = "-cg", .long_name = "--color-green", .has_arg = true, .arg_type=IFT_DBL_TYPE,
						.required = false, .help = COLOR_GREEN_DESC},
		{.short_name = "-cb", .long_name = "--color-blue", .has_arg = true, .arg_type=IFT_DBL_TYPE,
						.required = false, .help = COLOR_BLUE_DESC},
		{.short_name = "-o", .long_name = "--out-img", .has_arg = true, .arg_type=IFT_STR_TYPE,
						.required = true, .help = OUT_DESC},
	};
	n_opts = sizeof(cmd) / sizeof (iftCmdLineOpt);

	// Parser Setup
	parser = iftCreateCmdLineParser(descr, n_opts, cmd);
	args = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

	// Free
	iftDestroyCmdLineParser(&parser);

	return args;
}

void iftGetAndValReqArgs(  iftDict *args, char **img_path, char **segm_path, bool* is_label,  char **out_path)
{
	// Input Image
	// Get
	*img_path	= iftGetStrValFromDict("--input-img", args);

	// Validate
  if (!iftFileExists(*img_path)) {
  	iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *img_path);
  } 

  // Segmented Image
	// Get
	*segm_path	= iftGetStrValFromDict("--segm-img", args);

	// Validate
  if (!iftFileExists(*segm_path)) {
  	iftError("Non-existent image: \"%s\"", "iftGetAndValReqArgs", *segm_path);
  } 

  // Is Label?
	// Get
	if(iftDictContainKey("--border-img", args, NULL)) *is_label = false;
	else *is_label = true;

  // Output Image
	// Get
  *out_path = iftGetStrValFromDict("--out-img", args);

  // Validate
  if (!iftIsImagePathnameValid(*out_path))  {
		iftError("Invalid output image: \"%s\"", "iftGetAndValReqArgs", *out_path);
	}
}

void iftGetAndValOptArgs(  iftDict *args, double *thick, double *r, double *g, double *b)
{
	// Border thickness
  // Get
  if( iftDictContainKey("--thickness", args, NULL)) {
  	*thick = iftGetDblValFromDict("--thickness", args);	

  	// Validate
	  if( *thick < 0 ) {
	  	iftError("Negative thickness: \"%d\"", "iftGetAndValOptArgs", *thick);
	  }
  }
  else *thick = 1.0;
  
  // Color red
  // Get
  if( iftDictContainKey("--color-red", args, NULL)) {
  	*r = iftGetDblValFromDict("--color-red", args );
  	
  	// Validate
  	if( *r < 0.0 || *r > 1.0 ) {
	  	iftError("Negative red color value: \"%d\"", "iftGetAndValOptArgs", *g);
	  }
  }
	else *r = 0.0; // Black

	// Color green
  // Get
  if( iftDictContainKey("--color-green", args, NULL)) {
  	*g = iftGetDblValFromDict("--color-green", args );
  	
  	// Validate
  	if( *g < 0.0 || *g > 1.0 ) {
	  	iftError("Negative green color value: \"%d\"", "iftGetAndValOptArgs", *g);
	  }
  }
	else *g = 0.0; // Black

	// Color blue
  // Get
  if( iftDictContainKey("--color-blue", args, NULL)) {
  	*b = iftGetDblValFromDict("--color-blue", args );
  	
  	// Validate
  	if( *b < 0.0 || *b > 1.0 ) {
	  	iftError("Negative blue color value: \"%d\"", "iftGetAndValOptArgs", *b);
	  }
  }
	else *b = 0.0; // Black
}
