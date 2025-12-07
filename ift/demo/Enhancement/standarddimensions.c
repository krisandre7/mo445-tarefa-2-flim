#include "common.h"
#include "ift.h"

//And here it begins....
int main(int argc, char **argv) 
{
	if (argc != 4)	iftError("Please provide the following parameters:\n<FILES_TO_BE_PROCESSED> x y\n\n", "main");

 	int idxImg,nimages,x,y,i,j;
	FileNames * files = NULL;
	char fullPath[128],*file,*folder;
	
	folder = iftAllocCharArray(256);
	file = argv[1];
	x = atoi(argv[2]);
	y = atoi(argv[3]);

	folder = getFolderName(file);
	nimages = countImages(file);
	files = createFileNames(nimages);
	loadFileNames(files, file);

	for(idxImg=0;idxImg<nimages;idxImg++)
	{
		fullPath[0] = 0;
		strcpy(fullPath, folder);
		strcat(fullPath, files[idxImg].filename);
		iftImage  *img;
		if (strstr(fullPath,".pgm"))
		    img = iftReadImageP5(fullPath);
		else if (strstr(fullPath,".ppm") )
		    img = iftReadImageP6(fullPath);
		else
		    iftError("unexpected extension file image","main");

		fprintf(stdout,"%s: (%dx%d)\n",files[idxImg].filename,img->xsize,img->ysize);

		if ( ( (img->xsize > x) || (img->xsize > y) ) && ( (img->ysize > x) || (img->ysize > y) ) )
		{
			iftImage* imgout = iftCreateImage(x,y,1); // always considering 2d images
			if (iftIsColorImage(img)) {
				iftSetCbCr(imgout,0);
				if ( (img->xsize > x) && (img->ysize > y) ) {
					for(i=0;i<y;i++)
						for(j=0;j<x;j++) {
							imgout->val[i*imgout->xsize+j] = img->val[i*img->xsize+j];
							imgout->Cb [i*imgout->xsize+j] = img->Cb [i*img->xsize+j];
							imgout->Cr [i*imgout->xsize+j] = img->Cr [i*img->xsize+j];
						}
				} else if  ( (img->xsize > y) && (img->ysize > x) ) {
					for(i=0;i<x;i++)
						for(j=0;j<y;j++) {
							imgout->val[j*imgout->xsize+i] = img->val[i*img->xsize+j];
							imgout->Cb [j*imgout->xsize+i] = img->Cb [i*img->xsize+j];
							imgout->Cr [j*imgout->xsize+i] = img->Cr [i*img->xsize+j];
						}

				} else
					iftError("Unexpected image size","main");
			  
			} else {
				if ( (img->xsize > x) && (img->ysize > y) ) {
					for(i=0;i<y;i++)
						for(j=0;j<x;j++)
							imgout->val[i*imgout->xsize+j] = img->val[i*img->xsize+j];
				} else if  ( (img->xsize > y) && (img->ysize > x) ) {
					for(i=0;i<x;i++)
						for(j=0;j<y;j++)
							imgout->val[j*imgout->xsize+i] = img->val[i*img->xsize+j];

				} else
					iftError("Unexpected image size","main");
			}
		
			fullPath[0] = 0;
			strcpy(fullPath, folder);
			strcat(fullPath, files[idxImg].filename);
			if (iftIsColorImage(img)) {
				strcat(fullPath, ".ppm");
				iftWriteImageP6(imgout,fullPath);
			} else {
				strcat(fullPath, ".pgm");
				iftWriteImageP5(imgout,fullPath);
			}
			iftDestroyImage(&imgout);
		} 
		else
			iftError("Unexpected image size","main");

		iftDestroyImage(&img);
	}

	destroyFileNames(files);
	free(folder);

	return 0;
}
