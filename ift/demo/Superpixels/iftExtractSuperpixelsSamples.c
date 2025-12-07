#include<ift.h>
#include <sys/stat.h>

typedef struct ift_voxel_bounding_box
{
	iftVoxel min;
	iftVoxel max;
} iftVoxelBoundingBox;

iftVoxelBoundingBox* iftExtractSuperpixels(iftImage* superpixels) {
	
	float max = iftMaximumValue(superpixels);
	float min = iftMinimumValue(superpixels);

	int n = max-1;//the index will be from [0,max-1)

	if(min!=1)
		iftError("iftExtractSuperpixels", "Invalid superpixels image.");

	iftVoxelBoundingBox* boxes = (iftVoxelBoundingBox*)malloc(sizeof(iftVoxelBoundingBox)*n);
	iftVoxel u;
	int p, i;

	for (i = 0; i < n; ++i)
	{
		boxes[i].min.x = boxes[i].min.y = boxes[i].min.z = INT_MAX;
		boxes[i].max.x = boxes[i].max.y = boxes[i].max.z = 0;	
	}

	for (u.z=0; u.z < superpixels->zsize; u.z++)
    	for (u.y=0; u.y < superpixels->ysize; u.y++)
    		for (u.x=0; u.x < superpixels->xsize; u.x++){
				p = iftGetVoxelIndex(superpixels,u);

				i = superpixels->val[p] - 1;

				if (u.x < boxes[i].min.x) boxes[i].min.x = u.x;
				if (u.y < boxes[i].min.y) boxes[i].min.y = u.y;
				if (u.z < boxes[i].min.z) boxes[i].min.z = u.z;
				if (u.x > boxes[i].max.x) boxes[i].max.x = u.x;
				if (u.y > boxes[i].max.y) boxes[i].max.y = u.y;
				if (u.z > boxes[i].max.z) boxes[i].max.z = u.z;
			}

	return boxes;
}

//TO DO: beautify this, find a way bro, I know you can
void iftFixBoundingBoxLimits(iftVoxelBoundingBox* b, float mx, float my, float mz)
{
	if(b->min.x<0)//translates the squares that are out of the image 
    {
    	b->max.x += -b->min.x;
    	b->min.x = 0;
    }

    if(b->min.y<0)
    {
    	b->max.y += -b->min.y;
    	b->min.y = 0;
    }

    if(b->min.z<0)
    {
    	b->max.z += -b->min.z;
    	b->min.z = 0;
    }

    if(b->max.x>=mx)
    {
    	b->min.x -= mx - b->max.x -1;
    	b->max.x = mx - 1;
    }

    if(b->max.y>=my)
    {
    	b->min.y -= my - b->max.y - 1;
    	b->max.y = my - 1;
    }

    if(b->max.z>=mz)
    {
    	b->min.z -= mz - b->max.z - 1;
    	b->max.z = mz - 1;
    }
}

//fix the bounding boxes that are not inside the image
void iftAdjustBoundingBox(iftVoxelBoundingBox* b, float ax, float ay, float az)
{
	if(ax>0)
	{
		b->min.x-= floor(ax/2);
		b->max.x+= ceil(ax/2);
	}

	if(ay>0)
	{
		b->min.y-= floor(ay/2);
		b->max.y+= ceil(ay/2);
	}

	if(az>0)
	{
		b->min.z-= floor(az/2);
		b->max.z+= ceil(az/2);
	}
}

/**
** Makes the bounding box a bounding cube, according to the bigger dimension
**/
void iftCubeBoundingBox(iftVoxelBoundingBox* b) {
	float x, y, z;
	
	x = b->max.x - b->min.x;
	y = b->max.y - b->min.y;
	z = b->max.z - b->min.z;

	float max = iftMax(x, iftMax(y, z));

	if(z==0)//just to ignore Z in 2D images
		z = max;

	iftAdjustBoundingBox(b, max-x, max-y, max-z);
}

void iftContextBoundingBox(iftVoxelBoundingBox* b, float perc) {
	float x, y, z;
	
	x = b->max.x - b->min.x;
	y = b->max.y - b->min.y;
	z = b->max.z - b->min.z;

	float inc = iftMax(x, iftMax(y, z)) * perc;

	iftAdjustBoundingBox(b, inc, inc, z>0.0? inc : 0.0);
}

iftImage* iftSuperpixelMask(iftImage* superpixels, iftVoxelBoundingBox b, int id)
{
    float x, y, z;
    
    iftVoxel mv;//mask voxel
    iftVoxel sv;//superpixel voxel

    int mi;//mask index
    int si;//superpixel index

    int i;

    x = b.max.x - b.min.x + 1;
    y = b.max.y - b.min.y + 1;
    z = b.max.z - b.min.z + 1;

    // printf("\n>> %d [(%d,%d,%d),(%d,%d,%d)]\n", id, b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z);

    iftImage* maskImg = iftCreateImage(x,y,z);

    for(sv.z = b.min.z, mv.z = 0; mv.z < maskImg->zsize; ++sv.z, ++mv.z)
    {
        for (sv.y = b.min.y, mv.y = 0; mv.y < maskImg->ysize; ++sv.y, ++mv.y)
        {
            for(sv.x = b.min.x, mv.x = 0; mv.x < maskImg->xsize; ++sv.x, ++mv.x)
            {
                mi = iftGetVoxelIndex(maskImg, mv);
                si = iftGetVoxelIndex(superpixels, sv);

                // printf("%d (%d,%d,%d)\t", superpixels->val[si], sv.x, sv.y, sv.z);
                maskImg->val[mi] = (superpixels->val[si] == id)*255;
            }
            // printf("\n");
        }
    }

    return maskImg;
}

int* iftLabelsFromFile(char* filename, int nsamples)
{
    FILE* f = fopen(filename, "rt");
    int* labels = iftAllocIntArray(nsamples);
    int i=0;
    while(fscanf(f,"%d", &labels[i++])!=EOF);
    fclose(f);
    return labels;
}

int iftMaxArray(int* array, int n)
{
    //printf("Max array\n");
    int m = array[0];
    int i;
    for (i = 0; i < n; ++i){
        //printf("Hue %d\n", i);
        m = iftMax(m, array[i]);
    }
    return m;
}

int iftMinArray(int* array, int n)
{
    int m = array[0];
    int i;
    for (i = 0; i < n; ++i){
        m = iftMin(m, array[i]);
    }
    return m;
}

float iftMeanArray(int* array, int n)
{
    int i;
    float m = 0.0;
    for (i = 0; i < n; ++i){
        m += array[i]/n;
    }
    return m;
}

float iftMaxFArray(float* array, int n)
{
    // printf("Max array\n");
    float m = array[0];
    int i;
    for (i = 0; i < n; ++i){
        // printf("Hue %d\n", i);
        // printf("%f \n", array[i]);
        m = iftMax(m, array[i]);
    }
    return m;
}

float iftMinFArray(float* array, int n)
{
    float m = array[0];
    int i;
    for (i = 0; i < n; ++i){
        m = iftMin(m, array[i]);
    }
    return m;
}

float iftMeanFArray(float* array, int n)
{
    int i;
    float m = 0.0;
    for (i = 0; i < n; ++i){
        m += array[i]/n;
    }
    return m;
}


int main(int argc, char *argv[])
{
	if (argc<6)
     		iftError("Usage: iftExtractSuperpixels <image.ppm> <superpixels.pgm> <labels.txt> <context perc> <size> <output path>","main");
	
    float contextPerc;
    float size;

    char output[1000];
    iftImage* image;
    iftImage* superpixels;
    int* labels;
    int i;


    printf("Opening Image ...\n", argv[1]);

    image = iftReadImageP6(argv[1]);
    superpixels = iftReadImageP2(argv[2]);

	if(image==NULL || superpixels==NULL)
	{
		printf("Invalid image.\n");
		return 1;
	}

    sscanf(argv[4], "%f", &contextPerc);
    sscanf(argv[5], "%f", &size);
    strcpy(output, argv[6]);

    int nsamples = iftMaximumValue(superpixels)-1;
    iftVoxelBoundingBox* boxes = iftExtractSuperpixels(superpixels);

    int nlabels;
    labels = iftLabelsFromFile(argv[3], nsamples);
    nlabels = iftMaxArray(labels, nsamples);

    char sample_filename[1000], mask_filename[1000];

    sprintf(sample_filename, "%s/%s", output, "orig");
    sprintf(mask_filename, "%s/%s", output, "label");

    printf("Creating output directories ...\n");
    mkdir(sample_filename, 0700);
    mkdir(mask_filename, 0700);    

    printf("Extracting samples ...\n\n");

    float *xSize, *ySize;

    xSize = iftAllocFloatArray(nsamples);
    ySize = iftAllocFloatArray(nsamples);

    printf("0.0%%");fflush(stdout);
    for (i = 0; i < nsamples; ++i)
    {
	printf("]\n\033[F\033[J");
        printf("===> %04.02f%%", (100.0*(i+1))/nsamples);fflush(stdout);
        iftCubeBoundingBox(&boxes[i]);
        iftContextBoundingBox(&boxes[i], contextPerc);
        iftFixBoundingBoxLimits(&boxes[i], image->xsize, image->ysize, image->zsize);

        xSize[i] = boxes[i].max.x - boxes[i].min.x;
        ySize[i] = boxes[i].max.y - boxes[i].min.y;

        iftBoundingBox bb = {.begin=boxes[i].min, .end=boxes[i].max};
        iftImage* im = iftExtractROI(image, bb);
        sprintf(sample_filename, "%s/orig/%06d_%09d.ppm", output, labels[i], i);
        sprintf(mask_filename, "%s/label/%06d_%09d.pgm", output, labels[i], i);

        iftImage* interpIm = iftInterp2D(im, size/im->xsize, size/im->ysize);

        iftImage* mask = iftSuperpixelMask(superpixels, boxes[i], i+1);
        iftImage* interpMask = iftInterp2D(im, size/im->xsize, size/im->ysize);
        iftImage* binMask = iftThreshold(im, 127, 255, 255);

        // if(xSize[i]>1000)
        // {
        //     printf("%s\n", sample_filename);
        //     getchar();
        // }


        // store images
        iftWriteImageP6(interpIm, sample_filename);
        iftWriteImageP2(interpMask, mask_filename);
        
        //destroy images
        iftDestroyImage(&im);
        iftDestroyImage(&interpIm);

        iftDestroyImage(&mask);
        iftDestroyImage(&interpMask);
        iftDestroyImage(&binMask);
    }

    printf("\n");

    printf("Superpixel Mean Size = (%f,%f)\n", iftMeanFArray(xSize, nsamples), iftMeanFArray(ySize, nsamples));
    printf("Superpixel Max Size = (%f,%f)\n", iftMaxFArray(xSize, nsamples), iftMaxFArray(ySize, nsamples));
    printf("Superpixel Min Size = (%f,%f)\n", iftMinFArray(xSize, nsamples), iftMinFArray(ySize, nsamples));

	return 0;
}

