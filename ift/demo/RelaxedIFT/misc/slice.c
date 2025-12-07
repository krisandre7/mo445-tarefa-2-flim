#include "ift.h"

int imageIsNotEmpty (iftImage * img)
{
	int i;
	for (i=0; i < img->n; i++)
	{
		if (img->val[i] != 0)
			return 1;
	}
	return 0;
}

int main(int argc, char **argv) 
{
	if (argc != 3)
		iftError("<3dimage> <number of slices>", "Slices - main");

	char * filename = argv[1];
	int slicenbr = atoi(argv[2]);
	char outputString[64];
	int i;
	iftImage * img = iftReadImage(filename);
	iftImage * output = 0;

	if (slicenbr == 0)
	{
		for (i=0; i< img->zsize; i++)
		{
			output = iftGetXYSlice(img, i);
			if (imageIsNotEmpty(output))
			{
				output = iftNormalize(output, 0, 255);
				sprintf(outputString, "slice%d.pgm", i);
				iftWriteImageP5(output, outputString);
				printf("Saving slice %d\n", i);
			}
		}
	}
	else
	{
		output = iftGetXYSlice(img, slicenbr);
		if (imageIsNotEmpty(output))
		{
			output = iftNormalize(output, 0, 255);
			sprintf(outputString, "slice%d.pgm", slicenbr);
			iftWriteImageP5(output, outputString);
			printf ("%s saved\n", outputString);
		}
		else
		{
			printf("Empty image, choose a different slice\n");
		}
	}
	return 0;
}