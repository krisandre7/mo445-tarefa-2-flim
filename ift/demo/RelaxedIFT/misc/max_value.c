#include "ift.h"

int main(int argc, char **argv) 
{
	if (argc != 2)
		iftError("<image path>", "Max value - main");

	iftImage * img = iftReadImage(argv[1]);
	printf("Max value: %d\n", iftMaximumValue(img));
	iftDestroyImage(&img);

	return 0;
}
