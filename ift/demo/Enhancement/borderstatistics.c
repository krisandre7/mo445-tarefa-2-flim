#include "ift.h"
#include "common.h"

int main(int argc, char **argv) 
{
	// read input files
	if (argc != 3)	iftError("Please provide the following parameters:\n <FILE_IMAGE_NAMES_GT> <RADIUS> \n\n", "main");
	int   nimagesGT;
	char  *file_imageNamesGT,*folder_nameGT;
	char fullPath[256];
	float radius;
	FileNames * filesGT = NULL;
	timer *tic, *toc;

	file_imageNamesGT = argv[1];
	radius            = atof(argv[2]);

	tic = iftTic();

	folder_nameGT = getFolderName(file_imageNamesGT);
	nimagesGT = countImages(file_imageNamesGT);
	filesGT = createFileNames(nimagesGT);
	loadFileNames(filesGT, file_imageNamesGT);

	iftAdjRel* A = iftCircular(radius);
	
	int idxImg,p,q,i;
	int countNB=0,countB=0;
	float minNB=0,minB=0,maxNB=0,maxB=0;
	float SNB=0.0,S2NB=0.0,SB=0.0,S2B=0.0;
	for( idxImg = 0; idxImg < nimagesGT; idxImg++){
		// read image
		fullPath[0] = 0;
		strcpy(fullPath, folder_nameGT);
		strcat(fullPath, filesGT[idxImg].filename);
		iftImage *imgGT = iftReadImageP5(fullPath);

		// counting pixels
		countB = 0;countNB = 0;
		for(p=0;p<imgGT->n;p++) {
			iftVoxel pixel   = iftGetVoxelCoord(imgGT,p);
			if (imgGT->val[p] == 255)
			      countB++;
			else {
				int bBorder=0;
				for(i=1;i<A->n;i++)
				{
					iftVoxel neighbor   = iftGetAdjacentVoxel(A,pixel,i);
					if (iftValidVoxel(imgGT,neighbor))
						q = iftGetVoxelIndex(imgGT,neighbor);
					else
						continue;

					if (imgGT->val[q] == 255) {
						bBorder=1; break;
					}
				}
				if ( bBorder )
					countB++;
				else 
					countNB++;
			}
		}
		if ( (minB  > countB ) || (idxImg == 0) ) minB  = (float)countB;
		if ( (minNB > countNB) || (idxImg == 0) ) minNB = (float)countNB;
		if ( (maxB  < countB ) || (idxImg == 0) ) maxB  = (float)countB;
		if ( (maxNB < countNB) || (idxImg == 0) ) maxNB = (float)countNB;
		SB  += (float)countB ; S2B  += (float)countB  * countB ;
		SNB += (float)countNB; S2NB += (float)countNB * countNB;
		
		fprintf(stdout,"B: %8d (%6.2f%%), NB: %8d (%6.2f%%)\n",countB,100*(float)countB/(countB+countNB),countNB,100*(float)countNB/(countB+countNB));
		
		iftDestroyImage(&imgGT);
	}

	iftDestroyAdjRel(&A);

	float meanB = SB / nimagesGT, meanNB = SNB / nimagesGT;
	float stdB  = sqrt( (S2B  - 2*SB *meanB  + nimagesGT*meanB *meanB ) / (nimagesGT-1) ),
	      stdNB = sqrt( (S2NB - 2*SNB*meanNB + nimagesGT*meanNB*meanNB) / (nimagesGT-1) );
	int n=countB+countNB; /* considering that the size of all images are the same */
	fprintf(stdout,"B:  [%8d,%8d] (%10.2f+-%10.2f) --- ",(int)minB ,(int)maxB ,meanB   ,stdB   );
	fprintf(stdout,"[%6.2f,%6.2f] (%6.2f+-%6.2f)\n"    ,100*minB /n,100*maxB /n,100*meanB /n,100*stdB /n);
	fprintf(stdout,"NB: [%8d,%8d] (%10.2f+-%10.2f) --- ",(int)minNB,(int)maxNB,meanNB  ,stdNB  );
	fprintf(stdout,"[%6.2f,%6.2f] (%6.2f+-%6.2f)\n"    ,100*minNB/n,100*maxNB/n,100*meanNB/n,100*stdNB/n);

	toc = iftToc();
	printf("Finished: %f\n", iftCompTime(tic,toc));

	return 0;
}
