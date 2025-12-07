#include "ift.h"


int main(int argc, char *argv[]) {

    //argv[1] = dataset
    //argv[2] = filen name output

    char *extension = ".data";
    char *filename ;
    int fileNameSize = strlen(argv[2]);
    int extensionNameSize = strlen(extension);
    filename = malloc(fileNameSize+extensionNameSize+1);

    if(filename != NULL){
        filename[0] = '\0';   // ensures the memory is an empty string
        strcat(filename,argv[2]);
        strcat(filename,extension);
    }else{
        printf("malloc failed!\n");
        return -1;
    }


    iftDataSet *Zinput = iftReadOPFDataSet(argv[1]);
    FILE *fp = fopen(filename, "w+");
    fputs("DY\n", fp);
    fprintf(fp,"%d\n",Zinput->nsamples);
    fprintf(fp,"%d\n",Zinput->nfeats);

    int i,j;
    for(i=0; i < Zinput->nfeats; i++){
        fprintf(fp,"f%d",i);
        if(i != Zinput->nfeats-1){
            fputs(";", fp);
        }
    }

    fputs("\n", fp);
    for(i=0; i < Zinput->nsamples; i++){
        fprintf(fp,"s%d.png;",i);
        for(j=0; j<Zinput->nfeats; j++){
            fprintf(fp,"%f;",Zinput->sample[i].feat[j]);
        }
        fprintf(fp,"%d\n",Zinput->sample[i].truelabel);
    }

    //iftImage *outputImage = iftDraw2DFeatureSpace(Zoutput,CLASS,IFT_TEST);
    //iftWriteImageP6(outputImage,"output4.ppm");

    //iftDestroyDataSet(&Zinput);
    //fclose(fp);


    return 0;
}

