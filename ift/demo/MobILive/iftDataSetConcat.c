#include <stdio.h>
#include <stdlib.h>

//#define _SILENCE

#include "ift.h"

char *replace_str(const char *str, const char *old, const char *new)
{
	char *ret, *r;
	const char *p, *q;
	size_t oldlen = strlen(old), newlen = strlen(new);
	size_t count, retlen;

	if (oldlen != newlen) {
		for (count = 0, p = str; (q = strstr(p, old)) != NULL; p = q + oldlen)
			count++;
		/* this is undefined if p - str > PTRDIFF_MAX */
		retlen = p - str + strlen(p) + count * (newlen - oldlen);
	} else
		retlen = strlen(str);

	if ((ret = malloc(retlen + 1)) == NULL)
		return NULL;

	for (r = ret, p = str; (q = strstr(p, old)) != NULL; p = q + oldlen) {
		/* this is undefined if q - p > PTRDIFF_MAX */
		ptrdiff_t l = q - p;
		memcpy(r, p, l);
		r += l;
		memcpy(r, new, newlen);
		r += newlen;
	}
    strcpy(r, p);

    return ret;
}

int main(int argc, char* argv[])
{
    iftDataSet** Z_in = NULL,*Z_out = NULL;
    int ds,nds,nfeats,nsamples,nclasses;
    int f,fc,s;
    char tmp[200];

    if ( (argc != 4) ) {
        sprintf(tmp,"usage: %s <num_datasets> <input_OPFDataSets#.data> <output_OPFDataSet_.data>\n",argv[0]);
        iftError(tmp,"iftDataSetConcat");
    }

    nds = atoi(argv[1]);

    /* Initialization */
    if (strrchr(argv[2],'#') == NULL)
        iftError("wildcard # is missing in inputs_OPFDAtaSetst#.data","iftDataSetConcat");


    Z_in = (iftDataSet**)malloc(sizeof(iftDataSet*)*nds);
    if (Z_in == NULL)
        iftError("allocating Z_in vector","iftDataSetConcat");

    char *file,sds[10],fileTmp[200];
    for(ds=0;ds<nds;ds++) {
        sprintf(sds,"%d",ds);
        strcpy(fileTmp,argv[2]);
        file=replace_str(fileTmp,"#",sds);
#ifndef _SILENCE
        printf("loading %s dataset\n",file);
#endif
        Z_in[ds] = iftReadOPFDataSet(file); // Read dataset Z_in
        free(file);
    }
#ifndef _SILENCE
    printf("\n\n");
#endif


    int Ok=1;
    nfeats = 0;nsamples = 0,nclasses = 0;
    for(ds=0;ds<nds;ds++) {
#ifndef _SILENCE
        printf("***** DATASET %d *****\n",ds);
        printf("Total number of samples  %d\n"  ,Z_in[ds]->nsamples);
        printf("Total number of features %d\n"  ,Z_in[ds]->nfeats);
        printf("Total number of classes  %d\n\n",Z_in[ds]->nclasses);
#endif

        iftSetDistanceFunction(Z_in[ds], 1);
        iftSetStatus(Z_in[ds], IFT_TRAIN); // set all elements as IFT_TRAIN

        if (ds == 0) {
            nsamples = Z_in[ds]->nsamples;
            nclasses = Z_in[ds]->nclasses;
        }
        nfeats += Z_in[ds]->nfeats;

        // Each/every datasets has to have the same number of samples
        if ( (Z_in[ds]->nsamples != nsamples) || (Z_in[ds]->nclasses != nclasses) ) {
            iftError("impossible to concatenate datasets with different number of samples and/or classes","iftDataSetConcat");
            Ok=0;
        }
    }

    if (Ok) {
        Z_out = iftCreateDataSet(nsamples,nfeats);
        Z_out->nclasses = nclasses;

        fc=0;
        for(ds=0;ds<nds;ds++) {
            for(s=0;s<nsamples;s++) {
                for(f=0;f<Z_in[ds]->nfeats;f++) {
                    Z_out->sample[s].feat[fc+f] = Z_in[ds]->sample[s].feat[f];
                }
                if (ds == 0) Z_out->sample[s].class = Z_in[ds]->sample[s].class;
                if (Z_in[ds]->sample[s].class != Z_out->sample[s].class) {
                    iftError("impossible to concatenate datasets with samples of diferent classes","iftDataSetConcat");
                    Ok=0;
                }
            }
            fc += Z_in[ds]->nfeats;
        }

#ifndef _SILENCE
        printf("\nsaving %s dataset\n",argv[3]);
#endif
        iftWriteOPFDataSet(Z_out, argv[3]);
        iftDestroyDataSet(&Z_out);
    }

    for(ds=0;ds<nds;ds++)
        iftDestroyDataSet(&(Z_in[ds]));
    free(Z_in);

    return 0;
}
