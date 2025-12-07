

Report :: Report(){
  n  = 0;
  id = 0;
  nseeds = 0;
  label  = 0;
}

void Report :: addMarkerEvent(int id, int nseeds, int label){
  char *evt;

  if(n>=4096) return;

  if(this->id==id){
    this->nseeds += nseeds;
  }
  else if(this->id!=id){
    if(this->id!=0){
      evt = (char *)malloc(sizeof(char)*256);
      sprintf(evt,"Marker(id = %d, nseeds = %d, label = %d)\n",this->id,this->nseeds,this->label);
      eventLog[n] = evt;
      n++;
    }
    this->id = id;
    this->nseeds = nseeds;
    this->label = label;
  }
}

void Report :: addRunEvent(){
  Set  *S=NULL;
  char *evt;
  int mk;

  if(n>=4096) return;

  addMarkerEvent(0, 0, 0);

  S = AppData.delSet;
  while(S != NULL){
    mk = S->elem;
    evt = (char *)malloc(sizeof(char)*256);
    sprintf(evt,"Del Marker(id = %d)\n", GetMarkerID(mk));
    eventLog[n] = evt;
    n++;
    S = S->next;
  }
  evt = (char *)malloc(sizeof(char)*256);
  sprintf(evt,"Run\n");
  eventLog[n] = evt;
  n++;
}

void Report :: write(char *filename){
  FILE *fp;
  int i;

  fp = fopen(filename,"w");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }

  for(i=0; i<n; i++){
    fprintf(fp,"%s",eventLog[i]);
  }
  fclose(fp);
}

void Report :: clear(){
  for(int i=0; i<n; i++)
    free(eventLog[i]);
  n = 0;
  id = 0;
  nseeds = 0;
  label = 0;
}


