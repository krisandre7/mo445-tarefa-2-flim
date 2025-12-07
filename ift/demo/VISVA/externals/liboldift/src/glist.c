/* Insert */
#include "glist.h"

GList* CreateGList(void (*freeF)(void**))
{
    GList* s = (GList*) malloc(sizeof(GList));

    s->freeFunction = freeF;
    s->first = s->last = NULL;

    return s;
}

void GListInsert (GList * qp, void* t)
{
   GListNode * n;
   if(qp == NULL) return;
   n = (GListNode *) calloc(1,sizeof(GListNode));

   /* Check if malloc succeeded */

   if (n == NULL) {
      fprintf(stderr, "Out of memory\n");
      exit(1);
   }

   /* Copy the data */

   n->data = t;
   n->next = NULL;

   /* If the GList was empty, just add this one element */

   if (qp->last == NULL)
   {
      qp->first = qp->last = n;
   }else {
      qp->last->next = n;
      qp->last = n;
   }
}

void DestroyGList(GList** s)
{
    GListNode* t;
    GListNode* temp;
    if(s == NULL || *s == NULL) return;

    t= (*s)->first;

    while( t != NULL)
    {
        if(t->data != NULL)
            (*s)->freeFunction(&(t->data));
        temp = t;
        t = t->next;
        free(temp);
    }

    free(*s);

    *s = NULL;
}
