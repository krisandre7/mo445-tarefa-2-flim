
#ifndef _REGISTRATION3_H
#define	_REGISTRATION3_H


typedef struct _point3{
float x,y,z;
}Point3;
 

int MyOtsu(Scene *scn);
Scene *NormalizeAccHist3(Scene *scn);
Point3 CalcCenterOfGravity(Set *Sj, Scene *scn) ;
void calcTransformation(float *theta, float RT[4][4], float xc, float yc, float zc);
void printMatrix(float t[4][4]);
double CriteryFunction_registration_gradient(Scene *B, Set *Sj, Point3 centro, Scene *movel, float *theta, float *delta, int i, int dir);
float * buscaMSGD(float **Delta, Scene *B, Set *Sj,Point3 centro,Scene *movel);

Scene *TextGradient(Scene *scn);
Scene *getWaterGray3(Scene *scn, float di, float factor);
Scene *getWGBorder(Scene *label);
Set *getSetBorder(Scene *scn);

void DefineDelta(float ** Delta);

void Register3(Scene *fixa, Scene *movel, float T[4][4], float **best_theta);


// These functions are used for registering left and right hemispheres
Scene *FlipSceneAxial(Scene *scn);
Scene *GetLeftAxialHalf(Scene *scn);
Scene *GetRightAxialHalf(Scene *scn);
Scene *SelfRegisterAxial(Scene *scn, float T[4][4], float **best_theta);
void MytransformScene(Scene *scn, float T[4][4], Scene *scn_out);

#endif


