

Scene*  msp_Rotate3(Scene *scn, 
		    double thx, double thy, double thz, // angles to rotate
		    int cx, int cy, int cz, // center of the rotation
		    double transx, double transy, double transz); // translation after rotation




// Parameters:
// in: Input scene
// mask_in: consider just the voxels within this mask (0=bg / 1=fg)
// input_ori: 1=sagittal / 2=axial / 3=coronal / 0=unknown
// quality: 1=best accuracy/slowest / 2=moderate / 3=medium accuracy/fastest
// detected_plane: when input_ori=0, it returns the detected orientation, 1=X 2=Y 3=Z
Plane* FindSymmetryPlane(Scene *in, Scene *mask_in, int input_ori, int quality, int *detected_plane);


// Parameters:
// scn: Input scene
// mask: consider just the voxels within this mask (0=bg / 1=fg). or NULL
// input_ori: 1=sagittal / 2=axial / 3=coronal / 0=unknown
// quality: 1=best accuracy/slowest / 2=moderate / 3=medium accuracy/fastest
Scene* MSP_Align(Scene *in, Scene *mask, int input_ori, int quality);

Scene*  msp_RotateToMSP(Scene *scn, Plane *p);
