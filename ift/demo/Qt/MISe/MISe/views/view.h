#ifndef VIEW_H
#define VIEW_H

#include "global.h"
#include "marker.h"
#include <views/rendering/rendering.h>
#include <views/orientation.h>
#include <segmentation/forestwrapper.h>
#include <segmentation/segmentation.h>

class Segmentation;

class View
{
public:
    ~View();

    static View *instance();

    /*
     * Class Read/Write Methods
     */

    iftImage *getCurrentSlice( iftImage *img, int sliceType);
    int       getCurrentSliceNumber(int sliceType);

    iftImage *getCurrentAxialSlice();
    iftImage *getCurrentNormalizedAxialSlice();
    iftImage *getCurrentLabelAxialSlice();
    iftImage *getCurrentMarkerAxialSlice();
    iftImage *getCurrentBorderAxialSlice();
    iftImage *getCurrentGradientAxialSlice();

    iftImage *getCurrentCoronalSlice();
    iftImage *getCurrentNormalizedCoronalSlice();
    iftImage *getCurrentLabelCoronalSlice();
    iftImage *getCurrentMarkerCoronalSlice();
    iftImage *getCurrentBorderCoronalSlice();
    iftImage *getCurrentGradientCoronalSlice();

    iftImage *getCurrentSagittalSlice();
    iftImage *getCurrentNormalizedSagittalSlice();
    iftImage *getCurrentLabelSagittalSlice();
    iftImage *getCurrentMarkerSagittalSlice();
    iftImage *getCurrentBorderSagittalSlice();
    iftImage *getCurrentGradientSagittalSlice();

    iftImage *getCurrentRemovalMarkersAxialSlice();
    iftImage *getCurrentRemovalMarkersCoronalSlice();
    iftImage *getCurrentRemovalMarkersSagittalSlice();

    void setCurrentAxialSliceNum(int numSlice);
    int getCurrentAxialSliceNum();
    int getMaxAxialSlice();

    void setCurrentCoronalSliceNum(int numSlice);
    int getCurrentCoronalSliceNum();
    int getMaxCoronalSlice();

    void setCurrentSagittalSliceNum(int numSlice);
    int getCurrentSagittalSliceNum();
    int getMaxSagittalSlice();

    void createRendition();
    void destroyRendition();
    void performRendering();
    iftImage *getProjection();
    void incrementRenderingAngle(double tilt, double spin);
    void setRenderingMode(char mode);
    void getTiltSpinAngle(double *tilt, double *spin);
    void setRenditionLabel();

    QT_DEPRECATED
    void setImage(iftImage *i, QString filename);
    void setImage(iftImage *i);
    iftImage *getImage();
    iftImage *getNormalizedImage();
    bool isImageEmpty();
    void destroyImage();
    QString getFilename();

    void setImageSequence(iftImageSequence *imgseq, QString filename);
    iftImageSequence* getImageSequence();
    void displayImageSequenceTime(int time);
    iftImage *getThumbnail(int time);

    iftImageSequence* getLabelSequence();
    void displayLabelSequenceTime(int time);

    int getXsize();
    int getYsize();
    int getZsize();

    void setGradient(iftMImage *gradient, float adjRelRadius);
    iftMImage *getGradient();
    float getGradientAdjRelRadius();
    //TODO
    // const iftImage *getCalculatedGradient() const;

    void setLabel(iftImage *label, int index, iftColorTable *labelCtb = nullptr, bool render=true);
    iftImage *getLabel();
    iftImage *getBorder();
    bool isLabelEmpty();
    void destroyLabel();
    int getNumberOfLabels();
    iftImage *mergeLabels(iftImage *label1, iftImage *label2);

    iftColorTable *getColorTable();
    void addItemInColorTable();
    bool existColorTable();
    void setObjectColorInColorTable(int obj, iftColor color);
    void destroyColorTable();


    void setSegmentationMethod(Segmentation *method);
    Segmentation *currentSegmentationMethod();

    /*
     * Image Forest
     */

    ForestWrapper *getImageForest();
    void setImageForest(iftImageForest *forest);
    void setImageForest(iftDynTrees *forest);

    /*
     * IFT Methods
     */

    int iftGetImageIntensityAtVoxel(int x, int y, int z);
    int iftGetImageIntensityAtVoxel(iftVoxel u);

    /*
     * Public Variables
     */

    int maxValue, minValue;
    int maxNormalizedValue,minNormalizedValue;
    Rendering *rendition;
    Orientation *orientation;
    Marker *annotation;

private:

    /*
     * Variables Declarations
     */

    View();
    static View *_instance;

    void setLabel(iftImage *label, iftColorTable *labelCtb, bool render=true);

    iftImage *img, *normalized;
    iftImage *label,*border;
    iftImage *projection;

    iftImageSequence *imgseq;
    iftImageSequence *thumbnails;
    iftImageSequence *labelseq;
    int currentTime = 0;

    QString filename;

    ForestWrapper *forest;

    Segmentation *currentSegMethod;

    iftMImage *gradient;
    float      gradientAdjRelRadius = 1.0;
    iftImage  *gradientCache;

    iftColorTable *ctb;

    int currentAxialSliceNum, maxAxialSlice;
    int currentCoronalSliceNum, maxCoronalSlice;
    int currentSagittalSliceNum, maxSagittalSlice;

    int nlabels;

    friend class Segmentation;
};

#endif // VIEW_H
