#ifndef ZOOM_THUMBNAIL_COMPONENT_H
#define ZOOM_THUMBNAIL_COMPONENT_H


#include "../JuceLibraryCode/JuceHeader.h"
#include "SimpleThumbnailComponent.h"
#include "BeatIndexComponent.h"


class ZoomThumbnailComponent : public Component, private ChangeListener, public ChangeBroadcaster
{
public:
    ZoomThumbnailComponent (int sourceSamplesPerThumbnailSample,
                              AudioFormatManager& formatManager,
                              AudioThumbnailCache& cache,
                              BeatIndexComponent& beatIndexComp,
                              SimpleThumbnailComponent& thumbnailComp);

    void setFile (const File& file);

    void paint (Graphics& g) override;

    void paintIfNoFileLoaded (Graphics& g);

    void paintIfFileLoaded (Graphics& g);

    void changeListenerCallback (ChangeBroadcaster* source) override;

    void mouseDown(const MouseEvent& event) override;

    void mouseDrag(const MouseEvent& event) override;

    int clickPositionX { 0 };
    int clickPositionY { 0 };
    int clickPositionDragX;
    int clickPositionDragY;
    int clickPositionDifferenceX { 0 };
    int clickPositionDifferenceY { 0 };
    int width;
    int frameWidthAtClick;
    int frameWidth;
    int frameXAtClick;
    double zoomFactor { 1.0 };

private:
    void thumbnailChanged();

    AudioThumbnail thumbnail;
    SimpleThumbnailComponent* pThumbnailComp;
    BeatIndexComponent* pBeatIndexComp;


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ZoomThumbnailComponent)
};


#endif