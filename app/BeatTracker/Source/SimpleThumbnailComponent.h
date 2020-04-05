#ifndef SIMPLE_THUMBNAIL_COMPONENT_H
#define SIMPLE_THUMBNAIL_COMPONENT_H


#include "../JuceLibraryCode/JuceHeader.h"
#include "BeatIndexComponent.h"
#include "utils.h"


class SimpleThumbnailComponent : public Component, private ChangeListener, public ChangeBroadcaster
{
public:
    SimpleThumbnailComponent (int sourceSamplesPerThumbnailSample,
                              AudioFormatManager& formatManager,
                              AudioThumbnailCache& cache,
                              BeatIndexComponent& beatIndexComp);

    void setFile (const File& file);

    void paint (Graphics& g) override;

    void paintIfNoFileLoaded (Graphics& g);

    void paintIfFileLoaded (Graphics& g);

    void changeListenerCallback (ChangeBroadcaster* source) override;

    AudioThumbnail thumbnail;
    double startTime;
    double endTime;
    double newEndTime;
    double timeSpan;
    double timeDifference;
    double timeShift;
    
private:
    void thumbnailChanged();

    BeatIndexComponent* pBeatIndexComp;

    double positionDifferenceX;
    double positionDifferenceY;


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SimpleThumbnailComponent)
};

#endif