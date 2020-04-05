#ifndef SIMPLE_POSITION_OVERLAY_H
#define SIMPLE_POSITION_OVERLAY_H

#include "Metronome.h"
#include "ZoomThumbnailComponent.h"
#include "SimpleThumbnailComponent.h"
#include "BeatIndexComponent.h"


class SimplePositionOverlay : public Component, private Timer, private ChangeListener
{
public:
    SimplePositionOverlay(AudioTransportSource& transportSourceToUse, Metronome &metronomeToUse,
    	ZoomThumbnailComponent& zoomThumbnailComponent, BeatIndexComponent& beatIndexComp,
        SimpleThumbnailComponent& simpleThumbnailComp);

    void paint(Graphics& g) override;

    void paintTime(Graphics& g);

    void paintIfZooming(Graphics& g);     

    void mouseDown(const MouseEvent& event) override;

    void changeListenerCallback (ChangeBroadcaster* source) override;

private:
    void timerCallback() override;

    AudioTransportSource& transportSource;
    Metronome& metronome;
    ZoomThumbnailComponent* pZoomThumbnailComponent;
    SimpleThumbnailComponent* pSimpleThumbnailComp;
    BeatIndexComponent* pBeatIndexComp;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SimplePositionOverlay)
};


#endif