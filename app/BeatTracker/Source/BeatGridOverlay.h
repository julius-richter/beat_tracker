#ifndef BEAT_GRID_OVERLAY_H
#define BEAT_GRID_OVERLAY_H


#include "Metronome.h"
#include "ZoomThumbnailComponent.h"


class BeatGridOverlay : public Component, private ChangeListener
{
public:
    BeatGridOverlay(AudioTransportSource& transportSourceToUse, Metronome& metronomeToUse,
    	ZoomThumbnailComponent& zoomThumbnailComponent);

    void paint(Graphics& g) override;

    void mouseDown(const MouseEvent& event) override;

    void changeListenerCallback (ChangeBroadcaster* source) override;

    std::vector<double> beats;

private:
    AudioTransportSource& transportSource;
    Metronome& metronome;
    ZoomThumbnailComponent* pZoomThumbnailComponent;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BeatGridOverlay)
};


#endif