#include "../JuceLibraryCode/JuceHeader.h"
#include "BeatGridOverlay.h"


BeatGridOverlay::BeatGridOverlay (AudioTransportSource& transportSourceToUse, Metronome& metronomeToUse,
    ZoomThumbnailComponent& zoomThumbnailComponent)
   : transportSource (transportSourceToUse), metronome(metronomeToUse)
{
    zoomThumbnailComponent.addChangeListener(this);
    pZoomThumbnailComponent = &zoomThumbnailComponent;
}

void BeatGridOverlay::paint (Graphics& g) 
{
    auto duration = (float) transportSource.getLengthInSeconds();

    if (duration > 0.0)
    {
        // int width = pZoomThumbnailComponent->width;
        // int frameWidth = pZoomThumbnailComponent->frameWidth;
        // int frameLeftX = pZoomThumbnailComponent->frameX;
        // int frameRightX = pZoomThumbnailComponent->frameX + frameWidth;
        // double startTime = ((double) frameLeftX / (double) width) * duration;
        // double endTime = ((double) frameRightX / (double) width) * duration;

        // for (auto &beat : beats) 
        // {  
        //     if (beat > startTime && beat < endTime)
        //     {
        //         auto drawPosition = (beat - startTime ) / (endTime - startTime) * getWidth();
        //         g.setColour (Colours::red);
        //         g.drawLine (drawPosition, 0.0f, drawPosition, (float) getHeight(), 1.0f);
        //     }
        // }
    }
}

void BeatGridOverlay::mouseDown (const MouseEvent& event) 
{
    auto duration = transportSource.getLengthInSeconds();

    if (duration > 0.0)
    {
        auto clickPosition = event.position.x;
        auto audioPosition = (clickPosition / getWidth()) * duration;

        transportSource.setPosition (audioPosition);
        metronome.currentPosition = audioPosition;
        metronome.determineBeatIndex();
    }
}


void BeatGridOverlay::changeListenerCallback (ChangeBroadcaster* source)
{
    if (source == pZoomThumbnailComponent)
    {
        repaint();
    }
}

