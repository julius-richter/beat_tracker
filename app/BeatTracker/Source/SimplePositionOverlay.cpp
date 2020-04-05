#include "../JuceLibraryCode/JuceHeader.h"
#include "SimplePositionOverlay.h"


SimplePositionOverlay::SimplePositionOverlay (AudioTransportSource& transportSourceToUse, 
    Metronome& metronomeToUse, ZoomThumbnailComponent& zoomThumbnailComponent, BeatIndexComponent& beatIndexComp,
    SimpleThumbnailComponent& simpleThumbnailComp)
   : transportSource(transportSourceToUse), metronome(metronomeToUse)
{
    startTimer (10);
    zoomThumbnailComponent.addChangeListener(this);
    beatIndexComp.addChangeListener(this);
    pZoomThumbnailComponent = &zoomThumbnailComponent;
    pBeatIndexComp = &beatIndexComp;
    pSimpleThumbnailComp = &simpleThumbnailComp;
}

void SimplePositionOverlay::paint (Graphics& g) 
{
    g.setColour(Colour(0xff2d3342));

    if (pBeatIndexComp->isMouseButtonDown)
    {
        paintIfZooming(g);
    }
    else
        paintTime(g);
}

void SimplePositionOverlay::paintTime (Graphics& g) 
{
    auto duration = (float) transportSource.getLengthInSeconds();

    if (duration > 0.0)
    {
        double startTime = pSimpleThumbnailComp->startTime;
        double endTime = pSimpleThumbnailComp->endTime;
        
        auto audioPosition = (float) transportSource.getCurrentPosition();

        if (audioPosition > startTime && audioPosition < endTime)
        {
            auto drawPosition = ((audioPosition - startTime) / (endTime - startTime)) * getWidth();

            g.drawLine(drawPosition, 0.0f, drawPosition, (float) getHeight(), 2.0f);   
        }      
    }
}

void SimplePositionOverlay::paintIfZooming (Graphics& g) 
{
    double clickPositionX = pBeatIndexComp->clickPositionX;
    double clickPositionDifferenceX = pBeatIndexComp->clickPositionDifferenceX;

    double thumbnailWidth = pSimpleThumbnailComp->getLocalBounds().getWidth(); 
    double startTime = pSimpleThumbnailComp->startTime;
    double endTime = pSimpleThumbnailComp->endTime;

    double clickTime = clickPositionX / thumbnailWidth * (endTime-startTime) + startTime;

    auto drawPosition = ((clickTime - startTime) / (endTime - startTime) 
        * pSimpleThumbnailComp->getLocalBounds().getWidth()) + clickPositionDifferenceX;

    g.drawLine(drawPosition, 0.0f, drawPosition, (float) getHeight(), 2.0f);   

}

void SimplePositionOverlay::mouseDown (const MouseEvent& event) 
{
    auto duration = transportSource.getLengthInSeconds();

    if (duration > 0.0)
    {
        auto clickPosition = event.position.x;

        double startTime = pSimpleThumbnailComp->startTime;
        double endTime = pSimpleThumbnailComp->endTime;    
        auto audioPosition = startTime + (clickPosition / getWidth() * (endTime - startTime));

        transportSource.setPosition(audioPosition);
        metronome.currentPosition = audioPosition;
        metronome.determineBeatIndex();
    }
}

void SimplePositionOverlay::changeListenerCallback (ChangeBroadcaster* source)
{
    if (source == pZoomThumbnailComponent)
    {
        repaint();
    }
    if (source == pBeatIndexComp)
    {
        repaint();
    }
}

void SimplePositionOverlay::timerCallback() 
{
    repaint();
}
