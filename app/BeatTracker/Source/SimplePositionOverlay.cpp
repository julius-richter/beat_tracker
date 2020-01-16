#include "../JuceLibraryCode/JuceHeader.h"
#include "SimplePositionOverlay.h"


SimplePositionOverlay::SimplePositionOverlay (AudioTransportSource& transportSourceToUse)
   : transportSource (transportSourceToUse)
{
    startTimer (40);
}

void SimplePositionOverlay::paint (Graphics& g) 
{
    auto duration = (float) transportSource.getLengthInSeconds();

    if (duration > 0.0)
    {
        auto audioPosition = (float) transportSource.getCurrentPosition();
        auto drawPosition = (audioPosition / duration) * getWidth();

        g.setColour (Colours::green);
        g.drawLine (drawPosition, 0.0f, drawPosition, (float) getHeight(), 2.0f);
    }
}

void SimplePositionOverlay::mouseDown (const MouseEvent& event) 
{
    auto duration = transportSource.getLengthInSeconds();

    if (duration > 0.0)
    {
        auto clickPosition = event.position.x;
        auto audioPosition = (clickPosition / getWidth()) * duration;

        transportSource.setPosition (audioPosition);
    }
}

void SimplePositionOverlay::timerCallback() 
{
    repaint();
}
