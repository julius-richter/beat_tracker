#include "../JuceLibraryCode/JuceHeader.h"
#include "BeatIndexComponent.h"


BeatIndexComponent::BeatIndexComponent ()
{
    setMouseCursor(juce::MouseCursor::UpDownLeftRightResizeCursor);
}


void BeatIndexComponent::paint (Graphics& g)
{
    if (beats.size() > 0)
        paintIfBeatsEstimated (g);
    else
        paintIfNoBeatsEstimated (g);
}

void BeatIndexComponent::paintIfNoBeatsEstimated (Graphics& g)
{
    g.fillAll (Colour(0xff6c6c6c));
    drawGrid(g);
}

void BeatIndexComponent::paintIfBeatsEstimated (Graphics& g)
{
    g.fillAll(Colour(0xff6c6c6c));
}

void BeatIndexComponent::changeListenerCallback (ChangeBroadcaster* source)
{

}

void BeatIndexComponent::drawGrid (Graphics& g)
{
    float heigth = getLocalBounds().getHeight();
    float width = getLocalBounds().getWidth();

    int interval = 50;
    int numTicks = width / (float) interval;

    g.setColour(Colour(0xff000000));
    g.setFont(12.0f);

    for (int i = 0; i < numTicks+1; i++)
    {
        g.drawLine(i*interval+1, heigth-6, i*interval+1, heigth-1);
        g.drawLine (i*interval, heigth-6, i*interval+3, heigth-6);
        g.drawSingleLineText(std::to_string(i*4+1), i*interval+6, heigth-3);
    }

}

void BeatIndexComponent::mouseDown (const MouseEvent& event) 
{   
    isMouseButtonDown = TRUE;
    clickPositionX = event.position.x;
    clickPositionY = event.position.y;

    setMouseCursor(juce::MouseCursor::NoCursor); 
    sendChangeMessage();
}

void BeatIndexComponent::mouseUp (const MouseEvent& event) 
{   
    isMouseButtonDown = FALSE;
    setMouseCursor(juce::MouseCursor::UpDownLeftRightResizeCursor); 
    sendChangeMessage();
}

void BeatIndexComponent::mouseDrag (const MouseEvent& event) 
{   
    clickPositionDragX = event.position.x;
    clickPositionDragY = event.position.y;
    clickPositionDifferenceX = clickPositionDragX - clickPositionX;
    clickPositionDifferenceY = clickPositionDragY - clickPositionY;

    sendChangeMessage();
    repaint();
}


