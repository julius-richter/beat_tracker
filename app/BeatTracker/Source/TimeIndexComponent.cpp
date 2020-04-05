#include "TimeIndexComponent.h"


TimeIndexComponent::TimeIndexComponent(SimpleThumbnailComponent* thumbnail)
{
    pThumbnail = thumbnail;
}


void TimeIndexComponent::paint(Graphics& g)
{
    g.fillAll (Colour(0xff6c6c6c));

    double startTime = pThumbnail->startTime;
    double endTime = pThumbnail->endTime;
    double currentTimeSpan = endTime - startTime;

    float width = getLocalBounds().getWidth();

    int numTicks = 8;
    int interval = int (width / (double) numTicks);

    g.setColour(Colour(0xff000000));
    g.setFont(12.0f);

    for (int i = 0; i < numTicks; i++)
    {
        g.drawLine(i*interval+1, 1, i*interval+1, 6);
        g.drawLine (i*interval, 6, i*interval+3, 6);

        RelativeTime time(i * (int) (currentTimeSpan / (float) numTicks));
        auto minutes = ((int) time.inMinutes()) % 60;
        auto seconds = ((int) time.inSeconds()) % 60;
        auto timeString = String::formatted("%02d:%02d", minutes, seconds);
        g.drawSingleLineText(timeString, i*interval+6, 11);
    }
}


void TimeIndexComponent::changeListenerCallback (ChangeBroadcaster* source)
{

}

void TimeIndexComponent::drawGrid (Graphics& g)
{
    

}

void TimeIndexComponent::mouseDown (const MouseEvent& event) 
{   
    clickPositionX = event.position.x;
    clickPositionY = event.position.y;
    
    sendChangeMessage();
}

void TimeIndexComponent::mouseDrag (const MouseEvent& event) 
{   
    clickPositionDragX = event.position.x;
    clickPositionDragY = event.position.y;
    clickPositionDifferenceX = clickPositionDragX - clickPositionX;
    clickPositionDifferenceY = clickPositionDragY - clickPositionY;

    sendChangeMessage();
    repaint();
}


