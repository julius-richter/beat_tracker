#ifndef Time_INDEX_COMPONENT_H
#define Time_INDEX_COMPONENT_H


#include "../JuceLibraryCode/JuceHeader.h"
#include "SimpleThumbnailComponent.h"


class TimeIndexComponent : public Component, private ChangeListener, public ChangeBroadcaster
{
public:
    TimeIndexComponent(SimpleThumbnailComponent* thumbnail);

    void paint(Graphics& g) override;

    void changeListenerCallback(ChangeBroadcaster* source) override;   

    void mouseDown(const MouseEvent& event) override;

    void mouseDrag(const MouseEvent& event) override;

    void drawGrid(Graphics& g);

    int clickPositionX { 0 };
    int clickPositionY { 0 };
    int clickPositionDragX;
    int clickPositionDragY;
    int clickPositionDifferenceX { 0 };
    int clickPositionDifferenceY { 0 };

private:
    SimpleThumbnailComponent* pThumbnail;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TimeIndexComponent)
};


#endif