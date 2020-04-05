#ifndef BEAT_INDEX_COMPONENT_H
#define BEAT_INDEX_COMPONENT_H


class BeatIndexComponent : public Component, private ChangeListener, public ChangeBroadcaster
{
public:
    BeatIndexComponent();

    void paint(Graphics& g) override;

    void paintIfNoBeatsEstimated(Graphics& g);

    void paintIfBeatsEstimated(Graphics& g);

    void changeListenerCallback(ChangeBroadcaster* source) override;   

    void mouseDown(const MouseEvent& event) override;

    void mouseUp(const MouseEvent& event) override;

    void mouseDrag(const MouseEvent& event) override;

    void drawGrid(Graphics& g);

    int clickPositionX { 0 };
    int clickPositionY { 0 };
    int clickPositionDragX;
    int clickPositionDragY;
    int clickPositionDifferenceX { 0 };
    int clickPositionDifferenceY { 0 };

    std::vector<double> beats;
    bool isMouseButtonDown { FALSE };

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BeatIndexComponent)
};


#endif