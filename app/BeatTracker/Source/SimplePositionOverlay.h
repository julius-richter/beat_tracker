#pragma once


class SimplePositionOverlay : public Component,
                              private Timer
{
public:
    SimplePositionOverlay (AudioTransportSource& transportSourceToUse);

    void paint (Graphics& g) override;

    void mouseDown (const MouseEvent& event) override;


private:
    void timerCallback() override;

    AudioTransportSource& transportSource;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SimplePositionOverlay)
};