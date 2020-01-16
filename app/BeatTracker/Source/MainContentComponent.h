#pragma once


#include <torch/script.h>

#include "../JuceLibraryCode/JuceHeader.h"
#include "utils.h"
#include "SimplePositionOverlay.h"
#include "SimpleThumbnailComponent.h"
#include "SpectrogramComponent.h"
#include "BeatActivationComponent.h"


class MainContentComponent: public AudioAppComponent,
                            public ChangeListener,
                            public Timer
{
public:
    MainContentComponent();

    ~MainContentComponent() override;

    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;

    void getNextAudioBlock (const AudioSourceChannelInfo& bufferToFill) override;

    void releaseResources() override;

    void resized() override;

    void changeListenerCallback (ChangeBroadcaster* source) override;

    void timerCallback() override;

private:
    enum TransportState
    {
        Stopped,
        Starting,
        Playing,
        Pausing,
        Paused,
        Stopping
    };

    void changeState (TransportState newState);

    void openButtonClicked();

    void playButtonClicked();

    void stopButtonClicked();

    void textButtonClicked();

    void processButtonClicked();

    TextButton openButton;
    TextButton playButton;
    TextButton stopButton;
    Label currentPositionLabel;
    TextButton textButton;
    TextButton processButton;
    Label infoText;

    AudioFormatManager formatManager;
    AudioSampleBuffer fileBuffer;
    std::unique_ptr<AudioFormatReaderSource> readerSource;
    AudioTransportSource transportSource;
    TransportState state;

    AudioThumbnailCache thumbnailCache;
    SimpleThumbnailComponent thumbnailComp;
    SimplePositionOverlay positionOverlay;

    SpectrogramComponent spectrogramComp;
    BeatActivationComponent beatActivationComp;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainContentComponent)
};
