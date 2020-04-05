#ifndef MAIN_CONTENT_COMPONENT_H
#define MAIN_CONTENT_COMPONENT_H

#include <torch/script.h>
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Functions.hpp"
#include <valarray>

#include "../JuceLibraryCode/JuceHeader.h"
#include "utils.h"
#include "LayoutComponent.h"
#include "Metronome.h"
#include "SimplePositionOverlay.h"
#include "SimpleThumbnailComponent.h"
#include "SpectrogramComponent.h"
#include "BeatActivationComponent.h"
#include "BeatGridOverlay.h"
#include "BeatIndexComponent.h"
#include "TemporalDecoding.h"
#include "TimeIndexComponent.h"
#include "ZoomThumbnailComponent.h"


class MainContentComponent: public AudioAppComponent, public ChangeListener, public Timer
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

    bool keyPressed(const KeyPress &key) override;

    bool keyStateChanged(bool isKeyDown) override;

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

    enum MetronomeState
    {
        On,
        Off
    };

    void changeState (TransportState newState);

    void openButtonClicked();

    void playButtonClicked();

    void stopButtonClicked();

    void textButtonClicked();

    void processButtonClicked();

    void metronomeButtonClicked();

    void beatGridButtonClicked();

    LayoutComponent layoutComp;

    TextButton openButton;
    TextButton playButton;
    TextButton stopButton;
    Label currentPositionLabel;
    TextButton processButton;
    Label infoText;
    TextButton metronomeButton;
    TextButton beatGridButton;

    AudioFormatManager formatManager;
    AudioSampleBuffer fileBuffer;
    std::unique_ptr<AudioFormatReaderSource> readerSource;
    AudioTransportSource transportSource;
    TransportState state;
    MetronomeState metronomeState;

    AudioThumbnailCache thumbnailCache;
    SimpleThumbnailComponent thumbnailComp;
    ZoomThumbnailComponent zoomThumbnailComp;

    SimplePositionOverlay positionOverlay;
    BeatGridOverlay beatGridOverlay1;
    BeatGridOverlay beatGridOverlay2;
    BeatGridOverlay beatGridOverlay3;

    BeatIndexComponent beatIndexComp;
    TimeIndexComponent timeIndexComp;

    SpectrogramComponent spectrogramComp;
    BeatActivationComponent beatActivationComp;

    std::vector<double> beats;
    double leftBoundTime;
    double rightBoundTime;

    Metronome metronome;

    MixerAudioSource mixerAudioSource;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainContentComponent)
};

#endif