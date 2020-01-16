#pragma once


#include <torch/script.h>

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
    MainContentComponent()
      : state (Stopped),
        thumbnailCache (5), 
        thumbnailComp (512, formatManager, thumbnailCache),
        positionOverlay (transportSource),
        beatActivationComp (&spectrogramComp.filteredSpectogram)
    {
        addAndMakeVisible (&openButton);
        openButton.setButtonText ("Open...");
        openButton.onClick = [this] { openButtonClicked(); };

        addAndMakeVisible (&playButton);
        playButton.setButtonText ("Play");
        playButton.onClick = [this] { playButtonClicked(); };
        playButton.setColour (TextButton::buttonColourId, Colours::green);
        playButton.setEnabled (false);

        addAndMakeVisible (&stopButton);
        stopButton.setButtonText ("Stop");
        stopButton.onClick = [this] { stopButtonClicked(); };
        stopButton.setColour (TextButton::buttonColourId, Colours::red);
        stopButton.setEnabled (false);

        addAndMakeVisible (&currentPositionLabel);
        currentPositionLabel.setText ("00:00:000", dontSendNotification);

        addAndMakeVisible (&textButton);
        textButton.setButtonText ("Text");
        textButton.onClick = [this] { textButtonClicked(); };  

        addAndMakeVisible (&processButton);
        processButton.setButtonText ("Process");
        processButton.onClick = [this] { processButtonClicked(); };  
        processButton.setEnabled (false);

        addAndMakeVisible (&thumbnailComp);
        addAndMakeVisible (&positionOverlay);     

        addAndMakeVisible (infoText);
        infoText.setColour (Label::backgroundColourId, Colours::black);

        addAndMakeVisible (&spectrogramComp);

        addAndMakeVisible (&beatActivationComp);

        setSize (600, 500);

        formatManager.registerBasicFormats();
        transportSource.addChangeListener (this);

        setAudioChannels (2, 2);
        startTimer (20);
    }

    ~MainContentComponent() override
    {
        shutdownAudio();
    }

    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override
    {
        transportSource.prepareToPlay (samplesPerBlockExpected, sampleRate);
    }

    void getNextAudioBlock (const AudioSourceChannelInfo& bufferToFill) override
    {
        if (readerSource.get() == nullptr)
        {
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        transportSource.getNextAudioBlock (bufferToFill);
    }

    void releaseResources() override
    {
        transportSource.releaseResources();
    }

    void resized() override
    {
        openButton          .setBounds (10, 10,  70, 20);
        playButton          .setBounds (10, 40,  70, 20);
        stopButton          .setBounds (10, 70,  70, 20);
        currentPositionLabel.setBounds (8, 95, getWidth() - 20, 20);
        
        Rectangle<int> thumbnailBounds (90, 10, getWidth() - 100, 100);   
        thumbnailComp.setBounds (thumbnailBounds);
        positionOverlay.setBounds (thumbnailBounds);

        processButton       .setBounds (10, 125, 70, 20);
        Rectangle<int> spectrogramBounds (90, 125, getWidth() - 100, 81);   
        spectrogramComp.setBounds (spectrogramBounds);

        Rectangle<int> beatActivationBounds (90, 221, getWidth() - 100, 100);   
        beatActivationComp.setBounds (beatActivationBounds);

        textButton          .setBounds (10, 387, 70, 20);
        infoText            .setBounds (90, 387, getWidth() - 100, 80);

    }

    void changeListenerCallback (ChangeBroadcaster* source) override
    {
        if (source == &transportSource)
        {
            if (transportSource.isPlaying())
                changeState (Playing);
            else if ((state == Stopping) || (state == Playing))
                changeState (Stopped);
            else if (Pausing == state)
                changeState (Paused);
        }
    }

    void timerCallback() override
    {
        RelativeTime position (transportSource.getCurrentPosition());

        auto minutes = ((int) position.inMinutes()) % 60;
        auto seconds = ((int) position.inSeconds()) % 60;
        auto millis  = ((int) position.inMilliseconds()) % 1000;

        auto positionString = String::formatted ("%02d:%02d:%03d", minutes, seconds, millis);

        currentPositionLabel.setText (positionString, dontSendNotification);
    }


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

    void changeState (TransportState newState)
    {
        if (state != newState)
        {
            state = newState;

            switch (state)
            {
                case Stopped:
                    playButton.setButtonText ("Play");
                    stopButton.setEnabled (false);
                    playButton.setEnabled (true);
                    transportSource.setPosition (0.0);
                    break;

                case Starting:
                    transportSource.start();
                    break;

                case Playing:
                    playButton.setButtonText ("Pause");
                    stopButton.setEnabled (true);
                    break;

                case Pausing:
                    transportSource.stop();
                    break;
 
                case Paused:
                    playButton.setButtonText ("Resume");
                    break;

                case Stopping:
                    transportSource.stop();
                    currentPositionLabel.setText ("00:00:000", dontSendNotification);
                    break;
            }
        }
    }

    void openButtonClicked()
    {
        FileChooser chooser ("Select a Wave file to play...",
                             {},
                             "*.wav;*.mp3");

        if (chooser.browseForFileToOpen())
        {
            auto file = chooser.getResult();
            auto* reader = formatManager.createReaderFor (file);

            if (reader != nullptr)
            {
                fileBuffer.setSize ((int) reader->numChannels, (int) reader->lengthInSamples); 
                reader->read (&fileBuffer, 0, (int) reader->lengthInSamples, 0, true, true); 
                std::unique_ptr<AudioFormatReaderSource> newSource (new AudioFormatReaderSource (reader, true));
                transportSource.setSource (newSource.get(), 0, nullptr, reader->sampleRate);
                playButton.setEnabled (true);
                processButton.setEnabled(true);
                thumbnailComp.setFile (file);
                readerSource.reset (newSource.release());   
                spectrogramComp.initialize(fileBuffer);    
            }
        }
    }

    void playButtonClicked()
    {
        if ((state == Stopped) || (state == Paused))
            changeState (Starting);
        else if (state == Playing)
            changeState (Pausing);
    }

    void stopButtonClicked()
    {
        if (state == Paused)
            changeState (Stopped);
        else
            changeState (Stopping);
    }

    void textButtonClicked()
    {

    }

    void processButtonClicked()
    {
        spectrogramComp.calculateSTFT();
        spectrogramComp.filterSpectogram();
        spectrogramComp.generateSpectrogramImage();
        infoText.setText (std::to_string(beatActivationComp.getValue()), dontSendNotification);   
        processButton.setEnabled (false);
    }

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
