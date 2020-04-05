#include "MainContentComponent.h"


MainContentComponent::MainContentComponent()
  : state(Stopped),
    metronomeState(Off),
    thumbnailCache(5), 
    thumbnailComp(512, formatManager, thumbnailCache, beatIndexComp),
    zoomThumbnailComp(512, formatManager, thumbnailCache, beatIndexComp, thumbnailComp),
    positionOverlay(transportSource, metronome, zoomThumbnailComp, beatIndexComp, thumbnailComp),
    beatGridOverlay1(transportSource, metronome, zoomThumbnailComp),
    beatGridOverlay2(transportSource, metronome, zoomThumbnailComp),
    beatGridOverlay3(transportSource, metronome, zoomThumbnailComp),
    timeIndexComp(&thumbnailComp),
    beatActivationComp(&spectrogramComp.filteredSpectogram)
{
    addAndMakeVisible(&layoutComp);

    addAndMakeVisible(&openButton);
    openButton.setButtonText ("Open...");
    openButton.onClick = [this] { openButtonClicked(); };

    addAndMakeVisible(&playButton);
    playButton.setButtonText ("Play");
    playButton.onClick = [this] { playButtonClicked(); };
    playButton.setColour (TextButton::buttonColourId, Colours::green);
    playButton.setEnabled (false);

    addAndMakeVisible(&stopButton);
    stopButton.setButtonText ("Stop");
    stopButton.onClick = [this] { stopButtonClicked(); };
    stopButton.setColour (TextButton::buttonColourId, Colours::red);
    stopButton.setEnabled (false);

    addAndMakeVisible(&currentPositionLabel);
    currentPositionLabel.setText ("00:00:000", dontSendNotification); 
    currentPositionLabel.setColour(Label::textColourId, Colours::black);

    addAndMakeVisible(&processButton);
    processButton.setButtonText ("Process");
    processButton.onClick = [this] { processButtonClicked(); };  
    processButton.setEnabled (false);

    addAndMakeVisible(&metronomeButton);
    metronomeButton.setButtonText ("Metronome");
    metronomeButton.onClick = [this] { metronomeButtonClicked(); };  
    metronomeButton.setClickingTogglesState(true);
    metronomeButton.setToggleState(false, dontSendNotification);
    metronomeButton.setColour(TextButton::buttonColourId, Colour(0xff656c6e));
    metronomeButton.setColour(TextButton::buttonOnColourId, Colour(0xff1e3039));

    addAndMakeVisible(&beatGridButton);
    beatGridButton.setButtonText ("Grid");
    beatGridButton.onClick = [this] { beatGridButtonClicked(); };  
    beatGridButton.setClickingTogglesState(true);
    beatGridButton.setToggleState(false, dontSendNotification);
    beatGridButton.setColour(TextButton::buttonColourId, Colour(0xff656c6e));
    beatGridButton.setColour(TextButton::buttonOnColourId, Colour(0xff1e3039));

    addAndMakeVisible(&zoomThumbnailComp);
    addAndMakeVisible(&thumbnailComp);
    addAndMakeVisible(&beatGridOverlay1);
    addAndMakeVisible(&beatIndexComp);
    addAndMakeVisible(&timeIndexComp);
  
    addAndMakeVisible(infoText);
    infoText.setColour(Label::backgroundColourId, Colour(0xffe2e1e0));
    infoText.setColour(Label::textColourId, Colour(0xff2d3342));

    addAndMakeVisible(&spectrogramComp);
    addAndMakeVisible(&beatGridOverlay2);

    addAndMakeVisible(&beatActivationComp);
    addAndMakeVisible(&positionOverlay);
    addAndMakeVisible(&beatGridOverlay3);     

    setSize(600, 500);

    leftBoundTime = 0.0;
    rightBoundTime = 120.0;

    formatManager.registerBasicFormats();
    transportSource.addChangeListener(this);
    beatIndexComp.addChangeListener(this);
    thumbnailComp.addChangeListener(this);


    mixerAudioSource.addInputSource(&transportSource, false);
    mixerAudioSource.addInputSource(&metronome.transportSource, false);

    setAudioChannels(2, 2);
    startTimer(10);
}

MainContentComponent::~MainContentComponent() 
{
    shutdownAudio();
}

void MainContentComponent::prepareToPlay (int samplesPerBlockExpected, double sampleRate) 
{
    mixerAudioSource.prepareToPlay(samplesPerBlockExpected, sampleRate);
}

void MainContentComponent::getNextAudioBlock(const AudioSourceChannelInfo& bufferToFill) 
{
    if (readerSource.get() == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    mixerAudioSource.getNextAudioBlock(bufferToFill);

}

void MainContentComponent::releaseResources() 
{
    mixerAudioSource.releaseResources();
}

void MainContentComponent::resized() 
{
    int gap = 5;
    int headerHeight = 20;
    int beatIndexHeight = 25;
    int thumbnailHeight = 100;
    int zoomThumbnailHeight = 16;
    int spectrogramHeight = 81;
    int beatActivationHeight = 100;
    int buttonWidth = 70;
    int timeIndexHeight = 20;

    layoutComp.setBounds(0, 0, getWidth(), getHeight());
    
    Rectangle<int> currentPositionBounds(getWidth()/2 - buttonWidth/2, gap, buttonWidth, headerHeight);
    layoutComp.currentPositionRegion = currentPositionBounds;
    currentPositionLabel.setBounds(getWidth()/2 - buttonWidth/2, gap, buttonWidth, headerHeight);

    openButton.setBounds(gap, gap, buttonWidth, headerHeight);
    playButton.setBounds(gap + buttonWidth + gap, gap , buttonWidth, headerHeight);
    stopButton.setBounds(gap + buttonWidth + gap + buttonWidth + gap, gap , buttonWidth , headerHeight);
    
    Rectangle<int> playerRegionBounds(gap, gap + headerHeight + gap, getWidth() - 2*gap, 
        beatIndexHeight + zoomThumbnailHeight + gap + thumbnailHeight + gap + spectrogramHeight + gap 
        + beatActivationHeight + timeIndexHeight);
    layoutComp.playerRegion = playerRegionBounds;

    Rectangle<int> zoomThumbnailBounds(2*gap, gap + headerHeight + gap + gap, 
        getWidth() - 4*gap, zoomThumbnailHeight);   
    zoomThumbnailComp.setBounds(zoomThumbnailBounds);

    Rectangle<int> beatIndexBounds(2*gap, gap + headerHeight + gap + gap + zoomThumbnailHeight, 
        getWidth() - 4*gap, beatIndexHeight);   
    beatIndexComp.setBounds(beatIndexBounds);

    Rectangle<int> thumbnailBounds(2*gap, gap + headerHeight + gap + beatIndexHeight + zoomThumbnailHeight + gap, 
        getWidth() - 4*gap, thumbnailHeight);   
    thumbnailComp.setBounds(thumbnailBounds);
    beatGridOverlay1.setBounds(thumbnailBounds);

    Rectangle<int> spectrogramBounds(2*gap, gap + headerHeight + gap + beatIndexHeight + zoomThumbnailHeight + gap + 
        thumbnailHeight + gap, getWidth() - 4*gap, spectrogramHeight);   
    spectrogramComp.setBounds(spectrogramBounds);
    beatGridOverlay2.setBounds(spectrogramBounds);

    Rectangle<int> beatActivationBounds(2*gap, gap + headerHeight + gap + beatIndexHeight + zoomThumbnailHeight + gap + 
        thumbnailHeight + gap + spectrogramHeight + gap, getWidth() - 4*gap, beatActivationHeight);   
    beatActivationComp.setBounds(beatActivationBounds);
    beatGridOverlay3.setBounds(beatActivationBounds);

    Rectangle<int> timeIndexBounds(2*gap, gap + headerHeight + gap + beatIndexHeight + zoomThumbnailHeight + gap + 
        thumbnailHeight + gap + spectrogramHeight + gap + beatActivationHeight, getWidth() - 4*gap, timeIndexHeight);    
    timeIndexComp.setBounds(timeIndexBounds);

    Rectangle<int> positionOverlayBounds(2*gap, gap + headerHeight + gap + beatIndexHeight + zoomThumbnailHeight + gap, 
        getWidth() - 2 * gap, thumbnailHeight + gap + spectrogramHeight + gap + beatActivationHeight);
    positionOverlay.setBounds(positionOverlayBounds);

    processButton.setBounds(10, 405, 70, 20);
    metronomeButton.setBounds(10, 435, 70, 20);
    beatGridButton.setBounds(10, 465, 70, 20);

    infoText.setBounds(90, 400, getWidth() - 100, 80);

}

void MainContentComponent::changeListenerCallback(ChangeBroadcaster* source) 
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

    if (source == &beatIndexComp)
    {
        leftBoundTime = thumbnailComp.startTime;
        rightBoundTime = thumbnailComp.endTime;
    }
    if (source == &thumbnailComp)
        infoText.setText(std::to_string(zoomThumbnailComp.frameWidth), dontSendNotification); 

}

void MainContentComponent::timerCallback() 
{
    RelativeTime position(transportSource.getCurrentPosition());

    auto minutes = ((int) position.inMinutes()) % 60;
    auto seconds = ((int) position.inSeconds()) % 60;
    auto millis  = ((int) position.inMilliseconds()) % 1000;
    auto positionString = String::formatted("%02d:%02d:%03d", minutes, seconds, millis);

    currentPositionLabel.setText(positionString, dontSendNotification);

    if (metronomeState == MetronomeState::On)
    {
        if (beats.size() > 0)
        {
            if (transportSource.getCurrentPosition() > beats[metronome.beatIndex])
            {
                metronome.transportSource.stop();
                metronome.transportSource.setPosition(0.0);
                metronome.transportSource.start();
                metronome.beatIndex++;
            }  
        }       
    }
}


bool MainContentComponent::keyPressed(const KeyPress &key)
{
    if (key == KeyPress::spaceKey)
    {
        if ((state == Stopped) || (state == Paused))
            changeState(Starting);
        else if (state == Playing)
            changeState(Pausing);
        return true;
    }

    return false;
}

bool MainContentComponent::keyStateChanged(bool isKeyDown)
{
    return false;
}


void MainContentComponent::changeState(TransportState newState)
{
    if (state != newState)
    {
        state = newState;

        switch (state)
        {
            case Stopped:
                playButton.setButtonText("Play");
                stopButton.setEnabled(false);
                playButton.setEnabled(true);
                transportSource.setPosition(0.0);
                break;

            case Starting:
                transportSource.start();
                break;

            case Playing:
                playButton.setButtonText("Pause");
                stopButton.setEnabled(true);
                break;

            case Pausing:
                transportSource.stop();
                break;

            case Paused:
                playButton.setButtonText("Resume");
                break;

            case Stopping:
                transportSource.stop();

                currentPositionLabel.setText("00:00:000", dontSendNotification);
                break;
        }
    }
}

void MainContentComponent::openButtonClicked()
{
    FileChooser chooser("Select a Wave file to play...", {}, "*.wav;*.mp3");

    if (chooser.browseForFileToOpen())
    {
        auto file = chooser.getResult();
        auto* reader = formatManager.createReaderFor(file);

        if (reader != nullptr)
        {
            fileBuffer.setSize((int) reader->numChannels, (int) reader->lengthInSamples); 
            reader->read(&fileBuffer, 0, (int) reader->lengthInSamples, 0, true, true); 
            std::unique_ptr<AudioFormatReaderSource> newSource (new AudioFormatReaderSource (reader, true));
            transportSource.setSource(newSource.get(), 0, nullptr, reader->sampleRate);
            playButton.setEnabled(true);
            processButton.setEnabled(true); 
            zoomThumbnailComp.setFile(file);
            thumbnailComp.setFile(file);
            readerSource.reset(newSource.release());   
            spectrogramComp.initialize(fileBuffer);    
        }
    }
}

void MainContentComponent::playButtonClicked()
{
    if ((state == Stopped) || (state == Paused))
        changeState(Starting);
    else if (state == Playing)
        changeState(Pausing);
}

void MainContentComponent::stopButtonClicked()
{
    if (state == Paused)
        changeState(Stopped);
    else
        changeState(Stopping);
}


void MainContentComponent::metronomeButtonClicked()
{
    if (metronomeState == Off)
        metronomeState = MetronomeState::On;
    else
        metronomeState = MetronomeState::Off;
}

void MainContentComponent::beatGridButtonClicked()
{
    if (beatGridOverlay1.isVisible())
    {
        beatGridOverlay1.setVisible(false);
        beatGridOverlay2.setVisible(false);
        beatGridOverlay3.setVisible(false);
    }
    else
    {
        beatGridOverlay1.setVisible(true);
        beatGridOverlay2.setVisible(true);
        beatGridOverlay3.setVisible(true);
    }
}

void MainContentComponent::processButtonClicked()
{   
    processButton.setEnabled (false);

    spectrogramComp.calculateSTFT();
    spectrogramComp.filterSpectogram();
    spectrogramComp.generateSpectrogramImage();
    spectrogramComp.repaint();
    beatActivationComp.calculateBeatActivation();
    beatActivationComp.repaint();

    beats = activationsToBeats(beatActivationComp.activations);
    beatGridOverlay1.beats = beats;
    beatGridOverlay2.beats = beats;
    beatGridOverlay3.beats = beats;
    beatGridButton.setToggleState(true, dontSendNotification);
    metronome.beats = beats;
    metronome.determineBeatIndex();
    metronomeButton.setToggleState(false, dontSendNotification);

    auto numFrames = std::to_string(spectrogramComp.numFrames);
    auto duration = std::to_string(transportSource.getLengthInSeconds());
    auto numSamples = std::to_string(spectrogramComp.numSamples);
    auto activationSize = std::to_string(beatActivationComp.activations.size());

    std::string text = vectorToString(beats);
}

