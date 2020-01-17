#include "MainContentComponent.h"
#include <chrono> 


MainContentComponent::MainContentComponent()
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

MainContentComponent::~MainContentComponent() 
{
    shutdownAudio();
}

void MainContentComponent::prepareToPlay (int samplesPerBlockExpected, double sampleRate) 
{
    transportSource.prepareToPlay (samplesPerBlockExpected, sampleRate);
}

void MainContentComponent::getNextAudioBlock (const AudioSourceChannelInfo& bufferToFill) 
{
    if (readerSource.get() == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    transportSource.getNextAudioBlock (bufferToFill);
}

void MainContentComponent::releaseResources() 
{
    transportSource.releaseResources();
}

void MainContentComponent::resized() 
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

    textButton          .setBounds (10, 336, 70, 20);
    infoText            .setBounds (90, 336, getWidth() - 100, 160);

}

void MainContentComponent::changeListenerCallback (ChangeBroadcaster* source) 
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

void MainContentComponent::timerCallback() 
{
    RelativeTime position (transportSource.getCurrentPosition());

    auto minutes = ((int) position.inMinutes()) % 60;
    auto seconds = ((int) position.inSeconds()) % 60;
    auto millis  = ((int) position.inMilliseconds()) % 1000;

    auto positionString = String::formatted ("%02d:%02d:%03d", minutes, seconds, millis);

    currentPositionLabel.setText (positionString, dontSendNotification);
}


void MainContentComponent::changeState (TransportState newState)
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

void MainContentComponent::openButtonClicked()
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

void MainContentComponent::playButtonClicked()
{
    if ((state == Stopped) || (state == Paused))
        changeState (Starting);
    else if (state == Playing)
        changeState (Pausing);
}

void MainContentComponent::stopButtonClicked()
{
    if (state == Paused)
        changeState (Stopped);
    else
        changeState (Stopping);
}

void MainContentComponent::textButtonClicked()
{

}

void MainContentComponent::processButtonClicked()
{   
    processButton.setEnabled (false);

    auto start1 = std::chrono::high_resolution_clock::now(); 
    spectrogramComp.calculateSTFT();
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1).count()); 
 
    auto start2 = std::chrono::high_resolution_clock::now(); 
    spectrogramComp.filterSpectogram();
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2).count()); 

    spectrogramComp.generateSpectrogramImage();
    spectrogramComp.repaint();
    beatActivationComp.calculateBeatActivation();
    beatActivationComp.repaint();

    std::string text = "Calculate STFT: " + duration1 + "\nFilter Spectrogram: " + duration2 + "\n";

    infoText.setText (text + spectrogramComp.getText(), dontSendNotification);     
}

